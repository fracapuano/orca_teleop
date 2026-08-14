"""Clutch/engage state machine and the per-frame control pipeline.

Zero I/O: :meth:`ArmController.tick` is a pure function of (frame, events,
state) plus the controller's own latched origins, so every ordering below is
testable without a robot or a clock (CLAUDE.md hard rule 4). Anything the
pipeline needs from the outside world -- the measured flange pose for a first
latch -- arrives in :class:`RobotState` rather than being fetched here.

The rule this module exists to enforce (CLAUDE.md, "pose_epoch/stream_id
change"):

    origins latch lazily on the first frame while engaged, and are cleared on
    EVERY exit from engaged (clutch release, hold, reference change, fault).

The reason it is a state machine rather than a pair of ``if`` statements is that
``on_reference_change`` can arrive **late** -- attached to the first frame after
a dropout, i.e. *after* we have already re-latched against the new epoch. A
boolean "have origin" flag gets this wrong: it sees a valid origin and maps the
frame. Modelling ENGAGED_NO_ORIGIN as a distinct state makes the late callback
demote us back to it, so the next frame re-latches and no stale origin ever maps
a fresh frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Sequence

import numpy as np

from .config import ARMS, Arm, Config
from .mapping import ClutchMapper, WorkspaceClampResult, clamp_to_workspace
from .poses import Pose, PoseLike, as_pose
from .streamer import apply_step_clamp

__all__ = [
    "ArmController",
    "ArmState",
    "ArmStateMachine",
    "Diagnostics",
    "HoldReason",
    "RobotState",
    "TickEvents",
    "TickResult",
    "Transition",
]

#: Upstream's three hold reasons; all mean the same thing to us: hold position.
HoldReason = Literal["no_frames", "stale", "tracking_invalid"]
HOLD_REASONS: tuple[str, ...] = ("no_frames", "stale", "tracking_invalid")


class ArmState(Enum):
    """Where the clutch/origin logic currently stands.

    ENGAGED_NO_ORIGIN is deliberately distinct from ENGAGED: it is the state in
    which we are *allowed* to command the arm but have nothing to command it
    against yet. Collapsing the two is the bug described in the module docstring.
    """

    DISENGAGED = "disengaged"
    ENGAGED_NO_ORIGIN = "engaged_no_origin"
    ENGAGED = "engaged"
    HOLD = "hold"
    FAULT = "fault"

    @property
    def is_engaged(self) -> bool:
        """True for both engaged states -- "the clutch is held"."""
        return self in (ArmState.ENGAGED_NO_ORIGIN, ArmState.ENGAGED)

    @property
    def can_command(self) -> bool:
        """True only when a mapped target may be emitted."""
        return self is ArmState.ENGAGED


@dataclass(frozen=True)
class Transition:
    """One recorded state change, for tests and diagnostics."""

    source: ArmState
    target: ArmState
    trigger: str
    cleared_origins: bool = False
    latched: bool = False


class ArmStateMachine:
    """Clutch/engage/hold/fault states with origin-clearing as a side effect.

    ``on_clear_origins`` is called on every transition that invalidates the
    latched origins, and ``on_latch`` when a frame is consumed to establish
    them. Wiring the mapper through callbacks rather than reaching into it keeps
    this class testable on its own and makes "did the origins get cleared?" an
    observable event rather than something to infer.
    """

    def __init__(
        self,
        *,
        on_clear_origins: Callable[[], None] | None = None,
        on_latch: Callable[[], None] | None = None,
    ) -> None:
        self._state = ArmState.DISENGAGED
        self._clutch = False
        self._hold_reason: str | None = None
        self._fault_detail: str | None = None
        self._on_clear_origins = on_clear_origins
        self._on_latch = on_latch
        self.history: list[Transition] = []
        self.latch_count = 0
        self.clear_count = 0

    # -- observation -----------------------------------------------------
    @property
    def state(self) -> ArmState:
        return self._state

    @property
    def clutch_held(self) -> bool:
        return self._clutch

    @property
    def hold_reason(self) -> str | None:
        """Why we are in HOLD, or None. Cleared on leaving HOLD."""
        return self._hold_reason

    @property
    def fault_detail(self) -> str | None:
        return self._fault_detail

    # -- inputs ----------------------------------------------------------
    def set_clutch(self, held: bool) -> ArmState:
        """Engage or release. Releasing always clears origins."""
        held = bool(held)
        previous, self._clutch = self._clutch, held
        if self._state is ArmState.FAULT:
            return self._state  # only reset() leaves FAULT
        if held and not previous:
            return self._goto(ArmState.ENGAGED_NO_ORIGIN, "clutch_engage")
        if not held and previous:
            return self._goto(ArmState.DISENGAGED, "clutch_release")
        return self._state

    def on_frame(self) -> bool:
        """Feed one fresh frame. Returns whether this frame caused a latch.

        From HOLD with the clutch still held this re-enters ENGAGED_NO_ORIGIN and
        latches in the *same* step, so the first frame after a dropout is the one
        that re-establishes the origin -- there is no dead tick where a frame
        arrives and nothing is commanded but the state says ENGAGED.
        """
        if self._state is ArmState.FAULT:
            return False
        if self._state is ArmState.DISENGAGED:
            return False  # frames while released are simply not ours
        if self._state is ArmState.HOLD:
            if not self._clutch:
                self._goto(ArmState.DISENGAGED, "frame_while_released")
                return False
            self._goto(ArmState.ENGAGED_NO_ORIGIN, "frames_resumed")
        if self._state is ArmState.ENGAGED_NO_ORIGIN:
            self._goto(ArmState.ENGAGED, "lazy_latch", latched=True)
            return True
        return False  # already ENGAGED: map against the existing origin

    def on_hold(self, reason: str) -> ArmState:
        """Upstream said hold. Edge-triggered there, idempotent here."""
        if reason not in HOLD_REASONS:
            raise ValueError(f"unknown hold reason {reason!r}, expected one of {HOLD_REASONS}")
        if self._state in (ArmState.FAULT, ArmState.DISENGAGED):
            return self._state
        self._hold_reason = reason
        if self._state is ArmState.HOLD:
            return self._state
        return self._goto(ArmState.HOLD, f"hold:{reason}")

    def on_reference_change(self, stream_id: int = 0, pose_epoch: int = 0) -> ArmState:
        """The pose reference space changed; any latched origin is garbage.

        Safe to receive late and safe to receive twice. From ENGAGED this demotes
        to ENGAGED_NO_ORIGIN so the next frame re-latches; from HOLD it stays in
        HOLD but still clears, because HOLD's exit path latches anyway.
        """
        if self._state is ArmState.FAULT:
            return self._state
        trigger = f"reference_change:{stream_id}/{pose_epoch}"
        if self._state is ArmState.ENGAGED:
            return self._goto(ArmState.ENGAGED_NO_ORIGIN, trigger)
        # Already originless (or disengaged): clear anyway, cheaply and loudly.
        self._clear_origins()
        self.history.append(
            Transition(self._state, self._state, trigger, cleared_origins=True)
        )
        return self._state

    def on_fault(self, detail: str = "") -> ArmState:
        """Latch a fault. Only :meth:`reset` leaves FAULT."""
        self._fault_detail = detail or "unspecified"
        if self._state is ArmState.FAULT:
            return self._state
        return self._goto(ArmState.FAULT, f"fault:{self._fault_detail}")

    def reset(self) -> ArmState:
        """Return to DISENGAGED from anywhere, clearing everything.

        Also drops the clutch: recovering from a fault must not silently
        re-engage because the operator happened to still be holding the pedal.
        """
        self._clutch = False
        self._fault_detail = None
        return self._goto(ArmState.DISENGAGED, "reset")

    # -- internals -------------------------------------------------------
    def _goto(self, target: ArmState, trigger: str, *, latched: bool = False) -> ArmState:
        source = self._state
        # Entering ENGAGED_NO_ORIGIN from anywhere clears origins; so does every
        # exit from an engaged state. Both reduce to: clear unless we are
        # latching, and clear whenever the origin could not survive the move.
        clears = target is ArmState.ENGAGED_NO_ORIGIN or (
            source.is_engaged and target is not ArmState.ENGAGED
        )
        if target is not ArmState.HOLD:
            self._hold_reason = None
        self._state = target
        if clears:
            self._clear_origins()
        if latched:
            self.latch_count += 1
            if self._on_latch is not None:
                self._on_latch()
        self.history.append(
            Transition(source, target, trigger, cleared_origins=clears, latched=latched)
        )
        return target

    def _clear_origins(self) -> None:
        self.clear_count += 1
        if self._on_clear_origins is not None:
            self._on_clear_origins()


# -- the per-frame pipeline ---------------------------------------------
@dataclass(frozen=True)
class TickEvents:
    """Everything that can arrive alongside (or instead of) a frame.

    All optional; a tick with none of them set is the ordinary streaming case.
    """

    clutch: bool | None = None
    hold_reason: str | None = None
    reference_change: tuple[int, int] | None = None
    fault: str | None = None
    reset: bool = False

    def __post_init__(self) -> None:
        if self.hold_reason is not None and self.hold_reason not in HOLD_REASONS:
            raise ValueError(
                f"unknown hold reason {self.hold_reason!r}, expected one of {HOLD_REASONS}"
            )


@dataclass(frozen=True)
class RobotState:
    """What the pipeline needs from the robot, passed in rather than fetched.

    ``measured_flange`` is only consulted for the very first latch. Every later
    latch prefers the last COMMANDED target (CLAUDE.md), because the arm is
    still converging on that command and latching the measurement would bake its
    tracking error in as a step.
    """

    arm: Arm
    measured_flange: Pose

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}, expected one of {ARMS}")


@dataclass(frozen=True)
class Diagnostics:
    """Why this tick did what it did."""

    state: ArmState
    latched_this_tick: bool = False
    holding: bool = False
    hold_reason: str | None = None
    reason: str = ""
    workspace_clamped_axes: tuple[str, ...] = ()
    step_clamped_lin: bool = False
    step_clamped_ang: bool = False
    translation_frame: str = "body"
    scale: float = 1.0
    origin_source: str | None = None
    recv_monotonic_ns: int | None = None

    @property
    def workspace_clamped(self) -> bool:
        return bool(self.workspace_clamped_axes)

    @property
    def step_clamped(self) -> bool:
        return self.step_clamped_lin or self.step_clamped_ang


@dataclass(frozen=True)
class TickResult:
    """``target`` is None when the arm should hold its last command."""

    target: Pose | None
    diagnostics: Diagnostics

    @property
    def hold_last(self) -> bool:
        return self.target is None


class ArmController:
    """Wires the state machine, the mapper and both clamps into one tick.

    Documented order, applied every tick:

      1. events -> state machine, in severity order
         ``reset -> fault -> reference_change -> hold -> clutch``;
      2. bail out holding the last command unless the state can command -- and
         an explicit hold event this tick holds even if a frame came with it;
      3. lazy-latch the origin pair if we are ENGAGED_NO_ORIGIN and have a frame;
      4. map (this is where scale hits the delta translation, and the tool offset
         converts palm -> flange);
      5. workspace clamp -- position into the arm's box minus the margin;
      6. step clamp -- per-tick velocity ceiling, against the last command;
      7. record the result as the last command.

    Step 6 reuses :func:`tron_arm.streamer.apply_step_clamp` rather than
    reimplementing it, so the controller and the streamer cannot drift apart on
    what "too fast" means.
    """

    def __init__(
        self,
        config: Config,
        *,
        tool_offset: PoseLike | np.ndarray | None = None,
        mapper: ClutchMapper | None = None,
    ) -> None:
        self._config = config
        self._mapper = mapper if mapper is not None else ClutchMapper(config, tool_offset=tool_offset)
        self._machine = ArmStateMachine(on_clear_origins=self._mapper.clear)
        self._last_commanded: Pose | None = None
        self._max_lin, self._max_ang = config.max_step

    # -- observation -----------------------------------------------------
    @property
    def state(self) -> ArmState:
        return self._machine.state

    @property
    def machine(self) -> ArmStateMachine:
        return self._machine

    @property
    def mapper(self) -> ClutchMapper:
        return self._mapper

    @property
    def last_commanded(self) -> Pose | None:
        return self._last_commanded

    @property
    def max_step(self) -> tuple[float, float]:
        return self._max_lin, self._max_ang

    # -- the tick --------------------------------------------------------
    def tick(
        self,
        frame: Any | None,
        events: TickEvents | None,
        state: RobotState,
    ) -> TickResult:
        """Advance one frame. See the class docstring for the order.

        ``frame`` is anything exposing ``.wrist`` (an orca_teleop ``TeleopFrame``)
        or a bare pose; ``None`` means no fresh frame this tick.
        """
        events = events or TickEvents()
        self._apply_events(events)

        recv_ns = _recv_monotonic_ns(frame)
        machine = self._machine

        if machine.state is ArmState.FAULT:
            return self._hold("fault", recv_ns)
        if events.hold_reason is not None:
            # An explicit hold THIS tick outranks any frame arriving with it.
            # Upstream never emits both (arm_worker either dispatches or holds),
            # but if the two are ever combined, "hold position" has to win --
            # otherwise the frame would resume streaming through the hold.
            # A frame arriving on a LATER tick while still in HOLD is the
            # resume path and does re-latch; that is handled below.
            return self._hold(f"hold:{events.hold_reason}", recv_ns)
        if frame is None:
            return self._hold("no_frame", recv_ns)

        latched_this_tick = machine.on_frame()
        if not machine.state.can_command:
            return self._hold(f"state={machine.state.value}", recv_ns)

        origin_source: str | None = None
        if latched_this_tick:
            origin_source = "last_commanded" if self._last_commanded is not None else "measured"
            robot0 = self._last_commanded if self._last_commanded is not None else state.measured_flange
            self._mapper.latch(as_pose(_wrist_of(frame)).matrix, robot0)

        # 4 -> 5: map (scale applied to the delta inside), then workspace clamp.
        clamp: WorkspaceClampResult = self._mapper.map_clamped(_wrist_of(frame), state.arm)

        # 6: step clamp against the last command, reusing the streamer's maths.
        target, lin_hit, ang_hit = apply_step_clamp(
            self._last_commanded, clamp.pose, self._max_lin, self._max_ang
        )

        self._last_commanded = target  # 7
        return TickResult(
            target,
            Diagnostics(
                state=machine.state,
                latched_this_tick=latched_this_tick,
                reason="mapped",
                workspace_clamped_axes=clamp.clamped_axis_names,
                step_clamped_lin=lin_hit,
                step_clamped_ang=ang_hit,
                translation_frame=self._mapper.translation_frame,
                scale=self._mapper.scale,
                origin_source=origin_source,
                recv_monotonic_ns=recv_ns,
            ),
        )

    def _apply_events(self, events: TickEvents) -> None:
        """Severity order: reset, then fault, then the origin-invalidating ones.

        Fault after reset so a tick carrying both ends in FAULT rather than
        clearing it -- recovering from a fault should take a deliberate,
        fault-free tick.
        """
        if events.reset:
            self._machine.reset()
        if events.fault is not None:
            self._machine.on_fault(events.fault)
        if events.reference_change is not None:
            self._machine.on_reference_change(*events.reference_change)
        if events.hold_reason is not None:
            self._machine.on_hold(events.hold_reason)
        if events.clutch is not None:
            self._machine.set_clutch(events.clutch)

    def _hold(self, reason: str, recv_ns: int | None) -> TickResult:
        machine = self._machine
        return TickResult(
            None,
            Diagnostics(
                state=machine.state,
                holding=True,
                hold_reason=machine.hold_reason,
                reason=reason,
                translation_frame=self._mapper.translation_frame,
                scale=self._mapper.scale,
                recv_monotonic_ns=recv_ns,
            ),
        )


def _wrist_of(frame: Any) -> PoseLike:
    """Accept a TeleopFrame-shaped object or a bare pose."""
    wrist = getattr(frame, "wrist", None)
    return frame if wrist is None else wrist


def _recv_monotonic_ns(frame: Any) -> int | None:
    """The only clock allowed in control maths (hard rule 2).

    ``timestamp_ns`` is deliberately never read: it is a publisher wall clock
    that can run backwards.
    """
    if frame is None:
        return None
    value = getattr(frame, "recv_monotonic_ns", None)
    return int(value) if value is not None else None
