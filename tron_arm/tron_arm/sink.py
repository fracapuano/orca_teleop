"""``TronArmSink``: the orca_teleop ``ArmSink`` that drives a TRON 2.

Structurally an ``ArmSink`` (CLAUDE.md "Upstream contract", verified against
``orca_teleop`` @ ``feat/wrist-pose`` ``1b5c85e``): five methods, ``dispatch``
called on the arm worker thread at ingress rate.

**The threading split is the whole design.** ``dispatch`` runs on upstream's arm
worker thread and must not block: a slow sink stalls the thread that owns the
staleness deadline. So ``dispatch`` does pure maths -- state machine, mapper,
clamps -- and hands the result to :class:`~tron_arm.streamer.PoseStreamer` as a
plain in-memory submit. Every byte of network I/O happens on the asyncio loop
thread the streamer runs on. ``dispatch`` never awaits, never locks on the
socket, and never touches the event loop except through
``call_soon_threadsafe``.

This module does not subclass ``orca_teleop.arm.ArmSink`` at import time: this
package must stay importable without orca_teleop (the mock and CLI run
standalone). It satisfies the ABC structurally, and
:func:`assert_matches_upstream_abc` checks that claim against the real class when
a checkout is present -- so the duck typing is verified, not assumed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from .arm_state import ArmController, ArmState, RobotState, TickEvents
from .config import ARMS, Arm, Config
from .mapping import check_inside_workspace
from .poses import Pose, PoseLike, as_pose
from .session import SessionLogger, collect_meta
from .streamer import PoseStreamer, TickInfo
from .tron2_client import NOTIFY_SERVOP, NotifyRecord, Tron2Client, Tron2Error

logger = logging.getLogger(__name__)

__all__ = [
    "ArmDiagnostics",
    "SinkStats",
    "TronArmSink",
    "assert_matches_upstream_abc",
]

#: Upstream's hold reasons. Repeated rather than imported: see module docstring.
HOLD_REASONS = ("no_frames", "stale", "tracking_invalid")

#: How long close() keeps commanding the frozen pose before dropping the socket.
FREEZE_BEFORE_CLOSE_S = 1.0


@dataclass
class ArmDiagnostics:
    """Per-arm snapshot for the status line."""

    arm: str
    state: str = ArmState.DISENGAGED.value
    frames: int = 0
    latches: int = 0
    workspace_clamps: int = 0
    step_clamps: int = 0
    holds: int = 0
    reference_changes: int = 0
    last_age_s: float | None = None
    last_hold_reason: str | None = None
    orientation_frozen: bool = False


@dataclass
class SinkStats:
    """Aggregate counters, safe to read from another thread (ints and floats)."""

    dispatches: int = 0
    dispatch_ns: list[int] = field(default_factory=list, repr=False)
    dropped_no_arm_pose: int = 0
    unknown_handedness: int = 0
    faults: int = 0

    def record_dispatch(self, elapsed_ns: int) -> None:
        self.dispatches += 1
        # Bounded: this list is appended to at ingress rate for the whole run.
        if len(self.dispatch_ns) < 100_000:
            self.dispatch_ns.append(elapsed_ns)

    def dispatch_percentile_ms(self, percentile: float = 95.0) -> float:
        if not self.dispatch_ns:
            return 0.0
        return float(np.percentile(self.dispatch_ns, percentile)) / 1e6


class TronArmSink:
    """Drives a TRON 2 from operator wrist poses.

    Args:
        config: validated configuration.
        clutch: something with an ``engaged`` property. ``None`` means always
            engaged -- convenient for tests and headless runs, and the reason
            :mod:`tron_arm.clutch` exists for everything else.
        loop: the asyncio loop the client and streamer live on. Created and owned
            here unless supplied.
        tool_offset: optional ``T_FH`` flange -> palm.
    """

    def __init__(
        self,
        config: Config,
        *,
        clutch: Any | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        tool_offset: PoseLike | np.ndarray | None = None,
        url: str | None = None,
        session: SessionLogger | None = None,
    ) -> None:
        self._config = config
        self._clutch = clutch
        self._url = url
        self._tool_offset = tool_offset
        self._owns_loop = loop is None
        self._loop = loop
        self._loop_thread: threading.Thread | None = None
        self._client: Tron2Client | None = None
        self._streamer: PoseStreamer | None = None
        self._streamer_stop: asyncio.Event | None = None
        self._streamer_task: asyncio.Task[None] | None = None

        self._controllers: dict[str, ArmController] = {
            arm: ArmController(config, tool_offset=tool_offset) for arm in ARMS
        }
        self._measured: dict[str, Pose | None] = {arm: None for arm in ARMS}
        self._diagnostics: dict[str, ArmDiagnostics] = {
            arm: ArmDiagnostics(arm=arm) for arm in ARMS
        }
        self.stats = SinkStats()
        self.last_notify: NotifyRecord | None = None
        self._orientation_frozen = False
        self._forced_hold = False
        self._closed = False
        # A disabled logger accepts every call and does nothing, so the hot path
        # needs no `if self._session is not None` branches.
        self._session = session if session is not None else SessionLogger(enabled=False)

    # -- observation (status line reads these from another thread) --------
    @property
    def config(self) -> Config:
        return self._config

    @property
    def client(self) -> Tron2Client | None:
        return self._client

    @property
    def streamer(self) -> PoseStreamer | None:
        return self._streamer

    @property
    def controllers(self) -> Mapping[str, ArmController]:
        return dict(self._controllers)

    def diagnostics(self, arm: str) -> ArmDiagnostics:
        return self._diagnostics[arm]

    @property
    def session(self) -> SessionLogger:
        return self._session

    @property
    def orientation_frozen(self) -> bool:
        """Position-only mode: command the latched orientation, ignore the wrist's."""
        return self._orientation_frozen

    def set_orientation_frozen(self, frozen: bool) -> None:
        self._orientation_frozen = bool(frozen)
        for arm in ARMS:
            self._diagnostics[arm].orientation_frozen = self._orientation_frozen

    def force_hold(self, held: bool = True) -> None:
        """Operator panic key: hold both arms regardless of the clutch."""
        self._forced_hold = bool(held)
        if held:
            for arm in ARMS:
                self._controllers[arm].tick(None, TickEvents(hold_reason="no_frames"),
                                            self._robot_state(arm))
                self._diagnostics[arm].state = self._controllers[arm].state.value

    # -- ArmSink: connect -------------------------------------------------
    def connect(self) -> None:
        """Open the client, learn accid, freeze both arms, start the streamer.

        Raising here aborts pipeline startup, which is the correct outcome: a
        sink that cannot reach the robot must not let the operator start moving.
        """
        if self._owns_loop:
            self._loop = asyncio.new_event_loop()
            ready = threading.Event()
            self._loop_thread = threading.Thread(
                target=self._run_loop, args=(ready,), name="tron-arm-loop", daemon=True
            )
            self._loop_thread.start()
            if not ready.wait(timeout=5.0):
                raise Tron2Error("asyncio loop thread failed to start")
        assert self._loop is not None
        self._call(self._async_connect(), timeout=self._config.robot.connect_timeout_s + 10.0)

    async def _async_connect(self) -> None:
        # Checked before opening a socket: it needs no robot, and failing fast
        # is better than discovering it after priming.
        #
        # LimX: "the API does not support sending commands to only one arm
        # independently; commands must be sent to both arms simultaneously."
        # send_both=False therefore cannot work on hardware. It stays valid for
        # the mock, which can be told to accept single-side messages.
        url = self._url or self._config.robot.url
        if not self._config.servop.send_both and not _is_loopback(url):
            raise Tron2Error(
                f"servop.send_both is false, but this robot ({url}) requires both "
                "left_pos and right_pos in every request_servop -- LimX confirmed the API "
                "does not accept single-arm commands. Set servop.send_both: true. "
                "(To move one arm only, leave the other at its current pose; that is what "
                "send_both already does.)"
            )

        client = Tron2Client(self._config, url=self._url)
        await client.connect()
        client.on_notify("*", self._on_notify)
        # Freeze BOTH arms at their current poses before anything can move: the
        # idle arm needs a frozen target for servop.send_both, and the engaged
        # arm needs an origin that is where the robot actually is.
        poses = await client.prime_frozen_poses()

        # Refuse to stream if either arm is parked outside its box. send_both
        # commands the frozen pose to the IDLE arm every tick, and that pose
        # goes through the same clamp -- so an out-of-box rest pose moves BOTH
        # arms the moment streaming starts, before the operator touches
        # anything. Measured on hardware: 29 cm into the stand.
        violations = [v for arm, pose in poses.items()
                      if (v := check_inside_workspace(self._config, arm, pose)) is not None]
        if violations:
            await client.close()
            raise Tron2Error(
                "refusing to stream: "
                + "; ".join(v.describe() for v in violations)
                + ". The workspace box and the robot disagree about where the arm is. "
                  "Either correct workspace: in the config for this robot, or movej the "
                  "arm inside the box first. Nothing has been commanded."
            )

        for arm, pose in poses.items():
            self._measured[arm] = pose
        self._client = client

        streamer = PoseStreamer(self._config, client.send_servop,
                                on_tick=self._on_streamer_tick,
                                on_fault=self._on_stream_fault)
        streamer.seed_all(poses)  # first tick is velocity-limited, not a jump
        for arm, pose in poses.items():
            # Seed the tracks too, so the streamer commands the frozen pose from
            # tick one rather than idling until the first operator frame.
            streamer.submit(arm, pose, time.monotonic_ns())
        self._streamer_stop = asyncio.Event()
        self._streamer_task = await streamer.start(stop=self._streamer_stop)
        self._streamer = streamer
        self._session.write_meta(
            collect_meta(self._config, session_id=self._session.session_id,
                         accid=client.accid, url=client.url,
                         firmware_version=self._session.firmware_version)
        )
        logger.info(
            "TRON arm sink connected to %s (accid=%s, format=%s, send_both=%s, %.0f Hz)",
            client.url, client.accid, self._config.servop.format,
            self._config.servop.send_both, self._config.servop.rate_hz,
        )

    # -- ArmSink: dispatch (HOT PATH -- no I/O) ---------------------------
    def dispatch(self, frame: Any) -> None:
        """Command the arm from one fresh operator frame.

        Runs on upstream's arm worker thread. Pure maths plus one in-memory
        submit; the socket write happens on the streamer's loop. Anything that
        could block belongs in :meth:`connect` or :meth:`close`.
        """
        started = time.monotonic_ns()
        try:
            self._dispatch_inner(frame, started)
        finally:
            self.stats.record_dispatch(time.monotonic_ns() - started)

    def _dispatch_inner(self, frame: Any, t_dispatch_ns: int) -> None:
        arm = self._arm_of(frame)
        if arm is None:
            self.stats.unknown_handedness += 1
            return
        wrist = getattr(frame, "wrist", None)
        if wrist is None:
            # An older publisher with --no-arm-pose: there is no wrist pose to
            # map. Hold rather than crash; the arm keeps its last command.
            self.stats.dropped_no_arm_pose += 1
            self._hold_arm(arm, "no_frames")
            self._session.event("dispatch", arm=arm, t_dispatch=t_dispatch_ns,
                                recv_monotonic_ns=_recv_ns(frame), reason="no_arm_pose",
                                state=self._controllers[arm].state.value, commanded=False)
            return

        controller = self._controllers[arm]
        engaged = self._engaged()
        events = TickEvents(clutch=engaged)
        result = controller.tick(frame, events, self._robot_state(arm))

        diagnostics = self._diagnostics[arm]
        diagnostics.state = result.diagnostics.state.value
        diagnostics.frames += 1
        diagnostics.last_age_s = _age_s(frame)
        if result.diagnostics.latched_this_tick:
            diagnostics.latches += 1
        if result.diagnostics.workspace_clamped:
            diagnostics.workspace_clamps += 1
        if result.diagnostics.step_clamped:
            diagnostics.step_clamps += 1

        record = {
            "arm": arm,
            "t_dispatch": t_dispatch_ns,
            "recv_monotonic_ns": _recv_ns(frame),
            "state": result.diagnostics.state.value,
            "latched": result.diagnostics.latched_this_tick,
            "engaged": engaged,
            "age_s": diagnostics.last_age_s,
            "ws_clamped_axes": list(result.diagnostics.workspace_clamped_axes),
            "step_clamped_lin": result.diagnostics.step_clamped_lin,
            "step_clamped_ang": result.diagnostics.step_clamped_ang,
            "stream_id": getattr(frame, "stream_id", None),
            "pose_epoch": getattr(frame, "pose_epoch", None),
        }
        if result.target is None:
            self._session.event("dispatch", commanded=False,
                                reason=result.diagnostics.reason, **record)
            return  # hold: the streamer keeps commanding the last target
        target = result.target
        if self._orientation_frozen:
            last = controller.last_commanded
            frozen_q = (self._measured[arm] or target).orientation_wxyz
            target = Pose(target.position_m, frozen_q)
        self._session.event("dispatch", commanded=True, reason="mapped", **record)
        streamer = self._streamer
        if streamer is not None:
            # The meta rides with the sample so the tick record can name the
            # dispatch it came from -- the join is exact, not inferred.
            streamer.submit(arm, target, _recv_ns(frame),
                            meta={"t_dispatch": t_dispatch_ns, "arm": arm})

    # -- ArmSink: the callbacks ------------------------------------------
    def on_hold(self, reason: str) -> None:
        """Frames stopped or arrived unusable. Hold BOTH arms.

        Upstream's hold is not per-arm -- it means the operator stream itself is
        gone, so holding only the arm that happened to be moving would leave the
        other one live against a stale origin.
        """
        if reason not in HOLD_REASONS:
            logger.warning("unknown hold reason %r; holding anyway", reason)
        for arm in ARMS:
            self._hold_arm(arm, reason)
        logger.info("hold(%s): both arms frozen at their last command", reason)

    def on_reference_change(self, stream_id: int, pose_epoch: int) -> None:
        """The pose reference space changed; latched origins are garbage.

        May arrive LATE -- after we have already re-latched against the new
        epoch. The state machine handles that by demoting to ENGAGED_NO_ORIGIN,
        so the next frame re-latches; the double latch is harmless and the target
        stream stays continuous.
        """
        for arm in ARMS:
            controller = self._controllers[arm]
            controller.tick(None, TickEvents(reference_change=(stream_id, pose_epoch)),
                            self._robot_state(arm))
            self._diagnostics[arm].reference_changes += 1
            self._diagnostics[arm].state = controller.state.value
        self._session.event("reference_change", stream_id=stream_id, pose_epoch=pose_epoch)
        logger.info("reference change (stream=%d, epoch=%d): origins cleared",
                    stream_id, pose_epoch)

    def close(self) -> None:
        """Freeze for a second, then drop the socket.

        The freeze matters: cutting the servop stream instantly leaves the arm
        wherever its last command put it with no settling time. One second of
        the frozen target lets it converge before the stream stops. Always
        called, including on error paths, so every step is best-effort.
        """
        if self._closed:
            return
        self._closed = True
        for arm in ARMS:
            self._hold_arm(arm, "no_frames")
        try:
            if self._loop is not None and self._streamer is not None:
                self._call(self._async_freeze_and_close(), timeout=FREEZE_BEFORE_CLOSE_S + 10.0)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("error while closing the TRON arm sink")
        finally:
            self._shutdown_loop()
            rate = self._streamer.stats.achieved_rate_hz if self._streamer else None
            self._session.close(achieved_rate_hz=rate, stats={
                "sink": {"dispatches": self.stats.dispatches,
                         "dispatch_p50_ms": self.stats.dispatch_percentile_ms(50.0),
                         "dispatch_p95_ms": self.stats.dispatch_percentile_ms(95.0),
                         "dropped_no_arm_pose": self.stats.dropped_no_arm_pose},
                "client": self._client.stats.as_dict() if self._client else None,
                "streamer": self._streamer.stats_dict() if self._streamer else None,
                "arms": {a: dataclasses.asdict(d) for a, d in self._diagnostics.items()},
            })
        logger.info("TRON arm sink closed after %d dispatches (p95 %.3f ms)",
                    self.stats.dispatches, self.stats.dispatch_percentile_ms())

    async def _async_freeze_and_close(self) -> None:
        # The streamer holds its last command while nothing new is submitted, so
        # "freeze" is simply: keep running, submit nothing.
        await asyncio.sleep(FREEZE_BEFORE_CLOSE_S)
        if self._streamer_stop is not None:
            self._streamer_stop.set()
        if self._streamer is not None:
            await self._streamer.stop()
        if self._streamer_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._streamer_task
        if self._client is not None:
            await self._client.close()

    # -- internals -------------------------------------------------------
    def _engaged(self) -> bool:
        if self._forced_hold:
            return False
        if self._clutch is None:
            return True
        try:
            return bool(self._clutch.engaged)
        except Exception:  # noqa: BLE001 - a broken clutch reads as released
            logger.exception("clutch source raised; treating as released")
            return False

    def _arm_of(self, frame: Any) -> Arm | None:
        handedness = getattr(frame, "handedness", None)
        if handedness in ARMS:
            return handedness  # type: ignore[return-value]
        logger.warning("frame with unusable handedness %r; ignoring", handedness)
        return None

    def _robot_state(self, arm: str) -> RobotState:
        measured = self._measured[arm]
        if measured is None:
            # Before connect() primed us. Only reachable in unit tests; a real
            # run cannot dispatch before connect().
            measured = Pose.identity()
        return RobotState(arm, measured)  # type: ignore[arg-type]

    def _hold_arm(self, arm: str, reason: str) -> None:
        controller = self._controllers[arm]
        # An unrecognised reason still means hold. Upstream would swallow the
        # exception and we would simply not hold, which is the one outcome that
        # is unsafe, so coerce to a known reason and keep the original for the
        # status line.
        effective = reason if reason in HOLD_REASONS else "no_frames"
        controller.tick(None, TickEvents(hold_reason=effective), self._robot_state(arm))
        diagnostics = self._diagnostics[arm]
        diagnostics.holds += 1
        diagnostics.last_hold_reason = reason
        diagnostics.state = controller.state.value
        self._session.event("hold", arm=arm, reason=reason, effective=effective,
                            state=controller.state.value)

    def _on_stream_fault(self, exc: BaseException) -> None:
        """The servop stream died. FAULT both arms; only a reset re-arms.

        Deliberately no reconnect: resuming would stream against origins latched
        before the gap, and the robot's behaviour when a ServoP stream stops is
        undocumented (LimX Q3). A human decides whether it is safe to continue.
        """
        self.stats.faults += 1
        logger.error("servop stream fault: %s -- both arms FAULTED, manual re-arm required", exc)
        for arm in ARMS:
            self._controllers[arm].tick(None, TickEvents(fault=str(exc)),
                                        self._robot_state(arm))
            self._diagnostics[arm].state = self._controllers[arm].state.value
        self._session.event("fault", source="streamer", detail=str(exc))

    def _on_streamer_tick(self, info: TickInfo) -> None:
        """Runs on the loop thread, right after the ws send."""
        arms = {
            arm: {
                "alpha": round(detail.alpha, 6),
                "interpolation_gap_ns": detail.ingress_gap_ns,
                "src_recv_ns": detail.src_recv_ns,
                "t_dispatch": (detail.src_meta or {}).get("t_dispatch"),
                "lin_clamped": detail.lin_clamped,
                "ang_clamped": detail.ang_clamped,
                # The commanded position, so continuity on the wire can be
                # checked after the fact rather than trusted.
                "p": ([round(float(v), 6) for v in detail.pose.position_m]
                      if detail.pose is not None else None),
            }
            for arm, detail in info.arms.items()
        }
        if not arms and not info.sent:
            return  # idle tick with nothing to say
        self._session.write({
            "type": "tick",
            "t_mono_ns": info.t_streamer_tick_ns,
            "t_streamer_tick": info.t_streamer_tick_ns,
            "t_ws_send": info.t_ws_send_ns,
            "sent": info.sent,
            "arms": arms,
        })

    def _on_notify(self, record: NotifyRecord) -> None:
        self.last_notify = record
        self._session.event("notify", title=record.title, data=dict(record.data),
                            guid=record.guid, robot_timestamp_ms=record.robot_timestamp_ms)
        if record.title == "notify_robot_info":
            self._session.note_robot_info(record.data)
        if record.title == NOTIFY_SERVOP:
            logger.warning("notify_servop: %s", dict(record.data))

    def _run_loop(self, ready: threading.Event) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(ready.set)
        self._loop.run_forever()

    def _call(self, coro: Any, *, timeout: float) -> Any:
        """Run a coroutine on the sink's loop from this (other) thread."""
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _shutdown_loop(self) -> None:
        if not self._owns_loop or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self._loop_thread = None


#: Hosts we treat as "not real hardware" -- the mock, and only the mock.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _is_loopback(url: str) -> bool:
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    return host is not None and host in _LOOPBACK_HOSTS


def _recv_ns(frame: Any) -> int:
    """``recv_monotonic_ns``, the only clock allowed here (hard rule 2)."""
    value = getattr(frame, "recv_monotonic_ns", None)
    return int(value) if value is not None else time.monotonic_ns()


def _age_s(frame: Any) -> float | None:
    value = getattr(frame, "age_s", None)
    return float(value) if value is not None else None


def assert_matches_upstream_abc(sink_cls: type = TronArmSink) -> None:
    """Verify ``sink_cls`` satisfies the real ``ArmSink`` ABC.

    Imports orca_teleop, so it only runs where a checkout is installed. Raises
    ``ImportError`` if not -- callers decide whether that is a skip or a failure.
    """
    from orca_teleop.arm import ArmSink  # noqa: PLC0415 - optional dependency

    required = {
        name
        for name in dir(ArmSink)
        if getattr(getattr(ArmSink, name, None), "__isabstractmethod__", False)
    }
    missing = sorted(name for name in required if not callable(getattr(sink_cls, name, None)))
    if missing:
        raise TypeError(f"{sink_cls.__name__} is missing ArmSink methods: {missing}")

    import inspect

    for name in sorted(required):
        theirs = inspect.signature(getattr(ArmSink, name))
        ours = inspect.signature(getattr(sink_cls, name))
        if list(theirs.parameters) != list(ours.parameters):
            raise TypeError(
                f"{sink_cls.__name__}.{name}{ours} does not match ArmSink.{name}{theirs}"
            )
