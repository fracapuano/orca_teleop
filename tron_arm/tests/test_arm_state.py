"""State machine transition table and the tick pipeline.

The centrepiece is :class:`TestNastyOrdering` -- the measured sequence where the
``on_reference_change`` callback arrives *after* we have already re-latched
against the new epoch. A naive "do I have an origin?" boolean maps a fresh frame
against a stale origin there and the arm lurches.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tron_arm.arm_state import (
    ArmController,
    ArmState,
    ArmStateMachine,
    Diagnostics,
    RobotState,
    TickEvents,
)
from tron_arm.config import load_config
from tron_arm.poses import Pose, quat_angle

MEASURED = Pose([0.40, -0.20, 0.00], [1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def machine():
    return ArmStateMachine()


@pytest.fixture
def controller(cfg):
    return ArmController(cfg)


@pytest.fixture
def robot():
    return RobotState("right", MEASURED)


def wrist(x=0.10, y=0.20, z=0.30, *, yaw=0.0, recv_ns=0):
    """A TeleopFrame-shaped stand-in: only .wrist and .recv_monotonic_ns are read."""
    q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    @dataclasses.dataclass(frozen=True)
    class _Frame:
        wrist: Pose
        recv_monotonic_ns: int
        # Present on purpose: nothing in control maths may read it (hard rule 2).
        timestamp_ns: int = -1

    return _Frame(Pose([x, y, z], q), recv_ns)


# -- transition table ----------------------------------------------------
class TestTransitionTable:
    def test_starts_disengaged(self, machine):
        assert machine.state is ArmState.DISENGAGED
        assert not machine.clutch_held

    def test_engage_enters_engaged_no_origin_and_clears(self, machine):
        assert machine.set_clutch(True) is ArmState.ENGAGED_NO_ORIGIN
        assert machine.clear_count == 1
        assert machine.latch_count == 0

    def test_first_frame_latches_and_engages(self, machine):
        machine.set_clutch(True)
        assert machine.on_frame() is True
        assert machine.state is ArmState.ENGAGED
        assert machine.latch_count == 1

    def test_second_frame_does_not_relatch(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        assert machine.on_frame() is False
        assert machine.latch_count == 1

    def test_frames_while_disengaged_are_ignored(self, machine):
        assert machine.on_frame() is False
        assert machine.state is ArmState.DISENGAGED
        assert machine.latch_count == 0

    def test_release_from_engaged_clears(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        before = machine.clear_count
        assert machine.set_clutch(False) is ArmState.DISENGAGED
        assert machine.clear_count == before + 1

    def test_release_from_engaged_no_origin_clears(self, machine):
        machine.set_clutch(True)
        before = machine.clear_count
        machine.set_clutch(False)
        assert machine.clear_count == before + 1

    @pytest.mark.parametrize("reason", ["no_frames", "stale", "tracking_invalid"])
    def test_hold_from_engaged_clears(self, machine, reason):
        machine.set_clutch(True)
        machine.on_frame()
        before = machine.clear_count
        assert machine.on_hold(reason) is ArmState.HOLD
        assert machine.clear_count == before + 1
        assert machine.hold_reason == reason

    def test_hold_is_idempotent(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        machine.on_hold("stale")
        before = machine.clear_count
        machine.on_hold("stale")
        assert machine.clear_count == before
        assert machine.state is ArmState.HOLD

    def test_unknown_hold_reason_rejected(self, machine):
        with pytest.raises(ValueError, match="unknown hold reason"):
            machine.on_hold("bored")

    def test_hold_while_disengaged_is_a_no_op(self, machine):
        assert machine.on_hold("no_frames") is ArmState.DISENGAGED

    def test_reference_change_from_engaged_demotes_and_clears(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        before = machine.clear_count
        assert machine.on_reference_change(1, 8) is ArmState.ENGAGED_NO_ORIGIN
        assert machine.clear_count == before + 1

    def test_reference_change_while_originless_still_clears(self, machine):
        machine.set_clutch(True)
        before = machine.clear_count
        machine.on_reference_change(1, 9)
        assert machine.clear_count == before + 1
        assert machine.state is ArmState.ENGAGED_NO_ORIGIN

    def test_frames_resume_from_hold_relatches_in_one_step(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        machine.on_hold("stale")
        assert machine.on_frame() is True
        assert machine.state is ArmState.ENGAGED
        assert machine.latch_count == 2

    def test_frames_from_hold_with_clutch_released_disengage(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        machine.on_hold("stale")
        machine._clutch = False  # released while held in HOLD
        assert machine.on_frame() is False
        assert machine.state is ArmState.DISENGAGED

    def test_fault_latches_and_only_reset_leaves_it(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        assert machine.on_fault("ws closed") is ArmState.FAULT
        assert machine.fault_detail == "ws closed"
        for attempt in (
            lambda: machine.set_clutch(True),
            lambda: machine.on_frame(),
            lambda: machine.on_hold("stale"),
            lambda: machine.on_reference_change(1, 1),
        ):
            attempt()
            assert machine.state is ArmState.FAULT
        assert machine.reset() is ArmState.DISENGAGED
        assert machine.fault_detail is None

    def test_reset_drops_the_clutch(self, machine):
        """Recovering from a fault must not silently re-engage."""
        machine.set_clutch(True)
        machine.on_fault("x")
        machine.reset()
        assert not machine.clutch_held
        assert machine.state is ArmState.DISENGAGED

    def test_fault_clears_origins(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        before = machine.clear_count
        machine.on_fault("x")
        assert machine.clear_count == before + 1

    def test_state_helpers(self):
        assert ArmState.ENGAGED.can_command
        assert not ArmState.ENGAGED_NO_ORIGIN.can_command
        assert ArmState.ENGAGED_NO_ORIGIN.is_engaged and ArmState.ENGAGED.is_engaged
        assert not ArmState.HOLD.is_engaged and not ArmState.FAULT.is_engaged

    def test_history_records_every_transition(self, machine):
        machine.set_clutch(True)
        machine.on_frame()
        machine.on_hold("stale")
        triggers = [t.trigger for t in machine.history]
        assert triggers == ["clutch_engage", "lazy_latch", "hold:stale"]
        assert machine.history[1].latched
        assert machine.history[2].cleared_origins


class TestOriginCallbacks:
    def test_clear_and_latch_callbacks_fire(self):
        events = []
        m = ArmStateMachine(
            on_clear_origins=lambda: events.append("clear"),
            on_latch=lambda: events.append("latch"),
        )
        m.set_clutch(True)
        m.on_frame()
        m.on_hold("stale")
        m.on_frame()
        assert events == ["clear", "latch", "clear", "clear", "latch"]

    def test_controller_wires_clear_to_the_mapper(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        assert controller.mapper.latched
        controller.tick(None, TickEvents(hold_reason="stale"), robot)
        assert not controller.mapper.latched


# -- tick pipeline -------------------------------------------------------
class TestTick:
    def test_disengaged_holds(self, controller, robot):
        got = controller.tick(wrist(), None, robot)
        assert got.hold_last and got.target is None
        assert got.diagnostics.state is ArmState.DISENGAGED

    def test_no_frame_holds(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        got = controller.tick(None, None, robot)
        assert got.hold_last
        assert got.diagnostics.reason == "no_frame"

    def test_first_engaged_frame_latches_and_commands_the_measured_pose(self, controller, robot):
        got = controller.tick(wrist(), TickEvents(clutch=True), robot)
        assert got.target is not None
        assert got.diagnostics.latched_this_tick
        assert got.diagnostics.origin_source == "measured"
        np.testing.assert_allclose(got.target.position_m, MEASURED.position_m, atol=1e-9)

    def test_relatch_prefers_the_last_command_over_the_measurement(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        controller.tick(wrist(x=0.15), None, robot)
        commanded = controller.last_commanded
        got = controller.tick(wrist(x=0.15), TickEvents(hold_reason="stale"), robot)
        assert got.hold_last
        got = controller.tick(wrist(x=0.15), None, robot)
        assert got.diagnostics.origin_source == "last_commanded"
        np.testing.assert_allclose(got.target.position_m, commanded.position_m, atol=1e-9)

    def test_scale_reaches_the_target(self, cfg, robot):
        c = ArmController(dataclasses.replace(cfg, scale=0.5))
        c.tick(wrist(x=0.10), TickEvents(clutch=True), robot)
        # +4 mm of operator motion -> 2 mm scaled, safely under the 4 mm step clamp.
        got = c.tick(wrist(x=0.104), None, robot)
        assert not got.diagnostics.step_clamped
        assert got.target.position_m[0] == pytest.approx(0.40 + 0.002, abs=1e-9)

    def test_workspace_clamp_is_reported(self, cfg, robot):
        c = ArmController(dataclasses.replace(cfg, scale=1.0))
        c.tick(wrist(x=0.0), TickEvents(clutch=True), robot)
        for i in range(1, 400):  # walk far past the +x wall
            got = c.tick(wrist(x=0.01 * i), None, robot)
        assert "x" in got.diagnostics.workspace_clamped_axes
        assert got.diagnostics.workspace_clamped

    def test_step_clamp_is_applied_and_reported(self, cfg, robot):
        c = ArmController(dataclasses.replace(cfg, scale=1.0))
        c.tick(wrist(x=0.0), TickEvents(clutch=True), robot)
        got = c.tick(wrist(x=1.0), None, robot)  # a 1 m operator jump
        assert got.diagnostics.step_clamped_lin
        moved = float(np.linalg.norm(got.target.position_m - MEASURED.position_m))
        assert moved == pytest.approx(c.max_step[0], abs=1e-12)

    def test_step_clamp_reuses_the_streamer_maths(self, cfg, robot):
        """Same ceiling as the streamer -- not a second implementation."""
        c = ArmController(cfg)
        assert c.max_step == cfg.max_step == cfg.velocity.max_step(cfg.servop.rate_hz)

    def test_an_explicit_hold_outranks_a_frame_in_the_same_tick(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        got = controller.tick(wrist(x=0.9), TickEvents(hold_reason="stale"), robot)
        assert got.hold_last
        assert controller.state is ArmState.HOLD
        assert not controller.mapper.latched

    def test_a_frame_on_a_later_tick_still_resumes_from_hold(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        controller.tick(wrist(), TickEvents(hold_reason="stale"), robot)
        got = controller.tick(wrist(x=0.5), None, robot)
        assert got.target is not None
        assert got.diagnostics.latched_this_tick

    def test_fault_event_holds_immediately(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        got = controller.tick(wrist(), TickEvents(fault="socket closed"), robot)
        assert got.hold_last
        assert got.diagnostics.state is ArmState.FAULT

    def test_reset_then_fault_in_one_tick_stays_faulted(self, controller, robot):
        controller.tick(wrist(), TickEvents(fault="x"), robot)
        got = controller.tick(wrist(), TickEvents(reset=True, fault="again"), robot)
        assert got.diagnostics.state is ArmState.FAULT

    def test_reset_recovers_on_a_clean_tick(self, controller, robot):
        controller.tick(wrist(), TickEvents(fault="x"), robot)
        controller.tick(None, TickEvents(reset=True), robot)
        assert controller.state is ArmState.DISENGAGED
        got = controller.tick(wrist(), TickEvents(clutch=True), robot)
        assert got.target is not None

    def test_release_holds_and_clears(self, controller, robot):
        controller.tick(wrist(), TickEvents(clutch=True), robot)
        got = controller.tick(wrist(), TickEvents(clutch=False), robot)
        assert got.hold_last
        assert not controller.mapper.latched

    def test_diagnostics_carry_the_monotonic_clock_only(self, controller, robot):
        got = controller.tick(wrist(recv_ns=12345), TickEvents(clutch=True), robot)
        assert got.diagnostics.recv_monotonic_ns == 12345

    def test_accepts_a_bare_pose_as_well_as_a_frame(self, controller, robot):
        got = controller.tick(Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]), TickEvents(clutch=True), robot)
        assert got.target is not None

    def test_unknown_hold_reason_rejected_in_events(self):
        with pytest.raises(ValueError, match="unknown hold reason"):
            TickEvents(hold_reason="sleepy")

    def test_robot_state_rejects_unknown_arm(self):
        with pytest.raises(ValueError, match="unknown arm"):
            RobotState("middle", MEASURED)

    def test_diagnostics_helpers(self):
        d = Diagnostics(state=ArmState.ENGAGED, workspace_clamped_axes=("x",), step_clamped_ang=True)
        assert d.workspace_clamped and d.step_clamped


# -- the measured nasty ordering ----------------------------------------
class TestNastyOrdering:
    """dropout -> hold -> resume -> re-latch -> LATE reference_change -> re-latch.

    The operator pose jumps at the epoch change (WebXR re-pins its reference
    space). The emitted target stream must not.
    """

    def _run(self, cfg):
        c = ArmController(dataclasses.replace(cfg, scale=1.0))
        robot = RobotState("right", MEASURED)
        emitted: list[Pose] = []
        latch_ticks: list[int] = []
        mapped_after_clear_without_latch: list[int] = []
        cleared_since_last_target = False
        step = 0

        def run(frame, events=None):
            nonlocal step, cleared_since_last_target
            before_clears = c.machine.clear_count
            got = c.tick(frame, events, robot)
            if c.machine.clear_count > before_clears:
                cleared_since_last_target = True
            if got.diagnostics.latched_this_tick:
                latch_ticks.append(step)
            if got.target is not None:
                # A target emitted after a clear MUST have re-latched this tick.
                if cleared_since_last_target and not got.diagnostics.latched_this_tick:
                    mapped_after_clear_without_latch.append(step)
                cleared_since_last_target = False
                emitted.append(got.target)
            step += 1
            return got

        # 1. engage, 2. stream a few frames in epoch 1
        run(wrist(x=0.10), TickEvents(clutch=True))
        for i in range(1, 5):
            run(wrist(x=0.10 + 0.002 * i))

        # 3. dropout begins -> upstream holds on the staleness deadline
        run(None, TickEvents(hold_reason="stale"))
        run(None)

        # 4. frames resume, clutch still held. WebXR re-pinned: the operator pose
        #    has JUMPED half a metre, but the callback has not arrived yet.
        for i in range(4):
            run(wrist(x=0.60 + 0.002 * i))

        # 5. the LATE reference_change finally lands, after we already re-latched
        run(wrist(x=0.608), TickEvents(reference_change=(1, 2)))

        # 6. next frames re-latch in the new epoch
        for i in range(4):
            run(wrist(x=0.610 + 0.002 * i))

        return c, emitted, latch_ticks, mapped_after_clear_without_latch

    def test_all_three_latches_occur(self, cfg):
        c, _emitted, latch_ticks, _bad = self._run(cfg)
        # engage, frames-resumed, and after the late reference change.
        assert c.machine.latch_count == 3, [t.trigger for t in c.machine.history]
        assert len(latch_ticks) == 3

    def test_no_fresh_frame_is_ever_mapped_against_a_stale_origin(self, cfg):
        _c, _emitted, _latches, bad = self._run(cfg)
        assert bad == [], f"ticks that mapped after a clear without re-latching: {bad}"

    def test_the_emitted_target_stream_is_continuous(self, cfg):
        """No jump larger than the step clamp, despite a 0.5 m operator jump."""
        c, emitted, _latches, _bad = self._run(cfg)
        max_lin, max_ang = c.max_step
        assert len(emitted) >= 10
        jumps = [
            float(np.linalg.norm(b.position_m - a.position_m))
            for a, b in zip(emitted, emitted[1:])
        ]
        assert max(jumps) <= max_lin + 1e-12, f"largest jump {max(jumps):.6f} m > {max_lin}"
        turns = [quat_angle(a.orientation_wxyz, b.orientation_wxyz)
                 for a, b in zip(emitted, emitted[1:])]
        assert max(turns) <= max_ang + 1e-9

    def test_the_operator_jump_really_was_large(self, cfg):
        """Guard the guard: if the fixture stopped jumping the test proves nothing."""
        c = ArmController(dataclasses.replace(cfg, scale=1.0))
        robot = RobotState("right", MEASURED)
        c.tick(wrist(x=0.10), TickEvents(clutch=True), robot)
        naive_before = c.mapper.map(wrist(x=0.108).wrist).position_m
        naive_after = c.mapper.map(wrist(x=0.60).wrist).position_m
        assert float(np.linalg.norm(naive_after - naive_before)) > 0.4

    def test_reference_change_during_hold_still_forces_a_relatch(self, cfg):
        c = ArmController(cfg)
        robot = RobotState("right", MEASURED)
        c.tick(wrist(), TickEvents(clutch=True), robot)
        c.tick(None, TickEvents(hold_reason="stale"), robot)
        c.tick(None, TickEvents(reference_change=(2, 3)), robot)
        assert not c.mapper.latched
        got = c.tick(wrist(x=0.5), None, robot)
        assert got.diagnostics.latched_this_tick
        assert c.state is ArmState.ENGAGED

    def test_two_reference_changes_back_to_back(self, cfg):
        c = ArmController(cfg)
        robot = RobotState("right", MEASURED)
        c.tick(wrist(), TickEvents(clutch=True), robot)
        c.tick(wrist(), TickEvents(reference_change=(1, 2)), robot)
        c.tick(wrist(), TickEvents(reference_change=(1, 3)), robot)
        assert c.state is ArmState.ENGAGED
        assert c.machine.latch_count == 3

    def test_hold_and_reference_change_in_the_same_tick(self, cfg):
        c = ArmController(cfg)
        robot = RobotState("right", MEASURED)
        c.tick(wrist(), TickEvents(clutch=True), robot)
        got = c.tick(None, TickEvents(hold_reason="stale", reference_change=(1, 2)), robot)
        assert got.hold_last
        assert c.state is ArmState.HOLD
        assert not c.mapper.latched
