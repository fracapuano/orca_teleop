"""TronArmSink against the mock robot, without orca_teleop.

Covers the ArmSink surface, the connect/freeze/close lifecycle, handedness
routing and the non-blocking dispatch guarantee. The full pipeline版 lives in
tests/test_integration_pipeline.py and needs grpcio.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import threading
import time
from typing import Any, Iterator

import numpy as np
import pytest

from tests.conftest import at, run
from tron_arm.arm_state import ArmState
from tron_arm.clutch import ScriptedClutch
from tron_arm.config import ARMS, load_config
from tron_arm.mock_robot import MockTron2
from tron_arm.poses import Pose
from tron_arm.sink import TronArmSink, assert_matches_upstream_abc


@dataclasses.dataclass(frozen=True)
class FakeFrame:
    """TeleopFrame-shaped: only the fields the sink reads."""

    wrist: Pose
    handedness: str = "right"
    recv_monotonic_ns: int = 0
    stream_id: int = 1
    pose_epoch: int = 1
    tracking_valid: bool = True
    age_s: float = 0.001
    timestamp_ns: int = -1  # present on purpose; must never be read


@contextlib.contextmanager
def connected_sink(**sink_kwargs: Any) -> Iterator[tuple[MockTron2, TronArmSink]]:
    """A mock robot plus a connected sink, on a loop this fixture owns."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    assert ready.wait(5.0)
    robot = MockTron2(port=0, info_period_s=0.1)
    asyncio.run_coroutine_threadsafe(robot.start(), loop).result(10.0)
    config = at(load_config(), f"ws://127.0.0.1:{robot.bound_port}")
    config = dataclasses.replace(config, notify_log_path=None, scale=1.0)
    sink = TronArmSink(config, loop=loop, **sink_kwargs)
    sink.connect()
    try:
        yield robot, sink
    finally:
        with contextlib.suppress(Exception):
            sink.close()
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(robot.stop(), loop).result(5.0)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)


def frame(x: float = 0.10, arm: str = "right", **kw: Any) -> FakeFrame:
    return FakeFrame(Pose([x, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]), handedness=arm,
                     recv_monotonic_ns=time.monotonic_ns(), **kw)


class TestArmSinkSurface:
    def test_has_the_five_abc_methods(self):
        for name in ("connect", "dispatch", "on_hold", "on_reference_change", "close"):
            assert callable(getattr(TronArmSink, name))

    def test_signatures_match_the_documented_abc(self):
        expected = {
            "connect": ["self"],
            "dispatch": ["self", "frame"],
            "on_hold": ["self", "reason"],
            "on_reference_change": ["self", "stream_id", "pose_epoch"],
            "close": ["self"],
        }
        for name, params in expected.items():
            got = list(inspect.signature(getattr(TronArmSink, name)).parameters)
            assert got == params, f"{name}{tuple(got)} != {tuple(params)}"

    def test_matches_the_real_abc_when_orca_teleop_is_installed(self):
        pytest.importorskip("orca_teleop.arm", reason="orca_teleop not installed")
        assert_matches_upstream_abc(TronArmSink)  # raises TypeError on mismatch

    def test_does_not_import_orca_teleop(self):
        """The package must stay standalone (mock + CLI run without it)."""
        source = inspect.getsource(__import__("tron_arm.sink", fromlist=["x"]))
        assert "from orca_teleop" not in source.split("def assert_matches_upstream_abc")[0]


class TestConnect:
    def test_freezes_both_arms_and_starts_the_streamer(self):
        with connected_sink() as (robot, sink):
            assert sink.client is not None and sink.client.accid
            assert sink.streamer is not None and sink.streamer.running
            for arm in ARMS:
                assert sink.client.frozen_poses[arm] is not None
            time.sleep(0.3)
            assert robot.servop_accepted > 5, "streamer is not sending"
            assert robot.servop_rejected == 0

    def test_streams_before_any_frame_arrives(self):
        """The idle arm must be commanded from tick one, not after first motion."""
        with connected_sink() as (robot, sink):
            time.sleep(0.3)
            assert robot.servop_accepted > 5
            assert sink.stats.dispatches == 0

    def test_connect_failure_propagates(self):
        config = at(load_config(), "ws://127.0.0.1:1")
        sink = TronArmSink(dataclasses.replace(config, notify_log_path=None))
        with pytest.raises(Exception):
            sink.connect()
        sink.close()


class TestDispatch:
    def test_engaged_dispatch_moves_the_arm(self):
        with connected_sink() as (robot, sink):
            start = np.array(robot.flange["right"].position_m)
            for i in range(40):
                sink.dispatch(frame(x=0.10 + 0.002 * i))
                time.sleep(0.01)
            time.sleep(0.2)
            moved = float(np.linalg.norm(np.array(robot.flange["right"].position_m) - start))
        assert moved > 0.005, f"arm moved only {moved:.5f} m"
        assert robot.servop_rejected == 0

    def test_dispatch_is_fast(self):
        with connected_sink() as (robot, sink):
            for i in range(300):
                sink.dispatch(frame(x=0.10 + 0.0005 * i))
            p95 = sink.stats.dispatch_percentile_ms(95.0)
        assert p95 < 1.0, f"dispatch p95 {p95:.3f} ms"

    def test_dispatch_does_no_network_io(self):
        source = inspect.getsource(TronArmSink._dispatch_inner)
        for forbidden in ("await", "run_coroutine_threadsafe", "self._call("):
            assert forbidden not in source

    def test_released_clutch_holds(self):
        clutch = ScriptedClutch(engaged=False)
        with connected_sink(clutch=clutch) as (robot, sink):
            start = np.array(robot.flange["right"].position_m)
            for i in range(30):
                sink.dispatch(frame(x=0.10 + 0.01 * i))
            time.sleep(0.2)
            moved = float(np.linalg.norm(np.array(robot.flange["right"].position_m) - start))
        assert moved < 1e-3, f"arm moved {moved:.5f} m with the clutch released"

    def test_engaging_mid_stream_latches_without_a_jump(self):
        clutch = ScriptedClutch(engaged=False)
        with connected_sink(clutch=clutch) as (robot, sink):
            for i in range(10):
                sink.dispatch(frame(x=0.5 + 0.01 * i))  # far away, but released
            time.sleep(0.15)
            before = np.array(robot.flange["right"].position_m)
            clutch.engaged = True
            sink.dispatch(frame(x=0.6))
            time.sleep(0.15)
            after = np.array(robot.flange["right"].position_m)
        assert float(np.linalg.norm(after - before)) < 0.02, "lurched on engage"

    def test_a_broken_clutch_reads_as_released(self):
        class Exploding:
            @property
            def engaged(self):
                raise RuntimeError("pedal on fire")

        with connected_sink(clutch=Exploding()) as (robot, sink):
            start = np.array(robot.flange["right"].position_m)
            for i in range(20):
                sink.dispatch(frame(x=0.10 + 0.01 * i))
            time.sleep(0.2)
            moved = float(np.linalg.norm(np.array(robot.flange["right"].position_m) - start))
        assert moved < 1e-3

    def test_unknown_handedness_is_counted_not_crashed(self):
        with connected_sink() as (robot, sink):
            sink.dispatch(frame(arm="middle"))
            assert sink.stats.unknown_handedness == 1

    def test_frame_without_a_wrist_holds(self):
        """An old --no-arm-pose publisher must not crash the sink."""

        @dataclasses.dataclass(frozen=True)
        class NoWrist:
            handedness: str = "right"
            recv_monotonic_ns: int = 0
            age_s: float = 0.0

        with connected_sink() as (robot, sink):
            sink.dispatch(NoWrist())
            assert sink.stats.dropped_no_arm_pose == 1
            assert sink.diagnostics("right").holds >= 1

    def test_left_frames_do_not_touch_the_right_arm(self):
        with connected_sink() as (robot, sink):
            start = np.array(robot.flange["right"].position_m)
            for i in range(40):
                sink.dispatch(frame(x=0.10 + 0.002 * i, arm="left"))
                time.sleep(0.005)
            time.sleep(0.2)
            right_moved = float(np.linalg.norm(
                np.array(robot.flange["right"].position_m) - start))
            assert sink.diagnostics("left").frames == 40
            assert sink.diagnostics("right").frames == 0
        assert right_moved < 1e-3


class TestCallbacks:
    def test_on_hold_freezes_both_arms(self):
        with connected_sink() as (robot, sink):
            for i in range(20):
                sink.dispatch(frame(x=0.10 + 0.002 * i))
            time.sleep(0.15)
            sink.on_hold("stale")
            frozen = np.array(robot.flange["right"].position_m)
            # The engaged arm enters HOLD; the arm that was never engaged stays
            # DISENGAGED -- it has no origin to invalidate and is already frozen
            # at its last commanded pose by send_both.
            assert sink.controllers["right"].state is ArmState.HOLD
            assert sink.controllers["left"].state is ArmState.DISENGAGED
            for arm in ARMS:
                assert sink.diagnostics(arm).last_hold_reason == "stale"
            time.sleep(0.3)
            after = np.array(robot.flange["right"].position_m)
        assert float(np.linalg.norm(after - frozen)) < 1e-3

    def test_hold_does_not_stop_the_servop_stream(self):
        with connected_sink() as (robot, sink):
            sink.on_hold("no_frames")
            before = robot.servop_accepted
            time.sleep(0.3)
            assert robot.servop_accepted > before

    def test_unknown_hold_reason_is_logged_not_raised(self):
        with connected_sink() as (robot, sink):
            sink.on_hold("bored")  # must not raise
            assert sink.diagnostics("right").last_hold_reason == "bored"

    def test_on_reference_change_clears_both_arms(self):
        with connected_sink() as (robot, sink):
            sink.dispatch(frame())
            assert sink.controllers["right"].mapper.latched
            sink.on_reference_change(1, 7)
            for arm in ARMS:
                assert not sink.controllers[arm].mapper.latched
                assert sink.diagnostics(arm).reference_changes == 1

    def test_late_reference_change_relatches_on_the_next_frame(self):
        with connected_sink() as (robot, sink):
            sink.dispatch(frame(x=0.10))
            sink.on_hold("stale")
            sink.dispatch(frame(x=0.30))          # resumed: re-latch #2
            sink.on_reference_change(1, 2)        # LATE
            sink.dispatch(frame(x=0.32))          # re-latch #3
            assert sink.diagnostics("right").latches == 3
            assert sink.controllers["right"].state is ArmState.ENGAGED


class TestClose:
    def test_close_freezes_then_shuts_the_socket(self):
        with connected_sink() as (robot, sink):
            for i in range(20):
                sink.dispatch(frame(x=0.10 + 0.002 * i))
            time.sleep(0.15)
            started = time.monotonic()
            sink.close()
            elapsed = time.monotonic() - started
            assert elapsed >= 1.0, f"close returned in {elapsed:.2f}s; no freeze window"
            assert sink.client is not None and not sink.client.connected

    def test_close_is_idempotent(self):
        with connected_sink() as (robot, sink):
            sink.close()
            sink.close()

    def test_close_keeps_streaming_during_the_freeze(self):
        with connected_sink() as (robot, sink):
            sink.dispatch(frame())
            before = robot.servop_accepted
            sink.close()
            assert robot.servop_accepted > before + 50, "stream stopped before the freeze ended"


class TestOperatorControls:
    def test_force_hold_stops_motion_and_releases(self):
        with connected_sink() as (robot, sink):
            sink.dispatch(frame(x=0.10))
            sink.force_hold(True)
            frozen = np.array(robot.flange["right"].position_m)
            for i in range(30):
                sink.dispatch(frame(x=0.2 + 0.01 * i))
            time.sleep(0.3)
            held = np.array(robot.flange["right"].position_m)
            assert float(np.linalg.norm(held - frozen)) < 5e-3
            sink.force_hold(False)
            for i in range(30):
                sink.dispatch(frame(x=0.2 + 0.002 * i))
                time.sleep(0.005)
            time.sleep(0.2)
            resumed = np.array(robot.flange["right"].position_m)
        assert float(np.linalg.norm(resumed - held)) > 1e-3, "did not resume after release"

    def test_orientation_freeze_holds_rotation_but_allows_translation(self):
        with connected_sink() as (robot, sink):
            sink.set_orientation_frozen(True)
            assert sink.orientation_frozen
            assert sink.diagnostics("right").orientation_frozen
            q0 = np.array(robot.flange["right"].orientation_wxyz)
            for i in range(40):
                yaw = 0.02 * i
                f = FakeFrame(
                    Pose([0.10 + 0.002 * i, 0.2, 0.3],
                         [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                    recv_monotonic_ns=time.monotonic_ns(),
                )
                sink.dispatch(f)
                time.sleep(0.005)
            time.sleep(0.2)
            q1 = np.array(robot.flange["right"].orientation_wxyz)
            moved = float(np.linalg.norm(
                np.array(robot.flange["right"].position_m) - np.array([0.40, -0.20, 0.0])))
        np.testing.assert_allclose(q1, q0, atol=1e-6)
        assert moved > 1e-3, "position-only mode should still translate"

    def test_notify_records_are_captured(self):
        with connected_sink() as (robot, sink):
            time.sleep(0.3)
            assert sink.last_notify is not None
            assert sink.last_notify.title.startswith("notify_")
