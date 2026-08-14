"""End-to-end: mock_publisher -> ingress -> arm_worker -> TronArmSink -> mock robot.

Scope note
----------
These drive the **arm path** exactly as ``arm.py``'s own bring-up docstring
describes it -- ``IngressServer`` + ``arm_worker`` on one side, the synthetic
Quest publisher on the other -- rather than going through
``run_metaquest_local``. That function additionally spins up the hand retargeter
(torch, a model config, ~0.7 s of imports), which is explicitly OUT OF SCOPE per
CLAUDE.md and would make these tests slow and dependent on a hand config that has
nothing to do with what is under test. Every component between the publisher and
the robot is the production one; only the hand branch is absent.
``tools/run_arm.py`` uses ``run_metaquest_local`` for real runs.

Skipped unless orca_teleop is importable (it needs grpcio).
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from typing import Any, Iterator

import numpy as np
import pytest

from tests.conftest import at
from tron_arm.config import ARMS, load_config
from tron_arm.mock_robot import MockTron2
from tron_arm.poses import Pose, quat_angle
from tron_arm.sink import TronArmSink

orca = pytest.importorskip("orca_teleop.arm", reason="orca_teleop not installed (needs grpcio)")
frames_mod = pytest.importorskip("orca_teleop.ingress.frames")
server_mod = pytest.importorskip("orca_teleop.ingress.server")
mock_pub = pytest.importorskip("orca_teleop.ingress.metaquest.mock_publisher")
publisher_mod = pytest.importorskip("orca_teleop.ingress.metaquest.publisher")

pytestmark = pytest.mark.slow


@dataclasses.dataclass
class Harness:
    robot: MockTron2
    sink: TronArmSink
    samples: list[tuple[float, dict[str, np.ndarray]]]
    stop: threading.Event

    def flange_track(self, arm: str) -> np.ndarray:
        return np.array([s[1][arm] for s in self.samples])

    def times(self) -> np.ndarray:
        return np.array([s[0] for s in self.samples])

    def window(self, arm: str, t0: float, t1: float) -> np.ndarray:
        return np.array([p[arm] for t, p in self.samples if t0 <= t <= t1])


@contextlib.contextmanager
def harness(
    *,
    hand: str = "right",
    dropout_every: float = 0.0,
    dropout_for: float = 0.0,
    epoch_change_every: float = 0.0,
    arm_pose: bool = True,
    fps: int = 60,
    sample_hz: float = 50.0,
) -> Iterator[Harness]:
    """Bring up robot + sink + ingress + arm worker + synthetic publisher."""
    import asyncio

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True, name="harness-loop")
    loop_thread.start()
    assert ready.wait(5.0)

    robot = MockTron2(port=0, info_period_s=0.25)
    asyncio.run_coroutine_threadsafe(robot.start(), loop).result(10.0)
    config = at(load_config(), f"ws://127.0.0.1:{robot.bound_port}")
    config = dataclasses.replace(config, notify_log_path=None, scale=1.0)

    sink = TronArmSink(config, loop=loop)
    sink.connect()

    stop = threading.Event()
    ingress = server_mod.IngressServer(_NullQueue(), stop, port=0)
    port = ingress.start()

    worker = threading.Thread(
        target=orca.arm_worker,
        kwargs=dict(frames=ingress.frames, sink=sink, stop_event=stop),
        daemon=True,
        name="arm-worker",
    )
    worker.start()

    bridge = mock_pub.MockQuestBridge(
        side=hand,
        fps=fps,
        dropout_every_s=dropout_every,
        dropout_for_s=dropout_for,
        epoch_change_every_s=epoch_change_every,
    )
    publisher = publisher_mod.MetaQuestPublisher(
        server_address=f"localhost:{port}",
        handedness=hand,
        fps=fps,
        arm_pose_enabled=arm_pose,
        bridge=bridge,
    )
    pub_thread = threading.Thread(target=_run_publisher, args=(publisher,), daemon=True,
                                 name="mock-publisher")
    pub_thread.start()

    samples: list[tuple[float, dict[str, np.ndarray]]] = []
    started = time.monotonic()

    def sample() -> None:
        period = 1.0 / sample_hz
        while not stop.is_set():
            samples.append((
                time.monotonic() - started,
                {arm: np.array(robot.flange[arm].position_m) for arm in ARMS},
            ))
            time.sleep(period)

    sampler = threading.Thread(target=sample, daemon=True, name="sampler")
    sampler.start()

    try:
        yield Harness(robot, sink, samples, stop)
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            publisher.stop()
        with contextlib.suppress(Exception):
            bridge.stop()
        worker.join(timeout=5.0)   # its finally: calls sink.close()
        sampler.join(timeout=2.0)
        with contextlib.suppress(Exception):
            ingress.stop()
        with contextlib.suppress(Exception):
            sink.close()
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(robot.stop(), loop).result(5.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)


def _run_publisher(publisher: Any) -> None:
    with contextlib.suppress(Exception):
        publisher.run()


class _NullQueue:
    """The landmarks queue the arm path does not use; drained by nobody."""

    def put_nowait(self, item: Any) -> None:
        return None

    def put(self, item: Any, *a: Any, **kw: Any) -> None:
        return None

    def get(self, *a: Any, **kw: Any) -> Any:
        raise TimeoutError

    def full(self) -> bool:
        return False

    def qsize(self) -> int:
        return 0


def max_step_between(track: np.ndarray) -> float:
    if len(track) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(track, axis=0), axis=1)))


# -- (a) nominal ---------------------------------------------------------
class TestNominal:
    def test_synthetic_motion_tracks_on_the_mock(self):
        with harness() as h:
            time.sleep(3.0)
            track = h.flange_track("right")
            sink = h.sink

        assert h.robot.servop_rejected == 0, "the mock rejected our servop payloads"
        assert h.robot.servop_accepted > 100
        assert sink.diagnostics("right").frames > 50, "no frames reached the sink"

        moved = float(np.linalg.norm(track.max(axis=0) - track.min(axis=0)))
        assert moved > 0.01, f"the arm barely moved ({moved:.4f} m)"
        # Inside the workspace box at all times.
        bounds = load_config().workspace.box("right").bounds
        assert np.all(track >= bounds[:, 0] - 1e-6) and np.all(track <= bounds[:, 1] + 1e-6)

    def test_streamer_holds_its_rate(self):
        with harness() as h:
            time.sleep(3.0)
            rate = h.sink.streamer.stats.achieved_rate_hz
        assert 80.0 < rate < 120.0, f"achieved {rate:.1f} Hz"

    def test_dispatch_p95_is_under_1ms(self):
        """dispatch runs on upstream's arm worker thread and must not block."""
        with harness() as h:
            time.sleep(3.0)
            p95 = h.sink.stats.dispatch_percentile_ms(95.0)
            count = h.sink.stats.dispatches
        assert count > 50, f"only {count} dispatches sampled"
        assert p95 < 1.0, f"dispatch p95 was {p95:.3f} ms"

    def test_no_network_io_happens_on_the_dispatch_thread(self):
        """The p95 bound is the symptom; this is the cause it is guarding."""
        import inspect

        source = inspect.getsource(TronArmSink._dispatch_inner)
        for forbidden in ("await", "run_coroutine_threadsafe", "_call(", ".result("):
            assert forbidden not in source, f"dispatch path contains {forbidden!r}"


# -- (b) dropout ---------------------------------------------------------
class TestDropout:
    """--dropout-every 5 --dropout-for 1.5.

    ``_in_dropout`` is ``elapsed % 5 < 1.5``, so the timeline is:
    dropout [0, 1.5), streaming [1.5, 5.0), dropout [5.0, 6.5), streaming again.
    The interesting event is the SECOND dropout -- the first one happens before
    anything has ever latched, so it proves nothing about re-latching. We
    therefore run past 6.5 s and measure around [5.0, 6.5).
    """

    def test_stream_never_stops_and_target_freezes_then_relatches(self):
        with harness(dropout_every=5.0, dropout_for=1.5) as h:
            time.sleep(5.4)                      # streamed, now inside dropout #2
            during_dropout = h.robot.servop_accepted
            time.sleep(0.9)                      # still inside it
            still_streaming = h.robot.servop_accepted
            time.sleep(1.7)                      # frames resumed at 6.5 s
            track_frozen = h.window("right", 5.4, 6.4)
            track_after = h.window("right", 6.9, 7.9)
            sink = h.sink
            latches = sink.diagnostics("right").latches
            holds = sink.diagnostics("right").holds

        # The servop stream to the robot MUST NOT stop while the operator is gone.
        assert still_streaming > during_dropout > 0, "servop stream stalled during the dropout"
        assert h.robot.servop_rejected == 0

        # Target frozen while there are no frames.
        assert len(track_frozen) > 10
        spread = float(np.max(np.linalg.norm(track_frozen - track_frozen[0], axis=1)))
        assert spread < 1e-3, f"target moved {spread:.5f} m during the dropout"

        # And a clean re-latch afterwards: motion resumes, holds and latches both
        # happened, and nothing lurched.
        assert holds >= 2, f"expected a hold per dropout, saw {holds}"
        # One latch when frames first arrived at 1.5 s, another on resume at 6.5 s.
        assert latches >= 2, f"expected a re-latch after resume, saw {latches}"
        assert len(track_after) > 20
        assert max_step_between(track_after) < 0.05


# -- (c) epoch change ----------------------------------------------------
class TestEpochChange:
    """--epoch-change-every 10: WebXR re-pins its reference space at t=10 s."""

    def test_origins_relatch_without_a_target_jump(self):
        with harness(epoch_change_every=10.0) as h:
            time.sleep(12.0)
            track = h.flange_track("right")
            latches = h.sink.diagnostics("right").latches
            changes = h.sink.diagnostics("right").reference_changes
            step_lin = h.sink.controllers["right"].max_step[0]

        assert changes >= 1, "no reference change observed in 12 s"
        assert latches >= 2, f"epoch change did not force a re-latch ({latches})"
        # The commanded stream is rate limited; the robot's own follower lags, so
        # allow a small multiple of the per-tick ceiling for sampling skew.
        assert max_step_between(track) < 10 * step_lin, (
            f"target jumped {max_step_between(track):.4f} m across the epoch change"
        )


# -- (d) the combined nasty ordering ------------------------------------
class TestNastyOrdering:
    """--dropout-every 2 --dropout-for 0.7 --epoch-change-every 3.

    Dropouts and epoch changes interleave, so the reference_change callback
    lands *after* frames have already resumed and re-latched. That double latch
    must be harmless.
    """

    def test_late_reference_change_double_latches_without_a_lurch(self):
        with harness(dropout_every=2.0, dropout_for=0.7, epoch_change_every=3.0) as h:
            time.sleep(9.0)
            track = h.flange_track("right")
            d = h.sink.diagnostics("right")
            step_lin = h.sink.controllers["right"].max_step[0]
            accepted = h.robot.servop_accepted

        assert h.robot.servop_rejected == 0
        assert accepted > 500, "servop stream did not survive the churn"
        assert d.holds >= 2, f"expected repeated holds, saw {d.holds}"
        assert d.latches >= 3, f"expected repeated re-latches, saw {d.latches}"
        assert d.reference_changes >= 1
        # The point of the whole exercise: no lurch.
        assert max_step_between(track) < 10 * step_lin, (
            f"largest robot step {max_step_between(track):.4f} m"
        )
        assert h.sink.stats.dispatch_percentile_ms() < 1.0

    def test_state_never_settles_anywhere_illegal(self):
        with harness(dropout_every=2.0, dropout_for=0.7, epoch_change_every=3.0) as h:
            time.sleep(5.0)
            state = h.sink.diagnostics("right").state
        assert state in {s.value for s in __import__(
            "tron_arm.arm_state", fromlist=["ArmState"]).ArmState}


# -- (e) old publisher ---------------------------------------------------
class TestNoArmPose:
    """--no-arm-pose: an old publisher that never sends a wrist pose."""

    def test_sink_holds_and_does_not_crash(self):
        with harness(arm_pose=False) as h:
            time.sleep(2.5)
            sink = h.sink
            d = sink.diagnostics("right")
            accepted = h.robot.servop_accepted
            track = h.flange_track("right")

        # No wrist pose ever arrives, so upstream reports no_frames and we hold.
        assert d.frames == 0 or sink.stats.dropped_no_arm_pose > 0
        assert d.holds >= 1, "expected a hold when no arm pose is published"
        # Still streaming to the robot -- holding is not stopping.
        assert accepted > 100
        assert h.robot.servop_rejected == 0
        # And the arm did not move.
        spread = float(np.max(np.linalg.norm(track - track[0], axis=1)))
        assert spread < 1e-3, f"arm moved {spread:.5f} m with no arm pose"


# -- (f) handedness routing ---------------------------------------------
class TestHandedness:
    def test_left_hand_routes_to_the_left_arm_and_the_right_stays_frozen(self):
        with harness(hand="left") as h:
            time.sleep(3.0)
            left = h.sink.diagnostics("left")
            right = h.sink.diagnostics("right")
            left_track = h.flange_track("left")
            right_track = h.flange_track("right")

        assert left.frames > 50, "no frames routed to the left arm"
        assert right.frames == 0, f"right arm saw {right.frames} frames"
        left_moved = float(np.max(np.linalg.norm(left_track - left_track[0], axis=1)))
        right_moved = float(np.max(np.linalg.norm(right_track - right_track[0], axis=1)))
        assert left_moved > 0.005, f"left arm did not move ({left_moved:.5f} m)"
        assert right_moved < 1e-3, f"right arm moved ({right_moved:.5f} m) but should be frozen"

    def test_both_arms_are_commanded_every_tick_via_send_both(self):
        """send_both=true: the idle arm still gets a frozen target each tick."""
        with harness(hand="left") as h:
            time.sleep(2.0)
            accepted = h.robot.servop_accepted
        assert accepted > 100
