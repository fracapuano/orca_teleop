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

from tests.conftest import at, fast_close, wait_until
from tron_arm.config import ARMS, load_config
from tron_arm.mock_robot import MockTron2
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
    freeze = fast_close(0.05)
    freeze.__enter__()

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
        freeze.__exit__(None, None, None)


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
@pytest.fixture(scope="module")
def nominal():
    """One nominal run, measured once.

    Three separate questions get asked of a nominal run (does it track, does the
    streamer hold its rate, does dispatch stay off the network). They were three
    identical 3 s runs; the run is the expensive part, not the assertions.
    """
    with harness() as h:
        # Wait for enough data to answer all three, rather than for a duration.
        # The dispatch bar is the binding one: stats accumulate from connect, so
        # a small sample lets the first few (cold) dispatches dominate the p95.
        # 150 frames is ~2.5 s at 60 fps, which is what the p95 assertion has
        # always effectively been measured over.
        assert wait_until(
            lambda: (h.robot.servop_accepted > 250
                     and h.sink.diagnostics("right").frames > 150
                     and h.sink.stats.dispatches > 150),
            timeout=20.0,
        ), "the pipeline never reached a steady state"
        yield {
            "track": h.flange_track("right"),
            "frames": h.sink.diagnostics("right").frames,
            "accepted": h.robot.servop_accepted,
            "rejected": h.robot.servop_rejected,
            "rate_hz": h.sink.streamer.stats.achieved_rate_hz,
            "p95_ms": h.sink.stats.dispatch_percentile_ms(95.0),
            "dispatches": h.sink.stats.dispatches,
        }


class TestNominal:
    def test_synthetic_motion_tracks_on_the_mock(self, nominal):
        assert nominal["rejected"] == 0, "the mock rejected our servop payloads"
        assert nominal["accepted"] > 100
        assert nominal["frames"] > 50, "no frames reached the sink"

        track = nominal["track"]
        moved = float(np.linalg.norm(track.max(axis=0) - track.min(axis=0)))
        assert moved > 0.01, f"the arm barely moved ({moved:.4f} m)"
        # Inside the workspace box at all times.
        bounds = load_config().workspace.box("right").bounds
        assert np.all(track >= bounds[:, 0] - 1e-6) and np.all(track <= bounds[:, 1] + 1e-6)

    def test_streamer_holds_its_rate(self, nominal):
        assert 80.0 < nominal["rate_hz"] < 120.0, f"achieved {nominal['rate_hz']:.1f} Hz"

    def test_dispatch_p95_is_under_1ms(self, nominal):
        """dispatch runs on upstream's arm worker thread and must not block."""
        assert nominal["dispatches"] > 50, f"only {nominal['dispatches']} dispatches sampled"
        assert nominal["p95_ms"] < 1.0, f"dispatch p95 was {nominal['p95_ms']:.3f} ms"


# -- (b) dropout ---------------------------------------------------------
class TestDropout:
    """--dropout-every 2 --dropout-for 0.8.

    ``_in_dropout`` is ``elapsed % 2 < 0.8``, so the timeline is:
    dropout [0, 0.8), streaming [0.8, 2.0), dropout [2.0, 2.8), streaming again.
    The interesting event is the SECOND dropout -- the first one happens before
    anything has ever latched, so it proves nothing about re-latching. We
    therefore run past 2.8 s and measure around [2.0, 2.8).

    The window only has to comfortably exceed upstream's staleness deadline
    (ARM_STALE_AFTER_S = 0.25 plus a 0.1 s poll); 0.8 s does, and the original
    1.5 s over 5 s only made the same run take two and a half times as long.
    Sampling is at 100 Hz so the shorter windows still carry plenty of points.
    """

    def test_stream_never_stops_and_target_freezes_then_relatches(self):
        with harness(dropout_every=2.0, dropout_for=0.8, sample_hz=100.0) as h:
            time.sleep(2.4)                      # streamed, now inside dropout #2
            during_dropout = h.robot.servop_accepted
            time.sleep(0.3)                      # still inside it
            still_streaming = h.robot.servop_accepted
            time.sleep(0.9)                      # frames resumed at 2.8 s
            track_frozen = h.window("right", 2.4, 2.75)
            track_after = h.window("right", 3.0, 3.6)
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
    """--epoch-change-every 1.5: WebXR re-pins its reference space at t=1.5 s.

    Epochs step at ``elapsed // every``, so one change is one interval. The test
    waits for the change and the re-latch it forces, not for a duration; running
    to 12 s to observe a change at 10 s was 8 s of nothing happening.
    """

    def test_origins_relatch_without_a_target_jump(self):
        with harness(epoch_change_every=1.5) as h:
            assert wait_until(
                lambda: (h.sink.diagnostics("right").reference_changes >= 1
                         and h.sink.diagnostics("right").latches >= 2),
                timeout=15.0,
            ), "the epoch change never forced a re-latch"
            track = h.flange_track("right")
            latches = h.sink.diagnostics("right").latches
            changes = h.sink.diagnostics("right").reference_changes
            step_lin = h.sink.controllers["right"].max_step[0]

        assert changes >= 1, "no reference change observed"
        assert latches >= 2, f"epoch change did not force a re-latch ({latches})"
        # The commanded stream is rate limited; the robot's own follower lags, so
        # allow a small multiple of the per-tick ceiling for sampling skew.
        assert max_step_between(track) < 10 * step_lin, (
            f"target jumped {max_step_between(track):.4f} m across the epoch change"
        )


# -- (d) the combined nasty ordering ------------------------------------
class TestNastyOrdering:
    """--dropout-every 1.2 --dropout-for 0.45 --epoch-change-every 1.8.

    Dropouts and epoch changes interleave, so the reference_change callback
    lands *after* frames have already resumed and re-latched. That double latch
    must be harmless. The intervals are scaled down together, so the same
    interleaving happens -- four dropouts and two epoch changes -- in 5 s
    instead of 9.
    """

    def test_late_reference_change_double_latches_without_a_lurch(self):
        with harness(dropout_every=1.2, dropout_for=0.45, epoch_change_every=1.8) as h:
            # Wait for the churn the assertions below describe, not for a
            # duration that merely tends to contain it.
            assert wait_until(
                lambda: (h.sink.diagnostics("right").holds >= 2
                         and h.sink.diagnostics("right").latches >= 3
                         and h.sink.diagnostics("right").reference_changes >= 1
                         and h.robot.servop_accepted > 300),
                timeout=20.0,
            ), f"the churn never materialised: {h.sink.diagnostics('right')}"
            track = h.flange_track("right")
            d = h.sink.diagnostics("right")
            step_lin = h.sink.controllers["right"].max_step[0]
            accepted = h.robot.servop_accepted

        assert h.robot.servop_rejected == 0
        assert accepted > 300, "servop stream did not survive the churn"
        assert d.holds >= 2, f"expected repeated holds, saw {d.holds}"
        assert d.latches >= 3, f"expected repeated re-latches, saw {d.latches}"
        assert d.reference_changes >= 1
        # The point of the whole exercise: no lurch.
        assert max_step_between(track) < 10 * step_lin, (
            f"largest robot step {max_step_between(track):.4f} m"
        )
        assert h.sink.stats.dispatch_percentile_ms() < 1.0


# -- (e) old publisher ---------------------------------------------------
class TestNoArmPose:
    """--no-arm-pose: an old publisher that never sends a wrist pose."""

    def test_sink_holds_and_does_not_crash(self):
        with harness(arm_pose=False) as h:
            assert wait_until(
                lambda: (h.sink.diagnostics("right").holds >= 1
                         and h.robot.servop_accepted > 150),
                timeout=15.0,
            ), "no hold, or the stream stopped"
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
            assert wait_until(
                lambda: (h.sink.diagnostics("left").frames > 60
                         and h.robot.servop_accepted > 150),
                timeout=15.0,
            ), "no frames routed to the left arm"
            accepted = h.robot.servop_accepted
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
        # send_both=true: the idle right arm is still commanded every tick, which
        # is why it stays frozen instead of drifting.
        assert accepted > 100
