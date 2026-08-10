"""Tests for the pose/frame contract the arm consumer is written against."""

import threading
import time

import numpy as np
import pytest

from orca_teleop.ingress.frames import LatestFrame, Pose, TeleopFrame


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _random_transform(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation_from_quaternion(quaternion)
    matrix[:3, 3] = rng.normal(size=3)
    return matrix


def _frame(**overrides) -> TeleopFrame:
    defaults = dict(
        timestamp_ns=1,
        recv_monotonic_ns=time.monotonic_ns(),
        stream_id=1,
        pose_epoch=0,
        handedness="right",
        tracking_valid=True,
        wrist=Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        head=None,
        wrist_angle_degrees=0.0,
    )
    return TeleopFrame(**{**defaults, **overrides})


# ----- Pose ---------------------------------------------------------------------------


def test_pose_matrix_round_trip():
    """Locks the rotation convention the arm side depends on."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        matrix = _random_transform(rng)
        worst = max(worst, float(np.abs(Pose.from_matrix(matrix).matrix - matrix).max()))
    assert worst < 1e-12


def test_pose_matrix_round_trip_at_180_degrees():
    """The cases a trace-only quaternion extraction gets wrong."""
    for axis in range(3):
        rotation = -np.eye(3)
        rotation[axis, axis] = 1.0
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        np.testing.assert_allclose(Pose.from_matrix(matrix).matrix, matrix, atol=1e-12)


def test_pose_rejects_zero_and_non_unit_quaternion():
    """An all-default proto Pose has qw=0; it must never become a NaN pose."""
    with pytest.raises(ValueError, match="unit quaternion"):
        Pose(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="unit quaternion"):
        Pose(np.zeros(3), np.array([2.0, 0.0, 0.0, 0.0]))


def test_pose_rejects_non_rigid_matrix():
    with pytest.raises(ValueError, match="orthonormal"):
        Pose.from_matrix(np.diag([2.0, 2.0, 2.0, 1.0]))


def test_pose_arrays_are_read_only():
    """A pose is shared across threads; nobody may mutate it in place."""
    pose = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        pose.position_m[0] = 1.0
    with pytest.raises(ValueError):
        pose.orientation_wxyz[0] = 0.5


def test_pose_as_xyz_wxyz_matches_tron_element_order():
    """Locks the LimX TRON left_pos/right_pos layout: xyz then w-first quat."""
    pose = Pose(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.5, 0.5, 0.5]))
    np.testing.assert_allclose(pose.as_xyz_wxyz(), [0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5])


# ----- LatestFrame --------------------------------------------------------------------


def test_latest_frame_keeps_only_the_newest():
    holder = LatestFrame()
    for index in range(5):
        holder.publish(_frame(timestamp_ns=index, stream_id=1))
    frame, seq = holder.get_unchecked()
    assert frame.timestamp_ns == 4
    assert seq == 5


def test_get_fresh_returns_none_when_stale():
    holder = LatestFrame()
    holder.publish(_frame(recv_monotonic_ns=time.monotonic_ns() - int(1e9)))
    assert holder.get_fresh(max_age_s=0.25) is None
    assert holder.get_unchecked()[0] is not None  # opt-in bypass still sees it


def test_get_fresh_returns_none_when_empty():
    assert LatestFrame().get_fresh() is None


def test_wait_for_next_wakes_on_publish():
    holder = LatestFrame()
    threading.Timer(0.05, lambda: holder.publish(_frame())).start()
    update = holder.wait_for_next(last_seq=0, timeout=5.0)
    assert update.fresh and not update.closed and update.frame is not None


def test_wait_for_next_wakes_on_close():
    """Shutdown must never strand the arm thread in a long wait."""
    holder = LatestFrame()
    threading.Timer(0.05, holder.close).start()
    started = time.monotonic()
    update = holder.wait_for_next(last_seq=0, timeout=5.0)
    assert update.closed
    assert time.monotonic() - started < 1.0


def test_wait_for_next_reports_timeout_as_not_fresh():
    update = LatestFrame().wait_for_next(last_seq=0, timeout=0.05)
    assert not update.fresh and not update.closed


def test_publish_is_not_backpressured_by_a_slow_consumer():
    """The arm path must never be able to slow the hand path."""
    holder = LatestFrame()
    stop = threading.Event()

    def slow_consumer():
        while not stop.is_set():
            holder.get_unchecked()
            time.sleep(0.02)

    consumer = threading.Thread(target=slow_consumer, daemon=True)
    consumer.start()
    try:
        started = time.perf_counter()
        for _ in range(500):
            holder.publish(_frame())
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        consumer.join(timeout=1.0)
    assert elapsed < 0.5  # ~microseconds each; nowhere near the 20 ms sleeps


def test_seq_is_strictly_increasing_under_contention():
    holder = LatestFrame()

    # One stream_id: ownership is about rejecting *other* publishers.
    def publish_many():
        for _ in range(500):
            holder.publish(_frame(stream_id=7))

    threads = [threading.Thread(target=publish_many) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert holder.get_unchecked()[1] == 2000


def test_second_stream_cannot_steal_the_slot():
    """Two publishers must not interleave two different wrists into one arm."""
    holder = LatestFrame()
    assert holder.publish(_frame(stream_id=1, timestamp_ns=10))
    assert not holder.publish(_frame(stream_id=2, timestamp_ns=20))
    assert holder.get_unchecked()[0].timestamp_ns == 10
    assert holder.stats["rejected"] == 1


def test_release_allows_immediate_takeover():
    holder = LatestFrame()
    holder.publish(_frame(stream_id=1, timestamp_ns=10))
    holder.release(stream_id=1)
    assert holder.publish(_frame(stream_id=2, timestamp_ns=20))
    assert holder.get_unchecked()[0].timestamp_ns == 20


def test_owner_silence_allows_takeover():
    holder = LatestFrame(takeover_after_s=0.0)
    holder.publish(_frame(stream_id=1, timestamp_ns=10))
    assert holder.publish(_frame(stream_id=2, timestamp_ns=20))


def test_publish_after_close_is_rejected():
    holder = LatestFrame()
    holder.close()
    assert not holder.publish(_frame())


def test_frame_age_uses_the_host_monotonic_clock():
    frame = _frame(recv_monotonic_ns=time.monotonic_ns() - int(0.5e9), timestamp_ns=0)
    assert frame.age_s == pytest.approx(0.5, abs=0.1)
