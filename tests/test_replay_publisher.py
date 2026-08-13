"""Tests for replaying a recorded Quest session.

The recording lives on the Hugging Face Hub, so nothing here touches the
network: every test builds a :class:`Recording` in memory. What is actually
worth pinning down is the basis conversion — the recording is in Unity's
left-handed frame and the wire is FLU, and getting that wrong produces poses
that look plausible and send an arm the wrong way.
"""

import time

import numpy as np
import pytest

from orca_teleop.ingress.metaquest.replay_publisher import (
    UNITY_LEFT_TO_FLU,
    Recording,
    ReplayPublisher,
    ReplaySample,
    unity_points_to_flu,
    unity_pose_to_flu_matrix,
)

IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def _sample(t_ns: int = 0, **overrides) -> ReplaySample:
    defaults = dict(
        t_ns=t_ns,
        position=np.array([0.1, 0.2, 0.3]),
        quaternion_xyzw=IDENTITY_QUAT_XYZW,
        landmarks=np.zeros((21, 3)),
    )
    return ReplaySample(**{**defaults, **overrides})


def _recording(samples=None, handedness: str = "right") -> Recording:
    return Recording(handedness=handedness, samples=samples or [_sample()])


def test_basis_maps_unity_axes_onto_flu():
    """Unity X=right, Y=up, Z=forward; FLU X=forward, Y=left, Z=up."""
    right, up, forward = np.eye(3)
    assert np.allclose(UNITY_LEFT_TO_FLU @ right, [0, -1, 0])  # right -> -left
    assert np.allclose(UNITY_LEFT_TO_FLU @ up, [0, 0, 1])  # up -> up
    assert np.allclose(UNITY_LEFT_TO_FLU @ forward, [1, 0, 0])  # forward -> forward


def test_basis_crosses_handedness():
    """det = -1 is the point: Unity is left-handed and FLU is not."""
    assert np.isclose(np.linalg.det(UNITY_LEFT_TO_FLU), -1.0)
    assert np.allclose(UNITY_LEFT_TO_FLU @ UNITY_LEFT_TO_FLU.T, np.eye(3))


def test_converted_pose_is_still_a_rotation():
    """Conjugation cancels the mirror, so the result must not be a reflection."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        matrix = unity_pose_to_flu_matrix(rng.normal(size=3), quaternion)
        rotation = matrix[:3, :3]
        assert np.isclose(np.linalg.det(rotation), 1.0)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_identity_rotation_stays_identity():
    matrix = unity_pose_to_flu_matrix(np.zeros(3), IDENTITY_QUAT_XYZW)
    assert np.allclose(matrix, np.eye(4))


def test_position_is_converted_like_the_landmarks():
    """One basis for both, or the wrist and the fingers disagree."""
    position = np.array([0.3, -0.2, 0.9])
    matrix = unity_pose_to_flu_matrix(position, IDENTITY_QUAT_XYZW)
    points = unity_points_to_flu(position.reshape(1, 3))
    assert np.allclose(matrix[:3, 3], points[0])


def test_from_table_drops_invisible_frames():
    pa = pytest.importorskip("pyarrow")
    rows = 4
    table = pa.table(
        {
            "t_ns": [i * 33_000_000 for i in range(rows)],
            "right_visible": [True, False, True, True],
            **{f"right_wrist_{axis}": [0.1] * rows for axis in ("x", "y", "z")},
            **{f"right_wrist_q{axis}": [0.0] * rows for axis in ("x", "y", "z")},
            "right_wrist_qw": [1.0] * rows,
            "right_landmarks": [[0.0] * 63] * rows,
        }
    )
    recording = Recording.from_table(table, "right")
    assert len(recording.samples) == 3
    assert recording.dropped == 1
    # Timestamps are kept, so the gap replays as silence rather than a jump cut.
    assert [s.t_ns for s in recording.samples] == [0, 66_000_000, 99_000_000]


def test_from_table_rejects_a_hand_that_is_never_visible():
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {
            "t_ns": [0],
            "left_visible": [False],
            **{f"left_wrist_{axis}": [float("nan")] for axis in ("x", "y", "z")},
            **{f"left_wrist_q{axis}": [float("nan")] for axis in ("x", "y", "z", "w")},
            "left_landmarks": [[float("nan")] * 63],
        }
    )
    with pytest.raises(ValueError, match="no visible left-hand frames"):
        Recording.from_table(table, "left")


def test_frames_carry_a_wrist_pose_and_the_recorded_handedness():
    publisher = ReplayPublisher(_recording(handedness="left"))
    frame = publisher._sample_to_proto(_sample(), pose_epoch=1, timestamp_ns=123)
    assert frame.handedness == "left"
    assert frame.timestamp_ns == 123
    assert frame.HasField("wrist_pose")
    assert frame.tracking_valid is True
    assert frame.pose_epoch == 1
    assert len(frame.keypoints) == 63
    # The recorder never captured the headset, so the field stays unset rather
    # than shipping a zero pose.
    assert not frame.HasField("head_pose")


def test_emitted_quaternion_is_a_unit_quaternion():
    publisher = ReplayPublisher(_recording())
    quaternion = np.array([0.3, -0.5, 0.2, 0.7])
    quaternion /= np.linalg.norm(quaternion)
    frame = publisher._sample_to_proto(
        _sample(quaternion_xyzw=quaternion), pose_epoch=1, timestamp_ns=0
    )
    pose = frame.wrist_pose
    assert np.isclose(np.linalg.norm([pose.qw, pose.qx, pose.qy, pose.qz]), 1.0)


def test_no_arm_pose_leaves_the_field_unset():
    """The compatibility case: a publisher that predates wrist_pose."""
    publisher = ReplayPublisher(_recording(), arm_pose_enabled=False)
    frame = publisher._sample_to_proto(_sample(), pose_epoch=1, timestamp_ns=0)
    assert not frame.HasField("wrist_pose")


def test_no_wrist_holds_the_hand_joint_at_zero():
    publisher = ReplayPublisher(_recording(), wrist_enabled=False)
    rotated = np.array([0.0, 0.3, 0.0, 0.954])
    rotated /= np.linalg.norm(rotated)
    frame = publisher._sample_to_proto(
        _sample(quaternion_xyzw=rotated), pose_epoch=1, timestamp_ns=0
    )
    assert frame.wrist_angle_degrees == 0.0
    # ...but the arm still gets the full orientation.
    assert frame.HasField("wrist_pose")


def test_replay_paces_itself_from_the_recorded_timestamps():
    samples = [_sample(t_ns=i * 20_000_000) for i in range(4)]  # 50 Hz
    publisher = ReplayPublisher(_recording(samples), speed=4.0)
    start = time.monotonic()
    frames = list(publisher._frame_generator())
    elapsed = time.monotonic() - start
    assert len(frames) == 4
    # 60 ms of tape at 4x is ~15 ms; anything near 60 ms means speed is ignored.
    assert elapsed < 0.05, elapsed


def test_looping_bumps_the_pose_epoch():
    """Restarting the take teleports the wrist, so any latched origin is stale."""
    publisher = ReplayPublisher(_recording([_sample()]), loop=True)
    epochs = []
    for frame in publisher._frame_generator():
        epochs.append(frame.pose_epoch)
        if len(epochs) == 3:
            publisher.stop()
    assert epochs == [1, 2, 3]


def test_speed_must_be_positive():
    with pytest.raises(ValueError, match="speed must be positive"):
        ReplayPublisher(_recording(), speed=0.0)


def test_from_table_rejects_an_unknown_hand():
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"t_ns": [0]})
    with pytest.raises(ValueError, match="handedness must be"):
        Recording.from_table(table, "third")
