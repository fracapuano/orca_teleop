"""Interop with orca_teleop's Pose -- the contract this package is built against.

Two layers, because each catches a different kind of drift:

* :class:`TranscribedPose` is a byte-for-byte transcription of upstream's ``Pose``
  (``orcahand/orca_teleop`` @ ``feat/wrist-pose``, commit ``1b5c85e``). It always
  runs, so CI keeps testing the documented shape even with no checkout around.
* The ``TestAgainstRealSource`` class loads the **actual** ``frames.py`` from a
  checkout and re-runs the same assertions. It skips when there is none. That is
  what would catch upstream changing the contract under us -- a transcription can
  only ever be as current as the day it was written.

Point ``ORCA_TELEOP_SRC`` at the ``src`` directory of a checkout to enable the
second layer::

    ORCA_TELEOP_SRC=/path/to/orca_teleop/src pytest tests/test_upstream_interop.py

``frames.py`` is loaded by file path rather than imported: it is numpy+stdlib
only, but ``orca_teleop.ingress.__init__`` eagerly imports the gRPC server, so a
plain import would need grpcio just to read a dataclass.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tron_arm.config import load_config
from tron_arm.poses import QUATERNION_NORM_TOLERANCE, Pose, as_pose, pose_lerp, slerp
from tron_arm.streamer import PoseStreamer, apply_step_clamp
from tron_arm.tron2_client import encode_servop_element, encode_servop_payload

UPSTREAM_COMMIT = "1b5c85e"
QUATERNION_NORM_TOLERANCE_UPSTREAM = 1e-3  # orca_teleop.constants


# -- layer 1: transcription ---------------------------------------------
def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Transcribed from frames.py -- Shepperd's method."""
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _rotation_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Transcribed from frames.py."""
    w, x, y, z = (float(v) for v in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class TranscribedPose:
    """orca_teleop.ingress.frames.Pose, transcribed at commit 1b5c85e."""

    position_m: np.ndarray
    orientation_wxyz: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64).reshape(-1)
        orientation = np.asarray(self.orientation_wxyz, dtype=np.float64).reshape(-1)
        if position.shape != (3,):
            raise ValueError(f"position_m must have shape (3,); got {position.shape}")
        if orientation.shape != (4,):
            raise ValueError(f"orientation_wxyz must have shape (4,); got {orientation.shape}")
        norm = float(np.linalg.norm(orientation))
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE_UPSTREAM:
            raise ValueError(f"orientation_wxyz must be a unit quaternion; |q|={norm:.6g}")
        orientation = orientation / norm
        position.setflags(write=False)
        orientation.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_wxyz", orientation)

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _rotation_from_quaternion_wxyz(self.orientation_wxyz)
        matrix[:3, 3] = self.position_m
        return matrix

    def as_xyz_wxyz(self) -> np.ndarray:
        return np.concatenate((self.position_m, self.orientation_wxyz))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "TranscribedPose":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a (4, 4) transform; got {matrix.shape}")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("Rotation block is not orthonormal")
        if float(np.linalg.det(rotation)) <= 0.0:
            raise ValueError("Rotation block is not a proper rotation (det <= 0)")
        return cls(matrix[:3, 3].copy(), _quaternion_wxyz_from_rotation(rotation))


# -- layer 2: the real thing --------------------------------------------
def _load_real_frames():
    """Load upstream ``frames.py`` by path, or return None."""
    candidates = []
    if os.environ.get("ORCA_TELEOP_SRC"):
        candidates.append(Path(os.environ["ORCA_TELEOP_SRC"]))
    candidates += [
        Path(__file__).resolve().parent.parent.parent / "orca_teleop" / "src",
        Path.home() / "orca_teleop" / "src",
    ]
    for src in candidates:
        frames = src / "orca_teleop" / "ingress" / "frames.py"
        if not frames.is_file():
            continue
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        spec = importlib.util.spec_from_file_location("_upstream_frames", frames)
        module = importlib.util.module_from_spec(spec)
        # Register before exec: the dataclass decorator resolves annotations
        # through sys.modules and blows up on a module that is not there yet.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - a checkout we cannot use
            sys.modules.pop(spec.name, None)
            continue
        return module
    return None


REAL_FRAMES = _load_real_frames()
requires_real = pytest.mark.skipif(
    REAL_FRAMES is None,
    reason="no orca_teleop checkout found; set ORCA_TELEOP_SRC to enable",
)

POSITION = np.array([0.40, -0.20, 0.05])
Q_Z90 = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])


def _contract_assertions(upstream_pose, cls) -> None:
    """The assertions both layers share."""
    # 1. as_pose consumes it without a matrix round-trip.
    ours = as_pose(upstream_pose)
    np.testing.assert_allclose(ours.position_m, upstream_pose.position_m)
    np.testing.assert_allclose(ours.orientation_wxyz, upstream_pose.orientation_wxyz)

    # 2. as_xyz_wxyz IS the servop element -- upstream says so explicitly.
    np.testing.assert_allclose(
        encode_servop_element(upstream_pose, "pos_quat"), upstream_pose.as_xyz_wxyz()
    )

    # 3. Both types agree on the homogeneous transform.
    np.testing.assert_allclose(ours.matrix, upstream_pose.matrix, atol=1e-12)

    # 4. Our Pose round-trips through theirs and back unchanged.
    mirrored = cls(ours.position_m, ours.orientation_wxyz)
    np.testing.assert_allclose(as_pose(mirrored).as_xyz_wxyz(), ours.as_xyz_wxyz())


class TestTranscribedContract:
    def test_field_names(self):
        pose = TranscribedPose(POSITION, Q_Z90)
        assert hasattr(pose, "position_m") and hasattr(pose, "orientation_wxyz")

    def test_our_pose_exposes_the_same_names(self):
        ours = Pose(POSITION, Q_Z90)
        upstream = TranscribedPose(POSITION, Q_Z90)
        for name in ("position_m", "orientation_wxyz", "matrix", "as_xyz_wxyz"):
            assert hasattr(ours, name), name
            assert hasattr(upstream, name), name

    def test_contract(self):
        _contract_assertions(TranscribedPose(POSITION, Q_Z90), TranscribedPose)

    def test_identical_validation_on_a_zero_quaternion(self):
        """The default-proto hazard: qw=0 must raise on both sides."""
        with pytest.raises(ValueError, match="unit quaternion"):
            TranscribedPose([0, 0, 0], [0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="unit quaternion"):
            Pose([0, 0, 0], [0.0, 0.0, 0.0, 0.0])

    def test_identical_norm_tolerance(self):
        assert QUATERNION_NORM_TOLERANCE == QUATERNION_NORM_TOLERANCE_UPSTREAM
        just_inside = 1.0 + 0.9 * QUATERNION_NORM_TOLERANCE
        just_outside = 1.0 + 1.1 * QUATERNION_NORM_TOLERANCE
        for cls in (Pose, TranscribedPose):
            cls([0, 0, 0], [just_inside, 0.0, 0.0, 0.0])
            with pytest.raises(ValueError):
                cls([0, 0, 0], [just_outside, 0.0, 0.0, 0.0])

    def test_identical_from_matrix_validation(self):
        for cls in (Pose, TranscribedPose):
            with pytest.raises(ValueError, match="orthonormal"):
                cls.from_matrix(np.diag([2.0, 2.0, 2.0, 1.0]))
            with pytest.raises(ValueError, match="proper rotation"):
                cls.from_matrix(np.diag([-1.0, 1.0, 1.0, 1.0]))

    def test_matrix_to_quat_agrees_with_upstream_shepperd(self):
        """All four branches, since they disagree about sign if transcribed wrong."""
        for q in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                  [0.5, 0.5, 0.5, 0.5], [0.2, -0.3, 0.6, 0.7]):
            q = np.asarray(q, dtype=np.float64)
            q = q / np.linalg.norm(q)
            rotation = _rotation_from_quaternion_wxyz(q)
            ours = Pose.from_matrix(
                np.block([[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
            )
            theirs = TranscribedPose.from_matrix(
                np.block([[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
            )
            np.testing.assert_allclose(ours.orientation_wxyz, theirs.orientation_wxyz, atol=1e-12)

    def test_upstream_poses_flow_through_the_whole_chain(self):
        """An upstream Pose must survive submit -> interpolate -> clamp -> encode."""
        config = load_config()
        streamer = PoseStreamer(config, lambda *_: None)
        a = TranscribedPose(np.array([0.40, -0.20, 0.00]), np.array([1.0, 0.0, 0.0, 0.0]))
        b = TranscribedPose(np.array([0.41, -0.21, 0.01]), Q_Z90)
        streamer.submit("right", a, 0)
        streamer.submit("right", b, 20_000_000)
        out = streamer.step(30_000_000)
        assert "right" in out
        assert len(encode_servop_element(out["right"], "pos_quat")) == 7

    def test_pose_lerp_accepts_upstream_poses_on_both_sides(self):
        a = TranscribedPose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        b = TranscribedPose(np.ones(3), Q_Z90)
        mid = pose_lerp(a, b, 0.5)
        np.testing.assert_allclose(mid.position_m, [0.5, 0.5, 0.5])

    def test_step_clamp_accepts_upstream_poses(self):
        last = TranscribedPose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        target = TranscribedPose(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        got, lin, _ang = apply_step_clamp(last, target, 0.004, 0.012)
        assert lin and float(np.linalg.norm(got.position_m)) == pytest.approx(0.004)

    def test_encode_payload_from_upstream_poses(self):
        pose = TranscribedPose(POSITION, Q_Z90)
        payload = encode_servop_payload("pos_quat", pose, pose)
        assert sorted(payload) == ["left_pos", "right_pos"]
        assert len(payload["left_pos"]) == 7

    def test_sign_continuity_is_preserved_across_the_boundary(self):
        """A negative-w quaternion must not be canonicalised on the way in."""
        q = np.array([-0.9987502603949663, 0.0, 0.0, -0.04997916927067833])
        upstream = TranscribedPose(np.zeros(3), q)
        assert as_pose(upstream).orientation_wxyz[0] < 0.0
        assert encode_servop_element(upstream, "pos_quat")[3] < 0.0


@requires_real
class TestAgainstRealSource:
    """Same contract, run against a real checkout of frames.py."""

    def test_loaded_from_a_checkout(self):
        assert REAL_FRAMES is not None
        assert hasattr(REAL_FRAMES, "Pose")

    def test_contract(self):
        _contract_assertions(REAL_FRAMES.Pose(POSITION, Q_Z90), REAL_FRAMES.Pose)

    def test_transcription_still_matches_the_source(self):
        """If this fails, upstream moved and the transcription above is stale."""
        real = REAL_FRAMES.Pose(POSITION, Q_Z90)
        copy = TranscribedPose(POSITION, Q_Z90)
        np.testing.assert_allclose(real.as_xyz_wxyz(), copy.as_xyz_wxyz(), atol=1e-15)
        np.testing.assert_allclose(real.matrix, copy.matrix, atol=1e-15)
        assert type(real).__dataclass_fields__.keys() == type(copy).__dataclass_fields__.keys()

    def test_real_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="unit quaternion"):
            REAL_FRAMES.Pose([0, 0, 0], [0.0, 0.0, 0.0, 0.0])

    def test_real_from_matrix_validation_matches(self):
        with pytest.raises(ValueError, match="orthonormal"):
            REAL_FRAMES.Pose.from_matrix(np.diag([2.0, 2.0, 2.0, 1.0]))
        with pytest.raises(ValueError, match="proper rotation"):
            REAL_FRAMES.Pose.from_matrix(np.diag([-1.0, 1.0, 1.0, 1.0]))

    def test_real_teleop_frame_exposes_the_fields_claude_md_documents(self):
        fields = REAL_FRAMES.TeleopFrame.__dataclass_fields__
        for name in ("timestamp_ns", "recv_monotonic_ns", "stream_id", "pose_epoch",
                     "handedness", "tracking_valid", "wrist", "head", "wrist_angle_degrees"):
            assert name in fields, name

    def test_real_pose_flows_through_the_whole_chain(self):
        config = load_config()
        streamer = PoseStreamer(config, lambda *_: None)
        a = REAL_FRAMES.Pose(np.array([0.40, -0.20, 0.00]), np.array([1.0, 0.0, 0.0, 0.0]))
        b = REAL_FRAMES.Pose(np.array([0.41, -0.21, 0.01]), Q_Z90)
        streamer.submit("right", a, 0)
        streamer.submit("right", b, 20_000_000)
        out = streamer.step(30_000_000)
        assert len(encode_servop_element(out["right"], "pos_quat")) == 7
