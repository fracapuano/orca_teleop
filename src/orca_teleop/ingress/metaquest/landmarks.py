"""Landmark and wrist adapters for the repository-owned Meta Quest WebXR client."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

WEBXR_HAND_JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)

# WebXR has one extra metacarpal joint for every non-thumb finger. The
# retargeter consumes the MediaPipe/MANO-like 21-point layout, whose MCP point
# corresponds to WebXR's phalanx-proximal joint.
WEBXR_TO_RETARGETER_LANDMARK_INDICES = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
)

# WebXR: right-handed, X right, Y up, -Z forward.
# Robot/MuJoCo FLU: X forward, Y left, Z up.
XR_TO_FLU_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def retargeter_landmarks_from_webxr(points: np.ndarray, handedness: str) -> np.ndarray:
    """Reduce live WebXR's 25 joints to the retargeter's 21-joint layout.

    WebXR already reports a right-handed point cloud, so this adapter only
    removes the four extra non-thumb metacarpal joints. It deliberately does
    not mirror either hand.
    """
    if handedness not in ("left", "right"):
        raise ValueError(f"Unsupported Meta Quest handedness: {handedness!r}")
    landmarks = np.asarray(points, dtype=np.float64)
    if landmarks.shape != (25, 3):
        raise ValueError(f"Expected WebXR landmarks with shape (25, 3), got {landmarks.shape}")
    return landmarks[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)].copy()


def retargeter_landmarks_from_quest(points: np.ndarray, handedness: str) -> np.ndarray:
    """Adapt legacy recorded HTS/Unity landmarks.

    This remains for replaying the old ``quest-poses`` datasets. Live Quest
    Browser frames must use :func:`retargeter_landmarks_from_webxr` instead.
    """
    if handedness not in ("left", "right"):
        raise ValueError(f"Unsupported Meta Quest handedness: {handedness!r}")
    landmarks = np.asarray(points, dtype=np.float64).copy()
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"Meta Quest landmarks must have shape (N, 3); got {landmarks.shape}")
    if handedness == "right":
        landmarks[:, 1] *= -1.0
    return landmarks


def xr_flat_matrix_to_numpy(raw_matrix: Sequence[float]) -> np.ndarray:
    """Decode a column-major WebXR transform matrix."""
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"Expected 16 values for a WebXR pose matrix, got {matrix.size}")
    return matrix.reshape(4, 4).T


def xr_matrix_to_flu_matrix(raw_matrix: Sequence[float]) -> np.ndarray:
    """Convert a WebXR transform to the robot's forward-left-up basis."""
    xr_matrix = xr_flat_matrix_to_numpy(raw_matrix)
    flu_matrix = np.eye(4, dtype=np.float64)
    flu_matrix[:3, :3] = XR_TO_FLU_BASIS @ xr_matrix[:3, :3] @ XR_TO_FLU_BASIS.T
    flu_matrix[:3, 3] = XR_TO_FLU_BASIS @ xr_matrix[:3, 3]
    return flu_matrix


class QuaternionContinuity:
    """Keep the sign of successive quaternions consistent along a stream.

    ``q`` and ``-q`` are the same rotation, and a matrix-to-quaternion
    extraction picks between them by whichever matrix element happened to be
    largest. A consumer that differences or slerps successive poses therefore
    sees a full 360° jump for a rotation that never happened.

    Note this deliberately does *not* canonicalize to ``w >= 0``: that rule
    manufactures exactly the same discontinuity whenever the wrist passes 180°
    from its reference, which an operator reaches routinely. Only the seed
    frame is canonicalized; after that each quaternion is aligned to its
    predecessor.
    """

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None
        self._pose_epoch: int | None = None

    def reset(self) -> None:
        self._previous = None
        self._pose_epoch = None

    def update(self, quaternion_wxyz: np.ndarray, pose_epoch: int = 0) -> np.ndarray:
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        # Across a reference-space change there is no continuity to preserve.
        if self._pose_epoch != pose_epoch:
            self._pose_epoch = pose_epoch
            self._previous = None

        if self._previous is None:
            if quaternion[0] < 0.0:
                quaternion = -quaternion
        elif float(np.dot(quaternion, self._previous)) < 0.0:
            quaternion = -quaternion

        self._previous = quaternion
        return quaternion


class WristAngleEstimator:
    """Reproduce the WebXR demonstration pipeline's relative wrist control."""

    def __init__(self) -> None:
        self._zero_pitch_rad: float | None = None

    @staticmethod
    def _hand_x_pitch_rad(wrist_matrix: np.ndarray) -> float:
        matrix = np.asarray(wrist_matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected wrist matrix with shape (4, 4), got {matrix.shape}")
        x_in_world = matrix[:3, 0]
        return float(np.arcsin(np.clip(x_in_world[2], -1.0, 1.0)))

    def reset(self) -> None:
        self._zero_pitch_rad = None

    def update(self, wrist_matrix: np.ndarray | None) -> float:
        if wrist_matrix is None:
            return 0.0
        pitch = self._hand_x_pitch_rad(wrist_matrix)
        if self._zero_pitch_rad is None:
            self._zero_pitch_rad = pitch
            return 0.0
        return float(np.degrees(self._zero_pitch_rad - pitch))
