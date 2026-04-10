"""Utilities for converting raw sensor data into canonical key vectors.

These functions sit between the ingress layer (sensor data) and the
retargeter (OrcaJointPositions). The main entry point for most callers
is ``get_canonical_key_vectors``, which converts raw joint positions from
any supported source into the palm-to-fingertip key vectors expected by
``TargetPose``.
"""

import numpy as np
import torch

from orca_teleop.retargeting.utils import get_hand_center_and_rotation


def preprocess_avp_data(data: dict, hand_type: str) -> tuple[np.ndarray, float]:
    """Extract joint translations and wrist angle from Apple Vision Pro data."""
    if hand_type not in ("left", "right"):
        raise ValueError(f"Invalid hand_type: {hand_type!r}. Must be 'left' or 'right'.")
    wrist = data[f"{hand_type}_wrist"]
    fingers = data[f"{hand_type}_fingers"]

    _, wrist_angle, _ = compute_roll_pitch_yaw(wrist)
    translations = fingers[:, :3, 3]
    indices_to_remove = [5, 10, 15, 20]
    translations = np.delete(translations, indices_to_remove, axis=0)
    first_row = translations[0:1]
    translations = np.vstack((first_row, translations))
    return translations, np.rad2deg(wrist_angle)


def preprocess_mediapipe_data(data: dict) -> tuple[np.ndarray, float]:
    """Extract joint positions from MediaPipe landmarks (21 MANO points)."""
    landmarks = data["hand_landmarks"]
    joints = landmarks * 1.2
    wrist_angle = 0.0  # MediaPipe does not provide wrist rotation
    return joints, wrist_angle


def compute_roll_pitch_yaw(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Compute roll, pitch, yaw from a 4×4 rotation matrix."""
    rotation_matrix = np.squeeze(rotation_matrix)
    R = rotation_matrix[:3, :3]
    pitch = -np.arcsin(R[2, 0])
    if np.abs(R[2, 0]) != 1.0:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = 0.0
        if R[2, 0] == -1:
            roll = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            roll = np.arctan2(R[0, 1], R[0, 2])
    return roll, -pitch, yaw


def get_mano_joints_dict(joints: np.ndarray | torch.Tensor, source: str) -> dict:
    """Split a flat joint array into a per-finger dict based on source convention."""
    if source == "mediapipe":
        return {
            "wrist": joints[0, :],
            "thumb": joints[1:5, :],
            "index": joints[5:9, :],
            "middle": joints[9:13, :],
            "ring": joints[13:17, :],
            "pinky": joints[17:21, :],
        }
    elif source == "avp":
        return {
            "wrist": joints[1, :],
            "thumb": joints[2:6, :],
            "index": joints[6:10, :],
            "middle": joints[10:14, :],
            "ring": joints[14:18, :],
            "pinky": joints[18:22, :],
        }
    else:
        raise ValueError(f"Unsupported source: {source!r}")


def extract_mano_fingertips_and_palm(
    joints: torch.Tensor,
    fingers: list[str],
    source: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Extract fingertips and palm position from a MANO joint tensor."""
    joints_dict = get_mano_joints_dict(joints, source)
    fingertips, bases = {}, {}
    for finger in fingers:
        finger_joints = joints_dict[finger]
        fingertips[finger] = finger_joints[[-1], :]  # last joint = fingertip
        bases[finger] = finger_joints[[0], :]  # first joint = base
    palm = torch.mean(torch.cat([bases["thumb"], bases["pinky"]], dim=0), dim=0, keepdim=True)
    return fingertips, palm


def get_normalized_local_manohand_joint_pos(joint_pos: np.ndarray, source: str) -> np.ndarray:
    """Express joint positions in the hand's local coordinate frame.

    Removes global translation and rotation so that the resulting key vectors
    are invariant to where and how the hand is positioned in the world.
    """
    joint_dict = get_mano_joints_dict(joint_pos, source)
    hand_center, hand_rot = get_hand_center_and_rotation(
        thumb_base=joint_dict["thumb"][0],
        index_base=joint_dict["index"][0],
        middle_base=joint_dict["middle"][0],
        ring_base=joint_dict["ring"][0],
        pinky_base=joint_dict["pinky"][0],
        wrist=joint_dict["wrist"],
    )
    joint_pos = joint_pos - hand_center
    joint_pos = joint_pos @ hand_rot
    return joint_pos


def get_canonical_key_vectors(
    joint_pos: np.ndarray,
    source: str,
    fingers: list[str] | None = None,
) -> np.ndarray:
    """Convert raw sensor joint positions to palm-to-fingertip key vectors.

    This is the main entry point for preparing a ``TargetPose``. It normalises
    the joint positions to a local hand frame, then returns the five
    palm-to-fingertip vectors stacked as a ``(5, 3)`` array.

    Args:
        joint_pos: Raw joint positions from the sensor (shape depends on source).
        source: ``"mediapipe"`` or ``"avp"``.
        fingers: Finger order to use. Defaults to
            ``["thumb", "index", "middle", "ring", "pinky"]``.

    Returns:
        ``(5, 3)`` float32 ndarray of palm-to-fingertip key vectors, ready to
        pass as ``TargetPose(key_vectors=...)``.
    """
    if fingers is None:
        fingers = ["thumb", "index", "middle", "ring", "pinky"]

    normalized = get_normalized_local_manohand_joint_pos(joint_pos, source)
    joint_tensor = torch.tensor(normalized, dtype=torch.float32)
    fingertips, palm = extract_mano_fingertips_and_palm(joint_tensor, fingers, source)
    key_vectors = [fingertips[f] - palm for f in fingers]
    return torch.cat(key_vectors, dim=0).cpu().numpy()
