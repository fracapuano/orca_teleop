import numpy as np
import torch

MANO_OFFSET = torch.tensor([0, 0, 0.015])


def get_hand_center_and_rotation(
    thumb_base: np.ndarray,
    index_base: np.ndarray,
    middle_base: np.ndarray,
    ring_base: np.ndarray,
    pinky_base: np.ndarray,
    wrist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hand center and a right-handed rotation matrix from finger bases.

    Axes:
        x — ring-base to index-base (lateral)
        y — wrist to middle-base (proximal)
        z — cross(x, y), points out of the palm for a right hand
    """
    hand_center = (thumb_base + pinky_base) / 2
    if wrist is None:
        wrist = hand_center

    y_axis = middle_base - wrist
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = index_base - ring_base
    x_axis -= (x_axis @ y_axis.T) * y_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    rot_matrix = np.concatenate(
        (x_axis.reshape(1, 3), y_axis.reshape(1, 3), z_axis.reshape(1, 3)), axis=0
    ).T
    assert np.allclose(
        rot_matrix @ rot_matrix.T, np.eye(3), atol=1e-6
    ), "Rotation matrix is not orthogonal"
    return hand_center, rot_matrix


def get_fingertip_urdf_name(hand_type: str, finger: str) -> str:
    return f"{hand_type}_{finger}_fingertip"


def get_finger_base_urdf_name(hand_type: str, finger: str) -> str:
    return f"{hand_type}_{finger}_mp"


def get_keyvectors(fingertips: dict[str, torch.Tensor], palm: torch.Tensor) -> list[torch.Tensor]:
    """Return palm-to-fingertip vectors for all five fingers."""
    return [
        fingertips["thumb"] - palm,
        fingertips["index"] - palm,
        fingertips["middle"] - palm,
        fingertips["ring"] - palm,
        fingertips["pinky"] - palm,
    ]


def extract_orca_fingertips_and_palm(
    chain,
    urdfhand_joint_angles: torch.Tensor,
    optimization_frames: list[int],
    hand_type: str,
    fingers: list[str],
    root: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run FK and return fingertip positions and palm for the Orca Hand."""
    chain_transforms = chain.forward_kinematics(
        urdfhand_joint_angles, frame_indices=optimization_frames
    )
    fingertips = {
        finger: chain_transforms[get_fingertip_urdf_name(hand_type, finger)].transform_points(root)
        for finger in fingers
    }
    thumb_base = chain_transforms[get_finger_base_urdf_name(hand_type, "thumb")].transform_points(
        root
    )
    pinky_base = chain_transforms[get_finger_base_urdf_name(hand_type, "pinky")].transform_points(
        root
    )
    palm = torch.mean(torch.cat([thumb_base, pinky_base], dim=0), dim=0, keepdim=True)

    # Hardcoded offset to align Orca Hand palm with MANO palm definition
    palm = palm - MANO_OFFSET.to(palm.device)
    return fingertips, palm


def get_urdf_model_params(
    chain,
    hand_type: str,
    fingers: list[str],
    root_point: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One-time init: compute hand geometry and FK frame indices for optimization."""
    transforms = chain.forward_kinematics(torch.zeros(chain.n_joints, device=root_point.device))
    finger_base_positions = {
        f"{finger}_base": transforms[get_finger_base_urdf_name(hand_type, finger)]
        .transform_points(root_point)
        .cpu()
        .numpy()
        for finger in fingers
    }
    wrist_pos = transforms[f"{hand_type}_palm"].transform_points(root_point).cpu().numpy()
    center, rot_matrix = get_hand_center_and_rotation(**finger_base_positions, wrist=wrist_pos)

    urdf_frame_names = [
        get_finger_base_urdf_name(hand_type, "thumb"),
        get_finger_base_urdf_name(hand_type, "pinky"),
    ] + [get_fingertip_urdf_name(hand_type, finger) for finger in fingers]
    optimization_frames = chain.get_frame_indices(*urdf_frame_names)

    return center, rot_matrix, optimization_frames
