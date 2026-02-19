from typing import Dict, Tuple, List, Union
import torch
import numpy as np

def preprocess_avp_data(data: Dict, hand_type: str) -> Tuple[np.ndarray, float]:
    """Extract joint translations and wrist angle from Apple Vision Pro data dict."""
    
    if hand_type not in ["left", "right"]:
        raise ValueError(f"Invalid hand_type: {hand_type}. Must be 'left' or 'right'")
    wrist = data[f"{hand_type}_wrist"]
    fingers = data[f"{hand_type}_fingers"]
    
    _, wrist_angle, _ = compute_roll_pitch_yaw(wrist)
    translations = fingers[:, :3, 3]
    indices_to_remove = [5, 10, 15, 20]
    translations = np.delete(translations, indices_to_remove, axis=0)
    first_row = translations[0:1]  # Add the first index to the beginning of the translations
    translations = np.vstack((first_row, translations))  
    return translations, np.rad2deg(wrist_angle)


def preprocess_mediapipe_data(data: Dict) -> Tuple[np.ndarray, float]:
    """Extract joint positions from MediaPipe landmarks (assumes 21 MANO points)."""

    landmarks = data["hand_landmarks"]
    joints = landmarks * 1.2
    wrist_angle = 0.0  # Default to no rotation for MediaPipe
    return joints, wrist_angle


def preprocess_manus_data(data: Dict) -> Tuple[np.ndarray, float]:
    """Extract joint positions and wrist angle from Manus glove skeleton data."""
    skeleton = data["skeleton"]  # (25, 7): x, y, z, qx, qy, qz, qw
    joints = skeleton[:, :3].copy()
    joints[:, 2] *= -1  # Negate Z to match retargeter curl convention
    wrist_quat = skeleton[0, 3:7]  # qx, qy, qz, qw
    rot_matrix = quaternion_to_rotation_matrix(wrist_quat)
    _, wrist_angle, _ = compute_roll_pitch_yaw(rot_matrix)
    return joints, np.rad2deg(wrist_angle)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qx, qy, qz, qw) to a 4x4 rotation matrix."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ])
    mat = np.eye(4)
    mat[:3, :3] = R
    return mat


def compute_roll_pitch_yaw(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Compute roll, pitch, yaw angles from a 4x4 rotation matrix."""
    
    rotation_matrix = np.squeeze(rotation_matrix)
    R = rotation_matrix[:3, :3]
    pitch = -np.arcsin(R[2, 0])

    if np.abs(R[2, 0]) != 1.0:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = 0
        if R[2, 0] == -1:
            roll = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            roll = np.arctan2(R[0, 1], R[0, 2])
    return roll, -pitch, yaw


def get_mano_joints_dict(joints: torch.Tensor, source: str):
    """Return MANO joint dictionary (wrist, thumb, index, middle, ring, pinky) based on source."""
    
    if source == "mediapipe":
        # MediaPipe structure
        return {
            "wrist": joints[0, :],  # Wrist is joint 0
            "thumb": joints[1:5, :],  # Thumb joints 1-4
            "index": joints[5:9, :],  # Index joints 5-8  
            "middle": joints[9:13, :],  # Middle joints 9-12
            "ring": joints[13:17, :],  # Ring joints 13-16
            "pinky": joints[17:21, :],  # Pinky joints 17-20
        }
    elif source == "avp":
        # AVP structure
        return {
            "wrist": joints[1, :],
            "thumb": joints[2:6, :],
            "index": joints[6:10, :],
            "middle": joints[10:14, :],
            "ring": joints[14:18, :],
            "pinky": joints[18:22, :],
    }
    elif source == "manus":
        # Manus 25-node layout: 0=Wrist, 1-4=Index, 5-9=Middle, 10-14=Pinky, 15-19=Ring, 20-24=Thumb
        # Skip CMC node (first in each 5-node finger chain) to get 4 joints per finger
        return {
            "wrist": joints[0, :],
            "thumb": joints[21:25, :],
            "index": joints[1:5, :],
            "middle": joints[6:10, :],
            "ring": joints[16:20, :],
            "pinky": joints[11:15, :],
        }
    else:
        raise ValueError(f"Unsupported source right now: {source}")


def extract_mano_fingertips_and_palm(joints: torch.Tensor, fingers: List[str], source: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract fingertips and palm position from MANO joints tensor for given fingers."""

    mano_joints_dict = get_mano_joints_dict(joints, source)
    mano_fingertips, mano_bases = {}, {}
    for finger in fingers:
        finger_joints = mano_joints_dict[finger]
        mano_fingertips[finger] = finger_joints[[-1], :]  # Last joint (fingertip)
        mano_bases[finger] = finger_joints[[0], :]  # First joint (base)
    mano_palm = torch.mean(torch.cat([mano_bases["thumb"], mano_bases["pinky"]], dim=0), dim=0, keepdim=True)  # Calculate palm position
    return mano_fingertips, mano_palm


def extract_orca_fingertips_and_palm(chain, urdfhand_joint_angles: torch.Tensor, optimization_frames: List[int], hand_type: str, fingers: List[str], root: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Extract fingertips and palm from Orca hand chain transforms for given fingers."""
    chain_transforms = chain.forward_kinematics(urdfhand_joint_angles, frame_indices=optimization_frames)
    orca_fingertips = {finger: chain_transforms[get_fingertip_urdf_name(hand_type, finger)].transform_points(root) for finger in fingers}
    # Calculate palm position (same way as MANO)
    orca_thumb_base = chain_transforms[get_finger_base_urdf_name(hand_type, "thumb")].transform_points(root)
    orca_pinky_base = chain_transforms[get_finger_base_urdf_name(hand_type, "pinky")].transform_points(root)
    orca_palm = torch.mean(torch.cat([orca_thumb_base, orca_pinky_base], dim=0), dim=0, keepdim=True)
    palm_offset = torch.tensor([0, 0, 0.015], device=orca_palm.device) # Hardcoded offset to align the keypoints better
    orca_palm = orca_palm - palm_offset

    return orca_fingertips, orca_palm


def get_keyvectors(fingertips: Dict[str, torch.Tensor], palm: torch.Tensor) -> List[torch.Tensor]:
    """Return key vectors from palm to fingertips for all fingers."""
    return [
        fingertips["thumb"] - palm,
        fingertips["index"] - palm,
        fingertips["middle"] - palm,
        fingertips["ring"] - palm,
        fingertips["pinky"] - palm,
    ]


def rotate_points_around_y(joints: np.ndarray, angle_degrees: float, source: str, hand_type: str = "right") -> np.ndarray:
    """Rotate joint positions around the y-axis by a given angle (degrees)."""
    
    joint_dict = get_mano_joints_dict(joints, source)
    wrist = joint_dict["wrist"]
    
    # # Invert angle for left hand
    # if hand_type == "left":
    #     angle_degrees *= -1
        
    angle_radians = -np.radians(angle_degrees)  
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])
    translated_joints = joints - wrist
    rotated_joints = translated_joints @ rotation_matrix.T
    rotated_joints += wrist
    return rotated_joints


def get_fingertip_urdf_name(hand_type: str, finger: str) -> str:
    """Return URDF name for fingertip of a given finger and hand type."""
    return f"{hand_type}_{finger}_fingertip"


def get_finger_base_urdf_name(hand_type: str, finger: str) -> str:
    """Return URDF name for base joint of a given finger and hand type."""
    return f"{hand_type}_{finger}_mp"


def get_hand_center_and_rotation(
    thumb_base: np.ndarray, index_base: np.ndarray, middle_base: np.ndarray, ring_base: np.ndarray, pinky_base: np.ndarray, wrist: Union[np.ndarray, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hand center and rotation matrix from finger base joints and wrist.
    x axis is the direction from ring to index finger base
    y axis is the direction from wrist to middle finger base
    z axis goes from the palm if the hand is right hand, otherwise it goes to the palm
    """
    hand_center = (thumb_base + pinky_base) / 2
    hand_center = hand_center
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
            (rot_matrix @ rot_matrix.T), 
            np.eye(3), 
            atol=1e-6
    ), "Model rotation matrix is not orthogonal"

    return hand_center, rot_matrix


def get_normalized_local_manohand_joint_pos(joint_pos: np.ndarray, source: str) -> np.ndarray:
    """Normalize MANO joint positions to local hand coordinate system (translation + rotation)."""

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


def to_geort_canonical_frame(joints: np.ndarray, source: str) -> np.ndarray:
    """Transform landmarks to GeoRT's wrist-centered canonical frame.

    Matches GeoRT's MediaPipeHandProcessor.forward() convention:
      Z: wrist → middle finger MCP
      X: palm normal (cross of index→ring direction with Z)
      Y: cross(Z, X)

    The origin is placed at the wrist joint.
    """
    joint_dict = get_mano_joints_dict(joints, source)
    wrist = joint_dict["wrist"]

    middle_base = joint_dict["middle"][0]
    index_base = joint_dict["index"][0]
    ring_base = joint_dict["ring"][0]

    # Z-axis: wrist → middle finger base
    z_axis = middle_base - wrist
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Auxiliary Y direction: index base → ring base (across the palm)
    y_aux = index_base - ring_base
    y_aux = y_aux / np.linalg.norm(y_aux)

    # X-axis: palm normal
    x_axis = np.cross(y_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Y-axis: orthogonal completion
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3) columns are axes

    # Transform: translate to wrist origin, rotate into canonical frame
    canonical = (joints - wrist) @ rot
    return canonical


def get_urdf_model_params(chain, hand_type: str, fingers: List[str], root_point: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Initialize URDF hand model: center, rotation, and frame indices for optimization."""

    transforms = chain.forward_kinematics(torch.zeros(chain.n_joints, device=root_point.device))
    finger_base_positions = {f"{finger}_base": transforms[get_finger_base_urdf_name(hand_type, finger)].transform_points(root_point).cpu().numpy() for finger in fingers}
    wrist_pos = transforms[f"{hand_type}_palm"].transform_points(root_point).cpu().numpy()
    urdfhand_center, urdfhand_rot_matrix = get_hand_center_and_rotation(**finger_base_positions, wrist=wrist_pos)
    # Get relevant urdf hand model frame indices for optimization
    urdf_frame_names = [get_finger_base_urdf_name(hand_type, "thumb"), get_finger_base_urdf_name(hand_type, "pinky")] + [get_fingertip_urdf_name(hand_type, finger) for finger in fingers]
    optimization_frames = chain.get_frame_indices(*urdf_frame_names)
    return urdfhand_center, urdfhand_rot_matrix, optimization_frames
