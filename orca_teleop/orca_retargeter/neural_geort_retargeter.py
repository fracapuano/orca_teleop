import os
import sys
import json
from typing import Dict, Union
import numpy as np
import torch
from orca_core import OrcaHand
from .utils import retargeter_utils

# Add GeoRT to path so we can import its modules
_GEORT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GeoRT")
if _GEORT_ROOT not in sys.path:
    sys.path.insert(0, _GEORT_ROOT)

from geort.model import IKModel


class NeuralGeoRTRetargeter:
    """Neural retargeter using GeoRT's pre-trained IK model (single MLP forward pass)."""

    def __init__(self, model_path: Union[OrcaHand, str] = None, urdf_path: Union[str, None] = None,
                 geort_checkpoint: str = None, geort_config: str = None, source: str = "none") -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = source
        self.target_angles = None
        self.mano_points = None

        # Load ORCA hand for joint IDs and ROM limits
        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type
        self.joint_ids = hand.joint_ids
        self.urdf_joint_ids = [f"{hand.type}_{joint_id}" for joint_id in self.joint_ids]
        lower_limits, upper_limits = map(list, zip(*hand.joint_roms_dict.values()))
        self.wrist_limit_lower = lower_limits[16]
        self.wrist_limit_upper = upper_limits[16]

        # Load GeoRT config
        with open(geort_config, 'r') as f:
            config = json.load(f)

        # Parse keypoint info from config (same logic as geort.utils.config_utils)
        joint_order = config["joint_order"]
        keypoint_joints = []
        self.human_ids = []
        for info in config["fingertip_link"]:
            self.human_ids.append(info["human_hand_id"])
            keypoint_joints.append([joint_order.index(j) for j in info["joint"]])

        # Joint limits from trained config (added during training)
        joint_lower = np.array(config["joint"]["lower"])
        joint_upper = np.array(config["joint"]["upper"])
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper

        # Build IK model and load checkpoint
        self.ik_model = IKModel(keypoint_joints=keypoint_joints).to(self.device)
        state_dict = torch.load(geort_checkpoint, map_location=self.device, weights_only=True)
        self.ik_model.load_state_dict(state_dict)
        self.ik_model.eval()

        # Build mapping from GeoRT's joint_order (16 joints) to ORCA's urdf_joint_ids (17 joints, last is wrist)
        # GeoRT outputs joints in config's joint_order; we need to reorder to match self.urdf_joint_ids
        self.geort_to_orca_indices = []
        for geort_idx, geort_joint_name in enumerate(joint_order):
            # Strip hand prefix to get the bare joint id (e.g. "right_thumb_abd" -> "thumb_abd")
            bare_name = geort_joint_name.split("_", 1)[1] if geort_joint_name.startswith(("left_", "right_")) else geort_joint_name
            orca_idx = self.joint_ids.index(bare_name)
            self.geort_to_orca_indices.append(orca_idx)

        # Source-specific fingertip indices for extracting from canonical frame landmarks
        # Mapping: finger name → fingertip index in the source landmark array
        finger_names = [info["name"] for info in config["fingertip_link"]]
        self._fingertip_indices = self._get_fingertip_indices(finger_names)

    def _get_fingertip_indices(self, finger_names):
        """Get fingertip landmark indices per source, in config's finger order."""
        # Fingertip index lookup per source
        mediapipe_tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        manus_tips = {"thumb": 24, "index": 4, "middle": 9, "ring": 19, "pinky": 14}

        if self.source == "manus":
            return [manus_tips[name] for name in finger_names]
        else:
            # MediaPipe and AVP both use MediaPipe-style indexing
            return [mediapipe_tips[name] for name in finger_names]

    def retarget(self, data: np.ndarray, manual_wrist_angle: Union[float, None] = None) -> Dict[str, float]:

        if self.source == "avp":
            joints, computed_wrist_angle = retargeter_utils.preprocess_avp_data(data, self.hand_type)
        elif self.source == "mediapipe":
            joints, computed_wrist_angle = retargeter_utils.preprocess_mediapipe_data(data)
        elif self.source == "manus":
            joints, computed_wrist_angle = retargeter_utils.preprocess_manus_data(data)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

        final_wrist_angle = manual_wrist_angle if manual_wrist_angle is not None else computed_wrist_angle

        # Transform to GeoRT's canonical wrist-centered frame
        canonical = retargeter_utils.to_geort_canonical_frame(joints, self.source)

        # Extract fingertip positions in config's finger order
        fingertips = canonical[self._fingertip_indices]  # (N_fingers, 3)

        # IK model forward pass
        with torch.no_grad():
            fingertips_t = torch.from_numpy(fingertips).unsqueeze(0).float().to(self.device)  # (1, N_fingers, 3)
            normalized_angles = self.ik_model(fingertips_t)  # (1, 16) in [-1, 1]
            normalized_angles = normalized_angles[0].cpu().numpy()  # (16,)

        # Unnormalize: [-1, 1] → [lower, upper] (GeoRT convention: tanh output)
        raw_angles = (normalized_angles / 2.0 + 0.5) * (self.joint_upper - self.joint_lower) + self.joint_lower

        # GeoRT outputs radians (URDF convention). ORCA retargeters work in degrees internally.
        angles_deg = np.rad2deg(raw_angles)

        # Reorder from GeoRT's joint_order to ORCA's joint_ids order
        orca_angles = np.zeros(len(self.urdf_joint_ids))
        for geort_idx, orca_idx in enumerate(self.geort_to_orca_indices):
            orca_angles[orca_idx] = angles_deg[geort_idx]

        # Wrist angle (handled separately, not part of GeoRT model)
        final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
        orca_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
        self.target_angles = orca_angles

        # Store visualization points (canonical frame joints for viewer compatibility)
        self.mano_points = joints

        return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, orca_angles)}
