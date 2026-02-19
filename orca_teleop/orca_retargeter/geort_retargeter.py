import os
from typing import Dict, Tuple, Union
import numpy as np
import pytorch_kinematics as pk
import torch
import yaml
from orca_core import OrcaHand
from .utils import retargeter_utils

class GeoRTRetargeter:
    """GeoRT-inspired retargeter using normalized key vectors (direction-only) and pinch distance matching."""

    def __init__(self, model_path: Union[OrcaHand, str] = None, urdf_path: Union[str, None] = None, source: str = "none") -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = source
        self.target_angles = None

        if not os.path.exists(urdf_path):
            raise ValueError(f"URDF file not found at {urdf_path}")
        with open(urdf_path, 'r') as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(device=self.device)

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type
        self.joint_ids = hand.joint_ids
        self.urdf_joint_ids = [f"{hand.type}_{joint_id}" for joint_id in self.joint_ids]
        self.fingers = ["thumb", "index", "middle", "ring", "pinky"]
        lower_limits, upper_limits = map(list, zip(*hand.joint_roms_dict.values()))
        self.wrist_limit_lower = lower_limits[16]
        self.wrist_limit_upper = upper_limits[16]
        lower_limits[16] = upper_limits[16] = 0.0
        self.joint_angle_limits_lower = torch.tensor(lower_limits, device=self.device)
        self.joint_angle_limits_upper = torch.tensor(upper_limits, device=self.device)

        urdf_joint_parameter_names = self.chain.get_joint_parameter_names()
        assert set(self.urdf_joint_ids) == set(urdf_joint_parameter_names), "Joint name mismatch between the user defined urdf joint_ids and the actual joint names in the URDF file. Please check if your config.yaml and URDF file have the same hand type (left/right) and are up to date."
        self.joint_reorder_indices = [urdf_joint_parameter_names.index(name) for name in self.urdf_joint_ids]

        with open(os.path.join(os.path.dirname(__file__), "utils", "geort_retargeter.yaml"), 'r') as file:
            cfg = yaml.safe_load(file)
        self.lr = cfg["lr"]
        self.direction_loss_coeffs = torch.tensor(cfg["direction_loss_coeffs"], device=self.device)
        self.pinch_loss_weight = cfg["pinch_loss_weight"]

        finger_to_idx = {f: i for i, f in enumerate(self.fingers)}
        self.pinch_pairs = [(finger_to_idx[a], finger_to_idx[b], w) for a, b, w in cfg["pinch_pairs"]]

        self.orcahand_joint_angles = torch.zeros(len(self.urdf_joint_ids), device=self.device, requires_grad=True)
        self.opt = torch.optim.RMSprop([self.orcahand_joint_angles], lr=self.lr)

        self.root = torch.zeros(1, 3, device=self.device)
        self.regularizer_zeros = torch.zeros(len(self.urdf_joint_ids), device=self.device)
        self.regularizer_weights = torch.zeros(len(self.urdf_joint_ids), device=self.device)
        for joint_id, zero_val, weight in cfg["joint_regularizers"]:
            idx = self.joint_ids.index(joint_id)
            self.regularizer_zeros[idx] = zero_val
            self.regularizer_weights[idx] = weight

        self.urdfhand_center, self.urdfhand_rot_matrix, self.optimization_frames = retargeter_utils.get_urdf_model_params(
            self.chain, self.hand_type, self.fingers, self.root)

        # Compute URDF key vector magnitudes at zero config (for auto-scaling)
        zero_angles = torch.zeros(self.chain.n_joints, device=self.device)
        urdf_fingertips, urdf_palm = retargeter_utils.extract_orca_fingertips_and_palm(
            self.chain, zero_angles, self.optimization_frames, self.hand_type, self.fingers, self.root)
        urdf_keyvectors = retargeter_utils.get_keyvectors(urdf_fingertips, urdf_palm)
        self._urdf_keyvector_mags = np.array([kv.detach().cpu().norm().item() for kv in urdf_keyvectors])

        # Auto-scale calibration state (global scale only, no per-finger corrections)
        self.mano_scale = 1.0
        self._calibration_frames = 30
        self._calibration_mags = []


    def optimize_orcahand_joint_angles(self, manohand_joint_pos: np.ndarray, opt_steps: int = 2) -> np.ndarray:

        manohand_joint_pos = torch.from_numpy(manohand_joint_pos).to(self.device)
        manohand_fingertips, manohand_palm = retargeter_utils.extract_mano_fingertips_and_palm(manohand_joint_pos, self.fingers, self.source)
        mano_kvs = retargeter_utils.get_keyvectors(manohand_fingertips, manohand_palm)

        # Normalize MANO key vectors (direction only)
        mano_dirs = [kv / kv.norm().clamp(min=1e-6) for kv in mano_kvs]

        # Extract MANO fingertip positions as a list for pinch loss
        mano_ft_list = [manohand_fingertips[f] for f in self.fingers]

        for _ in range(opt_steps):

            urdfhand_joint_angles = torch.zeros(self.chain.n_joints, device=self.device)
            urdfhand_joint_angles[self.joint_reorder_indices] = self.orcahand_joint_angles / (180.0 / np.pi)
            urdfhand_fingertips, urdfhand_palm = retargeter_utils.extract_orca_fingertips_and_palm(self.chain, urdfhand_joint_angles, self.optimization_frames, self.hand_type, self.fingers, self.root)
            urdf_kvs = retargeter_utils.get_keyvectors(urdfhand_fingertips, urdfhand_palm)

            # Normalize URDF key vectors (direction only)
            urdf_dirs = [kv / kv.norm().clamp(min=1e-6) for kv in urdf_kvs]

            # Direction loss: ||dir_mano - dir_urdf||²
            loss = sum(
                self.direction_loss_coeffs[i] * torch.norm(mano_dirs[i] - urdf_dirs[i]) ** 2
                for i in range(5)
            )

            # Pinch distance loss: ||(||ft_h_i - ft_h_j|| - ||ft_r_i - ft_r_j||)||²
            urdf_ft_list = [urdfhand_fingertips[f] for f in self.fingers]
            for idx_a, idx_b, weight in self.pinch_pairs:
                dist_mano = torch.norm(mano_ft_list[idx_a] - mano_ft_list[idx_b])
                dist_urdf = torch.norm(urdf_ft_list[idx_a] - urdf_ft_list[idx_b])
                loss += self.pinch_loss_weight * weight * (dist_mano - dist_urdf) ** 2

            # Joint regularization
            loss += torch.sum(self.regularizer_weights * (self.orcahand_joint_angles - self.regularizer_zeros) ** 2)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.orcahand_joint_angles.clamp_(self.joint_angle_limits_lower, self.joint_angle_limits_upper)

        return self.orcahand_joint_angles.detach().cpu().numpy()


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
        manohand_joint_pos = retargeter_utils.get_normalized_local_manohand_joint_pos(joints, self.source)

        # Auto-scale calibration: collect MANO key vector magnitudes for first N valid frames
        if len(self._calibration_mags) < self._calibration_frames:
            mano_t = torch.from_numpy(manohand_joint_pos).to(self.device)
            mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(mano_t, self.fingers, self.source)
            mano_kvs = retargeter_utils.get_keyvectors(mano_ft, mano_palm)
            mano_mags = np.array([kv.detach().cpu().norm().item() for kv in mano_kvs])
            self._calibration_mags.append(mano_mags)
            if len(self._calibration_mags) == self._calibration_frames:
                all_mags = np.array(self._calibration_mags)
                median_mano_mags = np.median(all_mags, axis=0)
                ratios = self._urdf_keyvector_mags / np.clip(median_mano_mags, 1e-6, None)
                self.mano_scale = float(np.median(ratios))
                print(f"GeoRT auto-scale calibrated: {self.mano_scale:.4f}")
                for i, finger in enumerate(self.fingers):
                    print(f"  {finger}: URDF={self._urdf_keyvector_mags[i]:.4f} MANO={median_mano_mags[i]:.4f} ratio={ratios[i]:.4f}")
                with torch.no_grad():
                    self.orcahand_joint_angles.zero_()
                self.opt = torch.optim.RMSprop([self.orcahand_joint_angles], lr=self.lr)

        manohand_joint_pos = manohand_joint_pos * self.mano_scale
        manohand_joint_pos = manohand_joint_pos @ self.urdfhand_rot_matrix.T + self.urdfhand_center + np.array([0, 0, -0.02])

        if len(self._calibration_mags) < self._calibration_frames:
            zero_angles = np.zeros(len(self.urdf_joint_ids))
            final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
            zero_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
            self.target_angles = zero_angles
            self.mano_points = retargeter_utils.rotate_points_around_y(manohand_joint_pos, final_wrist_angle, self.source, self.hand_type)
            return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, zero_angles)}

        optimized_angles = self.optimize_orcahand_joint_angles(manohand_joint_pos)

        final_wrist_angle = np.clip(final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper)
        optimized_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
        self.target_angles = optimized_angles

        self.mano_points = retargeter_utils.rotate_points_around_y(manohand_joint_pos, final_wrist_angle, self.source, self.hand_type)

        return {urdf_joint_id: np.deg2rad(angle) for urdf_joint_id, angle in zip(self.urdf_joint_ids, optimized_angles)}
