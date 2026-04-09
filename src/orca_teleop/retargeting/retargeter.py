import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytorch_kinematics as pk
import torch
import yaml
from orca_core import OrcaHand

from . import retargeter_utils
from .constants import KEY_VECTORS_SHAPE


@dataclass(frozen=True)
class TargetPose:
    """Canonical retargeting input: palm-to-fingertip key vectors, and wrist angle."""

    key_vectors: np.ndarray
    wrist_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        key_vectors = np.array(self.key_vectors, dtype=float, copy=True)
        if key_vectors.shape != KEY_VECTORS_SHAPE:
            raise ValueError(f"TargetPose.key_vectors must have shape {KEY_VECTORS_SHAPE}")

        # Prevents modifications to the key_vectors
        key_vectors.setflags(write=False)
        object.__setattr__(self, "key_vectors", key_vectors)
        object.__setattr__(self, "wrist_angle_degrees", float(self.wrist_angle_degrees))


class Retargeter:
    """Retargeter class for Orca Hand to retarget MANO joint angles to Orca Hand joint angles."""

    def __init__(self, model_path: OrcaHand | str = None, urdf_path: str | None = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_angles = None
        self.mano_points = None

        if not os.path.exists(urdf_path):
            raise ValueError(f"URDF file not found at {urdf_path}")
        with open(urdf_path) as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(device=self.device)

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError(
                "hand.type must be 'left' or 'right'. Update config.yaml with type field."
            )
        self.hand_type = hand.type
        self.joint_ids = hand.joint_ids
        self.urdf_joint_ids = [f"{hand.type}_{joint_id}" for joint_id in self.joint_ids]
        self.fingers = ["thumb", "index", "middle", "ring", "pinky"]
        lower_limits, upper_limits = map(list, zip(*hand.joint_roms_dict.values(), strict=False))
        self.wrist_limit_lower = lower_limits[16]
        self.wrist_limit_upper = upper_limits[16]
        lower_limits[16] = upper_limits[16] = (
            0.0  # Keep wrist constrained to zero during optimization
        )
        self.joint_angle_limits_lower = torch.tensor(lower_limits, device=self.device)
        self.joint_angle_limits_upper = torch.tensor(upper_limits, device=self.device)

        urdf_joint_parameter_names = self.chain.get_joint_parameter_names()
        assert (
            set(self.urdf_joint_ids) == set(urdf_joint_parameter_names)
        ), "Joint name mismatch between the user defined urdf joint_ids and the actual joint names in the URDF file. Please check if your config.yaml and URDF file have the same hand type (left/right) and are up to date."
        self.joint_reorder_indices = [
            urdf_joint_parameter_names.index(name) for name in self.urdf_joint_ids
        ]

        with open(os.path.join(os.path.dirname(__file__), "retargeter.yaml")) as file:
            cfg = yaml.safe_load(file)
        self.lr = cfg["lr"]
        self.use_scalar_distance = (
            [False, True, True, True, True] if cfg["use_scalar_distance_palm"] else [False] * 5
        )
        self.joint_regularizers = cfg["joint_regularizers"]
        self.loss_coeffs = torch.tensor(cfg["loss_coeffs"], device=self.device)

        self.orcahand_joint_angles = torch.zeros(
            len(self.urdf_joint_ids), device=self.device, requires_grad=True
        )
        self.opt = torch.optim.RMSprop([self.orcahand_joint_angles], lr=self.lr)

        self.root = torch.zeros(1, 3, device=self.device)
        self.regularizer_zeros = torch.zeros(len(self.urdf_joint_ids), device=self.device)
        self.regularizer_weights = torch.zeros(len(self.urdf_joint_ids), device=self.device)
        for joint_id, zero_val, weight in self.joint_regularizers:
            idx = self.joint_ids.index(joint_id)
            self.regularizer_zeros[idx] = zero_val
            self.regularizer_weights[idx] = weight

        _, _, self.optimization_frames = retargeter_utils.get_urdf_model_params(
            self.chain, self.hand_type, self.fingers, self.root
        )

    def optimize_orcahand_joint_angles(
        self, target_key_vectors: np.ndarray | Sequence[Sequence[float]], opt_steps: int = 2
    ) -> np.ndarray:
        target_key_vectors = torch.tensor(
            target_key_vectors, dtype=torch.float32, device=self.device
        )
        if target_key_vectors.shape != (len(self.fingers), 3):
            raise ValueError(f"target_key_vectors must have shape ({len(self.fingers)}, 3)")

        for _ in range(opt_steps):
            urdfhand_joint_angles = torch.zeros(self.chain.n_joints, device=self.device)
            urdfhand_joint_angles[self.joint_reorder_indices] = self.orcahand_joint_angles / (
                180.0 / np.pi
            )
            urdfhand_fingertips, urdfhand_palm = retargeter_utils.extract_orca_fingertips_and_palm(
                self.chain,
                urdfhand_joint_angles,
                self.optimization_frames,
                self.hand_type,
                self.fingers,
                self.root,
            )
            keyvectors_urdfhand = retargeter_utils.get_keyvectors(
                urdfhand_fingertips, urdfhand_palm
            )

            # Compute loss between MANO and hand key vectors
            loss = sum(
                self.loss_coeffs[i]
                * (
                    torch.norm(target_key_vectors[i] - keyvector_urdfhand.squeeze(0)) ** 2
                    if not self.use_scalar_distance[i]
                    else (
                        torch.norm(target_key_vectors[i])
                        - torch.norm(keyvector_urdfhand.squeeze(0))
                    )
                    ** 2
                )
                for i, keyvector_urdfhand in enumerate(keyvectors_urdfhand)
            )
            # Add regularization term (tunable in retargeter.yaml)
            loss += torch.sum(
                self.regularizer_weights
                * (self.orcahand_joint_angles - self.regularizer_zeros) ** 2
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.orcahand_joint_angles.clamp_(
                    self.joint_angle_limits_lower, self.joint_angle_limits_upper
                )

        return self.orcahand_joint_angles.detach().cpu().numpy()

    def retarget(self, target_pose: TargetPose) -> dict[str, float]:
        """Retarget canonical hand key vectors to Orca Hand joint angles."""

        final_wrist_angle = target_pose.wrist_angle_degrees
        optimized_angles = self.optimize_orcahand_joint_angles(target_pose.key_vectors)

        # Wrist angle is inverted for right hand due to URDF inconsistency, should be fixed/standardized in future URDF update
        final_wrist_angle = np.clip(
            final_wrist_angle, self.wrist_limit_lower, self.wrist_limit_upper
        )
        optimized_angles[-1] = final_wrist_angle if self.hand_type == "left" else -final_wrist_angle
        self.target_angles = optimized_angles

        return {
            urdf_joint_id: np.deg2rad(angle)
            for urdf_joint_id, angle in zip(self.urdf_joint_ids, optimized_angles, strict=False)
        }
