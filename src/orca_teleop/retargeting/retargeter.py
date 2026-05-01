import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from os import PathLike

import numpy as np
import pytorch_kinematics as pk
import torch
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import WRIST_MOTOR_IDX
from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.constants import (
    CALIBRATION_FRAMES,
    DEFAULT_LOSS_COEFFS,
    MANO_TO_URDF_TRANSLATION,
)
from orca_teleop.retargeting.urdf_offsets import load_ref_offsets

logger = logging.getLogger(__name__)

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


_ORCAHAND_DESCRIPTION_DIR_ENV = "ORCAHAND_DESCRIPTION_DIR"
_ORCAHAND_DESCRIPTION_DIR_DEFAULT = os.path.join(
    os.path.expanduser("~"), "Documents", "orcahand_description"
)
_ORCAHAND_URDF_SUBPATH = os.path.join("v1", "models", "urdf")


def _default_urdf_path(hand_type: str) -> str:
    base = os.environ.get(_ORCAHAND_DESCRIPTION_DIR_ENV, _ORCAHAND_DESCRIPTION_DIR_DEFAULT)
    path = os.path.join(base, _ORCAHAND_URDF_SUBPATH, f"orcahand_{hand_type}.urdf")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Default URDF not found at {path!r}. "
            f"Clone https://github.com/orcahand/orcahand_description to "
            f"{_ORCAHAND_DESCRIPTION_DIR_DEFAULT!r} or set the "
            f"{_ORCAHAND_DESCRIPTION_DIR_ENV} environment variable, "
            "or pass urdf_path explicitly to Retargeter.from_paths()."
        )
    return path


IKLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_DEFAULT_LR: float = 5.0
_DEFAULT_JOINT_REGULARIZERS: tuple[tuple[str, float, float], ...] = (
    ("index_abd", 0.0, 1e-6),
    ("middle_abd", 0.0, 1e-6),
    ("ring_abd", 0.0, 1e-6),
    ("pinky_abd", 0.0, 1e-6),
)
# Keep the historical overall regularization magnitude after normalizing the
# per-joint weights to sum to 1.0.
_DEFAULT_REGULARIZATION_WEIGHT: float = sum(weight for _, _, weight in _DEFAULT_JOINT_REGULARIZERS)


def get_device(best: bool = False) -> str:
    if not best:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def weighted_vector_loss(
    coeffs: tuple[float, ...] = DEFAULT_LOSS_COEFFS,
) -> IKLossFn:
    """Return an IK loss function: weighted sum of squared distances between
    the five target and robot palm-to-fingertip key vectors.
    """
    coeffs_tensor: torch.Tensor | None = None

    def loss(target: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        nonlocal coeffs_tensor
        if coeffs_tensor is None or coeffs_tensor.device != target.device:
            coeffs_tensor = torch.tensor(coeffs, dtype=target.dtype, device=target.device)
        diffs_sq = torch.norm(target - robot, dim=-1) ** 2  # (5,)
        return (coeffs_tensor * diffs_sq).sum()

    return loss


def _normalize_regularizer_weights(weights: torch.Tensor) -> torch.Tensor:
    """Normalize non-negative regularizer weights to sum to 1 when present."""
    if torch.any(weights < 0):
        raise ValueError("joint_regularizers weights must be non-negative")

    total = weights.sum()
    if total.item() == 0.0:
        return weights
    return weights / total


@dataclass(frozen=True)
class TargetPose:
    """Raw hand-pose input to the retargeter.

    The retargeter is responsible for normalizing the joints into the URDF
    base frame, applying the auto-scale calibration, and computing key
    vectors — none of that can happen in the ingress layer because it
    requires URDF knowledge and stateful calibration.
    """

    joint_positions: np.ndarray
    source: str = "mediapipe"
    wrist_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        joints = np.array(self.joint_positions, dtype=float, copy=True)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError(
                f"TargetPose.joint_positions must have shape (N, 3); got {joints.shape}"
            )
        joints.setflags(write=False)
        object.__setattr__(self, "joint_positions", joints)
        object.__setattr__(self, "wrist_angle_degrees", float(self.wrist_angle_degrees))


@dataclass(frozen=True)
class RetargeterConfig:
    """Immutable configuration for a Retargeter.

    Fields without defaults are derived from the hand model and URDF and must
    be provided (use ``from_paths`` to build them automatically). Fields with
    defaults are optimizer hyperparameters that can be freely tuned.
    """

    chain: object  # pk.Chain
    hand_type: str
    finger_joint_ids: list[str]
    finger_urdf_joint_ids: list[str]
    finger_reorder_indices: list[int]
    finger_limits_lower_urdf: torch.Tensor  # URDF-space (physical - ref_offset)
    finger_limits_upper_urdf: torch.Tensor
    finger_ref_offsets_deg: np.ndarray  # physical_deg - urdf_deg, in finger order
    wrist_joint_id: str
    wrist_urdf_joint_id: str
    wrist_limit_lower: float
    wrist_limit_upper: float
    optimization_frames: object
    urdfhand_center: np.ndarray  # MANO frame → URDF base frame translation
    urdfhand_rot_matrix: np.ndarray  # MANO frame → URDF base frame rotation
    fingertip_offsets: dict[str, torch.Tensor]
    urdf_keyvector_mags: np.ndarray  # neutral-pose magnitudes for autoscale
    device: str

    lr: float = _DEFAULT_LR
    ik_loss: IKLossFn = field(default_factory=weighted_vector_loss)
    regularization_weight: float = _DEFAULT_REGULARIZATION_WEIGHT
    joint_regularizers: tuple[tuple[str, float, float], ...] = field(
        default=_DEFAULT_JOINT_REGULARIZERS
    )

    @classmethod
    def from_paths(
        cls,
        hand_config_path: PathLike | None = None,
        urdf_path: PathLike | None = None,
        *,
        hand_type_override: str | None = None,
        lr: float = _DEFAULT_LR,
        ik_loss: IKLossFn | None = None,
        regularization_weight: float = _DEFAULT_REGULARIZATION_WEIGHT,
        joint_regularizers: tuple[tuple[str, float, float], ...] = _DEFAULT_JOINT_REGULARIZERS,
    ) -> "RetargeterConfig":
        device = get_device()

        hand = OrcaHand(hand_config_path)
        hand_type = hand.config.type if hand_type_override is None else hand_type_override
        if hand_type not in ("left", "right"):
            raise ValueError(
                "hand type must be 'left' or 'right'. Check hand.config.type "
                f"or hand_type_override for {hand_config_path}"
            )
        if hand_type_override is not None and hand.config.type != hand_type_override:
            logger.info(
                "Retargeter hand type override: config has %r, using %r",
                hand.config.type,
                hand_type_override,
            )

        if urdf_path is None:
            urdf_path = _default_urdf_path(hand_type)
        if not os.path.exists(urdf_path):
            raise ValueError(f"URDF file not found at {urdf_path}")

        import pytorch_kinematics.urdf_parser_py.xml_reflection as _xmlr
        import pytorch_kinematics.urdf_parser_py.xml_reflection.core as _urdf_core

        with open(urdf_path) as f:
            urdf_text = f.read()
        _orig_core_on_error = _urdf_core.on_error
        _orig_xmlr_on_error = _xmlr.on_error
        _urdf_core.on_error = lambda _: None
        _xmlr.on_error = lambda _: None
        try:
            chain = pk.build_chain_from_urdf(urdf_text).to(device=device)
        finally:
            _urdf_core.on_error = _orig_core_on_error
            _xmlr.on_error = _orig_xmlr_on_error

        joint_ids = list(hand.config.joint_ids)
        urdf_joint_ids = [f"{hand_type}_{jid}" for jid in joint_ids]
        physical_lower, physical_upper = map(
            list, zip(*hand.config.joint_roms_dict.values(), strict=False)
        )

        wrist_idx = joint_ids.index("wrist") if "wrist" in joint_ids else WRIST_MOTOR_IDX

        # Ref offsets: physical zero pose expressed in URDF radians
        ref_offsets_dict = load_ref_offsets(str(urdf_path), hand_type) or {}
        ref_offsets_rad = np.array(
            [ref_offsets_dict.get(jid, 0.0) for jid in joint_ids], dtype=float
        )
        ref_offsets_deg = np.rad2deg(ref_offsets_rad)

        # Split wrist from finger joints — wrist is a passthrough, not optimized
        wrist_joint_id = joint_ids[wrist_idx]
        wrist_urdf_joint_id = urdf_joint_ids[wrist_idx]
        finger_joint_ids = [jid for i, jid in enumerate(joint_ids) if i != wrist_idx]
        finger_urdf_joint_ids = [jid for i, jid in enumerate(urdf_joint_ids) if i != wrist_idx]
        finger_lower_phys = np.array(
            [v for i, v in enumerate(physical_lower) if i != wrist_idx], dtype=float
        )
        finger_upper_phys = np.array(
            [v for i, v in enumerate(physical_upper) if i != wrist_idx], dtype=float
        )
        finger_ref_deg = np.array(
            [v for i, v in enumerate(ref_offsets_deg) if i != wrist_idx], dtype=float
        )
        # Optimization is in URDF degrees; subtract physical→URDF offset.
        finger_lower_urdf = finger_lower_phys - finger_ref_deg
        finger_upper_urdf = finger_upper_phys - finger_ref_deg

        urdf_joint_names = chain.get_joint_parameter_names()
        assert set(urdf_joint_ids) == set(urdf_joint_names), (
            "Joint name mismatch between config.yaml and URDF — "
            "check that hand type (left/right) is consistent and files are up to date."
        )
        all_reorder_indices = [urdf_joint_names.index(name) for name in urdf_joint_ids]
        finger_reorder_indices = [
            idx for i, idx in enumerate(all_reorder_indices) if i != wrist_idx
        ]

        root = torch.zeros(1, 3, device=device)
        urdfhand_center, urdfhand_rot_matrix, optimization_frames = (
            retargeter_utils.get_urdf_model_params(chain, hand_type, list(FINGERS), root)
        )

        fingertip_offsets = retargeter_utils.get_fingertip_offset_tensors(list(FINGERS), device)

        # URDF key-vector magnitudes at the half-curled neutral pose. Used as
        # the target scale for autocalibration: mano_scale = urdf_mag / mano_mag.
        neutral_angles = torch.zeros(chain.n_joints, device=device)
        ref_rad_tensor = torch.tensor(ref_offsets_rad, device=device, dtype=torch.float32)
        all_reorder_tensor_idx = torch.tensor(all_reorder_indices, device=device, dtype=torch.long)
        neutral_angles[all_reorder_tensor_idx] = -0.5 * ref_rad_tensor.float()
        with torch.no_grad():
            urdf_fingertips, urdf_palm = retargeter_utils.extract_orca_fingertips_and_palm(
                chain,
                neutral_angles,
                optimization_frames,
                hand_type,
                list(FINGERS),
                root,
                fingertip_offsets=fingertip_offsets,
            )
            urdf_keyvectors = retargeter_utils.get_keyvectors(urdf_fingertips, urdf_palm)
            urdf_keyvector_mags = 0.9 * np.array(
                [kv.detach().cpu().norm().item() for kv in urdf_keyvectors]
            )

        return cls(
            chain=chain,
            hand_type=hand_type,
            finger_joint_ids=finger_joint_ids,
            finger_urdf_joint_ids=finger_urdf_joint_ids,
            finger_reorder_indices=finger_reorder_indices,
            finger_limits_lower_urdf=torch.tensor(finger_lower_urdf, device=device),
            finger_limits_upper_urdf=torch.tensor(finger_upper_urdf, device=device),
            finger_ref_offsets_deg=finger_ref_deg,
            wrist_joint_id=wrist_joint_id,
            wrist_urdf_joint_id=wrist_urdf_joint_id,
            wrist_limit_lower=physical_lower[WRIST_MOTOR_IDX],
            wrist_limit_upper=physical_upper[WRIST_MOTOR_IDX],
            optimization_frames=optimization_frames,
            urdfhand_center=urdfhand_center,
            urdfhand_rot_matrix=urdfhand_rot_matrix,
            fingertip_offsets=fingertip_offsets,
            urdf_keyvector_mags=urdf_keyvector_mags,
            device=device,
            lr=lr,
            ik_loss=ik_loss if ik_loss is not None else weighted_vector_loss(),
            regularization_weight=regularization_weight,
            joint_regularizers=joint_regularizers,
        )


class Retargeter:
    """Maps a TargetPose (raw 3D hand joints) to Orca Hand joint angles.

    Pipeline per call:
        1. Source-specific preprocess (mediapipe / metaquest / avp)
        2. Normalize joints to a local hand frame
        3. (First N frames) auto-scale calibration: collect MANO key-vector
           magnitudes, then set ``mano_scale`` so MANO ≈ metres
        4. Scale → rotate into URDF base frame → compute key vectors
        5. RMSprop IK (2 steps) in URDF degrees, clamped to URDF-space limits
        6. Convert to physical degrees by adding the per-joint ref offset

    Typical usage::

        retargeter = Retargeter.from_paths(model_path, urdf_path)
        joint_angles = retargeter.retarget(target_pose)
    """

    def __init__(self, config: RetargeterConfig) -> None:
        self.config = config

        self._joint_angles = torch.zeros(
            len(config.finger_urdf_joint_ids), device=config.device, requires_grad=True
        )
        self._optimizer = torch.optim.RMSprop([self._joint_angles], lr=config.lr)
        self._root = torch.zeros(1, 3, device=config.device)

        # Regularizer zeros are specified in physical degrees; the optimizer
        # runs in URDF degrees, so subtract the per-joint ref offset.
        n_fingers = len(config.finger_joint_ids)
        self._regularizer_zeros = torch.zeros(n_fingers, device=config.device)
        self._regularizer_weights = torch.zeros(n_fingers, device=config.device)
        for joint_id, zero_val, weight in config.joint_regularizers:
            idx = config.finger_joint_ids.index(joint_id)
            shifted = zero_val - float(config.finger_ref_offsets_deg[idx])
            self._regularizer_zeros[idx] = shifted
            self._regularizer_weights[idx] = weight
        self._regularizer_weights = _normalize_regularizer_weights(self._regularizer_weights)

        # Auto-scale calibration state
        self._mano_scale: float = 1.0
        self._calibration_mags: list[np.ndarray] = []
        self._calibration_done: bool = False

        # Pre-shape the URDF-to-physical conversion: physical_deg = urdf_deg + ref_deg.
        self._finger_ref_offsets_deg = config.finger_ref_offsets_deg

    @classmethod
    def from_paths(
        cls,
        model_path: str | None = None,
        urdf_path: str | None = None,
        **kwargs,
    ) -> "Retargeter":
        return cls(RetargeterConfig.from_paths(model_path, urdf_path, **kwargs))

    def _ik_loss(self, target_key_vectors: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        urdf_angles = torch.zeros(cfg.chain.n_joints, device=cfg.device)
        urdf_angles[cfg.finger_reorder_indices] = self._joint_angles / (180.0 / np.pi)

        fingertips, palm = retargeter_utils.extract_orca_fingertips_and_palm(
            cfg.chain,
            urdf_angles,
            cfg.optimization_frames,
            cfg.hand_type,
            list(FINGERS),
            self._root,
            fingertip_offsets=cfg.fingertip_offsets,
        )
        robot_kvs = torch.stack(
            [kv.squeeze(0) for kv in retargeter_utils.get_keyvectors(fingertips, palm)]
        )

        matching = cfg.ik_loss(target_key_vectors, robot_kvs)
        regularization = torch.sum(
            self._regularizer_weights * (self._joint_angles - self._regularizer_zeros) ** 2
        )
        return matching + cfg.regularization_weight * regularization

    def _optimize(self, target_key_vectors: torch.Tensor, n_steps: int = 2) -> np.ndarray:
        for _ in range(n_steps):
            self._optimizer.zero_grad()
            self._ik_loss(target_key_vectors).backward()
            self._optimizer.step()
            with torch.no_grad():
                self._joint_angles.clamp_(
                    self.config.finger_limits_lower_urdf,
                    self.config.finger_limits_upper_urdf,
                )
        return self._joint_angles.detach().cpu().numpy()

    def _ingest_calibration_frame(self, normalized_joints: np.ndarray, source: str) -> None:
        cfg = self.config
        joint_t = torch.tensor(normalized_joints, dtype=torch.float32, device=cfg.device)
        ft, palm = retargeter_utils.extract_mano_fingertips_and_palm(joint_t, list(FINGERS), source)
        kvs = retargeter_utils.get_keyvectors(ft, palm)
        mags = np.array([kv.detach().cpu().norm().item() for kv in kvs])
        self._calibration_mags.append(mags)

        if len(self._calibration_mags) >= CALIBRATION_FRAMES:
            all_mags = np.array(self._calibration_mags)
            median_mano = np.median(all_mags, axis=0)
            ratios = cfg.urdf_keyvector_mags / np.clip(median_mano, 1e-6, None)
            self._mano_scale = float(np.median(ratios))
            self._calibration_done = True
            logger.info(
                "Retargeter auto-scale calibrated: mano_scale=%.4f (per-finger ratios=%s)",
                self._mano_scale,
                np.round(ratios, 3).tolist(),
            )
            with torch.no_grad():
                self._joint_angles.zero_()
            self._optimizer = torch.optim.RMSprop([self._joint_angles], lr=cfg.lr)

    def _finger_angles_urdf_to_physical(self, urdf_deg: np.ndarray) -> np.ndarray:
        return urdf_deg + self._finger_ref_offsets_deg

    def retarget(self, target_pose: TargetPose) -> OrcaJointPositions | None:
        """Map a TargetPose to OrcaJointPositions ready to send to the robot.

        Returns ``None`` during the calibration window so the pipeline can keep
        the hand at whatever pose ``init_joints()`` established until the scale
        estimate is ready.
        """
        cfg = self.config
        source = target_pose.source

        # Step 1: source-specific raw preprocessing
        if source in ("mediapipe", "metaquest"):
            joints = np.asarray(target_pose.joint_positions, dtype=float)
        elif source == "avp":
            joints = np.asarray(target_pose.joint_positions, dtype=float)
        else:
            raise ValueError(f"Unsupported source: {source!r}")

        # Step 2: normalize into the local canonical hand frame
        normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(joints, source)

        wrist_phys_deg = float(
            np.clip(target_pose.wrist_angle_degrees, cfg.wrist_limit_lower, cfg.wrist_limit_upper)
        )

        # Step 3: calibration window — collect frames only, emit nothing yet
        if not self._calibration_done:
            self._ingest_calibration_frame(normalized, source)
            return None

        # Step 4: scale → rotate into URDF base frame → key vectors
        scaled = normalized * self._mano_scale
        in_urdf_frame = (
            scaled @ cfg.urdfhand_rot_matrix.T
            + cfg.urdfhand_center
            + np.array(MANO_TO_URDF_TRANSLATION)
        )
        joint_t = torch.tensor(in_urdf_frame, dtype=torch.float32, device=cfg.device)
        mano_ft, mano_palm = retargeter_utils.extract_mano_fingertips_and_palm(
            joint_t, list(FINGERS), source
        )
        target_kvs = torch.stack([(mano_ft[f] - mano_palm).squeeze(0) for f in FINGERS])

        # Step 5: IK in URDF degrees, clamped to URDF-space limits
        finger_urdf_deg = self._optimize(target_kvs)

        # Step 6: convert to physical degrees and assemble the action
        finger_phys_deg = self._finger_angles_urdf_to_physical(finger_urdf_deg)
        wrist_signed = wrist_phys_deg if cfg.hand_type == "left" else -wrist_phys_deg
        return OrcaJointPositions(
            {
                **dict(zip(cfg.finger_joint_ids, finger_phys_deg, strict=True)),
                cfg.wrist_joint_id: wrist_signed,
            }
        )
