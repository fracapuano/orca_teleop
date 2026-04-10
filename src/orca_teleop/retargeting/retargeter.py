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
from orca_teleop.retargeting.constants import KEY_VECTORS_SHAPE

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


_ORCAHAND_DESCRIPTION_DIR_ENV = "ORCAHAND_DESCRIPTION_DIR"
_ORCAHAND_DESCRIPTION_DIR_DEFAULT = os.path.join(
    os.path.expanduser("~"), "Documents", "orcahand_description"
)
_ORCAHAND_URDF_SUBPATH = os.path.join("v1", "models", "urdf")


def _default_urdf_path(hand_type: str) -> str:
    """Return the URDF path for *hand_type* from the local orcahand_description clone.

    Resolves the base directory from the ``ORCAHAND_DESCRIPTION_DIR`` environment
    variable, falling back to ``~/Documents/orcahand_description``. The expected
    file is ``<base>/{_ORCAHAND_URDF_SUBPATH}/orcahand_{hand_type}.urdf``.

    Raises:
        RuntimeError: if the resolved URDF file does not exist.
    """
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

_DEFAULT_LR: float = 2.5
_DEFAULT_JOINT_REGULARIZERS: tuple[tuple[str, float, float], ...] = (
    ("index_abd", 0.0, 1e-6),
    ("middle_abd", 0.0, 1e-6),
    ("ring_abd", 0.0, 1e-6),
    ("pinky_abd", 0.0, 1e-6),
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def weighted_vector_loss(
    coeffs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
) -> IKLossFn:
    """Return an IK loss function that computes a weighted sum of squared
    Euclidean distances between target and robot key vectors.

    This is the default loss used by the Retargeter. Increase a finger's
    coefficient to make the solver prioritize matching that finger more tightly.

    Args:
        coeffs: Per-finger weights in (thumb, index, middle, ring, pinky) order.

    Returns:
        loss(target, robot) → scalar tensor, where target and robot are both
        (5, 3) tensors of palm-to-fingertip key vectors.
    """
    coeffs_tensor: torch.Tensor | None = None  # lazily initialised on first call

    def loss(target: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        nonlocal coeffs_tensor
        if coeffs_tensor is None or coeffs_tensor.device != target.device:
            coeffs_tensor = torch.tensor(coeffs, dtype=target.dtype, device=target.device)
        diffs_sq = torch.norm(target - robot, dim=-1) ** 2  # (5,)
        return (coeffs_tensor * diffs_sq).sum()

    return loss


@dataclass(frozen=True)
class TargetPose:
    """Canonical retargeting input: palm-to-fingertip key vectors and wrist angle."""

    key_vectors: np.ndarray
    wrist_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        key_vectors = np.array(self.key_vectors, dtype=float, copy=True)
        if key_vectors.shape != KEY_VECTORS_SHAPE:
            raise ValueError(f"TargetPose.key_vectors must have shape {KEY_VECTORS_SHAPE}")
        key_vectors.setflags(write=False)
        object.__setattr__(self, "key_vectors", key_vectors)
        object.__setattr__(self, "wrist_angle_degrees", float(self.wrist_angle_degrees))


@dataclass(frozen=True)
class RetargeterConfig:
    """Immutable configuration for a Retargeter.
    Fields without defaults are derived from the hand model and URDF and must
    be provided (use ``from_paths`` to build them automatically). Fields with
    defaults are optimizer hyperparameters that can be freely tuned.
    """

    chain: object  # pk.Chain
    hand_type: str  # "left" or "right"
    finger_joint_ids: list[str]
    finger_urdf_joint_ids: list[str]
    finger_reorder_indices: list[int]
    finger_limits_lower: torch.Tensor
    finger_limits_upper: torch.Tensor
    wrist_joint_id: str  # non-prefixed, used for OrcaJointPositions
    wrist_urdf_joint_id: str  # prefixed, used for FK
    wrist_limit_lower: float
    wrist_limit_upper: float
    optimization_frames: object  # frame indices for FK
    device: str

    lr: float = _DEFAULT_LR
    ik_loss: IKLossFn = field(default_factory=weighted_vector_loss)
    joint_regularizers: tuple[tuple[str, float, float], ...] = field(
        default=_DEFAULT_JOINT_REGULARIZERS
    )

    @classmethod
    def from_paths(
        cls,
        hand_config_path: PathLike | None = None,
        urdf_path: PathLike | None = None,
        *,
        lr: float = _DEFAULT_LR,
        ik_loss: IKLossFn | None = None,
        joint_regularizers: tuple[tuple[str, float, float], ...] = _DEFAULT_JOINT_REGULARIZERS,
    ) -> "RetargeterConfig":
        """Build a RetargeterConfig from an OrcaHand model path and a URDF path.

        Args:
            hand_config_path: Path to the OrcaHand model directory. ``None`` uses
                the default model bundled with ``orca_core``.
            urdf_path: Path to the hand URDF file. ``None`` resolves the path
                automatically from the ``orcahand_description`` package using the
                hand type declared in the model config.
            lr: RMSprop learning rate in degrees/step.
            ik_loss: A callable ``(target_kvs, robot_kvs) → scalar`` that
                measures how well the robot key vectors match the target.
                Both tensors have shape ``(5, 3)``. Defaults to
                ``weighted_vector_loss()``. Use this to swap in any
                differentiable loss — e.g. ``weighted_vector_loss((10, 10, 5, 5, 5))``
                to prioritise thumb and index.
            joint_regularizers: Sequence of ``(joint_id, target_degrees, weight)``
                triples. Each adds ``weight * (angle - target)²`` to the loss,
                pulling that joint toward a resting pose when the IK signal
                is weak.
        """
        device = get_device()

        hand = OrcaHand(hand_config_path)
        hand_type = hand.config.type
        if hand_type not in ("left", "right"):
            raise ValueError(
                f"hand.config.type must be 'left' or 'right'. Check {hand_config_path}"
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

        joint_ids = hand.config.joint_ids
        urdf_joint_ids = [f"{hand_type}_{jid}" for jid in joint_ids]
        lower, upper = map(list, zip(*hand.config.joint_roms_dict.values(), strict=False))

        # Split wrist from finger joints — wrist is a passthrough, not optimized
        wrist_joint_id = joint_ids[WRIST_MOTOR_IDX]
        wrist_urdf_joint_id = urdf_joint_ids[WRIST_MOTOR_IDX]
        finger_joint_ids = [jid for i, jid in enumerate(joint_ids) if i != WRIST_MOTOR_IDX]
        finger_urdf_joint_ids = [
            jid for i, jid in enumerate(urdf_joint_ids) if i != WRIST_MOTOR_IDX
        ]
        finger_lower = [v for i, v in enumerate(lower) if i != WRIST_MOTOR_IDX]
        finger_upper = [v for i, v in enumerate(upper) if i != WRIST_MOTOR_IDX]

        urdf_joint_names = chain.get_joint_parameter_names()
        assert set(urdf_joint_ids) == set(urdf_joint_names), (
            "Joint name mismatch between config.yaml and URDF — "
            "check that hand type (left/right) is consistent and files are up to date."
        )
        all_reorder_indices = [urdf_joint_names.index(name) for name in urdf_joint_ids]
        finger_reorder_indices = [
            idx for i, idx in enumerate(all_reorder_indices) if i != WRIST_MOTOR_IDX
        ]

        root = torch.zeros(1, 3, device=device)
        _, _, optimization_frames = retargeter_utils.get_urdf_model_params(
            chain, hand_type, list(FINGERS), root
        )

        return cls(
            chain=chain,
            hand_type=hand_type,
            finger_joint_ids=finger_joint_ids,
            finger_urdf_joint_ids=finger_urdf_joint_ids,
            finger_reorder_indices=finger_reorder_indices,
            finger_limits_lower=torch.tensor(finger_lower, device=device),
            finger_limits_upper=torch.tensor(finger_upper, device=device),
            wrist_joint_id=wrist_joint_id,
            wrist_urdf_joint_id=wrist_urdf_joint_id,
            wrist_limit_lower=lower[WRIST_MOTOR_IDX],
            wrist_limit_upper=upper[WRIST_MOTOR_IDX],
            optimization_frames=optimization_frames,
            device=device,
            lr=lr,
            ik_loss=ik_loss if ik_loss is not None else weighted_vector_loss(),
            joint_regularizers=joint_regularizers,
        )


class Retargeter:
    """Maps a TargetPose (palm-to-fingertip key vectors) to Orca Hand joint angles.

    Typical usage::

        retargeter = Retargeter.from_paths(model_path, urdf_path)
        joint_angles = retargeter.retarget(target_pose)
    """

    def __init__(self, config: RetargeterConfig) -> None:
        self.config = config

        # Only finger joints are optimized; wrist is handled as a passthrough
        self._joint_angles = torch.zeros(
            len(config.finger_urdf_joint_ids), device=config.device, requires_grad=True
        )
        self._optimizer = torch.optim.RMSprop([self._joint_angles], lr=config.lr)
        self._root = torch.zeros(1, 3, device=config.device)

        # Pre-build regularization tensors once
        n_fingers = len(config.finger_joint_ids)
        self._regularizer_zeros = torch.zeros(n_fingers, device=config.device)
        self._regularizer_weights = torch.zeros(n_fingers, device=config.device)
        for joint_id, zero_val, weight in config.joint_regularizers:
            idx = config.finger_joint_ids.index(joint_id)
            self._regularizer_zeros[idx] = zero_val
            self._regularizer_weights[idx] = weight

    @classmethod
    def from_paths(
        cls,
        model_path: str | None = None,
        urdf_path: str | None = None,
        **kwargs,
    ) -> "Retargeter":
        """Construct a Retargeter from an OrcaHand model path and a URDF path.

        Both arguments default to ``None``: ``model_path=None`` uses the default
        model bundled with ``orca_core``; ``urdf_path=None`` resolves the URDF
        automatically from the ``orcahand_description`` package.

        Any keyword arguments are forwarded to ``RetargeterConfig.from_paths``
        (e.g. ``lr``, ``ik_loss``, ``joint_regularizers``).
        """
        return cls(RetargeterConfig.from_paths(model_path, urdf_path, **kwargs))

    def _ik_loss(self, target_key_vectors: torch.Tensor) -> torch.Tensor:
        """FK-based IK loss: user-provided key-vector matching + regularization."""
        cfg = self.config

        # Place finger joints at their URDF positions (deg → rad); wrist stays at 0 in FK
        urdf_angles = torch.zeros(cfg.chain.n_joints, device=cfg.device)
        urdf_angles[cfg.finger_reorder_indices] = self._joint_angles / (180.0 / np.pi)

        fingertips, palm = retargeter_utils.extract_orca_fingertips_and_palm(
            cfg.chain,
            urdf_angles,
            cfg.optimization_frames,
            cfg.hand_type,
            list(FINGERS),
            self._root,
        )
        robot_kvs = torch.stack(
            [kv.squeeze(0) for kv in retargeter_utils.get_keyvectors(fingertips, palm)]
        )  # (5, 3)

        matching_loss = cfg.ik_loss(target_key_vectors, robot_kvs)
        regularization = torch.sum(
            self._regularizer_weights * (self._joint_angles - self._regularizer_zeros) ** 2
        )
        return matching_loss + regularization

    def _optimize(self, target_key_vectors: torch.Tensor, n_steps: int = 2) -> np.ndarray:
        """Run n_steps of gradient descent; return finger joint angles in degrees."""
        for _ in range(n_steps):
            self._optimizer.zero_grad()
            self._ik_loss(target_key_vectors).backward()
            self._optimizer.step()
            with torch.no_grad():
                self._joint_angles.clamp_(
                    self.config.finger_limits_lower, self.config.finger_limits_upper
                )
        return self._joint_angles.detach().cpu().numpy()

    def retarget(self, target_pose: TargetPose) -> OrcaJointPositions:
        """Map a TargetPose to OrcaJointPositions ready to send to the robot.

        Finger joints are solved via gradient-based IK. The wrist is not
        optimized — it is passed through directly from
        ``target_pose.wrist_angle_degrees`` and clipped to its ROM.

        Note: the wrist sign is flipped for the right hand due to a URDF
        convention inconsistency (to be fixed in a future URDF update).
        """
        cfg = self.config
        target_kv = torch.tensor(target_pose.key_vectors, dtype=torch.float32, device=cfg.device)
        finger_angles_deg = self._optimize(target_kv)

        wrist_deg = np.clip(
            target_pose.wrist_angle_degrees, cfg.wrist_limit_lower, cfg.wrist_limit_upper
        )
        wrist_deg = wrist_deg if cfg.hand_type == "left" else -wrist_deg

        return OrcaJointPositions(
            {
                **dict(zip(cfg.finger_joint_ids, finger_angles_deg, strict=True)),
                cfg.wrist_joint_id: wrist_deg,
            }
        )
