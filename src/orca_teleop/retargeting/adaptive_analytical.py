"""Orca-native port of Wuji's adaptive analytical retargeting strategy."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import WRIST_MOTOR_IDX
from orca_teleop.retargeting import utils as retargeter_utils
from orca_teleop.retargeting.constants import (
    CALIBRATION_FRAMES,
    FINGERTIP_OFFSETS,
    MANO_TO_URDF_TRANSLATION,
)
from orca_teleop.retargeting.retargeter import FINGERS, TargetPose
from orca_teleop.retargeting.urdf_offsets import load_ref_offsets

M_TO_CM = 100.0
MANO_OFFSET_CM = np.array([0.0, 0.0, 1.5], dtype=np.float64)
FINGERTIP_OFFSETS_M = {
    finger: np.array(offset, dtype=np.float64) for finger, offset in FINGERTIP_OFFSETS.items()
}
logger = logging.getLogger(__name__)

_ORCAHAND_DESCRIPTION_DIR_ENV = "ORCAHAND_DESCRIPTION_DIR"
_ORCAHAND_DESCRIPTION_DIR_DEFAULT = os.path.join(
    os.path.expanduser("~"), "Documents", "orcahand_description"
)

_MP_TIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
_MP_PIP_INDICES = np.array([2, 6, 10, 14, 18], dtype=np.int64)
_MP_DIP_INDICES = np.array([3, 7, 11, 15, 19], dtype=np.int64)

_OPERATOR2MANO_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_OPERATOR2MANO_LEFT = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _infer_model_version(path: str | os.PathLike | None) -> str:
    if path is None:
        return "v1"
    parts = Path(path).parts
    for part in parts:
        if part.startswith("v") and part[1:].isdigit():
            return part
    return "v1"


def _default_urdf_path(hand_type: str, version: str = "v1") -> str:
    base = os.environ.get(_ORCAHAND_DESCRIPTION_DIR_ENV, _ORCAHAND_DESCRIPTION_DIR_DEFAULT)
    path = os.path.join(base, version, "models", "urdf", f"orcahand_{hand_type}.urdf")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Default URDF not found at {path!r}. Set {_ORCAHAND_DESCRIPTION_DIR_ENV} "
            "or pass urdf_path explicitly."
        )
    return path


def _optional_imports():
    try:
        import nlopt
        import pinocchio as pin
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "backend='adaptive_analytical' requires optional dependencies. Install the "
            "project with its adaptive extra, e.g. `uv sync --extra adaptive`."
        ) from exc
    return nlopt, pin, yaml


def _default_config_path() -> Path:
    return resources.files("orca_teleop.retargeting") / "configs" / "adaptive_analytical_orca.yaml"


def _huber_loss(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))


def _huber_grad(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, x, delta * np.sign(x))


def _estimate_mediapipe_frame(keypoints: np.ndarray) -> np.ndarray:
    if keypoints.shape != (21, 3):
        raise ValueError(f"Expected MediaPipe keypoints with shape (21, 3), got {keypoints.shape}")

    points = keypoints[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]

    x_axis = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Degenerate MediaPipe hand frame: wrist-middle axis collapsed")
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Degenerate MediaPipe hand frame: palm normal collapsed")
    z_axis = z_axis / z_norm

    if np.sum(z_axis * (centered[1] - centered[2])) < 0:
        normal *= -1.0
        z_axis *= -1.0
    return np.stack([x_axis, normal, z_axis], axis=1)


def _transform_mediapipe_keypoints(keypoints: np.ndarray, hand_type: str) -> np.ndarray:
    centered = np.asarray(keypoints, dtype=np.float64) - keypoints[0:1, :]
    wrist_frame = _estimate_mediapipe_frame(centered)
    operator2mano = _OPERATOR2MANO_RIGHT if hand_type == "right" else _OPERATOR2MANO_LEFT
    return centered @ wrist_frame @ operator2mano


def _rotation_matrix_xyz(rotation_degrees: dict[str, float]) -> np.ndarray:
    rx = math.radians(float(rotation_degrees.get("x", 0.0)))
    ry = math.radians(float(rotation_degrees.get("y", 0.0)))
    rz = math.radians(float(rotation_degrees.get("z", 0.0)))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


@dataclass(frozen=True)
class _FrameMap:
    palm: str
    pip: list[str]
    dip: list[str]
    tip: list[str]


class _LPFilter:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"lp_alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    def next(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(np.float64, copy=True)
        else:
            self._state = self._state + self.alpha * (x - self._state)
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


class AdaptiveAnalyticalRetargeter:
    """Wuji-style adaptive retargeter adapted to OrcaHand's kinematics and API."""

    def __init__(
        self,
        *,
        hand_config_path: str | None,
        urdf_path: str | None,
        config_path: str | None,
        hand_model: object | None = None,
    ) -> None:
        self._nlopt, self._pin, yaml = _optional_imports()

        # ``hand_model`` (e.g. orca_teleop.hand_model.HandModel) sidesteps
        # constructing an OrcaHand — the config.yaml may come from a different
        # orca_core branch than the one installed here.
        self._hand = hand_model if hand_model is not None else OrcaHand(hand_config_path)
        self.hand_type = self._hand.config.type
        if self.hand_type not in ("left", "right"):
            raise ValueError(f"hand.config.type must be 'left' or 'right', got {self.hand_type!r}")
        self.model_version = _infer_model_version(getattr(self._hand.config, "model_path", None))

        if urdf_path is None:
            urdf_path = _default_urdf_path(self.hand_type, self.model_version)
        self.urdf_path = str(urdf_path)

        cfg_path = Path(config_path) if config_path is not None else _default_config_path()
        with open(cfg_path) as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

        self._retarget_cfg = self.config.get("retarget", {})
        self._frame_map = self._load_frame_map(self.config)
        self._joint_aliases = (
            self.config.get("joint_name_aliases", {}).get(self.hand_type, {}) or {}
        )

        self._robot = _PinocchioOrcaRobot(self.urdf_path, self._pin)
        self._build_joint_mapping()
        self._build_frame_indices()
        self._build_retarget_params()

        self._filter = _LPFilter(float(self._retarget_cfg.get("lp_alpha", 0.15)))
        self._neutral_qpos_phys = self._build_neutral_qpos()
        self._build_target_alignment_params()
        self._mano_scale: float = 1.0
        self._calibration_mags: list[np.ndarray] = []
        self._calibration_done: bool = False
        self._last_qpos_phys: np.ndarray | None = None

    @classmethod
    def from_paths(
        cls,
        model_path: str | None = None,
        urdf_path: str | None = None,
        config_path: str | None = None,
        hand_model: object | None = None,
    ) -> AdaptiveAnalyticalRetargeter:
        return cls(hand_config_path=model_path, urdf_path=urdf_path,
                   config_path=config_path, hand_model=hand_model)

    def reset(self) -> None:
        self._filter.reset()
        self._calibration_mags.clear()
        self._calibration_done = False
        self._mano_scale = 1.0
        self._last_qpos_phys = None

    def retarget(self, target_pose: TargetPose) -> OrcaJointPositions | None:
        if target_pose.source != "mediapipe":
            raise ValueError(
                "Adaptive analytical backend currently expects MediaPipe-format "
                f"(21, 3) keypoints; got source={target_pose.source!r}."
            )

        normalized = retargeter_utils.get_normalized_local_manohand_joint_pos(
            target_pose.joint_positions,
            target_pose.source,
        )
        if self._rotation_matrix is not None:
            normalized = normalized @ self._rotation_matrix.T

        if not self._calibration_done:
            self._ingest_calibration_frame(normalized, target_pose.source)
            return None

        keypoints = self._target_keypoints_in_urdf_frame(normalized)

        qpos_phys = self._solve(keypoints)
        qpos_phys = self._filter.next(qpos_phys)

        finger_phys_deg = np.rad2deg(qpos_phys)
        wrist_phys_deg = float(
            np.clip(target_pose.wrist_angle_degrees, self._wrist_lower_deg, self._wrist_upper_deg)
        )
        wrist_signed = wrist_phys_deg if self.hand_type == "left" else -wrist_phys_deg

        return OrcaJointPositions(
            {
                **dict(zip(self._finger_joint_ids, finger_phys_deg, strict=True)),
                self._wrist_joint_id: wrist_signed,
            }
        )

    def _load_frame_map(self, config: dict[str, Any]) -> _FrameMap:
        versioned_cfg = config.get("robot_frames_by_version", {}).get(self.model_version, {})
        side_cfg = versioned_cfg.get(self.hand_type)
        if side_cfg is None:
            side_cfg = config.get("robot_frames", {}).get(self.hand_type)
        if not side_cfg:
            raise ValueError(f"robot_frames.{self.hand_type} is required in retarget config")
        fingers_cfg = side_cfg.get("fingers", {})
        missing = [finger for finger in FINGERS if finger not in fingers_cfg]
        if missing:
            raise ValueError(f"Missing robot frame config for fingers: {missing}")

        def read_frame(finger: str, key: str) -> str:
            value = fingers_cfg[finger].get(key)
            if not value:
                raise ValueError(
                    f"robot_frames.{self.hand_type}.fingers.{finger}.{key} is required"
                )
            return str(value)

        return _FrameMap(
            palm=str(side_cfg["palm"]),
            pip=[read_frame(finger, "pip") for finger in FINGERS],
            dip=[read_frame(finger, "dip") for finger in FINGERS],
            tip=[read_frame(finger, "tip") for finger in FINGERS],
        )

    def _build_joint_mapping(self) -> None:
        joint_ids = list(self._hand.config.joint_ids)
        physical_lower, physical_upper = map(
            list, zip(*self._hand.config.joint_roms_dict.values(), strict=False)
        )
        wrist_idx = joint_ids.index("wrist") if "wrist" in joint_ids else WRIST_MOTOR_IDX

        self._wrist_joint_id = joint_ids[wrist_idx]
        self._wrist_lower_deg = float(physical_lower[wrist_idx])
        self._wrist_upper_deg = float(physical_upper[wrist_idx])

        self._finger_joint_ids = [jid for i, jid in enumerate(joint_ids) if i != wrist_idx]
        lower_deg = np.array(
            [v for i, v in enumerate(physical_lower) if i != wrist_idx], dtype=np.float64
        )
        upper_deg = np.array(
            [v for i, v in enumerate(physical_upper) if i != wrist_idx], dtype=np.float64
        )
        self._lower_phys = np.deg2rad(lower_deg)
        self._upper_phys = np.deg2rad(upper_deg)

        ref_offsets = load_ref_offsets(self.urdf_path, self.hand_type) or {}
        self._ref_offsets = np.array(
            [
                ref_offsets.get(jid, ref_offsets.get(self._joint_aliases.get(jid, jid), 0.0))
                for jid in self._finger_joint_ids
            ],
            dtype=np.float64,
        )

        model_joint_names = self._robot.dof_joint_names
        self._model_q_indices: list[int] = []
        missing = []
        for jid in self._finger_joint_ids:
            urdf_suffix = self._joint_aliases.get(jid, jid)
            urdf_joint_name = f"{self.hand_type}_{urdf_suffix}"
            try:
                self._model_q_indices.append(model_joint_names.index(urdf_joint_name))
            except ValueError:
                missing.append(urdf_joint_name)
        if missing:
            raise ValueError(
                "URDF is missing joints required by the Orca config: "
                f"{missing}. Check joint_name_aliases in the retarget config."
            )
        self._model_q_indices_np = np.array(self._model_q_indices, dtype=np.int64)

    def _build_frame_indices(self) -> None:
        self._base_frame_names = [f"{self.hand_type}_{finger}_mp" for finger in FINGERS]
        names = (
            [self._frame_map.palm]
            + self._base_frame_names
            + self._frame_map.tip
            + self._frame_map.pip
            + self._frame_map.dip
        )
        self._computed_frame_names = list(dict.fromkeys(names))
        missing = [name for name in self._computed_frame_names if not self._robot.has_frame(name)]
        if missing:
            raise ValueError(
                f"URDF is missing retarget frames {missing}. Check robot_frames in the config."
            )

        self._computed_frame_indices = [
            self._robot.get_frame_index(name) for name in self._computed_frame_names
        ]
        self._palm_index = self._computed_frame_names.index(self._frame_map.palm)
        self._base_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._base_frame_names],
            dtype=np.int64,
        )
        self._thumb_base_index = self._computed_frame_names.index(f"{self.hand_type}_thumb_mp")
        self._pinky_base_index = self._computed_frame_names.index(f"{self.hand_type}_pinky_mp")
        self._tip_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._frame_map.tip], dtype=np.int64
        )
        self._pip_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._frame_map.pip], dtype=np.int64
        )
        self._dip_indices = np.array(
            [self._computed_frame_names.index(name) for name in self._frame_map.dip], dtype=np.int64
        )
        self._frame_offsets = [np.zeros(3, dtype=np.float64) for _ in self._computed_frame_names]
        for finger, frame_name in zip(FINGERS, self._frame_map.tip, strict=True):
            frame_idx = self._computed_frame_names.index(frame_name)
            self._frame_offsets[frame_idx] = FINGERTIP_OFFSETS_M[finger]

    def _build_retarget_params(self) -> None:
        cfg = self._retarget_cfg
        self._huber_delta = float(cfg.get("huber_delta_cm", 2.0))
        self._huber_delta_dir = float(cfg.get("huber_delta_dir", 0.5))
        self._norm_delta = float(cfg.get("norm_delta", 50.0))
        self._w_pos = float(cfg.get("w_pos", 1.0))
        self._w_dir = float(cfg.get("w_dir", 10.0))
        self._w_full_hand = float(cfg.get("w_full_hand", 1.0))
        self._scaling = float(cfg.get("scaling", 1.0))

        segment_cfg = cfg.get("segment_scaling", {})
        self._segment_scaling = np.ones((len(FINGERS), 3), dtype=np.float64)
        for i, finger in enumerate(FINGERS):
            if finger not in segment_cfg:
                continue
            values = np.asarray(segment_cfg[finger], dtype=np.float64)
            if values.shape != (3,):
                raise ValueError(f"segment_scaling.{finger} must be a length-3 list")
            self._segment_scaling[i] = values

        pinch_cfg = cfg.get("pinch_thresholds_cm", {})
        self._pinch_d1 = np.array(
            [pinch_cfg.get(f, {}).get("d1", 2.0) for f in FINGERS[1:]], dtype=np.float64
        )
        self._pinch_d2 = np.array(
            [pinch_cfg.get(f, {}).get("d2", 5.0) for f in FINGERS[1:]], dtype=np.float64
        )

        rotation = cfg.get("mediapipe_rotation", {})
        if any(float(rotation.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z")):
            self._rotation_matrix = _rotation_matrix_xyz(rotation)
        else:
            self._rotation_matrix = None

    def _build_target_alignment_params(self) -> None:
        """Build the same human-to-URDF frame alignment used by the RMSprop backend."""
        zero_qpos = np.zeros(self._robot.nq, dtype=np.float64)
        positions, _ = self._robot.compute_positions_and_jacobians(
            zero_qpos,
            self._computed_frame_indices,
            self._frame_offsets,
        )
        base_pos = {
            finger: positions[self._computed_frame_names.index(f"{self.hand_type}_{finger}_mp")]
            for finger in FINGERS
        }
        palm_pos = positions[self._palm_index]
        center, rot_matrix = retargeter_utils.get_hand_center_and_rotation(
            thumb_base=base_pos["thumb"],
            index_base=base_pos["index"],
            middle_base=base_pos["middle"],
            ring_base=base_pos["ring"],
            pinky_base=base_pos["pinky"],
            wrist=palm_pos,
        )
        self._urdfhand_center = np.asarray(center, dtype=np.float64).reshape(3)
        self._urdfhand_rot_matrix = np.asarray(rot_matrix, dtype=np.float64)

        neutral_positions, _ = self._robot.compute_positions_and_jacobians(
            self._full_model_qpos(self._neutral_qpos_phys),
            self._computed_frame_indices,
            self._frame_offsets,
        )
        tip_pos = neutral_positions[self._tip_indices]
        palm = self._robot_palm_from_positions(neutral_positions)
        urdf_keyvectors = tip_pos - palm
        self._urdf_keyvector_mags = 0.9 * np.linalg.norm(urdf_keyvectors, axis=1)

    def _ingest_calibration_frame(self, normalized_joints: np.ndarray, source: str) -> None:
        joint_t = torch.tensor(normalized_joints, dtype=torch.float32)
        fingertips, palm = retargeter_utils.extract_mano_fingertips_and_palm(
            joint_t,
            list(FINGERS),
            source,
        )
        keyvectors = retargeter_utils.get_keyvectors(fingertips, palm)
        mags = np.array([kv.detach().cpu().norm().item() for kv in keyvectors])
        self._calibration_mags.append(mags)

        if len(self._calibration_mags) < CALIBRATION_FRAMES:
            return

        all_mags = np.array(self._calibration_mags)
        median_mano = np.median(all_mags, axis=0)
        ratios = self._urdf_keyvector_mags / np.clip(median_mano, 1e-6, None)
        self._mano_scale = float(np.median(ratios))
        self._calibration_done = True
        self._last_qpos_phys = None
        self._filter.reset()
        logger.info(
            "Adaptive retargeter auto-scale calibrated: mano_scale=%.4f " "(per-finger ratios=%s)",
            self._mano_scale,
            np.round(ratios, 3).tolist(),
        )

    def _target_keypoints_in_urdf_frame(self, normalized_joints: np.ndarray) -> np.ndarray:
        scaled = normalized_joints * self._mano_scale
        return (
            scaled @ self._urdfhand_rot_matrix.T
            + self._urdfhand_center
            + np.array(MANO_TO_URDF_TRANSLATION)
        )

    def _solve(self, keypoints: np.ndarray) -> np.ndarray:
        init_qpos = self._initial_qpos()
        reg_qpos = self._last_qpos_phys

        alphas = self._compute_pinch_alpha(keypoints)
        target_tip_vectors = self._compute_tip_vectors(keypoints)
        target_tip_dirs = self._compute_tip_dirs(keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(keypoints)

        opt = self._nlopt.opt(self._nlopt.LD_SLSQP, len(self._finger_joint_ids))
        opt.set_maxeval(int(self._retarget_cfg.get("maxeval", 50)))
        opt.set_ftol_abs(float(self._retarget_cfg.get("ftol_abs", 1e-4)))
        opt.set_lower_bounds(self._lower_phys.tolist())
        opt.set_upper_bounds(self._upper_phys.tolist())

        def objective(x, grad_out):
            loss, grad = self._loss_and_grad(
                np.asarray(x, dtype=np.float64),
                target_tip_vectors,
                target_tip_dirs,
                target_full_hand_vectors,
                alphas,
                reg_qpos,
            )
            if grad_out.size > 0:
                grad_out[:] = grad
            return float(loss)

        opt.set_min_objective(objective)
        try:
            result = np.asarray(opt.optimize(init_qpos.tolist()), dtype=np.float64)
        except RuntimeError:
            result = init_qpos
        result = np.clip(result, self._lower_phys, self._upper_phys)
        self._last_qpos_phys = result.copy()
        return result

    def _initial_qpos(self) -> np.ndarray:
        if self._last_qpos_phys is not None:
            init = self._last_qpos_phys
        else:
            init = self._neutral_qpos_phys
        return np.clip(init, self._lower_phys, self._upper_phys)

    def _build_neutral_qpos(self) -> np.ndarray:
        neutral_deg = np.array(
            [
                float(self._hand.config.neutral_position.get(jid, 0.0))
                for jid in self._finger_joint_ids
            ],
            dtype=np.float64,
        )
        neutral_rad = np.deg2rad(neutral_deg)
        return np.clip(neutral_rad, self._lower_phys, self._upper_phys)

    def _full_model_qpos(self, qpos_phys: np.ndarray) -> np.ndarray:
        qpos = np.zeros(self._robot.nq, dtype=np.float64)
        qpos[self._model_q_indices_np] = qpos_phys - self._ref_offsets
        return qpos

    def _compute_pinch_alpha(self, keypoints: np.ndarray) -> np.ndarray:
        thumb_tip = keypoints[_MP_TIP_INDICES[0]]
        finger_tips = keypoints[_MP_TIP_INDICES[1:]]
        distances = np.linalg.norm(finger_tips - thumb_tip, axis=1) * M_TO_CM
        alphas_4 = np.clip(
            (self._pinch_d2 - distances) / (self._pinch_d2 - self._pinch_d1 + 1e-8),
            0.0,
            0.7,
        )
        return np.concatenate([[float(np.max(alphas_4))], alphas_4])

    def _target_palm(self, keypoints: np.ndarray) -> np.ndarray:
        return 0.5 * (keypoints[1] + keypoints[17])

    def _compute_tip_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        palm = self._target_palm(keypoints)
        return ((keypoints[_MP_TIP_INDICES] - palm) * self._scaling * M_TO_CM).astype(np.float64)

    def _compute_tip_dirs(self, keypoints: np.ndarray) -> np.ndarray:
        vectors = keypoints[_MP_TIP_INDICES] - keypoints[_MP_DIP_INDICES]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def _compute_full_hand_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        palm = self._target_palm(keypoints)
        pip_vectors = (keypoints[_MP_PIP_INDICES] - palm) * self._segment_scaling[:, 0:1]
        dip_vectors = (keypoints[_MP_DIP_INDICES] - palm) * self._segment_scaling[:, 1:2]
        tip_vectors = (keypoints[_MP_TIP_INDICES] - palm) * self._segment_scaling[:, 2:3]
        return np.vstack([pip_vectors, dip_vectors, tip_vectors]) * M_TO_CM

    def _robot_palm_from_positions(self, positions_m: np.ndarray) -> np.ndarray:
        return (
            0.5 * (positions_m[self._thumb_base_index] + positions_m[self._pinky_base_index])
            - MANO_OFFSET_CM / M_TO_CM
        )

    def _robot_palm_from_positions_cm(self, positions_cm: np.ndarray) -> np.ndarray:
        return (
            0.5 * (positions_cm[self._thumb_base_index] + positions_cm[self._pinky_base_index])
            - MANO_OFFSET_CM
        )

    def _loss_and_grad(
        self,
        qpos_phys: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        alphas: np.ndarray,
        reg_qpos: np.ndarray | None,
    ) -> tuple[float, np.ndarray]:
        qpos_full = self._full_model_qpos(qpos_phys)
        positions, jacobians = self._robot.compute_positions_and_jacobians(
            qpos_full,
            self._computed_frame_indices,
            self._frame_offsets,
        )
        positions_cm = positions * M_TO_CM
        jacobians_cm = jacobians[:, :, self._model_q_indices_np] * M_TO_CM

        palm_pos = self._robot_palm_from_positions_cm(positions_cm)
        tip_pos = positions_cm[self._tip_indices]
        pip_pos = positions_cm[self._pip_indices]
        dip_pos = positions_cm[self._dip_indices]

        j_palm = 0.5 * (jacobians_cm[self._thumb_base_index] + jacobians_cm[self._pinky_base_index])
        j_tip = jacobians_cm[self._tip_indices]
        j_pip = jacobians_cm[self._pip_indices]
        j_dip = jacobians_cm[self._dip_indices]

        total_grad = np.zeros_like(qpos_phys, dtype=np.float64)

        robot_tip_vec = tip_pos - palm_pos
        diff_pos = robot_tip_vec - target_tip_vectors
        dist_pos = np.linalg.norm(diff_pos, axis=1)
        loss_tip_pos = _huber_loss(dist_pos, self._huber_delta)
        grad_pos = _huber_grad(dist_pos, self._huber_delta)
        diff_normed_pos = diff_pos / (dist_pos[:, None] + 1e-8)

        for i in range(len(FINGERS)):
            j_diff = j_tip[i] - j_palm
            total_grad += alphas[i] * self._w_pos * grad_pos[i] * (diff_normed_pos[i] @ j_diff)

        robot_tip_dir_vec = tip_pos - dip_pos
        robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec, axis=1, keepdims=True)
        robot_tip_dirs = robot_tip_dir_vec / (robot_tip_dir_norm + 1e-8)
        diff_dir = robot_tip_dirs - target_tip_dirs
        dist_dir = np.linalg.norm(diff_dir, axis=1)
        loss_tip_dir = _huber_loss(dist_dir, self._huber_delta_dir)
        grad_dir = _huber_grad(dist_dir, self._huber_delta_dir)
        diff_normed_dir = diff_dir / (dist_dir[:, None] + 1e-8)

        for i in range(len(FINGERS)):
            unit_dir = robot_tip_dirs[i]
            norm = robot_tip_dir_norm[i, 0]
            j_norm = (np.eye(3) - np.outer(unit_dir, unit_dir)) / (norm + 1e-8)
            j_diff = j_tip[i] - j_dip[i]
            total_grad += (
                alphas[i] * self._w_dir * grad_dir[i] * (diff_normed_dir[i] @ j_norm @ j_diff)
            )

        target_pip = target_full_hand_vectors[:5]
        target_dip = target_full_hand_vectors[5:10]
        target_tip = target_full_hand_vectors[10:15]

        diff_pip = (pip_pos - palm_pos) - target_pip
        diff_dip = (dip_pos - palm_pos) - target_dip
        diff_tip = (tip_pos - palm_pos) - target_tip

        dist_pip = np.linalg.norm(diff_pip, axis=1)
        dist_dip = np.linalg.norm(diff_dip, axis=1)
        dist_tip = np.linalg.norm(diff_tip, axis=1)

        loss_pip = _huber_loss(dist_pip, self._huber_delta)
        loss_dip = _huber_loss(dist_dip, self._huber_delta)
        loss_tip_full = _huber_loss(dist_tip, self._huber_delta)
        loss_full_hand = (loss_pip + loss_dip + loss_tip_full) / 3.0

        grad_pip = _huber_grad(dist_pip, self._huber_delta)
        grad_dip = _huber_grad(dist_dip, self._huber_delta)
        grad_tip_full = _huber_grad(dist_tip, self._huber_delta)

        diff_normed_pip = diff_pip / (dist_pip[:, None] + 1e-8)
        diff_normed_dip = diff_dip / (dist_dip[:, None] + 1e-8)
        diff_normed_tip = diff_tip / (dist_tip[:, None] + 1e-8)

        for i in range(len(FINGERS)):
            grad_coeff = (1.0 - alphas[i]) * self._w_full_hand / 3.0
            total_grad += grad_coeff * grad_pip[i] * (diff_normed_pip[i] @ (j_pip[i] - j_palm))
            total_grad += grad_coeff * grad_dip[i] * (diff_normed_dip[i] @ (j_dip[i] - j_palm))
            total_grad += grad_coeff * grad_tip_full[i] * (diff_normed_tip[i] @ (j_tip[i] - j_palm))

        loss_tip_dir_vec = self._w_pos * loss_tip_pos + self._w_dir * loss_tip_dir
        loss_full = self._w_full_hand * loss_full_hand
        total_loss = float(np.sum(alphas * loss_tip_dir_vec + (1.0 - alphas) * loss_full))

        if reg_qpos is not None:
            diff_reg = qpos_phys - reg_qpos
            total_loss += self._norm_delta * float(np.sum(diff_reg**2))
            total_grad += 2.0 * self._norm_delta * diff_reg

        return total_loss, total_grad


class _PinocchioOrcaRobot:
    def __init__(self, urdf_path: str, pin_module) -> None:
        self._pin = pin_module
        self.model = self._pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if self.model.nq != self.model.nv:
            raise NotImplementedError("Adaptive analytical backend requires one-DoF joints only")

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def dof_joint_names(self) -> list[str]:
        return [name for i, name in enumerate(self.model.names) if self.model.nqs[i] > 0]

    def has_frame(self, name: str) -> bool:
        return self.model.getFrameId(name) < self.model.nframes

    def get_frame_index(self, name: str) -> int:
        idx = self.model.getFrameId(name)
        if idx >= self.model.nframes:
            raise RuntimeError(f"Frame {name!r} not found in URDF")
        return int(idx)

    def compute_positions_and_jacobians(
        self,
        qpos: np.ndarray,
        frame_indices: list[int],
        frame_offsets: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=np.float64)
        self._pin.forwardKinematics(self.model, self.data, qpos)
        self._pin.computeJointJacobians(self.model, self.data, qpos)
        self._pin.updateFramePlacements(self.model, self.data)
        if frame_offsets is None:
            frame_offsets = [np.zeros(3, dtype=np.float64) for _ in frame_indices]

        positions = []
        jacobians = []
        for idx, offset in zip(frame_indices, frame_offsets, strict=True):
            offset = np.asarray(offset, dtype=np.float64)
            placement = self.data.oMf[idx]
            positions.append(placement.translation + placement.rotation @ offset)
            j_local = self._pin.getFrameJacobian(self.model, self.data, idx, self._pin.LOCAL)
            linear_local = j_local[:3, :]
            angular_local = j_local[3:, :]
            offset_linear_local = (
                linear_local
                + np.cross(
                    angular_local.T,
                    offset,
                ).T
            )
            jacobians.append(placement.rotation @ offset_linear_local)
        return np.stack(positions, axis=0), np.stack(jacobians, axis=0)
