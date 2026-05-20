"""Stateful teleop controller for the bimanual OrcaArm.

The :class:`OrcaArmTeleopController` owns the per-side startup anchoring,
fixed translation scaling, finger retargeting, IK solve, and post-IK
joint-step clamp that ``scripts/teleop_arm_quest.py`` previously inlined.
It accepts already-converted ``pin.SE3`` wrist poses (in robot world / FLU
coords) so downstream consumers — including simulators with no Quest/gRPC
layer — can drive the same control logic.

The controller does NOT own a queue, a gRPC server, MeshCat, or a
publisher: callers are responsible for plumbing frames in and pushing
:class:`OrcaArmTeleopResult` to whatever sink they choose.

A MetaQuest-specific helper, :func:`metaquest_wrist_pose_to_robot_se3`,
is provided alongside the controller so the live Quest pipeline can keep
converting Unity left-handed wrist poses to FLU SE3 without dragging the
basis transform back into the script.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pinocchio as pin
from orca_core import OrcaJointPositions

from orca_teleop.constants import (
    BOOTSTRAP_SCALE,
    CLUTCH_GRACE_S,
    CUTOFF_MIN,
    MAX_JOINT_STEP_RAD,
    MIN_SPAN_SAMPLES,
    OPERATOR_NEUTRAL_WRIST_POSES_FLU,
    OPERATOR_WRIST_WORKSPACE_LIMITS_M,
    SPAN_CHANGE_THRESHOLD,
    SPAN_REFIT_PERIOD_S,
    STILL_THRESHOLD_M,
    STILL_WINDOW_SAMPLES,
    WORKSPACE_DELTA_LIMITS_M,
)
from orca_teleop.orca_arm_ik import BimanualIKSolver, IKResult
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
KeypointSource = Literal["metaquest", "mediapipe", "canonical"]
IKMode = Literal["pose", "position"]
SideStatus = Literal["missing", "awaiting_neutral", "awaiting_anchor", "tracking", "ik_failed"]
TranslationFrame = Literal["operator_local", "world"]


def _default_orientation_cost() -> np.ndarray:
    """Match ``teleop_arm_quest.py`` default: cost=1 on X/Y, free Z (5-DOF)."""
    cost = np.full(3, 1.0, dtype=np.float64)
    cost[2] = 0.0
    return cost


def _default_hand_model_path() -> str:
    """Prefer the installed v2 hand model topology for finger retargeting."""
    import orca_core

    models_dir = Path(orca_core.__file__).resolve().parent / "models"
    candidates = (
        models_dir / "v2" / "orcahand_right" / "config.yaml",
        models_dir / "v1" / "orcahand_right" / "config.yaml",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    raise RuntimeError(f"No bundled OrcaHand config.yaml found under {models_dir}")


def _relative_flu_z_angle_degrees(T_zero: pin.SE3, T_now: pin.SE3) -> float:
    """Signed relative rotation around local FLU +Z, in degrees."""
    dR = T_zero.rotation.T @ T_now.rotation
    return float(np.rad2deg(np.arctan2(dR[1, 0], dR[0, 0])))


def _retargeted_action_for_side(
    action: OrcaJointPositions,
    side: str,
) -> OrcaJointPositions:
    """Adapt shared right-hand retargeter output to the embedded OrcaArm side.

    The standalone ``orcahand_left.urdf`` mirrors the flexion axes, but the
    OrcaArm's embedded left hand keeps MCP/PIP/DIP flexion signs aligned with
    the right subtree. Use the right-hand retargeter convention for fingers on
    both sides, then only mirror the wrist command for the left side.
    """
    if side != "left":
        return action

    data = action.as_dict()
    if "wrist" in data:
        data["wrist"] = -data["wrist"]
    return OrcaJointPositions(data)


def _hand_urdf_path_for_side(hand_urdf_path: str | None, side: str) -> str | None:
    """Resolve an optional CLI/config URDF path for a side.

    ``None`` lets :class:`Retargeter` choose ``orcahand_{side}.urdf``. When a
    single explicit right/left URDF path is provided, mirror its filename for
    the opposite side if that sibling exists. A ``{side}`` placeholder is also
    accepted for explicit templates.
    """
    if hand_urdf_path is None:
        return None

    if "{side}" in hand_urdf_path:
        return hand_urdf_path.format(side=side)

    path = Path(hand_urdf_path)
    if side == "left" and "right" in path.name:
        candidate = path.with_name(path.name.replace("right", "left"))
        if candidate.exists():
            return str(candidate)
    if side == "right" and "left" in path.name:
        candidate = path.with_name(path.name.replace("left", "right"))
        if candidate.exists():
            return str(candidate)
    return str(path)


def metaquest_wrist_pose_to_robot_se3(position: np.ndarray, rotation: np.ndarray) -> pin.SE3:
    """Quest wrist pose (Unity left-handed) → ``pin.SE3`` in robot world (FLU) coords.

    Unity LH → robot FLU change-of-basis is delegated to
    ``hand_tracking_sdk.convert`` so live teleop and SDK pose conversion stay
    aligned.
    """

    from hand_tracking_sdk import convert as hts_convert

    p = np.asarray(
        hts_convert.basis_transform_position(
            tuple(np.asarray(position, dtype=np.float64).reshape(3)),
            hts_convert.BASIS_UNITY_LEFT_TO_FLU,
        ),
        dtype=np.float64,
    )

    qx, qy, qz, qw = pin.Quaternion(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).coeffs()
    R = np.asarray(
        hts_convert.basis_transform_rotation_matrix(
            float(qx),
            float(qy),
            float(qz),
            float(qw),
            hts_convert.BASIS_UNITY_LEFT_TO_FLU,
        ),
        dtype=np.float64,
    )
    return pin.SE3(R, p)


@dataclass
class OrcaArmTeleopConfig:
    """Tuning knobs for :class:`OrcaArmTeleopController`.

    Defaults mirror the values used by ``scripts/teleop_arm_quest.py`` so
    the controller is drop-in equivalent. ``orientation_cost`` defaults to
    ``[1, 1, 0]`` — free Z rotation — matching the script's
    ``--orientation-cost 1.0 --free-roll-axis Z`` default.
    """

    manual_scale: float | np.ndarray | dict[str, np.ndarray] | None = None
    workspace_delta_limits_m: (
        dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None
    ) = field(default_factory=lambda: dict(WORKSPACE_DELTA_LIMITS_M))
    operator_workspace_limits_m: dict[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = field(default_factory=lambda: dict(OPERATOR_WRIST_WORKSPACE_LIMITS_M))
    min_span_samples: int = MIN_SPAN_SAMPLES
    span_refit_period_s: float = SPAN_REFIT_PERIOD_S
    span_change_threshold: float = SPAN_CHANGE_THRESHOLD
    still_threshold_m: float = STILL_THRESHOLD_M
    still_window_samples: int = STILL_WINDOW_SAMPLES
    clutch_grace_s: float = CLUTCH_GRACE_S
    bootstrap_scale: float = BOOTSTRAP_SCALE
    cutoff_min: float = CUTOFF_MIN
    max_joint_step_rad: float = MAX_JOINT_STEP_RAD
    hand_model_path: str | None = None
    hand_urdf_path: str | None = None
    hand_type_override: str | None = "right"
    ik_mode: IKMode = "pose"
    orientation_cost: float | np.ndarray = field(default_factory=_default_orientation_cost)
    posture_cost: float = 1e-3
    position_damping: float = 1e-4
    position_step_size: float = 0.7
    position_posture_gain: float = 1e-5
    position_rotation_axes: str = "XZ"
    position_rotation_gain: float = 3e-6
    active_sides: tuple[str, ...] | None = None
    translation_frame: TranslationFrame = "operator_local"
    use_workspace_calibration: bool = True
    require_operator_neutral: bool = False
    operator_neutral_position_tolerance_m: float = 0.05
    operator_neutral_orientation_tolerance_rad: float = np.deg2rad(15.0)
    workspace_position_clip: bool = False
    workspace_position_clip_tolerance_m: float = 0.005
    pose_filter_alpha: float = 1.0
    max_operator_translation_speed_mps: float = 2.0
    max_operator_rotation_speed_radps: float = 12.0
    operator_neutral_poses_flu: dict[
        str,
        tuple[
            tuple[float, float, float],
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ],
        ],
    ] = field(default_factory=lambda: dict(OPERATOR_NEUTRAL_WRIST_POSES_FLU))


@dataclass
class OrcaArmTeleopFrame:
    """One operator wrist sample, already converted to robot-world coords.

    The controller is source-agnostic: pass already-converted ``pin.SE3``
    wrist poses so simulators can feed canonical poses directly without
    going through MetaQuest. Finger keypoints are optional — if absent,
    finger retargeting is skipped for this frame.
    """

    handedness: Literal["left", "right"]
    timestamp_ns: int
    wrist_pose: pin.SE3
    keypoints: np.ndarray | None = None
    keypoint_source: KeypointSource = "metaquest"


@dataclass
class OrcaArmTeleopResult:
    """One controller tick's output.

    ``arm_angles`` and ``target_poses`` are keyed only by sides with an
    active IK target this tick. ``operator_poses`` exposes a pose suitable for
    renderer debugging: normally the incoming FLU wrist pose, but during the
    optional neutral gate it is the live operator pose reconstructed relative
    to the configured recorded-neutral pose and mapped into robot-home space.
    ``hand_positions`` carries the most recent finger command per side.
    ``statuses`` covers all known sides so callers can drive logging.
    """

    arm_angles: dict[str, np.ndarray]
    hand_positions: dict[str, OrcaJointPositions]
    target_poses: dict[str, pin.SE3]
    operator_poses: dict[str, pin.SE3]
    q: np.ndarray
    converged: dict[str, bool]
    position_error: dict[str, float]
    orientation_error: dict[str, float]
    statuses: dict[str, SideStatus]
    new_target_acquire_ns: dict[str, int]


class OrcaArmTeleopController:
    """Per-side startup-anchor + IK pipeline for the bimanual OrcaArm.

    Lifecycle::

        controller = OrcaArmTeleopController()
        controller.reset()  # called by __init__ already
        while running:
            frames = [_to_frame(item) for item in drain_queue(...)]
            result = controller.step(frames)
            sink.update(result.arm_angles,
                        hand_positions=result.hand_positions,
                        target_Ts={s: T.homogeneous for s, T in result.target_poses.items()})

    The controller carries ``q_prev`` internally between ``step`` calls.
    Pass ``q_current`` to ``step`` to override that seed (useful for sim
    backends that simulate joint dynamics and want the next IK solve
    seeded from the actual joint state, not the previous IK output).
    """

    def __init__(
        self,
        config: OrcaArmTeleopConfig | None = None,
        ik: BimanualIKSolver | None = None,
    ) -> None:
        self.config = config if config is not None else OrcaArmTeleopConfig()
        self.ik = (
            ik
            if ik is not None
            else BimanualIKSolver(
                orientation_cost=self.config.orientation_cost,
                posture_cost=self.config.posture_cost,
            )
        )
        self.active_sides = tuple(
            self.config.active_sides
            if self.config.active_sides is not None
            else self.ik.arm_joint_indices.keys()
        )

        self._q_home = self._build_q_home()
        self._home_poses: dict[str, pin.SE3] = {
            side: pin.SE3(self.ik.forward_kinematics_full(self._q_home, side))
            for side in self.active_sides
        }

        self._retargeters: dict[str, Retargeter] = {}
        # Lazy-built hand model path; resolved on first retargeter need so
        # tests / sim users without a bundled OrcaHand model can drive the
        # controller with ``keypoints=None`` frames and never trip this.
        self._resolved_hand_model_path: str | None = self.config.hand_model_path

        self.reset()

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #
    @property
    def q_home(self) -> np.ndarray:
        """Full pinocchio q at the side-biased ready/home pose."""
        return self._q_home.copy()

    @property
    def home_poses(self) -> dict[str, pin.SE3]:
        """Per-side carpals SE3 at ``q_home`` (FK once at construction)."""
        return {side: pin.SE3(T) for side, T in self._home_poses.items()}

    @property
    def q(self) -> np.ndarray:
        """Most recent solved q. Equals ``q_home`` until the first IK tick."""
        return self._q_prev.copy()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, q_seed: np.ndarray | None = None) -> None:
        """Reset all per-side state and seed ``q_prev``.

        Defaults to ``q_home``; pass ``q_seed`` to start tracking from a
        different configuration (e.g. the sim's current joint state).
        """
        self._q_prev = (q_seed if q_seed is not None else self._q_home).copy()

        self._T_first: dict[str, pin.SE3] = {}
        self._scale: dict[str, np.ndarray] = {}
        self._targets: dict[str, pin.SE3] = {}
        self._operator_display_poses: dict[str, pin.SE3] = {}
        self._hand_targets: dict[str, OrcaJointPositions] = {}
        self._filtered_operator_poses: dict[str, pin.SE3] = {}
        self._filtered_operator_timestamps_ns: dict[str, int] = {}

    def set_home_configuration(self, q_home: np.ndarray) -> None:
        """Replace the teleop home pose and reset startup anchoring.

        Sim-backed embodiments can own their home pose via a MuJoCo keyframe.
        This lets the shared controller adopt that environment-defined pose
        instead of assuming Pinocchio neutral or a hardcoded ready posture.
        """
        q = np.asarray(q_home, dtype=np.float64).copy()
        if q.shape != self._q_home.shape:
            raise ValueError(f"Expected q_home shape {self._q_home.shape}, got {q.shape}")
        self._q_home = np.clip(
            q,
            self.ik._model.lowerPositionLimit,
            self.ik._model.upperPositionLimit,
        )
        self._home_poses = {
            side: pin.SE3(self.ik.forward_kinematics_full(self._q_home, side))
            for side in self.active_sides
        }
        self.reset(self._q_home)

    # ------------------------------------------------------------------ #
    # Main entrypoint
    # ------------------------------------------------------------------ #
    def step(
        self,
        frames: Iterable[OrcaArmTeleopFrame],
        q_current: np.ndarray | None = None,
    ) -> OrcaArmTeleopResult:
        """Process the latest frame per side, run IK, return the result.

        If multiple frames per side are passed, only the last one (in
        iteration order) is used — matching the original loop's
        latest-frame-per-side queue drain.
        """
        if q_current is not None:
            self._q_prev = np.asarray(q_current, dtype=np.float64).copy()

        new_target_acquire_ns: dict[str, int] = {}
        statuses: dict[str, SideStatus] = {side: "missing" for side in self.active_sides}

        latest_by_side: dict[str, OrcaArmTeleopFrame] = {}
        for frame in frames:
            if frame.handedness not in self.active_sides:
                continue
            if frame.wrist_pose is None:
                continue
            latest_by_side[frame.handedness] = frame

        for side in self.active_sides:
            frame = latest_by_side.get(side)
            if frame is None:
                continue
            statuses[side] = self._update_side(side, frame, new_target_acquire_ns)

        solve_result = self._solve_and_clamp(statuses)

        arm_angles = {
            side: np.array([self._q_prev[idx] for idx in self.ik.arm_joint_indices[side]])
            for side in self._targets
        }
        target_poses = {side: pin.SE3(T) for side, T in self._targets.items()}
        operator_poses = {}
        for side, frame in latest_by_side.items():
            if side not in self.active_sides:
                continue
            operator_poses[side] = pin.SE3(self._operator_display_poses.get(side, frame.wrist_pose))
        hand_positions = dict(self._hand_targets)

        return OrcaArmTeleopResult(
            arm_angles=arm_angles,
            hand_positions=hand_positions,
            target_poses=target_poses,
            operator_poses=operator_poses,
            q=self._q_prev.copy(),
            converged=solve_result.converged,
            position_error=solve_result.position_error,
            orientation_error=solve_result.orientation_error,
            statuses=statuses,
            new_target_acquire_ns=new_target_acquire_ns,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_q_home(self) -> np.ndarray:
        """Build the side-biased anchor pose: forearms forward, palms down.

        Indices into the per-side joint array are 0..4 → joint1..joint5:
          joint1 (shoulder yaw, ±0.6): rotate ~34° outward so the wrists
                 land at shoulder width when the elbow flexes forward.
          joint4 (elbow, π/2):         right-angle flex.
          joint5 (wrist roll, ∓1.43):  palm-down on both sides; sign is
                 mirrored because the L/R carpals frames are 180° apart
                 in the URDF. Pulled in ~0.14 rad from the joint limit
                 (±π/2) to leave headroom for the operator to roll.
        """
        q = self.ik.neutral_q.copy()
        side_bias = {
            "left": {1: +0.004, 3: +1.520, 4: +1.571},
            "right": {1: +0.005, 3: +1.530, 4: -1.571},
        }
        for side, bias in side_bias.items():
            if side not in self.active_sides:
                continue
            idx_q = self.ik.arm_joint_indices[side]
            if len(idx_q) != 5 or not all(
                name.startswith(f"openarm_{side}_joint") for name in self.ik.arm_joint_names[side]
            ):
                continue
            for k, v in bias.items():
                q[idx_q[k]] = v

        # Clip to URDF position limits so values typed at the limit (e.g.
        # 1.571 vs the truncated 1.570796) don't trip pink's check_limits
        # on the very first IK call. Margin is well below any meaningful
        # operator precision.
        return np.clip(q, self.ik._model.lowerPositionLimit, self.ik._model.upperPositionLimit)

    def _retargeter_for(self, side: str) -> Retargeter:
        if side in self._retargeters:
            return self._retargeters[side]
        if self._resolved_hand_model_path is None:
            self._resolved_hand_model_path = _default_hand_model_path()
            logger.info("Finger retargeters resolving from %s", self._resolved_hand_model_path)
        self._retargeters[side] = Retargeter.from_paths(
            self._resolved_hand_model_path,
            _hand_urdf_path_for_side(self.config.hand_urdf_path, "right"),
            hand_type_override=self.config.hand_type_override,
        )
        return self._retargeters[side]

    def _manual_translation_scale_for_side(self, side: str) -> np.ndarray:
        scale = self.config.manual_scale
        if scale is None:
            raise ValueError("manual_scale is not configured")
        if isinstance(scale, dict):
            if side not in scale:
                raise ValueError(f"manual_scale is missing side {side!r}")
            arr = np.asarray(scale[side], dtype=np.float64)
        else:
            arr = np.asarray(scale, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(3, float(arr), dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"translation scale for {side!r} must be scalar or shape (3,), " f"got {arr.shape}"
            )
        return arr.copy()

    def _translation_delta_for_side(self, side: str, T_op: pin.SE3) -> np.ndarray:
        """Map operator anchor-relative translation into robot workspace coordinates.

        If ``manual_scale`` is unset, endpoint calibration maps each calibrated
        operator axis endpoint exactly onto the corresponding robot workspace
        endpoint. Manual scale keeps the simpler local/world delta path for
        quick experiments and explicit overrides.
        """
        return self._translation_delta_from_reference(
            side,
            T_op,
            self._T_first[side],
        )

    def _translation_delta_from_reference(
        self,
        side: str,
        T_op: pin.SE3,
        T_ref: pin.SE3,
    ) -> np.ndarray:
        raw_dp = T_op.translation - T_ref.translation
        workspace_limits = self.config.workspace_delta_limits_m
        if workspace_limits is None:
            robot_lo = robot_hi = None
        else:
            robot_lo = np.asarray(workspace_limits[side][0], dtype=np.float64)
            robot_hi = np.asarray(workspace_limits[side][1], dtype=np.float64)

        if self.config.manual_scale is not None or not self.config.use_workspace_calibration:
            if self.config.translation_frame == "operator_local":
                operator_delta = T_ref.rotation.T @ raw_dp
            else:
                operator_delta = raw_dp
            scale = (
                np.full(3, self.config.bootstrap_scale, dtype=np.float64)
                if self.config.manual_scale is None
                else self._manual_translation_scale_for_side(side)
            )
            if self.config.translation_frame == "operator_local":
                scaled = self._home_poses[side].rotation @ (scale * operator_delta)
            else:
                scaled = scale * operator_delta
            if robot_lo is None or robot_hi is None:
                return scaled
            return np.clip(scaled, robot_lo, robot_hi)

        if robot_lo is None or robot_hi is None:
            raise ValueError(
                "workspace_delta_limits_m is required when " "use_workspace_calibration=True"
            )

        operator_lo = np.asarray(self.config.operator_workspace_limits_m[side][0], dtype=np.float64)
        operator_hi = np.asarray(self.config.operator_workspace_limits_m[side][1], dtype=np.float64)
        anchor = T_ref.translation
        out = np.zeros(3, dtype=np.float64)
        eps = 1e-6

        for axis in range(3):
            if raw_dp[axis] >= 0.0:
                span = operator_hi[axis] - anchor[axis]
                if span <= eps:
                    out[axis] = robot_hi[axis] if raw_dp[axis] > 0.0 else 0.0
                    continue
                out[axis] = np.clip(raw_dp[axis] / span, 0.0, 1.0) * robot_hi[axis]
            else:
                span = anchor[axis] - operator_lo[axis]
                if span <= eps:
                    out[axis] = robot_lo[axis]
                    continue
                out[axis] = np.clip((-raw_dp[axis]) / span, 0.0, 1.0) * robot_lo[axis]

        return out

    def _operator_neutral_pose_for_side(self, side: str) -> pin.SE3:
        if side not in self.config.operator_neutral_poses_flu:
            raise ValueError(f"No operator neutral pose configured for side {side!r}")
        p, R = self.config.operator_neutral_poses_flu[side]
        return pin.SE3(
            np.asarray(R, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
        )

    def _candidate_target_from_reference(
        self,
        side: str,
        T_op: pin.SE3,
        T_ref: pin.SE3,
    ) -> pin.SE3:
        dR = T_ref.rotation.T @ T_op.rotation
        dp = self._translation_delta_from_reference(side, T_op, T_ref)
        return pin.SE3(
            self._home_poses[side].rotation @ dR,
            self._home_poses[side].translation + dp,
        )

    def _neutral_lock_errors(self, side: str, candidate: pin.SE3) -> tuple[float, float]:
        home = self._home_poses[side]
        position_error = float(np.linalg.norm(candidate.translation - home.translation))
        orientation_error = float(np.linalg.norm(pin.log3(home.rotation.T @ candidate.rotation)))
        return position_error, orientation_error

    def _filtered_operator_pose(
        self,
        side: str,
        raw_pose: pin.SE3,
        timestamp_ns: int,
    ) -> pin.SE3:
        """Reject single-frame tracking jumps and low-pass the operator wrist pose."""
        if self.config.pose_filter_alpha >= 1.0:
            return raw_pose

        previous = self._filtered_operator_poses.get(side)
        previous_timestamp_ns = self._filtered_operator_timestamps_ns.get(side)
        if previous is None or previous_timestamp_ns is None:
            self._filtered_operator_poses[side] = pin.SE3(raw_pose)
            self._filtered_operator_timestamps_ns[side] = int(timestamp_ns)
            return raw_pose

        dt = max(1e-3, (int(timestamp_ns) - previous_timestamp_ns) * 1e-9)
        dp = raw_pose.translation - previous.translation
        angular_delta = pin.log3(previous.rotation.T @ raw_pose.rotation)
        translation_speed = float(np.linalg.norm(dp) / dt)
        rotation_speed = float(np.linalg.norm(angular_delta) / dt)

        if (
            translation_speed > self.config.max_operator_translation_speed_mps
            or rotation_speed > self.config.max_operator_rotation_speed_radps
        ):
            logger.debug(
                "Holding %s operator pose after jump: %.2fm/s %.1frad/s",
                side,
                translation_speed,
                rotation_speed,
            )
            self._filtered_operator_timestamps_ns[side] = int(timestamp_ns)
            return pin.SE3(previous)

        alpha = float(np.clip(self.config.pose_filter_alpha, 0.0, 1.0))
        filtered = pin.SE3(
            previous.rotation @ pin.exp3(alpha * angular_delta),
            previous.translation + alpha * dp,
        )
        self._filtered_operator_poses[side] = filtered
        self._filtered_operator_timestamps_ns[side] = int(timestamp_ns)
        return filtered

    def _update_side(
        self,
        side: str,
        frame: OrcaArmTeleopFrame,
        new_target_acquire_ns: dict[str, int],
    ) -> SideStatus:
        """Anchor once on the first frame, then track every pose relative to it."""
        T_op = self._filtered_operator_pose(side, frame.wrist_pose, frame.timestamp_ns)

        if side not in self._T_first:
            if self.config.require_operator_neutral:
                T_neutral = self._operator_neutral_pose_for_side(side)
                candidate = self._candidate_target_from_reference(side, T_op, T_neutral)
                self._operator_display_poses[side] = candidate
                self._targets[side] = pin.SE3(
                    self._home_poses[side].rotation.copy(),
                    self._home_poses[side].translation.copy(),
                )
                pos_err, ori_err = self._neutral_lock_errors(side, candidate)
                if (
                    pos_err > self.config.operator_neutral_position_tolerance_m
                    or ori_err > self.config.operator_neutral_orientation_tolerance_rad
                ):
                    logger.debug(
                        "Awaiting %s neutral lock: pos_err=%.3fm ori_err=%.1fdeg",
                        side,
                        pos_err,
                        np.rad2deg(ori_err),
                    )
                    return "awaiting_neutral"

                logger.info(
                    "Operator neutral lock acquired for %s " "(pos_err=%.3fm ori_err=%.1fdeg)",
                    side,
                    pos_err,
                    np.rad2deg(ori_err),
                )
                self._operator_display_poses.pop(side, None)

            self._T_first[side] = pin.SE3(T_op.rotation.copy(), T_op.translation.copy())
            self._targets[side] = pin.SE3(
                self._home_poses[side].rotation.copy(),
                self._home_poses[side].translation.copy(),
            )
            if self.config.manual_scale is not None:
                self._scale[side] = self._manual_translation_scale_for_side(side)
            logger.info(
                "Anchored %s on first operator pose (position=%s)",
                side,
                np.round(T_op.translation, 3).tolist(),
            )
            new_target_acquire_ns[side] = int(frame.timestamp_ns)
            return "tracking"

        wrist_angle_degrees = _relative_flu_z_angle_degrees(self._T_first[side], T_op)

        # Finger retargeter construction/calibration can be slow on the first
        # use. Do not let it block the initial anchor frame, especially
        # for dataset replay where the ingress queue would otherwise drop the
        # opening anchor frames while the hand model initializes.
        if frame.keypoints is not None:
            try:
                if frame.keypoint_source == "metaquest":
                    from orca_teleop.ingress.metaquest.landmarks import (
                        retargeter_landmarks_from_quest,
                    )

                    kp = retargeter_landmarks_from_quest(frame.keypoints, side)
                    source: str = "metaquest"
                elif frame.keypoint_source == "mediapipe":
                    kp = frame.keypoints
                    source = "mediapipe"
                else:  # "canonical": already in retargeter frame, source-agnostic
                    kp = frame.keypoints
                    source = "metaquest"
                hand_action = self._retargeter_for(side).retarget(
                    TargetPose(
                        joint_positions=kp,
                        source=source,
                        wrist_angle_degrees=wrist_angle_degrees,
                    )
                )
                if hand_action is not None:
                    self._hand_targets[side] = _retargeted_action_for_side(hand_action, side)
            except (AssertionError, ValueError):
                logger.debug("Skipping degenerate %s hand landmark frame.", side)

        # Apply the operator's wrist motion as a local relative rotation:
        # R_target = R_robot_home * (R_operator_first^-1 * R_operator_now).
        # Pre-multiplying by a world-frame delta would rotate the robot wrist
        # around Quest/world axes, which only works when the operator and robot
        # frames are already aligned.
        dR = self._T_first[side].rotation.T @ T_op.rotation
        dp = self._translation_delta_for_side(side, T_op)

        self._targets[side] = pin.SE3(
            self._home_poses[side].rotation @ dR,
            self._home_poses[side].translation + dp,
        )
        new_target_acquire_ns[side] = int(frame.timestamp_ns)
        return "tracking"

    def _position_clipped_targets(self) -> dict[str, pin.SE3]:
        """Project final target translations onto position-reachable IK poses.

        The translation workspace constants are an axis-aligned envelope, not
        the true reachable set. This optional pass solves a position-only IK
        problem first and only rewrites the target translation when the target
        is outside what the arm can reach from the current seed. Orientation is
        deliberately left untouched for the subsequent pose IK.
        """
        if (
            not self.config.workspace_position_clip
            or self.config.ik_mode != "pose"
            or not self._targets
        ):
            return self._targets

        try:
            projected = self.ik.solve_position(
                self._targets,
                self._q_prev,
                damping=self.config.position_damping,
                step_size=self.config.position_step_size,
                posture_gain=self.config.position_posture_gain,
                rotation_axes="",
                rotation_gain=0.0,
            )
        except Exception:
            logger.exception("Workspace position clipping failed; using unclipped targets.")
            return self._targets

        clipped: dict[str, pin.SE3] = {}
        for side, target in self._targets.items():
            solved_T = pin.SE3(self.ik.forward_kinematics_full(projected.q, side))
            position_error = float(np.linalg.norm(target.translation - solved_T.translation))
            if position_error <= self.config.workspace_position_clip_tolerance_m:
                clipped[side] = target
                continue
            clipped[side] = pin.SE3(target.rotation.copy(), solved_T.translation.copy())
            logger.debug(
                "Workspace-clipped %s target position by %.1f mm",
                side,
                1000.0 * position_error,
            )
        return clipped

    def _solve_and_clamp(self, statuses: dict[str, SideStatus]) -> IKResult:
        """IK solve + per-joint step clamp. Marks ``ik_failed`` on exception.

        Per-joint Δq clamp: caps how far any single joint can move in one
        IK tick. Catches startup anchors and tracking blips that would
        otherwise integrate into a large one-shot jump.
        Caller-visible state (``q_prev``) is the clamped value, so the
        next solve starts from where the arm actually is, not from the
        unclamped IK output.
        """
        if not self._targets:
            return IKResult(
                q=self._q_prev.copy(),
                position_error={},
                orientation_error={},
                converged={},
            )

        targets_for_solve = self._position_clipped_targets()

        try:
            if self.config.ik_mode == "position":
                result = self.ik.solve_position(
                    targets_for_solve,
                    self._q_prev,
                    damping=self.config.position_damping,
                    step_size=self.config.position_step_size,
                    posture_gain=self.config.position_posture_gain,
                    rotation_axes=self.config.position_rotation_axes,
                    rotation_gain=self.config.position_rotation_gain,
                )
            else:
                result = self.ik.solve(targets_for_solve, self._q_prev)
        except Exception:
            logger.exception("IK solve failed; sides=%s", sorted(self._targets))
            for side in self._targets:
                statuses[side] = "ik_failed"
            return IKResult(
                q=self._q_prev.copy(),
                position_error={},
                orientation_error={},
                converged=dict.fromkeys(self._targets, False),
            )

        dq = np.clip(
            result.q - self._q_prev,
            -self.config.max_joint_step_rad,
            self.config.max_joint_step_rad,
        )
        self._q_prev = self._q_prev + dq
        self._targets = targets_for_solve
        return self.ik.evaluate(
            targets_for_solve,
            self._q_prev,
            position_only=self.config.ik_mode == "position",
        )
