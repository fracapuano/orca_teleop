"""Sim-backed ``RobotSink`` driving an ``orca_sim`` Gymnasium env.

Mirrors the physical-robot path of the teleop pipeline: the ingress and
retargeter stages are unchanged, streaming``OrcaJointPositions``to a
``OrcaHandSimSink``, stepping a MuJoCo environment from ``orca_sim``.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from orca_core import OrcaJointPositions
from orca_sim.envs import RENDER_FPS, BaseOrcaHandEnv

from orca_teleop.pipeline import _SHUTDOWN, RecordableSink, TeleopAction
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)

_ORCA_ARM_SIDES = ("left", "right")
_ORCA_ARM_JOINTS_PER_SIDE = 5
_ORCA_PANDA_SIDES = ("right",)
_ORCA_PANDA_JOINTS_PER_SIDE = 7
_CARPALS_SIDE_PREFIX = {"left": "L", "right": "R"}
_DEFAULT_TASK_MAX_EPISODE_STEPS = 10_000
_DEBUG_TRIAD_AXIS_LEN = 0.14
_DEBUG_TRIAD_AXIS_R = 0.006
_DEBUG_TRIAD_AXIS_LEN_SCALE = {
    "current": 0.82,
    "operator": 1.0,
    "target": 1.25,
}
_DEBUG_ORIGIN_R = {
    "current": 0.010,
    "operator": 0.014,
    "target": 0.018,
}
_DEBUG_ORIGIN_RGBA = {
    "current": np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32),
    "operator": np.array([1.0, 0.58, 0.08, 0.95], dtype=np.float32),
    "target": np.array([0.85, 0.18, 1.0, 0.95], dtype=np.float32),
}
_DEBUG_AXIS_RGBA = (
    np.array([1.0, 0.08, 0.08, 0.92], dtype=np.float32),
    np.array([0.1, 0.82, 0.22, 0.92], dtype=np.float32),
    np.array([0.16, 0.35, 1.0, 0.92], dtype=np.float32),
)
_DEBUG_CURRENT_TO_TARGET_RGBA = np.array([0.9, 0.35, 1.0, 0.75], dtype=np.float32)


# Retargeter joint IDs -> generated OrcaArm MJCF joint-name fragments. Kept
# local to avoid importing the arm visualization sink just to resolve controls.
_ORCA_ARM_HAND_JOINT_MARKERS = {
    "thumb_mcp": {"left": "T-TP-L_92b8100b_to_", "right": "T-TP-R_1c2b802d_to_"},
    "thumb_abd": {"left": "L-T-AP_58680c44_to_", "right": "R-T-AP_a9723101_to_"},
    "thumb_cmc": {"left": "T-PP_ef067304_to_", "right": "T-PP_68395e98_to_"},
    "thumb_pip": {"left": "T-PP_ef067304_to_", "right": "T-PP_68395e98_to_"},
    "thumb_dip": {"left": "T-DP_307db3cc_to_", "right": "T-DP_b7429e50_to_"},
    "index_abd": {"left": "I-AP-L_57ce92f7_to_", "right": "I-AP-R_d95d02d1_to_"},
    "index_mcp": {"left": "I-PP_3df4f91d_to_", "right": "I-PP_bacbd481_to_"},
    "index_pip": {
        "left": "I-FingerTipAssembly_ed91b18a_to_",
        "right": "I-FingerTipAssembly_ec49c16c_to_",
    },
    "middle_abd": {"left": "M-AP_e04a96f2_to_", "right": "M-AP_e04a96f2_to_"},
    "middle_mcp": {"left": "M-PP_08efa608_to_", "right": "M-PP_08efa608_to_"},
    "middle_pip": {
        "left": "M-FingerTipAssembly_34afb748_to_",
        "right": "M-FingerTipAssembly_34afb748_to_",
    },
    "ring_abd": {"left": "M-AP_6ec59111_to_", "right": "M-AP_6ec59111_to_"},
    "ring_mcp": {"left": "M-PP_8660a1eb_to_", "right": "M-PP_8660a1eb_to_"},
    "ring_pip": {
        "left": "M-FingerTipAssembly_424a8e75_to_",
        "right": "M-FingerTipAssembly_424a8e75_to_",
    },
    "pinky_abd": {"left": "P-AP_f5e42b61_to_", "right": "P-AP_f5e42b61_to_"},
    "pinky_mcp": {"left": "P-PP_1d411b9b_to_", "right": "P-PP_1d411b9b_to_"},
    "pinky_pip": {
        "left": "P-FingerTipAssembly_cd219176_to_",
        "right": "P-FingerTipAssembly_cd219176_to_",
    },
}


@dataclass(frozen=True)
class SimCameraConfig:
    name: str = "frontal"
    width: int = 320
    height: int = 240


class OrcaHandSimSink(RecordableSink):
    """``RobotSink`` that steps an ``orca_sim`` env from the actions queue.

    ``OrcaJointPositions`` values arrive in physical degrees (retargeter
    convention); ``_to_action_array`` converts them to radians before writing
    to the MuJoCo ctrl vector, which uses radians throughout.
    """

    def __init__(
        self,
        env_name: str = "right",
        version: str | None = None,
        render_mode: str = "human",
        camera_config: SimCameraConfig | None = None,
    ) -> None:
        self._env_name = env_name
        self._version = version
        self._render_mode = render_mode
        self._camera_config = SimCameraConfig() if camera_config is None else camera_config
        self._env: BaseOrcaHandEnv = None
        self._actuator_joint_names: list[str] = []
        self._last_action: OrcaJointPositions | None = None
        self._dt: float = 1.0 / RENDER_FPS
        self._renderer: Any | None = None
        self._record_camera: Any | None = None

    def connect(self) -> None:
        import mujoco
        from orca_sim import OrcaHandLeft, OrcaHandRight

        builders = {"left": OrcaHandLeft, "right": OrcaHandRight}
        if self._env_name not in builders:
            raise ValueError(
                f"Unknown orca_sim env '{self._env_name}'. " f"Choices: {sorted(builders)}"
            )

        kwargs: dict[str, Any] = {"render_mode": self._render_mode}
        if self._version is not None:
            kwargs["version"] = self._version
        env = builders[self._env_name](**kwargs)
        env.reset()

        self._env = env

        self._actuator_joint_names = list(env.hand.config.joint_ids)

        # Hold neutral pose until the first retargeted command arrives.
        self._last_action = OrcaJointPositions(env.hand.config.neutral_position)

        self._dt = 1.0 / float(env.metadata.get("render_fps", RENDER_FPS))

        self._renderer = mujoco.Renderer(
            env.model,
            height=self._camera_config.height,
            width=self._camera_config.width,
        )
        self._record_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self._record_camera)

        logger.info(
            "SimSink connected: env=%s version=%s actuators=%d dt=%.3fs camera=%s",
            self._env_name,
            getattr(env, "version", "?"),
            len(self._actuator_joint_names),
            self._dt,
            self.camera_shapes,
        )

    @property
    def joint_ids(self) -> list[str]:
        return list(self._actuator_joint_names)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        return {
            self._camera_config.name: (
                self._camera_config.height,
                self._camera_config.width,
                3,
            )
        }

    def get_joint_state(self) -> np.ndarray:
        assert self._env is not None, "connect() must be called before get_joint_state()"
        state_rad = self._env.hand.get_joint_position().as_array(self._actuator_joint_names)
        return np.rad2deg(state_rad).astype(np.float32)

    def capture_frames(self) -> dict[str, np.ndarray]:
        assert self._env is not None, "connect() must be called before capture_frames()"
        assert self._renderer is not None
        assert self._record_camera is not None

        self._renderer.update_scene(self._env.data, camera=self._record_camera)
        return {
            self._camera_config.name: np.asarray(
                self._renderer.render(),
                dtype=np.uint8,
            )
        }

    def dispatch_action(self, action: OrcaJointPositions) -> None:
        assert self._env is not None, "connect() must be called before dispatch_action()"
        self._env.step(self._to_action_array(action))

    def run_loop(
        self,
        actions_q: queue.Queue[TeleopAction | object],
        stop_event: threading.Event,
    ) -> None:
        assert self._env is not None, "connect() must be called before run_loop()"
        assert self._last_action is not None

        ticker = RateTicker(dt=self._dt)

        while not stop_event.is_set():
            shutdown_received = False
            try:
                item = actions_q.get_nowait()
                if item is _SHUTDOWN:
                    shutdown_received = True
                elif isinstance(item, TeleopAction):
                    try:
                        self._last_action = item.joint_positions
                    except Exception:
                        logger.exception(
                            "SimSink failed to convert TeleopAction; holding last action"
                        )
            except queue.Empty:
                pass

            if shutdown_received:
                break

            try:
                self.dispatch_action(self._last_action)

            except Exception as e:
                logger.exception("orca_sim env.step() failed: %s", e)
                break

            ticker.tick()  # sleeps to control frequency

    def close(self) -> None:
        if self._env is None:
            return
        try:
            if self._renderer is not None:
                self._renderer.close()
            self._env.close()
        except Exception:
            logger.exception(".close() encountered an error")
        finally:
            self._env = None
            self._renderer = None
            self._record_camera = None

    def _to_action_array(self, positions: OrcaJointPositions) -> np.ndarray:
        # Retargeter outputs degrees; MuJoCo accepts radians
        return np.deg2rad(positions.as_array(self._actuator_joint_names))


class OrcaArmCubeStackingSink:
    """Task-backed sink for Quest-driven OrcaArm cube stacking teleop.

    The sink consumes the same solved arm-angle / retargeted-hand stream as
    the arm visualization sinks, but writes those commands into
    ``orca_sim.OrcaArmCubeStacking`` so teleop runs against the actual task
    environment rather than a viewer-only scene.
    """

    def __init__(
        self,
        *,
        render_mode: str | None = "human",
        version: str | None = None,
        camera_names: tuple[str, ...] | None = None,
        camera_width: int = 128,
        camera_height: int = 128,
        render_camera: str = "chest_table_camera",
        max_episode_steps: int = _DEFAULT_TASK_MAX_EPISODE_STEPS,
        reset_on_done: bool = True,
        seed: int | None = None,
        instant_qpos: bool = False,
        frame_skip: int | None = None,
        scene_file: str | None = None,
        debug_visuals: bool = False,
    ) -> None:
        self._render_mode = render_mode
        self._version = version
        self._scene_file = scene_file
        self._camera_names = camera_names
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._render_camera = render_camera
        self._max_episode_steps = int(max_episode_steps)
        self._reset_on_done = bool(reset_on_done)
        self._seed = seed
        self._instant_qpos = bool(instant_qpos)
        self._frame_skip = None if frame_skip is None else int(frame_skip)
        self._debug_visuals = bool(debug_visuals)

        self._env: Any | None = None
        self._arm_action_indices: dict[str, list[int]] = {}
        self._hand_action_indices: dict[str, dict[str, int]] = {}
        self._carpals_body_ids: dict[str, int] = {}
        self._home_arm_angles: dict[str, np.ndarray] = {}
        self._debug_target_Ts: dict[str, np.ndarray] = {}
        self._debug_operator_Ts: dict[str, np.ndarray] = {}
        self._debug_ik_frame_in_current_Ts: dict[str, np.ndarray] = {}
        self._last_action: np.ndarray | None = None
        self._home_action: np.ndarray | None = None
        self._dt = 1.0 / RENDER_FPS
        self._last_physics_wall_time: float | None = None
        self._physics_time_debt_s = 0.0
        self._max_physics_steps_per_update = 20

    @property
    def arm_joint_names(self) -> dict[str, list[str]]:
        return self._arm_joint_names_by_side()

    def connect(self) -> None:
        self.launch()

    def launch(self) -> None:
        kwargs: dict[str, Any] = {
            "render_mode": self._render_mode,
            "version": self._version,
            "camera_width": self._camera_width,
            "camera_height": self._camera_height,
            "render_camera": self._render_camera,
            "max_episode_steps": self._max_episode_steps,
        }
        if self._scene_file is not None:
            kwargs["scene_file"] = self._scene_file
        if self._frame_skip is not None:
            kwargs["frame_skip"] = self._frame_skip
        if self._camera_names is not None:
            kwargs["camera_names"] = self._camera_names

        env = self._make_env(**kwargs)
        env.reset(seed=self._seed)
        self._env = env

        self._arm_action_indices = {
            side: [self._action_index_for_joint(joint_name) for joint_name in joint_names]
            for side, joint_names in self.arm_joint_names.items()
        }
        self._hand_action_indices = {
            side: self._resolve_hand_action_indices(side) for side in self.arm_joint_names
        }
        self._carpals_body_ids = {
            side: self._find_body_id_by_prefix(self._carpals_body_prefix(side))
            for side in self.arm_joint_names
        }
        self._home_arm_angles = {
            side: np.array(
                [self._qpos_for_joint(joint_name) for joint_name in joint_names],
                dtype=np.float64,
            )
            for side, joint_names in self.arm_joint_names.items()
        }

        self._home_action = np.asarray(env.data.ctrl[list(env.actuator_ids)], dtype=np.float32)
        self._last_action = self._home_action.copy()
        self._dt = 1.0 / float(env.metadata.get("render_fps", RENDER_FPS))
        self._render()
        self._last_physics_wall_time = time.monotonic()
        self._physics_time_debt_s = 0.0

        logger.info(
            "%s connected: version=%s actuators=%d dt=%.3fs "
            "physics_dt=%.3fs frame_skip=%d render_mode=%s cameras=%s "
            "debug_visuals=%s scene=%s",
            type(self).__name__,
            getattr(env, "version", "?"),
            len(env.actuator_names),
            self._dt,
            float(env.model.opt.timestep) * int(env.frame_skip),
            int(env.frame_skip),
            self._render_mode,
            getattr(env, "camera_names", ()),
            self._debug_visuals,
            getattr(env, "scene_file", "?"),
        )

    @property
    def home_arm_angles(self) -> dict[str, np.ndarray]:
        """Per-side arm q at the environment-defined reset/home pose."""
        return {side: values.copy() for side, values in self._home_arm_angles.items()}

    @property
    def current_arm_angles(self) -> dict[str, np.ndarray]:
        """Per-side arm q measured from the live MuJoCo state."""
        self._require_env()
        return {
            side: np.array(
                [self._qpos_for_joint(joint_name) for joint_name in joint_names],
                dtype=np.float64,
            )
            for side, joint_names in self.arm_joint_names.items()
        }

    def to_neutral_configuration(self, arm_angles: dict[str, np.ndarray] | None = None) -> None:
        self._require_env()
        if arm_angles is None:
            self._last_action = self._home_action.copy()
            self._render()
            return

        action = self._compose_action(arm_angles=arm_angles, hand_positions=None)
        self._step_env(action)

    def update(
        self,
        arm_angles: dict[str, np.ndarray],
        hand_positions: dict[str, OrcaJointPositions] | None = None,
        target_Ts: dict[str, np.ndarray] | None = None,
        operator_Ts: dict[str, np.ndarray] | None = None,
    ) -> None:
        if target_Ts is not None:
            self._debug_target_Ts = {
                side: np.asarray(T, dtype=np.float64).copy() for side, T in target_Ts.items()
            }
        if operator_Ts is not None:
            self._debug_operator_Ts = {
                side: np.asarray(T, dtype=np.float64).copy() for side, T in operator_Ts.items()
            }
        action = self._compose_action(arm_angles=arm_angles, hand_positions=hand_positions)
        self._step_env(action)

    def set_debug_target_frame_offsets(self, target_home_Ts: dict[str, np.ndarray]) -> None:
        """Align IK target-frame triads with the MuJoCo body triads.

        The IK target pose comes from the Pinocchio/URDF frame, while the
        renderer's current pose is the MuJoCo body origin. In OrcaPanda those
        frames share a name but are not colocated, so record their home-pose
        offset and draw future targets in the comparable MuJoCo body frame.
        """
        offsets: dict[str, np.ndarray] = {}
        for side, target_home_T in target_home_Ts.items():
            if side not in self._carpals_body_ids:
                continue
            current_home_T = self._current_carpals_T(side)
            offsets[side] = np.linalg.inv(current_home_T) @ np.asarray(
                target_home_T,
                dtype=np.float64,
            )
        self._debug_ik_frame_in_current_Ts = offsets
        if offsets:
            logger.info(
                "Debug target triads aligned to MuJoCo carpals bodies: %s",
                {
                    side: round(float(np.linalg.norm(offset[:3, 3])), 4)
                    for side, offset in offsets.items()
                },
            )

    def run_loop(
        self,
        actions_q: queue.Queue[TeleopAction | object],
        stop_event: threading.Event,
    ) -> None:
        self._require_env()
        ticker = RateTicker(dt=self._dt)

        while not stop_event.is_set():
            shutdown_received = False
            try:
                item = actions_q.get_nowait()
                if item is _SHUTDOWN:
                    shutdown_received = True
                elif isinstance(item, TeleopAction):
                    self._apply_teleop_action(item)
            except queue.Empty:
                pass

            if shutdown_received:
                break

            assert self._last_action is not None
            self._step_env(self._last_action)
            ticker.tick()

    def close(self) -> None:
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception:
            logger.exception("OrcaArmCubeStackingSink.close() encountered an error")
        finally:
            self._env = None
            self._last_action = None
            self._home_action = None
            self._last_physics_wall_time = None
            self._physics_time_debt_s = 0.0

    def _apply_teleop_action(self, item: TeleopAction) -> None:
        arm_angles = {}
        if item.handedness in self.arm_joint_names and item.arm_angles is not None:
            arm_angles[item.handedness] = item.arm_angles
        hand_positions = (
            {item.handedness: item.joint_positions}
            if item.handedness in self.arm_joint_names
            else None
        )
        self._last_action = self._compose_action(
            arm_angles=arm_angles,
            hand_positions=hand_positions,
        )

    def _compose_action(
        self,
        *,
        arm_angles: dict[str, np.ndarray],
        hand_positions: dict[str, OrcaJointPositions] | None,
    ) -> np.ndarray:
        env = self._require_env()
        assert self._last_action is not None

        action = self._last_action.copy()
        for side, angles in arm_angles.items():
            if side not in self._arm_action_indices:
                raise ValueError(f"Unknown arm side {side!r}")
            resolved = np.asarray(angles, dtype=np.float32)
            expected_shape = (len(self._arm_action_indices[side]),)
            if resolved.shape != expected_shape:
                raise ValueError(
                    f"Expected {side} arm angles shape {expected_shape}, got {resolved.shape}"
                )
            action[self._arm_action_indices[side]] = resolved

        if hand_positions is not None:
            for side, positions in hand_positions.items():
                for joint_id, value_deg in positions:
                    action_idx = self._hand_action_indices.get(side, {}).get(joint_id)
                    if action_idx is None:
                        continue
                    action[action_idx] = np.deg2rad(value_deg)

        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

    def _step_env(self, action: np.ndarray) -> None:
        if self._instant_qpos:
            self._teleport_action_to_qpos(action)
            return

        env = self._require_env()
        now = time.monotonic()
        physics_dt = self._physics_dt()
        if self._last_physics_wall_time is None:
            self._last_physics_wall_time = now
            elapsed_s = physics_dt
        else:
            elapsed_s = max(0.0, now - self._last_physics_wall_time)
            self._last_physics_wall_time = now

        max_debt_s = self._max_physics_steps_per_update * physics_dt
        self._physics_time_debt_s = min(
            max_debt_s,
            self._physics_time_debt_s + elapsed_s,
        )
        steps_to_run = int(self._physics_time_debt_s / physics_dt)
        steps_to_run = min(self._max_physics_steps_per_update, max(1, steps_to_run))
        self._physics_time_debt_s = max(
            0.0,
            self._physics_time_debt_s - steps_to_run * physics_dt,
        )

        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for _ in range(steps_to_run):
            _, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        self._last_action = action.copy()
        self._render()

        if (terminated or truncated) and self._reset_on_done:
            logger.info(
                "OrcaArmCubeStacking episode done: reward=%.3f terminated=%s "
                "truncated=%s success=%s elapsed_steps=%s; resetting",
                reward,
                terminated,
                truncated,
                info.get("is_success"),
                info.get("elapsed_steps"),
            )
            env.reset()
            assert self._home_action is not None
            self._last_action = self._home_action.copy()
            self._last_physics_wall_time = time.monotonic()
            self._physics_time_debt_s = 0.0
            self._render()

    def _physics_dt(self) -> float:
        env = self._require_env()
        return float(env.model.opt.timestep) * int(env.frame_skip)

    def _teleport_action_to_qpos(self, action: np.ndarray) -> None:
        """Debug mode: write actuator position commands straight into qpos."""
        env = self._require_env()
        import mujoco

        clipped = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        env.data.ctrl[list(env.actuator_ids)] = clipped
        for action_index, actuator_id in enumerate(env.actuator_ids):
            joint_id = int(env.model.actuator_trnid[actuator_id, 0])
            qpos_adr = int(env.model.jnt_qposadr[joint_id])
            qvel_adr = int(env.model.jnt_dofadr[joint_id])
            env.data.qpos[qpos_adr] = float(clipped[action_index])
            env.data.qvel[qvel_adr] = 0.0
        mujoco.mj_forward(env.model, env.data)
        self._last_action = clipped.copy()
        self._render()

    def _render(self) -> None:
        env = self._require_env()
        if self._render_mode is not None:
            env.render()
            if self._debug_visuals:
                self._draw_debug_triads()

    def _action_index_for_joint(self, joint_name: str) -> int:
        env = self._require_env()
        for action_index, actuator_id in enumerate(env.actuator_ids):
            joint_id = int(env.model.actuator_trnid[actuator_id, 0])
            if env.model.joint(joint_id).name == joint_name:
                return action_index
        raise ValueError(f"Could not find actuator for joint {joint_name!r}")

    def _qpos_for_joint(self, joint_name: str) -> float:
        env = self._require_env()
        joint = env.model.joint(joint_name)
        qpos_adr = int(env.model.jnt_qposadr[joint.id])
        return float(env.data.qpos[qpos_adr])

    def _arm_joint_names_by_side(self) -> dict[str, list[str]]:
        return {
            side: [f"openarm_{side}_joint{i}" for i in range(1, _ORCA_ARM_JOINTS_PER_SIDE + 1)]
            for side in _ORCA_ARM_SIDES
        }

    def _make_env(self, **kwargs: Any) -> Any:
        from orca_sim import OrcaArmCubeStacking

        return OrcaArmCubeStacking(**kwargs)

    def _carpals_body_prefix(self, side: str) -> str:
        return f"orcahand_{side}_{_CARPALS_SIDE_PREFIX[side]}-Carpals_"

    def _find_body_id_by_prefix(self, prefix: str) -> int:
        env = self._require_env()
        import mujoco

        matches = [
            body_id
            for body_id in range(env.model.nbody)
            if (name := mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            and name.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one MuJoCo body with prefix {prefix!r}, got {matches}"
            )
        return int(matches[0])

    def _current_carpals_T(self, side: str) -> np.ndarray:
        env = self._require_env()
        body_id = self._carpals_body_ids[side]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(env.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(env.data.xpos[body_id], dtype=np.float64)
        return T

    def _draw_debug_triads(self) -> None:
        env = self._require_env()
        viewer = getattr(env, "_viewer", None)
        if viewer is None:
            viewer = getattr(getattr(env, "hand", None), "_viewer", None)
        if viewer is None or viewer.user_scn is None:
            return

        scene = viewer.user_scn
        scene.ngeom = 0
        for side in self.arm_joint_names:
            current_T = self._current_carpals_T(side)
            self._add_debug_triad(scene, current_T, kind="current")
            if side in self._debug_operator_Ts:
                self._add_debug_triad(
                    scene,
                    self._debug_operator_T_for_current_frame(side),
                    kind="operator",
                )
            if side in self._debug_target_Ts:
                target_T = self._debug_target_T_for_current_frame(side)
                self._add_debug_triad(scene, target_T, kind="target")
                self._add_debug_pose_delta(
                    scene,
                    current_T,
                    target_T,
                )
        viewer.sync()

    def _debug_target_T_for_current_frame(self, side: str) -> np.ndarray:
        target_T = self._debug_target_Ts[side]
        if side not in self._debug_ik_frame_in_current_Ts:
            return target_T
        return target_T @ np.linalg.inv(self._debug_ik_frame_in_current_Ts[side])

    def _debug_operator_T_for_current_frame(self, side: str) -> np.ndarray:
        operator_T = self._debug_operator_Ts[side]
        if side not in self._debug_ik_frame_in_current_Ts:
            return operator_T
        return operator_T @ np.linalg.inv(self._debug_ik_frame_in_current_Ts[side])

    def _add_debug_triad(self, scene: Any, T_world: np.ndarray, *, kind: str) -> None:
        import mujoco

        T = np.asarray(T_world, dtype=np.float64)
        origin = T[:3, 3]
        R = T[:3, :3]
        self._add_debug_sphere(
            scene,
            origin,
            _DEBUG_ORIGIN_R[kind],
            _DEBUG_ORIGIN_RGBA[kind],
        )
        radius_scale = {"current": 0.85, "operator": 1.15, "target": 1.45}[kind]
        axis_len = _DEBUG_TRIAD_AXIS_LEN * _DEBUG_TRIAD_AXIS_LEN_SCALE[kind]
        for axis_index, rgba in enumerate(_DEBUG_AXIS_RGBA):
            axis_rgba = rgba.copy()
            if kind == "current":
                axis_rgba[3] = 0.65
            elif kind == "target":
                axis_rgba[3] = 1.0
            axis_end = origin + R[:, axis_index] * axis_len
            if kind == "operator":
                self._add_debug_dashed_connector(
                    scene,
                    origin,
                    axis_end,
                    _DEBUG_TRIAD_AXIS_R * radius_scale,
                    axis_rgba,
                    mujoco,
                )
            else:
                self._add_debug_connector(
                    scene,
                    origin,
                    axis_end,
                    _DEBUG_TRIAD_AXIS_R * radius_scale,
                    axis_rgba,
                    mujoco,
                )

    def _add_debug_pose_delta(
        self,
        scene: Any,
        current_T: np.ndarray,
        target_T: np.ndarray,
    ) -> None:
        import mujoco

        current_origin = np.asarray(current_T, dtype=np.float64)[:3, 3]
        target_origin = np.asarray(target_T, dtype=np.float64)[:3, 3]
        self._add_debug_connector(
            scene,
            current_origin,
            target_origin,
            _DEBUG_TRIAD_AXIS_R * 0.45,
            _DEBUG_CURRENT_TO_TARGET_RGBA,
            mujoco,
        )

    def _add_debug_sphere(
        self,
        scene: Any,
        pos: np.ndarray,
        radius: float,
        rgba: np.ndarray,
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            logger.warning("MuJoCo marker scene full; dropping debug sphere")
            return
        import mujoco

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            rgba,
        )
        scene.ngeom += 1

    def _add_debug_connector(
        self,
        scene: Any,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: np.ndarray,
        mujoco: Any,
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            logger.warning("MuJoCo marker scene full; dropping debug triad axis")
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    def _add_debug_dashed_connector(
        self,
        scene: Any,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: np.ndarray,
        mujoco: Any,
    ) -> None:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        for i in range(3):
            a = i / 3.0
            b = a + 0.18
            self._add_debug_connector(
                scene,
                start + (end - start) * a,
                start + (end - start) * b,
                radius,
                rgba,
                mujoco,
            )

    def _resolve_hand_action_indices(self, side: str) -> dict[str, int]:
        env = self._require_env()
        prefix = f"orcahand_{side}_"
        out: dict[str, int] = {}

        for joint_id, side_markers in _ORCA_ARM_HAND_JOINT_MARKERS.items():
            marker = side_markers[side]
            matches = [
                action_index
                for action_index, actuator_id in enumerate(env.actuator_ids)
                if (
                    joint_name := env.model.joint(
                        int(env.model.actuator_trnid[actuator_id, 0])
                    ).name
                )
                and joint_name.startswith(prefix)
                and marker in joint_name
            ]
            if len(matches) != 1:
                logger.warning(
                    "Could not resolve %s hand joint %s in OrcaArm cube-stacking env "
                    "(matches=%d)",
                    side,
                    joint_id,
                    len(matches),
                )
                continue
            out[joint_id] = matches[0]
        return out

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("connect()/launch() must be called before using the sink")
        return self._env


class OrcaPandaCubeStackingSink(OrcaArmCubeStackingSink):
    """Task-backed sink for right-hand Quest teleop into ``orca_sim`` OrcaPanda."""

    def __init__(
        self,
        *,
        render_mode: str | None = "human",
        version: str | None = None,
        camera_names: tuple[str, ...] | None = None,
        camera_width: int = 128,
        camera_height: int = 128,
        render_camera: str = "orcapanda_overview",
        max_episode_steps: int = _DEFAULT_TASK_MAX_EPISODE_STEPS,
        reset_on_done: bool = True,
        seed: int | None = None,
        instant_qpos: bool = False,
        frame_skip: int | None = None,
        scene_file: str | None = None,
        debug_visuals: bool = False,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            version=version,
            scene_file=scene_file,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            render_camera=render_camera,
            max_episode_steps=max_episode_steps,
            reset_on_done=reset_on_done,
            seed=seed,
            instant_qpos=instant_qpos,
            frame_skip=frame_skip,
            debug_visuals=debug_visuals,
        )

    def _arm_joint_names_by_side(self) -> dict[str, list[str]]:
        return {"right": [f"panda_joint{i}" for i in range(1, _ORCA_PANDA_JOINTS_PER_SIDE + 1)]}

    def _make_env(self, **kwargs: Any) -> Any:
        import inspect

        import orca_sim

        env_cls = getattr(orca_sim, "OrcaPanda", None)
        if env_cls is not None:
            signature = inspect.signature(env_cls)
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if accepts_kwargs:
                return env_cls(**kwargs)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
            return env_cls(**filtered_kwargs)
        return orca_sim.OrcaPandaCubeStacking(**kwargs)
