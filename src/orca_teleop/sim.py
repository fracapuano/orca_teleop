"""Sim-backed ``RobotSink`` driving an ``orca_sim`` Gymnasium env.

Mirrors the physical-robot path of the teleop pipeline: the ingress and
retargeter stages are unchanged, streaming``OrcaJointPositions``to a
``OrcaHandSimSink``, stepping a MuJoCo environment from ``orca_sim``.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from orca_core import OrcaJointPositions
from orca_sim.envs import RENDER_FPS, BaseOrcaHandEnv

from orca_teleop.pipeline import _SHUTDOWN, RecordableSink
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)


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
        actions_q: queue.Queue[OrcaJointPositions | object],
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
                elif isinstance(item, OrcaJointPositions):
                    try:
                        self._last_action = item
                    except Exception:
                        logger.exception(
                            "SimSink failed to convert OrcaJointPositions; holding last action"
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
