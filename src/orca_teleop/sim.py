"""Sim-backed ``RobotSink`` driving an ``orca_sim`` Gymnasium env.

Mirrors the physical-robot path of the teleop pipeline: the ingress and
retargeter stages are unchanged, streaming``OrcaJointPositions``to a
``OrcaHandSimSink``, stepping a MuJoCo environment from ``orca_sim``.
"""

import logging
import queue
import threading
from typing import Any

import numpy as np
from orca_core import OrcaJointPositions
from orca_sim.envs import RENDER_FPS, BaseOrcaHandEnv

from orca_teleop.pipeline import _SHUTDOWN, RobotSink
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)


class OrcaHandSimSink(RobotSink):
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
    ) -> None:
        self._env_name = env_name
        self._version = version
        self._render_mode = render_mode
        self._env: BaseOrcaHandEnv = None
        self._actuator_joint_names: list[str] = []
        self._last_action: np.ndarray | None = None
        self._dt: float = 1.0 / RENDER_FPS

    def connect(self) -> None:
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

        logger.info(
            "SimSink connected: env=%s version=%s actuators=%d dt=%.3fs",
            self._env_name,
            getattr(env, "version", "?"),
            len(self._actuator_joint_names),
            self._dt,
        )

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
                self._env.step(self._to_action_array(self._last_action))

            except Exception as e:
                logger.exception("orca_sim env.step() failed: %s", e)
                break

            ticker.tick()  # sleeps to control frequency

    def close(self) -> None:
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception:
            logger.exception(".close() encountered an error")
        finally:
            self._env = None

    def _to_action_array(self, positions: OrcaJointPositions) -> np.ndarray:
        # Retargeter outputs degrees; MuJoCo accepts radians
        return np.deg2rad(positions.as_array(self._actuator_joint_names))
