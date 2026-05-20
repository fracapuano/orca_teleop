"""Teleop pipeline: ingress (gRPC) -> adapter -> robot sink.

Architecture:

    Typical deployment spans two machines, with gRPC as the only network boundary:

        TELEOP MACHINE                                      ROBOT MACHINE
    +---------------------------+              +-------------------------------------------+
    |         publisher         |    gRPC      |              teleop pipeline              |
    |  (MediaPipe, Manus, etc.) +------------->|  +---------------+                        |
    |                           |  HandFrame   |  | IngressServer |                        |
    +---------------------------+              |  +-------+-------+                        |
                                               |          | landmarks_q                    |
                                               |          | HandLandmarks (+ WristPose)    |
                                               |          v                                |
                                               |  +---------------+                        |
                                               |  |    Adapter    |                        |
                                               |  | retarget + IK |                        |
                                               |  +-------+-------+                        |
                                               |          | actions_q                      |
                                               |          | TeleopAction                   |
                                               |          v                                |
                                               |  +---------------+                        |
                                               |  |   RobotSink   |                        |
                                               |  +---------------+                        |
                                               +-------------------------------------------+

The ingress is a gRPC server streaming ``HandFrame`` from a generic publisher (MediaPipe webcam,
Manus glove, VisionPro, replay file, ...).
Publishers are standalone scripts that know nothing about the robot, they just stream
``(21, 3)`` hand landmarks over the network.

The adapter stage runs as a worker thread; the robot sink owns the main-thread control loop.
"""

import logging
import os
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HAND,
    DEFAULT_PORT,
    HEARTBEAT_INTERVAL,
    JOIN_TIMEOUT,
    MOTION_NUM_STEPS,
    QUEUES_MAXSIZE,
)
from orca_teleop.ingress.server import HandLandmarks, IngressServer, WristPose
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

if TYPE_CHECKING:
    # Don't necessarily import IK for wrist, because teleop is also hand only.
    import pinocchio as pin

    from orca_teleop.orca_arm_ik import BimanualIKSolver

logger = logging.getLogger(__name__)

_SHUTDOWN = object()


def _default_model_config_for_hand(handedness: str) -> str:
    """Resolve an installed OrcaHand config for the requested side."""
    import orca_core

    models_dir = os.path.join(os.path.dirname(orca_core.__file__), "models")
    candidates = [
        os.path.join(models_dir, "v2", f"orcahand_{handedness}", "config.yaml"),
        os.path.join(models_dir, "v1", f"orcahand_{handedness}", "config.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise RuntimeError(
        f"No bundled OrcaHand config found for handedness={handedness!r} under {models_dir}"
    )


def _resolve_model_config_for_hand(model_path: str | None, handedness: str) -> str:
    """Resolve or validate the hand model config for the requested side."""
    if handedness not in ("left", "right"):
        raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")

    if model_path is None:
        resolved = _default_model_config_for_hand(handedness)
        logger.info("Using %s OrcaHand model config: %s", handedness, resolved)
        return resolved

    hand = OrcaHand(model_path)
    config_type = hand.config.type
    if config_type != handedness:
        raise ValueError(
            f"OrcaHand config type {config_type!r} does not match handedness {handedness!r}: "
            f"{model_path}"
        )
    return model_path


def _shutdown_queue(q: "queue.Queue[Any]") -> None:
    """Signal a downstream worker to stop, without blocking."""
    try:
        q.put_nowait(_SHUTDOWN)
    except queue.Full:
        pass


@dataclass(frozen=True)
class TeleopAction:
    """Single output frame from the adapter: finger joints + optional arm IK solution.

    ``arm_angles`` carries the 5-DOF IK solution for the side identified by
    ``handedness`` when the adapter has an IK solver and the incoming frame
    carried a wrist pose. ``wrist_pose`` is preserved for sinks that want to
    render the raw target alongside the solved configuration.
    """

    joint_positions: OrcaJointPositions
    handedness: str | None = None
    arm_angles: np.ndarray | None = None
    wrist_pose: WristPose | None = None


@dataclass(frozen=True)
class AdapterState:
    """State threaded across :meth:`Adapter.step` calls.

    ``q_seed`` is the full pinocchio config used as the IK seed for the
    next solve (re-anchored to the previous solution, hence the
    branch-stable behavior the IK posture-task already provides). It is
    ``None`` when no IK solver is configured.

    ``last_wrist`` buffers the most recent operator wrist pose per side so
    bimanual IK can solve with whatever sides have been observed so far.
    """

    q_seed: np.ndarray | None
    last_wrist: dict[str, "pin.SE3"]


@dataclass
class TeleopQueues:
    landmarks_q: "queue.Queue[HandLandmarks]"
    actions_q: "queue.Queue[TeleopAction | object]"


@dataclass(frozen=True)
class OpenCVCameraConfig:
    name: str
    index: int = 0


class RobotSink(ABC):
    """Pluggable consumer of ``OrcaJointPositions``.

    The sink owns the main-thread loop, routing actions to a sink (real robot
    or sim environment), which implements its own run_loop.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def run_loop(
        self,
        actions_q: "queue.Queue[OrcaJointPositions | object]",
        stop_event: threading.Event,
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class RecordableSink(RobotSink):
    """Robot sink that can expose synchronized state/images for dataset recording."""

    @property
    @abstractmethod
    def joint_ids(self) -> list[str]: ...

    @property
    @abstractmethod
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]: ...

    @abstractmethod
    def get_joint_state(self) -> np.ndarray: ...

    @abstractmethod
    def capture_frames(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def dispatch_action(self, action: OrcaJointPositions) -> None: ...


class OrcaHandSink(RecordableSink):
    """Default sink: streams actions to a physical ``OrcaHand``.

    Resolves ``OrcaHand`` via the module-level attribute so tests that
    monkeypatch ``orca_teleop.pipeline.OrcaHand`` still intercept construction.
    """

    def __init__(
        self,
        model_path: str | None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
    ) -> None:
        self._model_path = model_path
        self._hand = OrcaHand(model_path)
        self._camera_configs = [] if camera_configs is None else list(camera_configs)
        self._captures: dict[str, Any] = {}
        self._camera_shapes: dict[str, tuple[int, int, int]] = {}

    def connect(self) -> None:
        success, message = self._hand.connect()
        if not success:
            raise RuntimeError(f"Robot failed to connect: {message}")

        self._hand.init_joints()
        self._open_cameras()

    @property
    def joint_ids(self) -> list[str]:
        return list(self._hand.config.joint_ids)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        return dict(self._camera_shapes)

    def get_joint_state(self) -> np.ndarray:
        return self._hand.get_joint_position().as_array(self.joint_ids).astype(np.float32)

    def capture_frames(self) -> dict[str, np.ndarray]:
        if not self._captures:
            return {}

        import cv2  # lazy

        frames: dict[str, np.ndarray] = {}
        for name, cap in self._captures.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera {name!r} read failed mid-episode.")
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

    def dispatch_action(self, action: OrcaJointPositions) -> None:
        self._hand.set_joint_positions(action, num_steps=MOTION_NUM_STEPS)

    def run_loop(
        self,
        actions_q: "queue.Queue[TeleopAction | object]",
        stop_event: threading.Event,
    ) -> None:
        assert self._hand is not None, "connect() must be called before run_loop()"
        while not stop_event.is_set():
            try:
                action = actions_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if action is _SHUTDOWN:
                break
            assert isinstance(action, TeleopAction)
            self.dispatch_action(action.joint_positions)

    def close(self) -> None:
        if self._hand is None:
            return
        try:
            self._release_cameras()
            self._hand.set_zero_position()
            self._hand.disable_torque()
            self._hand.disconnect()
        except Exception:
            logger.exception("OrcaHandSink.close() encountered an error")
        finally:
            self._hand = None

    def _open_cameras(self) -> None:
        if not self._camera_configs:
            return

        import cv2  # lazy

        for config in self._camera_configs:
            cap = cv2.VideoCapture(config.index)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {config.name!r} (index {config.index}).")
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"Camera {config.name!r} returned no frame on probe read.")
            self._captures[config.name] = cap
            self._camera_shapes[config.name] = tuple(int(x) for x in frame.shape)

        logger.info("Opened cameras: %s", self._camera_shapes)

    def _release_cameras(self) -> None:
        for cap in self._captures.values():
            try:
                cap.release()
            except Exception:
                pass
        self._captures.clear()
        self._camera_shapes.clear()


class Adapter:
    """Publisher-consumer adaptation: hand retargeting [+ wrist-pose IK].

    Stateful shell wrapping the pure :meth:`step` function. Each call reads
    the current :class:`AdapterState` and a :class:`HandLandmarks` frame and
    produces a new state plus an optional :class:`TeleopAction`. IK runs iff
    an ``ik_solver`` is configured AND the incoming frame carries a
    ``wrist_pose``; otherwise the adapter behaves as the legacy
    retargeter-only stage and emits ``arm_angles=None``.

    Per-side retargeters are built lazily on the first frame for each side,
    enabling bimanual hand retargeting via chaining (one retargeter per
    side) without changing the single-hand action shape.
    """

    def __init__(
        self,
        model_path: str | None,
        urdf_path: str | None,
        ik_solver: "BimanualIKSolver | None" = None,
        hand_type_override: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._urdf_path = urdf_path
        self._hand_type_override = hand_type_override
        self._retargeters: dict[str, Retargeter] = {}
        self._ik = ik_solver
        self._state = AdapterState(
            q_seed=ik_solver.neutral_q.copy() if ik_solver is not None else None,
            last_wrist={},
        )

    def _retargeter_for(self, side: str) -> Retargeter:
        if side not in self._retargeters:
            self._retargeters[side] = Retargeter.from_paths(
                self._model_path,
                self._urdf_path,
                hand_type_override=self._hand_type_override,
            )
        return self._retargeters[side]

    def step(
        self,
        state: AdapterState,
        landmarks: HandLandmarks,
    ) -> tuple[AdapterState, TeleopAction | None]:
        """Pure step: ``(state, landmarks) → (new_state, action | None)``."""
        retargeter = self._retargeter_for(landmarks.handedness)  # one retargeter per side
        try:
            action = retargeter.retarget(
                TargetPose(
                    joint_positions=landmarks.keypoints, source="mediapipe"
                )  # TODO: add source from landmarks
            )
        except (AssertionError, ValueError):
            logger.debug("Skipping degenerate landmark frame.")
            return state, None

        if action is None:
            return state, None

        # Hand-only: no IK configured or no wrist pose on this frame.
        if self._ik is None or landmarks.wrist_pose is None:
            return state, TeleopAction(
                joint_positions=action,
                handedness=landmarks.handedness,
                wrist_pose=landmarks.wrist_pose,
            )

        # IK path.
        return self.step_ik(state, action, landmarks)

    def step_ik(
        self, state: AdapterState, action: OrcaJointPositions, landmarks: HandLandmarks
    ) -> tuple[AdapterState, TeleopAction | None]:
        import pinocchio as pin

        T_target = pin.SE3(
            np.asarray(landmarks.wrist_pose.rotation, dtype=np.float64),
            np.asarray(landmarks.wrist_pose.position, dtype=np.float64),
        )
        new_last_wrist = dict(state.last_wrist)
        new_last_wrist[landmarks.handedness] = T_target

        result = self._ik.solve(new_last_wrist, state.q_seed)
        idx_q = self._ik.arm_idx_q[landmarks.handedness]
        arm_angles = np.array([result.q[i] for i in idx_q])

        new_state = AdapterState(q_seed=result.q, last_wrist=new_last_wrist)
        return new_state, TeleopAction(
            joint_positions=action,
            handedness=landmarks.handedness,
            arm_angles=arm_angles,
            wrist_pose=landmarks.wrist_pose,
        )

    def process(self, landmarks: HandLandmarks) -> TeleopAction | None:
        """Stateful wrapper: thread state across calls and return the action."""
        new_state, action = self.step(self._state, landmarks)
        self._state = new_state
        return action


def adapter_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    model_path: str | None = None,
    urdf_path: str | None = None,
    hand_type_override: str | None = None,
    ik_solver: "BimanualIKSolver | None" = None,
) -> None:
    """Consume ``HandLandmarks`` from the gRPC ingress, adapt, push to actions_q.

    Builds an :class:`Adapter` (retargeter + optional bimanual IK) from the
    given paths and per-frame produces a :class:`TeleopAction`. During the
    retargeter's calibration window the adapter may return ``None``, in
    which case no robot command is enqueued yet.
    """
    _LOG_EVERY = 30
    _t_step_ms: list[float] = []
    _t_window_start: float = time.perf_counter()

    try:
        try:
            adapter = Adapter(
                model_path=model_path,
                urdf_path=urdf_path,
                ik_solver=ik_solver,
                hand_type_override=hand_type_override,
            )
        except Exception:
            logger.exception("Adapter init failed; shutting down worker.")
            return

        while not stop_event.is_set():
            try:
                item = queues.landmarks_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if item is _SHUTDOWN:
                break

            if not isinstance(item, HandLandmarks):
                raise ValueError(f"Expected instance of HandLandmarks, got {type(item)}")

            t_start = time.perf_counter()
            teleop_action = adapter.process(item)
            t_end = time.perf_counter()

            _t_step_ms.append((t_end - t_start) * 1e3)
            if teleop_action is None:
                continue

            try:
                queues.actions_q.put_nowait(teleop_action)
            except queue.Full:
                pass

            if len(_t_step_ms) >= _LOG_EVERY:
                num_samples = len(_t_step_ms)
                elapsed_s = time.perf_counter() - _t_window_start
                avg_step_ms = sum(_t_step_ms) / num_samples
                fps = num_samples / elapsed_s

                logger.info(
                    "Adapter | %.1f fps | step %.2f ms",
                    fps,
                    avg_step_ms,
                )
                _t_step_ms.clear()
                _t_window_start = time.perf_counter()
    finally:
        _shutdown_queue(queues.actions_q)


retargeter_worker = adapter_worker


def robot_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    ready_event: threading.Event,
    model_path: str,
) -> None:
    """Consume OrcaJointPositions and stream them to the OrcaHand."""
    hand: OrcaHand | None = None
    try:
        hand = OrcaHand(model_path)
        success, message = hand.connect()
        if not success:
            logger.error("Robot worker: failed to connect: %s", message)
            return
        hand.init_joints()
        ready_event.set()

        while not stop_event.is_set():
            try:
                action = queues.actions_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if action is _SHUTDOWN:
                break
            assert isinstance(action, TeleopAction)
            hand.set_joint_positions(action.joint_positions, num_steps=MOTION_NUM_STEPS)
    except Exception as e:
        logger.exception("Robot worker error: %s", e)
    finally:
        if hand is not None:
            try:
                hand.disable_torque()
                hand.disconnect()
            except Exception:
                pass


def run(
    model_path: str | None = None,
    urdf_path: str | None = None,
    port: int = DEFAULT_PORT,
    sink: RobotSink | None = None,
    hand_type_override: str | None = None,
    ik_solver: "BimanualIKSolver | None" = None,
) -> None:
    """Start the full teleop pipeline:
    - gRPC-ingress -> adapter -> robot consumer

    The robot-side machine runs this function. A publisher running on *any*
    machine (same host, a laptop across the room, etc.) connects via gRPC and
    streams hand-landmark frames. The main thread is handed to the sink's ``run_loop``.

    Args:
        model_path: Path to the OrcaHand model directory. ``None`` uses the
            default model bundled with ``orca_core``.
        urdf_path: Path to the hand URDF file. ``None`` resolves automatically
            from the ``orcahand_description`` package. Used for retargeting.
        port: TCP port for the gRPC ingress server, which streams HandLandmarks to the adapter.
        sink: Consumer of adapted actions. Defaults to ``OrcaHandSink(model_path)``,
            i.e. a physical OrcaHand.
        ik_solver: Optional bimanual IK solver. When provided, the adapter solves
            wrist IK for any landmark frame carrying a ``wrist_pose`` and emits
            ``TeleopAction.arm_angles`` alongside the retargeted hand joints.
    """
    if sink is None:
        sink = OrcaHandSink(model_path)

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()

    sink.connect()

    ingress_server = IngressServer(queues.landmarks_q, stop_event, port=port)
    ingress_server.start()

    adapter_thread = threading.Thread(
        target=adapter_worker,
        args=(queues, stop_event, model_path, urdf_path, hand_type_override, ik_solver),
        name="adapter",
    )
    adapter_thread.start()

    try:
        sink.run_loop(queues.actions_q, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ingress_server.stop()
        adapter_thread.join(timeout=JOIN_TIMEOUT)
        sink.close()


def _mediapipe_publisher(
    port: int,
    handedness: str,
    confidence: float,
    show_video: bool,
) -> None:
    """Entry point for the MediaPipe publisher."""
    from orca_teleop.ingress.mediapipe.publisher import MediaPipePublisher

    server_address = f"localhost:{port}"
    deadline = time.monotonic() + 10  # TODO: add to constants

    # Wait until the ingress server is actually accepting connections
    while True:
        try:
            with socket.create_connection(tuple(server_address.split(":")), timeout=0.5):
                break
        except OSError as err:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ingress server on {server_address} did not become ready"
                ) from err
            time.sleep(0.1)

    publisher = MediaPipePublisher(
        server_address=server_address,
        handedness=handedness,
        confidence=confidence,
        show_video=show_video,
    )
    publisher.run()


def run_local(
    model_path: str | None = None,
    urdf_path: str | None = None,
    port: int = DEFAULT_PORT,
    handedness: str = DEFAULT_HAND,
    confidence: float = DEFAULT_CONFIDENCE,
    show_video: bool = False,
    sink: RobotSink | None = None,
) -> None:
    """Run ``run()`` plus a local MediaPipe publisher for one-command teleop.
    Useful for prototyping.
    """
    import multiprocessing

    model_path = _resolve_model_config_for_hand(model_path, handedness)

    # Start the publisher in a child process so the webcam doesn't fight with main thread
    publisher_process = multiprocessing.Process(
        target=_mediapipe_publisher,
        args=(port, handedness, confidence, show_video),
        name="mediapipe-publisher",
        daemon=True,
    )

    publisher_process.start()
    logger.info(
        "Local MediaPipe publisher started (pid=%d, hand=%s)",
        publisher_process.pid,
        handedness,
    )

    try:
        run(
            model_path=model_path,
            urdf_path=urdf_path,
            port=port,
            sink=sink,
        )
    finally:
        if publisher_process.is_alive():
            publisher_process.terminate()
        publisher_process.join(timeout=3.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Orca Hand teleoperation pipeline. "
        "Launches a MediaPipe webcam publisher and the full retargeting pipeline.",
    )
    parser.add_argument("--model_path", default=None, help="OrcaHand model directory")
    parser.add_argument("--urdf_path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--hand", default="right", choices=["left", "right"], help="Hand to track (default: right)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_local(
        args.model_path,
        args.urdf_path,
        port=args.port,
        handedness=args.hand,
        confidence=args.confidence,
        show_video=args.show_video,
    )
