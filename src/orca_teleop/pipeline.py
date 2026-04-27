"""Teleop pipeline: ingress (gRPC) -> retargeter -> robot control.

Architecture:

    Typical deployment spans two machines, with gRPC as the only network boundary:

        TELEOP MACHINE                                      ROBOT MACHINE
    +---------------------------+              +--------------------------------------+
    |         publisher         |    gRPC      |            teleop pipeline           |
    |  (MediaPipe, Manus, etc.) +------------->|  +---------------+                   |
    |                           |  HandFrame   |  | IngressServer |                   |
    +---------------------------+              |  +-------+-------+                   |
                                               |          | landmarks_q               |
                                               |          v                           |
                                               |  +---------------+                   |
                                               |  |  retargeter   |                   |
                                               |  +-------+-------+                   |
                                               |          | actions_q                 |
                                               |          v                           |
                                               |  +---------------+                   |
                                               |  |     robot     |                   |
                                               |  +---------------+                   |
                                               +--------------------------------------+

The ingress is a gRPC server streaming ``HandFrame`` from a generic publisher (MediaPipe webcam,
Manus glove, VisionPro, replay file, ...).
Publishers are standalone scripts that know nothing about the robot, they just stream
``(21, 3)`` hand landmarks over the network.

The retargeter and robot stages run as threads on the robot-side machine.
"""

import logging
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

logger = logging.getLogger(__name__)

_SHUTDOWN = object()


def _shutdown_queue(q: "queue.Queue[Any]") -> None:
    """Signal a downstream worker to stop, without blocking."""
    try:
        q.put_nowait(_SHUTDOWN)
    except queue.Full:
        pass


@dataclass
class TeleopQueues:
    landmarks_q: "queue.Queue[HandLandmarks]"
    actions_q: "queue.Queue[OrcaJointPositions | object]"


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
        actions_q: "queue.Queue[OrcaJointPositions | object]",
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
            assert isinstance(action, OrcaJointPositions)
            self.dispatch_action(action)

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


def retargeter_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    model_path: str | None = None,
    urdf_path: str | None = None,
) -> None:
    """Consume ``HandLandmarks`` from the gRPC ingress, retarget, push to actions_q.

    Builds a Retargeter from model_path and urdf_path, then for each incoming
    ``HandLandmarks`` (21, 3) wraps the raw keypoints in a ``TargetPose`` and
    calls ``retargeter.retarget()`` — the retargeter handles MANO normalization,
    auto-scale calibration, and the URDF-frame transform internally. During the
    calibration window it may return ``None``, in which case no robot command is
    enqueued yet.
    """
    _LOG_EVERY = 30
    _t_retarget_ms: list[float] = []
    _t_window_start: float = time.perf_counter()

    try:
        try:
            retargeter = Retargeter.from_paths(model_path, urdf_path)
        except Exception:
            logger.exception("Retargeter init failed; shutting down worker.")
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

            t_retarget_start = time.perf_counter()
            try:
                target_pose = TargetPose(joint_positions=item.keypoints, source="mediapipe")
                action = retargeter.retarget(target_pose)
            except (AssertionError, ValueError):
                logger.debug("Skipping degenerate landmark frame.")
                continue
            t_retarget_end = time.perf_counter()

            _t_retarget_ms.append((t_retarget_end - t_retarget_start) * 1e3)
            if action is None:
                continue

            try:
                queues.actions_q.put_nowait(action)
            except queue.Full:
                pass

            if len(_t_retarget_ms) >= _LOG_EVERY:
                num_samples = len(_t_retarget_ms)
                elapsed_s = time.perf_counter() - _t_window_start
                avg_retarget_ms = sum(_t_retarget_ms) / num_samples
                fps = num_samples / elapsed_s

                logger.info(
                    "Retargeter | %.1f fps | retarget %.2f ms",
                    fps,
                    avg_retarget_ms,
                )
                _t_retarget_ms.clear()
                _t_window_start = time.perf_counter()
    finally:
        _shutdown_queue(queues.actions_q)


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
            assert isinstance(action, OrcaJointPositions)
            hand.set_joint_positions(action, num_steps=MOTION_NUM_STEPS)
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
) -> None:
    """Start the full teleop pipeline:
    - gRPC-ingress -> retargeter -> robot consumer

    The robot-side machine runs this function. A publisher running on *any*
    machine (same host, a laptop across the room, etc.) connects via gRPC and
    streams hand-landmark frames. The main thread is handed to the sink's ``run_loop``.

    Args:
        model_path: Path to the OrcaHand model directory. ``None`` uses the
            default model bundled with ``orca_core``.
        urdf_path: Path to the hand URDF file. ``None`` resolves automatically
            from the ``orcahand_description`` package. Used for retargeting.
        port: TCP port for the gRPC ingress server, which streams HandLandmarks to the retargeter.
        sink: Consumer of retargeted joint positions. Defaults to
            ``OrcaHandSink(model_path)``, i.e. a physical OrcaHand.
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

    retargeter_thread = threading.Thread(
        target=retargeter_worker,
        args=(queues, stop_event, model_path, urdf_path),
        name="retargeter",
    )
    retargeter_thread.start()

    try:
        sink.run_loop(queues.actions_q, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ingress_server.stop()
        retargeter_thread.join(timeout=JOIN_TIMEOUT)
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
    deadline = time.monotonic() + 10.0

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
        run(model_path=model_path, urdf_path=urdf_path, port=port, sink=sink)
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
