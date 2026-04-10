"""Teleop pipeline: ingress -> retargeter -> robot.

Three worker threads communicate exclusively through queues:

    [ingress] --landmarks_q--> [retargeter] --actions_q--> [robot]

Ingress captures landmarks and fills the landmarks_q with ``HandLandmark``
to be retargeted into joint commands.

Retargeter consumes targets from the ingress and fills the actions_q with
``OrcaJointPositions``.

The robot worker consumes ``OrcaJointPositions`` off the queue and
streams them to an ``OrcaHand``.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import (
    HEARTBEAT_INTERVAL,
    INGRESS_FPS,
    JOIN_TIMEOUT,
    MOTION_NUM_STEPS,
    QUEUES_MAXSIZE,
)
from orca_teleop.ingress import get_canonical_key_vectors
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
    landmarks_q: "queue.Queue[Any]"
    actions_q: "queue.Queue[OrcaJointPositions | object]"


def ingress_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
) -> None:
    """Produce random landmarks and push them onto landmarks_q.

    Emits np.ndarray of shape (21, 3) at ~30 Hz, matching the MediaPipe
    world-landmark format consumed by the retargeter.
    """
    rng = np.random.default_rng()
    try:
        while not stop_event.is_set():
            landmarks = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
            try:
                queues.landmarks_q.put_nowait(landmarks)
            except queue.Full:
                pass
            time.sleep(1.0 / INGRESS_FPS)
    finally:
        _shutdown_queue(queues.landmarks_q)


def retargeter_worker(
    queues: TeleopQueues,
    stop_event: threading.Event,
    model_path: str | None = None,
    urdf_path: str | None = None,
) -> None:
    """Consume landmarks, retarget them to OrcaJointPositions, push to actions_q.

    Builds a Retargeter from model_path and urdf_path, then for each incoming
    (21, 3) MediaPipe landmark array: computes canonical key vectors, wraps
    them in a TargetPose, and calls retargeter.retarget() to produce joint commands.

    Both model_path and urdf_path default to None: model_path=None uses the
    default OrcaHand model; urdf_path=None resolves from orcahand_description.
    """
    retargeter = Retargeter.from_paths(model_path, urdf_path)

    _LOG_EVERY = 30  # emit a timing summary every N processed frames
    _t_ingress_ms: list[float] = []
    _t_retarget_ms: list[float] = []
    _t_frame_start: float = time.perf_counter()

    try:
        while not stop_event.is_set():
            try:
                item = queues.landmarks_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if item is _SHUTDOWN:
                break

            t0 = time.perf_counter()
            try:
                key_vectors = get_canonical_key_vectors(item, "mediapipe")
            except AssertionError:
                # Degenerate frame: landmarks are geometrically implausible
                # (e.g. collinear finger bases). Skip and wait for the next frame.
                logger.debug("Skipping degenerate landmark frame.")
                continue
            t1 = time.perf_counter()

            target_pose = TargetPose(key_vectors=key_vectors)
            action = retargeter.retarget(target_pose)
            t2 = time.perf_counter()

            _t_ingress_ms.append((t1 - t0) * 1e3)
            _t_retarget_ms.append((t2 - t1) * 1e3)

            try:
                queues.actions_q.put_nowait(action)
            except queue.Full:
                pass

            if len(_t_retarget_ms) >= _LOG_EVERY:
                elapsed = time.perf_counter() - _t_frame_start
                fps = _LOG_EVERY / elapsed
                logger.info(
                    "Retargeter | %.1f fps | ingress %.2f ms | retarget %.2f ms | total %.2f ms",
                    fps,
                    sum(_t_ingress_ms) / len(_t_ingress_ms),
                    sum(_t_retarget_ms) / len(_t_retarget_ms),
                    sum(t_i + t_r for t_i, t_r in zip(_t_ingress_ms, _t_retarget_ms, strict=True))
                    / len(_t_retarget_ms),
                )
                _t_ingress_ms.clear()
                _t_retarget_ms.clear()
                _t_frame_start = time.perf_counter()
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


def run(model_path: str | None = None, urdf_path: str | None = None) -> None:
    """Start the full ingress -> retargeter -> robot pipeline and block.

    The main thread owns the robot: it connects, initializes, and consumes
    ``OrcaJointPositions`` from ``actions_q`` directly, so that
    ``KeyboardInterrupt`` triggers an immediate, synchronous cleanup.

    Args:
        model_path: Path to the OrcaHand model directory. ``None`` uses the
            default model bundled with ``orca_core``.
        urdf_path: Path to the hand URDF file. ``None`` resolves automatically
            from the ``orcahand_description`` package.

    Raises:
        RuntimeError: if the robot fails to connect.
    """
    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()

    hand = OrcaHand(model_path)
    success, message = hand.connect()
    if not success:
        raise RuntimeError(f"Robot failed to connect: {message}")
    hand.init_joints()

    retargeter_thread = threading.Thread(
        target=retargeter_worker,
        args=(queues, stop_event, model_path, urdf_path),
        name="retargeter",
    )
    ingress_thread = threading.Thread(
        target=ingress_worker,
        args=(queues, stop_event),
        name="ingress",
    )
    workers = [retargeter_thread, ingress_thread]
    for w in workers:
        w.start()

    try:
        while not stop_event.is_set():
            try:
                action = queues.actions_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if action is _SHUTDOWN:
                break
            assert isinstance(action, OrcaJointPositions)
            hand.set_joint_positions(action, num_steps=MOTION_NUM_STEPS)

    except KeyboardInterrupt:
        pass

    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=JOIN_TIMEOUT)

        hand.set_zero_position()
        hand.disable_torque()
        hand.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Threaded teleop pipeline")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to OrcaHand model directory (default: bundled orca_core model)",
    )
    parser.add_argument(
        "--urdf_path",
        default=None,
        help="Path to hand URDF file (default: resolved from orcahand_description)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.model_path, args.urdf_path)
