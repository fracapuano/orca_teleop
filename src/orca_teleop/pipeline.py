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
    model_path: str,
) -> None:
    """Consume landmarks, produce random OrcaJointPositions, push to actions_q.

    Uses the real joint names and ROM limits from OrcaHand so that the robot
    worker receives well-formed commands.
    """
    hand = OrcaHand(model_path)
    joint_ids = hand.config.joint_ids
    lower, upper = map(np.array, zip(*hand.config.joint_roms_dict.values(), strict=False))
    rng = np.random.default_rng()

    try:
        while not stop_event.is_set():
            try:
                item = queues.landmarks_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue
            if item is _SHUTDOWN:
                break
            angles = rng.uniform(lower, upper)
            action = OrcaJointPositions(dict(zip(joint_ids, angles, strict=False)))
            try:
                queues.actions_q.put_nowait(action)
            except queue.Full:
                pass
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


def run(model_path: str) -> None:
    """Start the full ingress -> retargeter -> robot pipeline and block.

    The main thread owns the robot: it connects, initializes, and consumes
    ``OrcaJointPositions`` from ``actions_q`` directly, so that
    ``KeyboardInterrupt`` triggers an immediate, synchronous cleanup.

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
        args=(queues, stop_event, model_path),
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

        hand.disable_torque()
        hand.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Threaded teleop pipeline")
    parser.add_argument("--model_path", default=None, help="Path to OrcaHand model directory")
    args = parser.parse_args()

    run(args.model_path)
