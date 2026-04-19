"""Record teleoperated episodes into a LeRobotDataset.

Wires the standard ingress + retargeter pipeline (without modifying it) and
runs a parallel recorder thread that writes ``(state, action, images)`` frames
to a LeRobotDataset on disk. Multi-episode capture, optional camera streams,
and optional ``--push-to-hub`` are supported.

Pipeline (per episode):

  - main thread: pull each ``OrcaJointPositions`` from ``actions_q``, read
    ``hand.get_joint_position()``, capture one frame from each opened camera,
    pack everything into a frame dict, push it onto ``rec_q``, then dispatch
    the action to the hand.
  - recorder thread: blocks on ``rec_q``, calls ``dataset.add_frame(...)`` for
    every dict, and ``dataset.save_episode()`` when it sees the SAVE sentinel
  - between episodes: a small rest pause; at the end: optional push to Hub

Example usage:

.. code-block:: bash

    HF_USERNAME=your_username python scripts/record_dataset.py --repo-id $HF_USERNAME/orca-teleop \
        --task "pick up the block" \\  # text description of the task
        --num-episodes 5 --episode-seconds 20 \
        --camera front:0 --camera wrist:1 \\  # camera specs, given as camera_name:camera_index
        --push-to-hub
"""

import argparse
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.constants import (
    HEARTBEAT_INTERVAL,
    JOIN_TIMEOUT,
    MOTION_NUM_STEPS,
    QUEUES_MAXSIZE,
)
from orca_teleop.ingress.server import DEFAULT_PORT, IngressServer
from orca_teleop.pipeline import _SHUTDOWN, TeleopQueues, retargeter_worker

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 30
_DEFAULT_NUM_EPISODES = 1
_DEFAULT_EPISODE_SECONDS = 30.0
_DEFAULT_REST_SECONDS = 3.0

# Sentinel pushed onto rec_q between episodes — the recorder calls
# dataset.save_episode() in response, then keeps draining the next episode.
_SAVE_EPISODE = object()


def _default_lerobot_root(repo_id: str) -> Path:
    """Mirror lerobot's default cache location."""
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base) / repo_id
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _open_cameras(specs: list[str]) -> tuple[dict, dict]:
    """Open every camera in *specs* (``"name[:index]"``) and probe its shape.

    Returns ``(captures, shapes)`` where ``captures`` is ``{name: cv2.VideoCapture}``
    and ``shapes`` is ``{name: (H, W, C)}``.
    """
    if not specs:
        return {}, {}
    import cv2  # lazy

    captures: dict = {}
    shapes: dict = {}
    for spec in specs:
        name, _, idx_str = spec.partition(":")
        if not name:
            raise ValueError(f"Camera spec {spec!r} is missing a name (use NAME[:INDEX]).")
        index = int(idx_str) if idx_str else 0
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {name!r} (index {index}).")
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {name!r} returned no frame on probe read.")
        captures[name] = cap
        shapes[name] = tuple(int(x) for x in frame.shape)
    return captures, shapes


def _capture_frames(captures: dict) -> dict:
    if not captures:
        return {}
    import cv2  # lazy

    out: dict = {}
    for name, cap in captures.items():
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Camera {name!r} read failed mid-episode.")
        out[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return out


def _release_cameras(captures: dict) -> None:
    for cap in captures.values():
        try:
            cap.release()
        except Exception:
            pass


def _stub_action_publisher(
    actions_q: "queue.Queue",
    stop_event: threading.Event,
    joint_ids: list[str],
    fps: float,
) -> None:
    """Push small random-walk ``OrcaJointPositions`` onto actions_q (dev only).

    These actions are *recorded* by the main loop but, in stub mode, are
    deliberately not dispatched to the device — see ``main()``.
    """
    rng = np.random.default_rng()
    n = len(joint_ids)
    pose = np.zeros(n, dtype=float)
    period = 1.0 / float(fps)
    while not stop_event.is_set():
        pose += rng.uniform(-0.2, 0.2, size=n)
        pose = np.clip(pose, -3.0, 3.0)
        action = OrcaJointPositions.from_dict(dict(zip(joint_ids, pose, strict=True)))
        try:
            actions_q.put_nowait(action)
        except queue.Full:
            try:
                actions_q.get_nowait()
            except queue.Empty:
                pass
            try:
                actions_q.put_nowait(action)
            except queue.Full:
                pass
        time.sleep(period)


def _recorder_loop(dataset, rec_q: "queue.Queue") -> None:
    """Drain rec_q. Frames are dicts; ``_SAVE_EPISODE`` flushes; ``None`` exits."""
    while True:
        item = rec_q.get()
        if item is None:
            break
        if item is _SAVE_EPISODE:
            try:
                dataset.save_episode()
                logger.info("Episode saved.")
            except Exception:
                logger.exception("Failed to save episode")
            continue
        try:
            dataset.add_frame(item)
        except Exception:
            logger.exception("Failed to add frame")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobotDataset repo id")
    parser.add_argument("--task", required=True, help="Task description for every episode")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=_DEFAULT_NUM_EPISODES,
        help=f"Number of episodes to record (default: {_DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=_DEFAULT_EPISODE_SECONDS,
        help=f"Length of each episode in seconds (default: {_DEFAULT_EPISODE_SECONDS})",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=_DEFAULT_REST_SECONDS,
        help=f"Pause between episodes (default: {_DEFAULT_REST_SECONDS})",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Camera spec NAME[:INDEX], repeatable. e.g. --camera front:0 --camera wrist:1",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After recording, push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe any existing dataset at the same root before creating.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root (defaults to lerobot's HF cache).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=_DEFAULT_FPS,
        help=f"Dataset fps metadata (default: {_DEFAULT_FPS})",
    )
    parser.add_argument("--model-path", default=None, help="OrcaHand model directory")
    parser.add_argument("--urdf-path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"gRPC ingress port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Dev only: skip ingress + retargeter, push random actions, do NOT dispatch to hand.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # TODO: Use constants key names from lerobot for frame key names

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
        actions_q=queue.Queue(maxsize=QUEUES_MAXSIZE),
    )
    stop_event = threading.Event()

    hand = OrcaHand(args.model_path)
    success, message = hand.connect()
    if not success:
        raise RuntimeError(f"Robot failed to connect: {message}")
    hand.init_joints()

    joint_ids = list(hand.config.joint_ids)
    n_joints = len(joint_ids)

    captures, camera_shapes = _open_cameras(args.camera)
    if captures:
        logger.info("Opened cameras: %s", camera_shapes)

    features = {
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": joint_ids},
        "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_ids},
    }
    for cam_name, shape in camera_shapes.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    target_root = args.root if args.root is not None else _default_lerobot_root(args.repo_id)
    if args.overwrite and target_root.exists():
        logger.info("--overwrite: removing existing dataset at %s", target_root)
        shutil.rmtree(target_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        use_videos=True,
    )
    logger.info("Dataset root: %s", dataset.root)

    rec_q: queue.Queue = queue.Queue(maxsize=64)

    ingress_server: IngressServer | None = None
    retargeter_thread: threading.Thread | None = None
    stub_thread: threading.Thread | None = None

    if args.stub:
        logger.info(
            "STUB mode — ingress + retargeter bypassed; actions are NOT dispatched to the hand."
        )
        stub_thread = threading.Thread(
            target=_stub_action_publisher,
            args=(queues.actions_q, stop_event, joint_ids, args.fps),
            name="stub-action-publisher",
        )
        stub_thread.start()
    else:
        ingress_server = IngressServer(queues.landmarks_q, stop_event, port=args.port)
        ingress_server.start()
        retargeter_thread = threading.Thread(
            target=retargeter_worker,
            args=(queues, stop_event, args.model_path, args.urdf_path),
            name="retargeter",
        )
        retargeter_thread.start()

    recorder_thread = threading.Thread(
        target=_recorder_loop, args=(dataset, rec_q), name="dataset-recorder"
    )
    recorder_thread.start()

    logger.info(
        "Recording %d episode(s) of ~%.1fs each — Ctrl+C to abort.",
        args.num_episodes,
        args.episode_seconds,
    )

    try:
        for ep_idx in range(args.num_episodes):
            if stop_event.is_set():
                break
            logger.info("=== Episode %d / %d ===", ep_idx + 1, args.num_episodes)
            ep_end = time.perf_counter() + args.episode_seconds
            n_frames = 0

            while time.perf_counter() < ep_end and not stop_event.is_set():
                try:
                    action = queues.actions_q.get(timeout=HEARTBEAT_INTERVAL)
                except queue.Empty:
                    continue
                if action is _SHUTDOWN:
                    stop_event.set()
                    break
                assert isinstance(action, OrcaJointPositions)

                state_arr = hand.get_joint_position().as_array(joint_ids).astype(np.float32)
                action_arr = action.as_array(joint_ids).astype(np.float32)

                try:
                    cam_images = _capture_frames(captures)
                except Exception:
                    logger.exception("Camera capture failed; aborting episode.")
                    stop_event.set()
                    break

                frame = {
                    "observation.state": state_arr,
                    "action": action_arr,
                    "task": args.task,
                }
                for cam_name, img in cam_images.items():
                    frame[f"observation.images.{cam_name}"] = img

                try:
                    rec_q.put_nowait(frame)
                    n_frames += 1
                except queue.Full:
                    logger.debug("rec_q full; dropping frame")

                if not args.stub:
                    hand.set_joint_positions(action, num_steps=MOTION_NUM_STEPS)

            logger.info("Episode %d captured %d frames.", ep_idx + 1, n_frames)
            rec_q.put(_SAVE_EPISODE)

            if ep_idx + 1 < args.num_episodes and not stop_event.is_set() and args.rest_seconds > 0:
                logger.info("Resting %.1fs before next episode...", args.rest_seconds)
                stop_event.wait(args.rest_seconds)

    except KeyboardInterrupt:
        logger.info("Interrupted — finalizing dataset.")

    finally:
        stop_event.set()
        rec_q.put(None)  # drain sentinel — recorder finishes pending items first

        if ingress_server is not None:
            ingress_server.stop()
        if retargeter_thread is not None:
            retargeter_thread.join(timeout=JOIN_TIMEOUT)
        if stub_thread is not None:
            stub_thread.join(timeout=JOIN_TIMEOUT)

        recorder_thread.join()

        _release_cameras(captures)

        try:
            num_eps = dataset.num_episodes
        except Exception:
            num_eps = "?"
        logger.info("Dataset now contains %s episode(s) at %s", num_eps, dataset.root)

        if args.push_to_hub:
            try:
                logger.info("Pushing %s to the Hugging Face Hub...", args.repo_id)
                dataset.push_to_hub()
                logger.info("Push complete.")
            except Exception:
                logger.exception("Failed to push to hub")

        try:
            hand.set_zero_position()
            hand.disable_torque()
        finally:
            hand.disconnect()


if __name__ == "__main__":
    main()
