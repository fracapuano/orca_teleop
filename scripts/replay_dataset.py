"""Replay a LeRobotDataset on the OrcaHand.

This script streams recorded ``action`` rows back to the hand open-loop:

.. code-block:: bash

    python scripts/replay_dataset.py --repo-id HF_USER/REPO_ID
    mjpython scripts/replay_dataset.py --backend sim --repo-id HF_USER/REPO_ID
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.pipeline import OrcaHandSink, RecordableSink

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a LeRobotDataset on the OrcaHand.",
    )
    parser.add_argument("--repo-id", required=True, help="LeRobotDataset repo id")
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Replay only this episode index. Default: replay all episodes.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local lerobot cache root (defaults to lerobot's HF cache).",
    )
    parser.add_argument("--model-path", default=None, help="OrcaHand model directory")
    parser.add_argument(
        "--backend",
        choices=["hardware", "sim"],
        default="hardware",
        help="Backend to replay against (default: hardware).",
    )
    parser.add_argument(
        "--sim-env",
        choices=["left", "right"],
        default="right",
        help="orca_sim hand environment to use with --backend sim (default: right).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the dataset's fps for playback (default: use dataset.fps).",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=1.0,
        help="Pause between episodes when replaying multiple (default: 1.0)",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Pop a cv2 window per recorded camera while replaying.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes_filter = [args.episode] if args.episode is not None else None
    logger.info("Loading dataset %s (episodes=%s)...", args.repo_id, episodes_filter or "all")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=episodes_filter,
        download_videos=args.show_cameras,
    )
    logger.info("Loaded %d frames at %d fps from %s", len(dataset), dataset.fps, dataset.root)

    fps = float(args.fps) if args.fps is not None else float(dataset.fps)
    period = 1.0 / fps

    joint_names = list(dataset.features["action"]["names"])

    image_keys: list[str] = []
    cv2 = None
    if args.show_cameras:
        import cv2 as _cv2

        cv2 = _cv2
        image_keys = [k for k in dataset.features if k.startswith("observation.images.")]
        if not image_keys:
            logger.warning("--show-cameras passed but no observation.images.* features in dataset.")

    if args.show_cameras:
        rows = (dataset[i] for i in range(len(dataset)))
        n_frames = len(dataset)
    else:
        rows = iter(dataset.hf_dataset)
        n_frames = len(dataset.hf_dataset)

    sink: RecordableSink
    if args.backend == "sim":
        from orca_teleop.sim import OrcaHandSimSink

        sink = OrcaHandSimSink(
            env_name=args.sim_env,
        )
    else:
        sink = OrcaHandSink(model_path=args.model_path)
    sink.connect()

    logger.info("Replaying %d frame(s) - Ctrl+C to abort.", n_frames)

    try:
        prev_episode: int | None = None
        next_t = time.perf_counter()

        for row in rows:
            episode_index = int(np.asarray(row["episode_index"]).item())

            if prev_episode is not None and episode_index != prev_episode:
                logger.info(
                    "Episode %d done - resting %.1fs before episode %d.",
                    prev_episode,
                    args.rest_seconds,
                    episode_index,
                )
                if args.rest_seconds > 0:
                    time.sleep(args.rest_seconds)
                next_t = time.perf_counter()
            if prev_episode != episode_index:
                logger.info("=== Episode %d ===", episode_index)
            prev_episode = episode_index

            action_arr = np.asarray(row["action"], dtype=float)
            action = OrcaJointPositions.from_dict(
                dict(zip(joint_names, action_arr.tolist(), strict=True))
            )
            sink.dispatch_action(action)

            if cv2 is not None:
                for key in image_keys:
                    img = row.get(key)
                    if img is None:
                        continue
                    arr = img.permute(1, 2, 0).contiguous().numpy() * 255.0
                    arr = arr.clip(0, 255).astype(np.uint8)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"replay: {key.rsplit('.', 1)[-1]}", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("'q' pressed - aborting replay.")
                    break

            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()

    except KeyboardInterrupt:
        logger.info("Interrupted.")

    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        sink.close()


if __name__ == "__main__":
    main(sys.argv[1:])
