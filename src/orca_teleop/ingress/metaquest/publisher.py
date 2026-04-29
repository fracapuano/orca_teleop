"""MetaQuest hand-pose gRPC publisher (replay from HF).

Pulls a recorded session from the Hugging Face dataset ``fracapuano/quest-poses``
and streams it to the ``IngressServer`` over gRPC, mimicking what a live Meta
Quest would emit. No conversions are applied — the wrist poses + 21 hand
landmarks per side go out raw (Unity left-handed). Downstream is responsible
for any handedness fix or frame alignment.

Two ``HandFrame`` messages are sent per tick: one for ``left`` and one for
``right``, each only when that side was visible in the recording.

Usage::

    # Stream to a local IngressServer
    python -m orca_teleop.ingress.metaquest.publisher

    # Remote ingress
    python -m orca_teleop.ingress.metaquest.publisher --server 192.168.1.42:50051

    # Force re-download from HF (ignore local cache)
    python -m orca_teleop.ingress.metaquest.publisher --refresh
"""

# TODO: Swap on actual meta quest stuff and make tracking go brrr

import argparse
import logging
import time

import grpc
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc

logger = logging.getLogger(__name__)

DEFAULT_REPO = "fracapuano/quest-poses"
DEFAULT_FILENAME = "data.parquet"
DEFAULT_SERVER = "localhost:50051"
DEFAULT_FPS = 30


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3x3 rotation matrix.

    No external deps, no checks beyond a degenerate-norm guard.
    """
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


class MetaQuestPublisher:
    """Replays a recorded HF dataset over the gRPC HandStream service."""

    def __init__(
        self,
        server_address: str = DEFAULT_SERVER,
        repo: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        fps: int = DEFAULT_FPS,
        loop: bool = True,
        refresh: bool = False,
    ) -> None:
        self._server_address = server_address
        self._repo = repo
        self._filename = filename
        self._fps = int(fps)
        self._loop = loop
        self._refresh = refresh
        self._period = 1.0 / self._fps

    def _load_columns(self) -> dict:
        path = hf_hub_download(
            repo_id=self._repo,
            filename=self._filename,
            repo_type="dataset",
            force_download=self._refresh,
        )
        logger.info("Loaded %s/%s → %s", self._repo, self._filename, path)
        table = pq.read_table(path)
        return {name: table.column(name).to_pylist() for name in table.column_names}

    def _frames_for_row(self, cols: dict, i: int) -> list:
        """Build up to two HandFrame protos (one per visible side) for row *i*."""
        out = []
        for side in ("left", "right"):
            if not cols[f"{side}_visible"][i]:
                continue
            R = _quat_to_rotmat(
                cols[f"{side}_wrist_qx"][i],
                cols[f"{side}_wrist_qy"][i],
                cols[f"{side}_wrist_qz"][i],
                cols[f"{side}_wrist_qw"][i],
            )
            out.append(
                hand_stream_pb2.HandFrame(
                    keypoints=cols[f"{side}_landmarks"][i],
                    handedness=side,
                    timestamp_ns=time.time_ns(),
                    wrist_pose=hand_stream_pb2.WristPose(
                        position=[
                            cols[f"{side}_wrist_x"][i],
                            cols[f"{side}_wrist_y"][i],
                            cols[f"{side}_wrist_z"][i],
                        ],
                        rotation=R.flatten().tolist(),
                    ),
                )
            )
        return out

    def _frame_generator(self):
        cols = self._load_columns()
        n = len(cols["t_ns"])
        if n == 0:
            logger.error("Dataset has 0 rows.")
            return

        logger.info(
            "Streaming %d rows at %d fps (loop=%s). Ctrl+C to stop.",
            n,
            self._fps,
            self._loop,
        )

        next_tick = time.monotonic()
        loops = 0
        emitted = 0
        log_every = self._fps * 5  # log roughly every 5 seconds

        while True:
            for i in range(n):
                next_tick += self._period
                for frame in self._frames_for_row(cols, i):
                    yield frame
                    emitted += 1
                    if emitted % log_every == 0:
                        logger.info("emitted=%d  loops=%d  row=%d/%d", emitted, loops, i + 1, n)
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
            loops += 1
            if not self._loop:
                logger.info("End of dataset (emitted=%d). Stopping.", emitted)
                return
            logger.info("Looped dataset (loops=%d).", loops)

    def run(self) -> None:
        logger.info("Connecting to %s", self._server_address)
        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        try:
            summary = stub.StreamHandFrames(self._frame_generator())
            logger.info("Stream closed: server received %d frames", summary.frames_received)
        except KeyboardInterrupt:
            logger.info("Interrupted; closing stream.")
        except grpc.RpcError as e:
            logger.error("gRPC error: %s", e)
        finally:
            channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ingress address (host:port)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="parquet filename in repo")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="replay rate (Hz)")
    parser.add_argument(
        "--no-loop", action="store_true", help="stop at end of file instead of looping"
    )
    parser.add_argument("--refresh", action="store_true", help="force re-download from HF")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    publisher = MetaQuestPublisher(
        server_address=args.server,
        repo=args.repo,
        filename=args.filename,
        fps=args.fps,
        loop=not args.no_loop,
        refresh=args.refresh,
    )
    publisher.run()


if __name__ == "__main__":
    main()
