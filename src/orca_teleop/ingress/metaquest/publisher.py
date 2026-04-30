"""MetaQuest hand-pose gRPC publishers.

Two flavors:

- :class:`MetaQuestPublisher` (default): streams live wrist + 21 hand
  landmarks from a real Meta Quest using ``hand_tracking_sdk.HTSClient``.
  A producer thread reads ``iter_events()`` as fast as the device sends;
  a paced consumer reads the latest per-side frame and emits gRPC frames
  at a user-controlled rate (capped at ``--fps``, never above the device
  rate). The publisher decides the wire rate, not the SDK.
- :class:`DummyMetaQuestPublisher`: replays a recorded session from the HF
  dataset ``fracapuano/quest-poses``. Used by ``scripts/teleop_arm_quest.py
  --local`` to drive the pipeline without a physical Quest.

Both publishers send wrist poses raw in Unity left-handed coords (no basis
transform). Downstream is responsible for the LH→FLU conversion.

Two ``HandFrame`` messages are sent per tick: one for ``left`` and one for
``right``, each only when that side was visible.

Usage::

    # Live Quest (default)
    python -m orca_teleop.ingress.metaquest.publisher \\
        --quest-port 9000 --transport udp

    # Dataset replay
    python -m orca_teleop.ingress.metaquest.publisher --dummy

    # Remote ingress
    python -m orca_teleop.ingress.metaquest.publisher --server 192.168.1.42:50051
"""

import argparse
import logging
import threading
import time

import grpc
import numpy as np
import pyarrow.parquet as pq
from hand_tracking_sdk import (
    ErrorPolicy,
    HandFilter,
    HandFrame,
    HandSide,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)
from huggingface_hub import hf_hub_download

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc

logger = logging.getLogger(__name__)

DEFAULT_REPO = "fracapuano/quest-poses"
DEFAULT_FILENAME = "data.parquet"
DEFAULT_SERVER = "localhost:50051"
DEFAULT_FPS = 30
DEFAULT_QUEST_HOST = "0.0.0.0"
DEFAULT_QUEST_PORT = 9000


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
    """Live MetaQuest publisher: HTS device → paced gRPC stream.

    Two-stage pipeline so the device's native rate doesn't dictate the
    pipeline rate:

    1. **Producer thread** runs ``HTSClient.iter_events()``, filters to
       :class:`HandFrame`, and overwrites the latest frame for each hand
       side in a one-slot per-side cache. Older frames are silently
       dropped — the consumer always sees fresh data.
    2. **Consumer (gRPC stream)** ticks at ``fps``. Each tick it snapshots
       the cache and emits one proto per side whose ``sequence_id`` has
       advanced since the last emission. Net effect: output is rate-capped
       at ``fps`` and never re-emits identical frames (re-emission would
       look like artificial stillness to the downstream clutch detector).
    """

    def __init__(
        self,
        server_address: str = DEFAULT_SERVER,
        *,
        transport_mode: TransportMode = TransportMode.UDP,
        quest_host: str = DEFAULT_QUEST_HOST,
        quest_port: int = DEFAULT_QUEST_PORT,
        fps: int = DEFAULT_FPS,
    ) -> None:
        self._server_address = server_address
        self._fps = int(fps)
        self._period = 1.0 / self._fps
        self._transport_mode = transport_mode
        self._quest_host = quest_host
        self._quest_port = quest_port
        self._client = HTSClient(
            HTSClientConfig(
                transport_mode=transport_mode,
                host=quest_host,
                port=quest_port,
                output=StreamOutput.FRAMES,
                hand_filter=HandFilter.BOTH,
                error_policy=ErrorPolicy.TOLERANT,
                include_wall_time=True,
            )
        )
        self._latest: dict[HandSide, HandFrame | None] = {
            HandSide.LEFT: None,
            HandSide.RIGHT: None,
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _producer_loop(self) -> None:
        """Drain HTS frames as fast as they arrive into the per-side cache."""
        try:
            for event in self._client.iter_events():
                if self._stop.is_set():
                    return
                if not isinstance(event, HandFrame):
                    continue
                if event.side not in (HandSide.LEFT, HandSide.RIGHT):
                    continue
                with self._lock:
                    self._latest[event.side] = event
        except Exception:
            logger.exception("HTS producer crashed; consumer will starve.")

    @staticmethod
    def _frame_to_proto(frame: HandFrame) -> hand_stream_pb2.HandFrame:
        """SDK ``HandFrame`` → wire ``HandFrame`` proto.

        Wrist pose is forwarded raw in Unity LH coords; downstream
        (``teleop_arm_quest._wrist_pose_to_robot_se3``) handles the basis
        transform to FLU.
        """
        wp = frame.wrist
        R = _quat_to_rotmat(wp.qx, wp.qy, wp.qz, wp.qw)
        keypoints: list[float] = []
        for x, y, z in frame.landmarks.points:
            keypoints.extend((x, y, z))
        return hand_stream_pb2.HandFrame(
            keypoints=keypoints,
            handedness=frame.side.value.lower(),
            timestamp_ns=frame.recv_time_unix_ns or time.time_ns(),
            wrist_pose=hand_stream_pb2.WristPose(
                position=[wp.x, wp.y, wp.z],
                rotation=R.flatten().tolist(),
            ),
        )

    def _frame_generator(self):
        """Paced consumer: snapshot per-side cache, yield only fresh frames."""
        last_seq: dict[HandSide, int | None] = {
            HandSide.LEFT: None,
            HandSide.RIGHT: None,
        }
        next_tick = time.monotonic()
        emitted = 0
        log_every = self._fps * 5
        logger.info(
            "Streaming live HTS → gRPC at %d fps (transport=%s, %s:%d). Ctrl+C to stop.",
            self._fps,
            self._transport_mode.value,
            self._quest_host,
            self._quest_port,
        )
        while not self._stop.is_set():
            next_tick += self._period
            with self._lock:
                snapshot = dict(self._latest)
            for side in (HandSide.LEFT, HandSide.RIGHT):
                frame = snapshot[side]
                if frame is None:
                    continue
                if last_seq[side] == frame.sequence_id:
                    continue
                last_seq[side] = frame.sequence_id
                yield self._frame_to_proto(frame)
                emitted += 1
                if emitted % log_every == 0:
                    logger.info(
                        "emitted=%d  L_seq=%s  R_seq=%s",
                        emitted,
                        last_seq[HandSide.LEFT],
                        last_seq[HandSide.RIGHT],
                    )
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def run(self) -> None:
        producer = threading.Thread(target=self._producer_loop, name="hts-producer", daemon=True)
        producer.start()
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
            self._stop.set()
            channel.close()


class DummyMetaQuestPublisher:
    """Replays a recorded HF dataset over the gRPC HandStream service.

    Used by ``scripts/teleop_arm_quest.py --local`` so the rest of the
    pipeline can run without a physical Quest. Wire format matches
    :class:`MetaQuestPublisher` exactly.
    """

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
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="publish rate cap (Hz)")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="replay HF dataset instead of streaming from a real Quest",
    )
    parser.add_argument(
        "--transport",
        choices=[m.value for m in TransportMode],
        default=TransportMode.UDP.value,
        help="HTS transport mode (live mode only)",
    )
    parser.add_argument(
        "--quest-host",
        default=DEFAULT_QUEST_HOST,
        help="HTS bind/connect host (live mode only)",
    )
    parser.add_argument(
        "--quest-port",
        type=int,
        default=DEFAULT_QUEST_PORT,
        help="HTS bind/connect port (live mode only)",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id (dummy only)")
    parser.add_argument(
        "--filename", default=DEFAULT_FILENAME, help="parquet filename in repo (dummy only)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="stop at end of file instead of looping (dummy only)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="force re-download from HF (dummy only)"
    )
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

    publisher: DummyMetaQuestPublisher | MetaQuestPublisher
    if args.dummy:
        publisher = DummyMetaQuestPublisher(
            server_address=args.server,
            repo=args.repo,
            filename=args.filename,
            fps=args.fps,
            loop=not args.no_loop,
            refresh=args.refresh,
        )
    else:
        publisher = MetaQuestPublisher(
            server_address=args.server,
            transport_mode=TransportMode(args.transport),
            quest_host=args.quest_host,
            quest_port=args.quest_port,
            fps=args.fps,
        )
    publisher.run()


if __name__ == "__main__":
    main()
