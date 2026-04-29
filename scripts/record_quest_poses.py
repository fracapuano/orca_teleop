"""Record Meta Quest hand poses to parquet.

Connects to a running Hand Tracking Streamer (HTS) on the Quest, waits until the
first hand frame arrives, then samples both hands at a fixed rate and appends rows
to a single parquet file.

    python scripts/record_quest_poses.py
    python scripts/record_quest_poses.py --no-upload --fps 30
"""

import argparse
import logging
import os
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from hand_tracking_sdk import (
    HandFilter,
    HandFrame,
    HandSide,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("recordings/data.parquet")
DEFAULT_REPO = "fracapuano/quest-poses"
DEFAULT_FPS = 30
DEFAULT_DURATION_S = 30.0
STALE_AFTER_S = 0.5  # if no new frame for a side in this long → mark not visible


def _flatten_landmarks(frame: HandFrame) -> list[float]:
    """21 x (x,y,z) tuples → flat list of 63 floats."""
    return [c for pt in frame.landmarks.points for c in pt]


_NAN_LANDMARKS = [float("nan")] * 63
_NAN_WRIST = (float("nan"),) * 7  # x,y,z,qx,qy,qz,qw


def _row_for_side(frame: HandFrame | None, recv_ts_ns: int | None, now_ns: int) -> dict:
    """Build the per-side fields. Returns NaNs if frame missing or stale."""
    visible = (
        frame is not None
        and recv_ts_ns is not None
        and (now_ns - recv_ts_ns) < int(STALE_AFTER_S * 1e9)
    )
    if not visible:
        x, y, z, qx, qy, qz, qw = _NAN_WRIST
        return {
            "visible": False,
            "wrist_x": x,
            "wrist_y": y,
            "wrist_z": z,
            "wrist_qx": qx,
            "wrist_qy": qy,
            "wrist_qz": qz,
            "wrist_qw": qw,
            "landmarks": _NAN_LANDMARKS,
            "frame_id": "",
            "sequence_id": -1,
            "source_ts_ns": -1,
        }
    w = frame.wrist
    return {
        "visible": True,
        "wrist_x": w.x,
        "wrist_y": w.y,
        "wrist_z": w.z,
        "wrist_qx": w.qx,
        "wrist_qy": w.qy,
        "wrist_qz": w.qz,
        "wrist_qw": w.qw,
        "landmarks": _flatten_landmarks(frame),
        "frame_id": frame.frame_id,
        "sequence_id": frame.sequence_id,
        "source_ts_ns": frame.source_ts_ns if frame.source_ts_ns is not None else -1,
    }


def _prefixed(side: str, fields: dict) -> dict:
    return {f"{side}_{k}": v for k, v in fields.items()}


class _LatestFrames:
    """Thread-safe holder for the most recent HandFrame per side."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[HandSide, HandFrame] = {}
        self._recv_ns: dict[HandSide, int] = {}

    def update(self, frame: HandFrame) -> None:
        with self._lock:
            self._latest[frame.side] = frame
            self._recv_ns[frame.side] = time.time_ns()

    def snapshot(self) -> tuple[dict[HandSide, HandFrame], dict[HandSide, int]]:
        with self._lock:
            return dict(self._latest), dict(self._recv_ns)


def _stream_into(latest: _LatestFrames, client: HTSClient, stop: threading.Event) -> None:
    """Background loop: pull events from the SDK, stash latest per side."""
    try:
        for event in client.iter_events():
            if stop.is_set():
                break
            if isinstance(event, HandFrame):
                latest.update(event)
    except Exception:
        logger.exception("Streaming thread crashed")
        stop.set()


def _wait_for_first_frame(
    latest: _LatestFrames,
    stop: threading.Event,
    poll_s: float = 0.02,
) -> bool:
    """Block until any HandFrame is stored in *latest*, or *stop* is set."""
    while not stop.is_set():
        frames, _ = latest.snapshot()
        if frames:
            return True
        time.sleep(poll_s)
    return False


def _build_table(rows: list[dict]) -> pa.Table:
    schema = pa.schema(
        [
            ("t_ns", pa.int64()),
            ("left_visible", pa.bool_()),
            ("left_wrist_x", pa.float64()),
            ("left_wrist_y", pa.float64()),
            ("left_wrist_z", pa.float64()),
            ("left_wrist_qx", pa.float64()),
            ("left_wrist_qy", pa.float64()),
            ("left_wrist_qz", pa.float64()),
            ("left_wrist_qw", pa.float64()),
            ("left_landmarks", pa.list_(pa.float64(), 63)),
            ("left_frame_id", pa.string()),
            ("left_sequence_id", pa.int64()),
            ("left_source_ts_ns", pa.int64()),
            ("right_visible", pa.bool_()),
            ("right_wrist_x", pa.float64()),
            ("right_wrist_y", pa.float64()),
            ("right_wrist_z", pa.float64()),
            ("right_wrist_qx", pa.float64()),
            ("right_wrist_qy", pa.float64()),
            ("right_wrist_qz", pa.float64()),
            ("right_wrist_qw", pa.float64()),
            ("right_landmarks", pa.list_(pa.float64(), 63)),
            ("right_frame_id", pa.string()),
            ("right_sequence_id", pa.int64()),
            ("right_source_ts_ns", pa.int64()),
        ]
    )
    columns = {name: [r[name] for r in rows] for name in schema.names}
    return pa.Table.from_pydict(columns, schema=schema)


def _write_parquet(rows: list[dict], out: Path) -> int:
    """Append rows to *out*, creating the file if missing. Returns total row count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    new_table = _build_table(rows)
    if out.exists():
        existing = pq.read_table(out)
        combined = pa.concat_tables([existing, new_table.cast(existing.schema)])
    else:
        combined = new_table
    pq.write_table(combined, out)
    return combined.num_rows


def _upload(out: Path, repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo="data.parquet",
        repo_id=repo,
        repo_type="dataset",
    )
    logger.info("Uploaded to https://huggingface.co/datasets/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="frames per second")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="recording duration in seconds; use <= 0 to record until Ctrl+C",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--transport",
        default="udp",
        choices=["udp", "tcp_server", "tcp_client"],
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

    config = HTSClientConfig(
        transport_mode=TransportMode(args.transport),
        host=args.host,
        port=args.port,
        output=StreamOutput.FRAMES,
        hand_filter=HandFilter.BOTH,
    )
    client = HTSClient(config)
    latest = _LatestFrames()
    stop = threading.Event()

    thread = threading.Thread(target=_stream_into, args=(latest, client, stop), daemon=True)
    thread.start()
    logger.info("Listening for HTS frames on %s://%s:%d", args.transport, args.host, args.port)

    logger.info("Waiting for first hand frame from HTS...")
    try:
        if not _wait_for_first_frame(latest, stop):
            logger.warning("Stream ended before any hand frame arrived; nothing to record.")
            return

    except KeyboardInterrupt:
        logger.info("Stopping before recording started.")
        stop.set()
        return

    logger.info("First frame received; starting capture.")

    period = 1.0 / args.fps if args.fps > 0 else 1.0
    rows: list[dict] = []
    next_tick = time.monotonic()
    last_log = time.monotonic()
    end_at = next_tick + args.duration if args.duration > 0 else None
    seen_any = False

    if end_at is None:
        logger.info("Recording until Ctrl+C.")
    else:
        logger.info("Recording for %.1f seconds.", args.duration)

    try:
        while not stop.is_set() and (end_at is None or time.monotonic() < end_at):
            frames, recv_ns = latest.snapshot()
            now_ns = time.time_ns()
            row = {"t_ns": now_ns}
            row.update(
                _prefixed(
                    "left",
                    _row_for_side(
                        frames.get(HandSide.LEFT),
                        recv_ns.get(HandSide.LEFT),
                        now_ns,
                    ),
                )
            )
            row.update(
                _prefixed(
                    "right",
                    _row_for_side(
                        frames.get(HandSide.RIGHT),
                        recv_ns.get(HandSide.RIGHT),
                        now_ns,
                    ),
                )
            )
            rows.append(row)
            seen_any = seen_any or row["left_visible"] or row["right_visible"]

            if args.fps > 0 and time.monotonic() - last_log > 5.0:
                n_last = args.fps * 5
                vis = sum(r["left_visible"] or r["right_visible"] for r in rows[-n_last:])
                logger.info(
                    "rows=%d  visible_in_last_5s=%d/%d%s",
                    len(rows),
                    vis,
                    n_last,
                    "" if seen_any else "  (no frames yet — is HTS app streaming?)",
                )
                last_log = time.monotonic()

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # we fell behind, resync

    except KeyboardInterrupt:
        logger.info("Stopping (Ctrl+C).")
    finally:
        stop.set()

    if end_at is not None and time.monotonic() >= end_at:
        logger.info("Recording duration elapsed.")

    if not rows:
        logger.warning("No rows recorded; nothing to write.")
        return

    total = _write_parquet(rows, args.out)
    size_mb = args.out.stat().st_size / 1e6
    logger.info("Wrote %d new rows → %s (%d total, %.2f MB)", len(rows), args.out, total, size_mb)

    if args.no_upload:
        logger.info("Skipping HF upload (--no-upload).")
        return
    if (
        not os.environ.get("HF_TOKEN")
        and not Path.home().joinpath(".cache/huggingface/token").exists()
    ):
        logger.warning(
            "No HF token found; run `huggingface-cli login` or set HF_TOKEN. Skipping upload."
        )
        return
    _upload(args.out, args.repo)


if __name__ == "__main__":
    main()
