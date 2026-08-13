"""Replay a recorded Quest session as a live publisher — no headset needed.

The recording is ``fracapuano/quest-poses`` on the Hugging Face Hub: 30 s of
both wrists captured from a real Quest through the Hand Tracking Streamer SDK
at 30 Hz, with per-side visibility, the 21 hand landmarks, and the wrist
position and orientation. It is real human motion — real jitter, real tracking
dropouts — which the synthetic
:mod:`orca_teleop.ingress.metaquest.mock_publisher` cannot fabricate.

    # terminal 1 — arm bring-up (prints poses; no hand, no retargeter)
    python -m orca_teleop.arm --port 50051

    # terminal 2 — replay a real operator, on a loop
    python -m orca_teleop.ingress.metaquest.replay_publisher \\
        --server localhost:50051 --loop

Pick your mock by what you are testing:

* ``replay_publisher`` — real motion, but it builds wire frames directly and so
  does not exercise the WebXR page, the bridge, or the 25->21 joint reduction.
  The recording was made with the SDK client, not the browser page.
* ``mock_publisher`` — synthetic motion, but every line downstream of the
  WebSocket is the production path, and dropouts and reference changes are
  flags rather than whatever happened to be on the tape.

The recording is in Unity's left-handed basis, which is what the SDK reports;
:data:`UNITY_LEFT_TO_FLU` converts it to the FLU basis this repository puts on
the wire. Everything after that conversion — the quaternion continuity tracker,
the wrist-angle estimator, the proto — is the same code the live publisher uses.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.frames import Pose
from orca_teleop.ingress.metaquest.landmarks import QuaternionContinuity, WristAngleEstimator
from orca_teleop.ingress.metaquest.publisher import DEFAULT_SERVER

logger = logging.getLogger(__name__)

DEFAULT_REPO = "fracapuano/quest-poses"
DEFAULT_FILENAME = "data.parquet"

# Unity left-handed (X right, Y up, Z forward) -> FLU (X forward, Y left, Z up).
# Verified equal to ``hand_tracking_sdk.convert.BASIS_UNITY_LEFT_TO_FLU``, which
# is what Francesco's live SDK path uses; kept as a literal here so replaying a
# recording does not drag in the SDK. det = -1, because this crosses handedness.
UNITY_LEFT_TO_FLU = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def _rotation_from_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """The parquet stores x,y,z,w; the wire wants w,x,y,z. Go via the matrix."""
    x, y, z, w = (float(component) for component in quaternion_xyzw)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero quaternion in the recording")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def unity_pose_to_flu_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Unity-LH position + quaternion -> a 4x4 rigid transform in FLU."""
    rotation = _rotation_from_quaternion_xyzw(quaternion_xyzw)
    matrix = np.eye(4)
    # Conjugating by the basis keeps det = +1 even though the basis itself
    # mirrors: the handedness flip cancels between the two factors.
    matrix[:3, :3] = UNITY_LEFT_TO_FLU @ rotation @ UNITY_LEFT_TO_FLU.T
    matrix[:3, 3] = UNITY_LEFT_TO_FLU @ np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


def unity_points_to_flu(points: np.ndarray) -> np.ndarray:
    """(N, 3) Unity-LH points -> (N, 3) FLU points."""
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ UNITY_LEFT_TO_FLU.T


@dataclass(frozen=True)
class ReplaySample:
    """One visible frame of one hand, still in the recording's Unity basis."""

    t_ns: int
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    landmarks: np.ndarray


@dataclass
class Recording:
    """Visible frames for one hand, in capture order.

    Frames where the hand was not visible are dropped rather than marked: the
    Quest cannot report "invalid", it simply stops reporting, so a dropout in
    the tape has to replay as silence for the arm worker's staleness path to
    see what it would see live.
    """

    handedness: str
    samples: list[ReplaySample]
    dropped: int = 0

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return (self.samples[-1].t_ns - self.samples[0].t_ns) / 1e9

    @classmethod
    def from_table(cls, table, handedness: str) -> Recording:
        """Build from a ``pyarrow`` table with Francesco's column layout."""
        if handedness not in ("left", "right"):
            raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        prefix = f"{handedness}_"
        samples: list[ReplaySample] = []
        dropped = 0
        for index, t_ns in enumerate(columns["t_ns"]):
            if not columns[f"{prefix}visible"][index]:
                dropped += 1
                continue
            position = np.array(
                [columns[f"{prefix}wrist_{axis}"][index] for axis in ("x", "y", "z")]
            )
            quaternion = np.array(
                [columns[f"{prefix}wrist_q{axis}"][index] for axis in ("x", "y", "z", "w")]
            )
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
                dropped += 1
                continue
            samples.append(
                ReplaySample(
                    t_ns=int(t_ns),
                    position=position,
                    quaternion_xyzw=quaternion,
                    landmarks=np.array(columns[f"{prefix}landmarks"][index]).reshape(-1, 3),
                )
            )
        if not samples:
            raise ValueError(f"the recording has no visible {handedness}-hand frames")
        return cls(handedness=handedness, samples=samples, dropped=dropped)

    @classmethod
    def load(
        cls,
        handedness: str = "right",
        *,
        parquet: Path | str | None = None,
        repo: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
    ) -> Recording:
        """Read a local parquet, or download one from the Hub and cache it."""
        import pyarrow.parquet as pq

        if parquet is None:
            from huggingface_hub import hf_hub_download

            parquet = hf_hub_download(repo, filename, repo_type="dataset")
            logger.info("Using recording %s/%s -> %s", repo, filename, parquet)
        else:
            logger.info("Using local recording %s", parquet)
        recording = cls.from_table(pq.read_table(parquet), handedness)
        logger.info(
            "Loaded %d visible %s-hand frames over %.1f s (%d not visible).",
            len(recording.samples),
            handedness,
            recording.duration_s,
            recording.dropped,
        )
        return recording


class ReplayPublisher:
    """Stream a :class:`Recording` to the gRPC ingress at its recorded rate."""

    def __init__(
        self,
        recording: Recording,
        server_address: str = DEFAULT_SERVER,
        *,
        speed: float = 1.0,
        loop: bool = False,
        wrist_enabled: bool = True,
        wrist_scale: float = 1.0,
        arm_pose_enabled: bool = True,
    ) -> None:
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self._recording = recording
        self._server_address = server_address
        self._speed = float(speed)
        self._loop = loop
        self._wrist_enabled = wrist_enabled
        self._wrist_scale = float(wrist_scale)
        self._arm_pose_enabled = arm_pose_enabled
        self._wrist_estimator = WristAngleEstimator()
        self._wrist_quaternion = QuaternionContinuity()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _sample_to_proto(
        self, sample: ReplaySample, pose_epoch: int, timestamp_ns: int
    ) -> hand_stream_pb2.HandFrame:
        flu_matrix = unity_pose_to_flu_matrix(sample.position, sample.quaternion_xyzw)

        wrist_angle_degrees = 0.0
        if self._wrist_enabled:
            wrist_angle_degrees = self._wrist_estimator.update(flu_matrix) * self._wrist_scale

        wrist_pose = None
        if self._arm_pose_enabled:
            pose = Pose.from_matrix(flu_matrix)
            quaternion = self._wrist_quaternion.update(pose.orientation_wxyz, pose_epoch)
            position = pose.position_m
            wrist_pose = hand_stream_pb2.Pose(
                px=position[0],
                py=position[1],
                pz=position[2],
                qw=quaternion[0],
                qx=quaternion[1],
                qy=quaternion[2],
                qz=quaternion[3],
            )

        # No head_pose: the recorder never captured the headset transform, and
        # an unset field is the honest way to say so.
        return hand_stream_pb2.HandFrame(
            keypoints=unity_points_to_flu(sample.landmarks).astype("float32").ravel().tolist(),
            handedness=self._recording.handedness,
            timestamp_ns=timestamp_ns,
            wrist_angle_degrees=wrist_angle_degrees,
            wrist_pose=wrist_pose,
            tracking_valid=True,
            pose_epoch=pose_epoch,
        )

    def _frame_generator(self):
        samples = self._recording.samples
        pose_epoch = 1
        emitted = 0
        while not self._stop.is_set():
            pass_start_monotonic = time.monotonic()
            pass_start_wall_ns = time.time_ns()
            first_t_ns = samples[0].t_ns

            for sample in samples:
                if self._stop.is_set():
                    return
                offset_s = (sample.t_ns - first_t_ns) / 1e9 / self._speed
                sleep_for = pass_start_monotonic + offset_s - time.monotonic()
                if sleep_for > 0:
                    # A gap in the tape (the hand left view) sleeps through
                    # here, so the ingress sees the same silence it saw live.
                    time.sleep(sleep_for)
                timestamp_ns = pass_start_wall_ns + int(offset_s * 1e9)
                yield self._sample_to_proto(sample, pose_epoch, timestamp_ns)
                emitted += 1
                if emitted == 1:
                    logger.info("Replaying the recorded %s hand.", self._recording.handedness)

            if not self._loop:
                logger.info("Recording finished after %d frames.", emitted)
                return
            # Restarting teleports the wrist back to the start of the take. That
            # is a new reference as far as any latched arm origin is concerned,
            # so bump the epoch and let the sink re-clutch instead of tracking a
            # jump.
            pose_epoch += 1
            self._wrist_estimator.reset()
            self._wrist_quaternion.reset()
            logger.info("Looping the recording (pose_epoch=%d).", pose_epoch)

    def run(self) -> None:
        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        logger.info("Connecting the replay publisher to %s.", self._server_address)
        try:
            summary = stub.StreamHandFrames(self._frame_generator(), wait_for_ready=True)
            logger.info(
                "Replay stream closed: ingress received %d frames.", summary.frames_received
            )
        except KeyboardInterrupt:
            logger.info("Replay publisher interrupted.")
        except grpc.RpcError as exc:
            if not self._stop.is_set():
                logger.error("Replay publisher gRPC error: %s", exc)
        finally:
            self._stop.set()
            channel.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC ingress host:port")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="parquet file inside the repo")
    parser.add_argument("--parquet", default=None, help="local parquet file instead of the Hub")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="repeat the recording forever")
    parser.add_argument(
        "--no-wrist", dest="wrist", action="store_false", help="hold the hand's wrist joint at 0"
    )
    parser.add_argument(
        "--no-arm-pose",
        dest="arm_pose",
        action="store_false",
        help="omit wrist_pose, i.e. behave like a publisher that predates it",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    recording = Recording.load(
        args.hand, parquet=args.parquet, repo=args.repo, filename=args.filename
    )
    ReplayPublisher(
        recording,
        args.server,
        speed=args.speed,
        loop=args.loop,
        wrist_enabled=args.wrist,
        arm_pose_enabled=args.arm_pose,
    ).run()


if __name__ == "__main__":
    main()
