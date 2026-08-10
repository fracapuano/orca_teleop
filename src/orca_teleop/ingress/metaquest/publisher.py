"""Serve the Orca WebXR client and publish Quest hand tracking over gRPC.

This is the live ingress used by the original in-hand demonstration pipeline:
the host serves a small page, Quest Browser supplies 25 WebXR hand joints over
a same-origin WebSocket, and this publisher reduces them to the retargeter's
21-joint layout. No third-party headset application is involved.
"""

from __future__ import annotations

import argparse
import logging
import ssl
import threading
import time
from typing import Any

import grpc

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.frames import Pose
from orca_teleop.ingress.metaquest.bridge import QuestTelemetryBridge, WebXRHandSample
from orca_teleop.ingress.metaquest.landmarks import (
    QuaternionContinuity,
    WristAngleEstimator,
    retargeter_landmarks_from_webxr,
)

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "localhost:50051"
DEFAULT_FPS = 30
DEFAULT_QUEST_HOST = "0.0.0.0"
DEFAULT_QUEST_PORT = 8765
WAIT_PERIOD_S = 5.0


def _ssl_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if certfile is None and keyfile is None:
        return None
    if certfile is None or keyfile is None:
        raise ValueError("ssl_cert and ssl_key must be provided together")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


class MetaQuestPublisher:
    """Repository-owned Quest Browser WebXR ingress to paced gRPC publisher."""

    def __init__(
        self,
        server_address: str = DEFAULT_SERVER,
        *,
        handedness: str = "right",
        quest_host: str = DEFAULT_QUEST_HOST,
        quest_port: int = DEFAULT_QUEST_PORT,
        fps: int = DEFAULT_FPS,
        ssl_cert: str | None = None,
        ssl_key: str | None = None,
        wrist_enabled: bool = True,
        wrist_scale: float = 1.0,
        arm_pose_enabled: bool = True,
        reset_event: Any | None = None,
        bridge: QuestTelemetryBridge | None = None,
    ) -> None:
        if handedness not in ("left", "right"):
            raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if wrist_scale < 0:
            raise ValueError(f"wrist_scale must be non-negative, got {wrist_scale}")

        self._server_address = server_address
        self._handedness = handedness
        self._fps = int(fps)
        self._period = 1.0 / self._fps
        self._wrist_enabled = wrist_enabled
        self._wrist_scale = float(wrist_scale)
        # Independent of ``wrist_enabled``: that one is the hand's wrist
        # flexion joint, this one is the arm's 6-DoF target.
        self._arm_pose_enabled = arm_pose_enabled
        self._reset_event = reset_event
        self._wrist_estimator = WristAngleEstimator()
        self._wrist_quaternion = QuaternionContinuity()
        self._head_quaternion = QuaternionContinuity()
        self._bridge = bridge or QuestTelemetryBridge(
            host=quest_host,
            port=quest_port,
            ssl_context=_ssl_context(ssl_cert, ssl_key),
        )
        self._stop = threading.Event()

    @property
    def bridge(self) -> QuestTelemetryBridge:
        return self._bridge

    def stop(self) -> None:
        self._stop.set()

    def _consume_reset(self) -> None:
        if self._reset_event is None or not self._reset_event.is_set():
            return
        self._reset_event.clear()
        self._wrist_estimator.reset()
        logger.info("Quest wrist zero reset for the next episode.")

    @property
    def arm_pose_enabled(self) -> bool:
        return self._arm_pose_enabled

    def _proto_pose(
        self,
        flu_matrix: Any,
        continuity: QuaternionContinuity,
        pose_epoch: int,
    ) -> hand_stream_pb2.Pose | None:
        """FLU 4x4 -> wire Pose, with a sign-continuous quaternion.

        Returns None (leaving the field unset) rather than shipping a
        degenerate pose if the matrix is not a rigid transform.
        """
        if flu_matrix is None:
            return None
        try:
            pose = Pose.from_matrix(flu_matrix)
        except ValueError:
            logger.warning("Discarding a non-rigid WebXR pose matrix.", exc_info=True)
            return None
        position = pose.position_m
        quaternion = continuity.update(pose.orientation_wxyz, pose_epoch)
        return hand_stream_pb2.Pose(
            px=position[0],
            py=position[1],
            pz=position[2],
            qw=quaternion[0],
            qx=quaternion[1],
            qy=quaternion[2],
            qz=quaternion[3],
        )

    def _sample_to_proto(self, sample: WebXRHandSample) -> hand_stream_pb2.HandFrame:
        keypoints = retargeter_landmarks_from_webxr(sample.landmarks, self._handedness)
        wrist_angle_degrees = 0.0
        if self._wrist_enabled:
            # Unchanged: an independent reduction of the same matrix, so the
            # hand's wrist joint behaves exactly as before regardless of the
            # arm path.
            wrist_angle_degrees = (
                self._wrist_estimator.update(sample.wrist_matrix) * self._wrist_scale
            )

        wrist_pose = head_pose = None
        if self._arm_pose_enabled:
            wrist_pose = self._proto_pose(
                sample.wrist_matrix, self._wrist_quaternion, sample.pose_epoch
            )
            head_pose = self._proto_pose(
                sample.head_matrix, self._head_quaternion, sample.pose_epoch
            )

        # Passing None for a message field leaves it unset, so the absent case
        # needs no branching here.
        return hand_stream_pb2.HandFrame(
            keypoints=keypoints.astype("float32").ravel().tolist(),
            handedness=self._handedness,
            timestamp_ns=sample.timestamp_ns,
            wrist_angle_degrees=wrist_angle_degrees,
            wrist_pose=wrist_pose,
            head_pose=head_pose,
            # A sample only exists when every joint was tracked, so any frame
            # we emit at all is a tracked one. Tracking loss is silence.
            tracking_valid=True,
            pose_epoch=sample.pose_epoch,
        )

    def _frame_generator(self):
        last_sequence_id: int | None = None
        next_tick = time.monotonic()
        next_silence_log = next_tick + WAIT_PERIOD_S
        emitted = 0

        while not self._stop.is_set():
            next_tick += self._period
            self._consume_reset()
            sample = self._bridge.state.get_hand_sample(self._handedness)
            now = time.monotonic()

            if sample is not None and sample.sequence_id != last_sequence_id:
                last_sequence_id = sample.sequence_id
                yield self._sample_to_proto(sample)
                emitted += 1
                next_silence_log = now + WAIT_PERIOD_S
                if emitted == 1:
                    logger.info("First Quest WebXR %s-hand frame received.", self._handedness)
                elif emitted % (self._fps * 5) == 0:
                    logger.info("Quest WebXR publisher emitted %d frames.", emitted)
            elif now >= next_silence_log:
                logger.warning(
                    "No fresh Quest WebXR %s-hand frames for %.0f s. Open the hosted "
                    "page in Quest Browser, tap Start hand tracking, and keep the hand visible.",
                    self._handedness,
                    WAIT_PERIOD_S,
                )
                next_silence_log = now + WAIT_PERIOD_S

            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def run(self) -> None:
        self._bridge.start()
        logger.info("Quest WebXR page listening at %s", self._bridge.url)
        if self._bridge.ssl_context is None:
            logger.info(
                "WebXR needs a secure context: expose port %d with HTTPS (for example, "
                "`ngrok http %d`) or use `adb reverse tcp:%d tcp:%d` and open "
                "http://localhost:%d in Quest Browser.",
                self._bridge.port,
                self._bridge.port,
                self._bridge.port,
                self._bridge.port,
                self._bridge.port,
            )

        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        logger.info(
            "Connecting Quest WebXR publisher to %s at up to %d fps.",
            self._server_address,
            self._fps,
        )
        try:
            summary = stub.StreamHandFrames(self._frame_generator(), wait_for_ready=True)
            logger.info("Quest stream closed: ingress received %d frames.", summary.frames_received)
        except KeyboardInterrupt:
            logger.info("Quest WebXR publisher interrupted.")
        except grpc.RpcError as exc:
            if not self._stop.is_set():
                logger.error("Quest WebXR publisher gRPC error: %s", exc)
        finally:
            self._stop.set()
            channel.close()
            self._bridge.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC ingress host:port")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="gRPC publish-rate cap")
    parser.add_argument("--quest-host", default=DEFAULT_QUEST_HOST, help="WebXR HTTP bind host")
    parser.add_argument(
        "--quest-port",
        type=int,
        default=DEFAULT_QUEST_PORT,
        help="WebXR HTTP port",
    )
    parser.add_argument("--ssl-cert", default=None, help="Optional HTTPS certificate")
    parser.add_argument("--ssl-key", default=None, help="Optional HTTPS private key")
    parser.add_argument(
        "--wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish relative WebXR wrist flexion (default: enabled)",
    )
    parser.add_argument("--wrist-scale", type=float, default=1.0)
    parser.add_argument(
        "--arm-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish the 6-DoF wrist and headset poses for an arm (default: enabled)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    MetaQuestPublisher(
        server_address=args.server,
        handedness=args.hand,
        quest_host=args.quest_host,
        quest_port=args.quest_port,
        fps=args.fps,
        ssl_cert=args.ssl_cert,
        ssl_key=args.ssl_key,
        wrist_enabled=args.wrist,
        wrist_scale=args.wrist_scale,
        arm_pose_enabled=args.arm_pose,
    ).run()


if __name__ == "__main__":
    main()
