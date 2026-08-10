"""gRPC ingress server.

The robot side runs this server. Any publisher connects as a gRPC client and
streams ``HandFrame``messages. The server drops each frame onto a queue
consumed by the retargeter.

Typical usage inside the pipeline::

```python
    from orca_teleop.ingress import IngressServer

    server = IngressServer(landmarks_q, stop_event)
    server.start(port=50051) # publishes HandLandmarks to be consumed, e.g. by the retargeter
    server.stop()
```
"""

import itertools
import logging
import queue
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from typing import Literal

import grpc
import numpy as np

from orca_teleop.constants import _COORDS_PER_POINT, _EXPECTED_LEN, _NUM_KEYPOINTS, DEFAULT_PORT
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.frames import LatestFrame, Pose, TeleopFrame

logger = logging.getLogger(__name__)


def _pose_from_proto(proto_pose: hand_stream_pb2.Pose) -> Pose | None:
    """Decode a wire ``Pose``, or None if its quaternion is not usable.

    Same class of guard as the keypoint-length check below: a malformed frame
    is dropped rather than propagated. The specific hole this closes is an
    all-default ``Pose()``, whose ``qw=0`` is the zero quaternion — normalizing
    it would hand the arm a NaN pose with nothing but a numpy warning.
    """
    try:
        return Pose(
            position_m=np.array([proto_pose.px, proto_pose.py, proto_pose.pz], dtype=np.float64),
            orientation_wxyz=np.array(
                [proto_pose.qw, proto_pose.qx, proto_pose.qy, proto_pose.qz],
                dtype=np.float64,
            ),
        )
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Canonical frame that travels on the landmarks queue.

    Every field added for the arm path is defaulted, so existing construction
    sites (MediaPipe, Manus, synthetic, tests) are unaffected.
    """

    keypoints: np.ndarray  # (21, 3) float32, wrist-relative, meters
    handedness: Literal["left", "right"]
    timestamp_ns: int
    wrist_angle_degrees: float = 0.0
    wrist_pose: Pose | None = None
    head_pose: Pose | None = None
    # Absent on the wire decodes as True: an old publisher tracks fine, it
    # just has no field in which to say so.
    tracking_valid: bool = True
    pose_epoch: int = 0
    stream_id: int = 0
    recv_monotonic_ns: int = 0


class _HandStreamServicer(hand_stream_pb2_grpc.HandStreamServicer):
    """Receives HandFrame streams from publishers and enqueues them."""

    def __init__(
        self,
        landmarks_q: queue.Queue,
        stop_event: threading.Event,
        frames: LatestFrame | None = None,
    ) -> None:
        self._q = landmarks_q
        self._stop = stop_event
        self._frames = frames
        self._stream_ids = itertools.count(1)
        self._bad_pose_logged = False

    def StreamHandFrames(self, request_iterator, context):
        frames_received = 0
        peer = context.peer() or "unknown"
        # Identifies this connection for the whole of its life, so an arm
        # consumer can tell "same operator, next frame" from "someone else
        # started publishing" — including the same publisher reconnecting.
        stream_id = next(self._stream_ids)
        logger.info("Publisher connected: %s (stream %d)", peer, stream_id)

        try:
            for frame in request_iterator:
                if self._stop.is_set():
                    break

                if len(frame.keypoints) != _EXPECTED_LEN:
                    logger.warning(
                        "Dropping frame: expected %d floats, got %d",
                        _EXPECTED_LEN,
                        len(frame.keypoints),
                    )
                    continue

                handedness = frame.handedness.lower()
                if handedness not in ("left", "right"):
                    logger.warning("Dropping frame: invalid handedness %r", frame.handedness)
                    continue

                # Stamped here, not by the publisher: timestamp_ns is a remote
                # wall clock and cannot be used for staleness or for dt.
                recv_monotonic_ns = time.monotonic_ns()

                wrist_pose = (
                    _pose_from_proto(frame.wrist_pose) if frame.HasField("wrist_pose") else None
                )
                if (
                    frame.HasField("wrist_pose")
                    and wrist_pose is None
                    and not self._bad_pose_logged
                ):
                    self._bad_pose_logged = True  # once per server, not per frame
                    logger.warning(
                        "Dropping wrist pose from %s: quaternion is not unit-norm. "
                        "Frames still reach the hand path.",
                        peer,
                    )
                head_pose = (
                    _pose_from_proto(frame.head_pose) if frame.HasField("head_pose") else None
                )
                tracking_valid = frame.tracking_valid if frame.HasField("tracking_valid") else True

                keypoints = np.array(frame.keypoints, dtype=np.float32).reshape(
                    _NUM_KEYPOINTS, _COORDS_PER_POINT
                )
                landmark = HandLandmarks(
                    keypoints=keypoints,
                    handedness=handedness,
                    timestamp_ns=frame.timestamp_ns,
                    wrist_angle_degrees=frame.wrist_angle_degrees,
                    wrist_pose=wrist_pose,
                    head_pose=head_pose,
                    tracking_valid=tracking_valid,
                    pose_epoch=frame.pose_epoch,
                    stream_id=stream_id,
                    recv_monotonic_ns=recv_monotonic_ns,
                )

                # Arm first: publishing is one uncontended lock, whereas the
                # queue path below can cost up to three queue operations. The
                # arm must never inherit the hand path's latency, and a full
                # queue must never cost the arm a frame.
                if self._frames is not None and wrist_pose is not None:
                    self._frames.publish(
                        TeleopFrame(
                            timestamp_ns=frame.timestamp_ns,
                            recv_monotonic_ns=recv_monotonic_ns,
                            stream_id=stream_id,
                            pose_epoch=frame.pose_epoch,
                            handedness=handedness,
                            tracking_valid=tracking_valid,
                            wrist=wrist_pose,
                            head=head_pose,
                            wrist_angle_degrees=frame.wrist_angle_degrees,
                        )
                    )

                # Always keep the latest frame; drop stale ones.
                try:
                    self._q.put_nowait(landmark)
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(landmark)
                    except queue.Full:
                        pass

                frames_received += 1
        except Exception:
            logger.exception("Error in stream from %s", peer)
        finally:
            # Hand the arm slot back immediately rather than making the next
            # publisher wait out the takeover timeout.
            if self._frames is not None:
                self._frames.release(stream_id)
            logger.info("Publisher disconnected: %s (received %d frames)", peer, frames_received)

        return hand_stream_pb2.StreamSummary(frames_received=frames_received)


class IngressServer:
    """gRPC server that accepts hand-frame streams from publishers.

    Args:
        landmarks_q: Queue onto which ``HandLandmarks`` are placed.
        stop_event: When set, the server drains and shuts down.
        port: TCP port to listen on.
        max_workers: gRPC thread-pool size.
    """

    def __init__(
        self,
        landmarks_q: queue.Queue,
        stop_event: threading.Event,
        port: int = DEFAULT_PORT,
        max_workers: int = 4,
    ) -> None:
        self._port = port
        self._executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._server = grpc.server(self._executor)
        self._stopped = False
        # Constructed unconditionally: it costs a lock and a slot, and having
        # it always present means an arm consumer can attach to any pipeline
        # without the server being configured for one up front.
        self._frames = LatestFrame()
        servicer = _HandStreamServicer(landmarks_q, stop_event, self._frames)
        hand_stream_pb2_grpc.add_HandStreamServicer_to_server(servicer, self._server)

    @property
    def frames(self) -> LatestFrame:
        """Latest-value slot of 6-DoF wrist poses, for an arm consumer.

        Fed at ingress rate from the same frames the hand path consumes, so
        the two never disagree about when the operator was sampled — but read
        independently, so the arm is not gated on the retargeter.
        """
        return self._frames

    def start(self) -> int:
        """Start listening. Returns the actual port (useful when port=0)."""
        bound_port = self._server.add_insecure_port(f"[::]:{self._port}")
        self._server.start()
        logger.info("Ingress server listening on port %d", bound_port)
        return bound_port

    def stop(self, grace: float = 2.0) -> None:
        """Gracefully shut down the server."""
        if self._stopped:
            return
        self._stopped = True
        termination = self._server.stop(grace=grace)
        termination.wait(timeout=grace + 1.0)
        self._executor.shutdown(wait=True)
        # After the servicer threads are gone, so no publish races the close;
        # this is what wakes an arm consumer blocked in wait_for_next().
        self._frames.close()
        logger.info("Ingress server stopped")

    def wait_for_termination(self, timeout: float | None = None) -> None:
        """Block until the server shuts down."""
        self._server.wait_for_termination(timeout=timeout)
