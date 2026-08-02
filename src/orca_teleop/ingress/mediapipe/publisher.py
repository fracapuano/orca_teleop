"""MediaPipe hand-tracking gRPC publisher.

Run this on the operator's machine (any OS with a webcam). It captures hand
landmarks via MediaPipe and streams them to the robot-side ``IngressServer``
over gRPC.

Usage::

    # Stream to robot on the same machine
    python -m orca_teleop.ingress.mediapipe.publisher

    # Stream to a remote robot
    python -m orca_teleop.ingress.mediapipe.publisher --server 192.168.1.42:50051

    # Left hand, high confidence
    python -m orca_teleop.ingress.mediapipe.publisher --hand left --confidence 0.9
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import cv2
import grpc
import mediapipe as mp
import numpy as np

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc

_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _draw_hand_landmarks(frame: np.ndarray, landmarks, color: tuple = (0, 255, 0)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1)


logger = logging.getLogger(__name__)


class MediaPipePublisher:
    """Captures hand landmarks from a webcam and streams them over gRPC."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        handedness: str = "right",
        confidence: float = 0.7,
        show_video: bool = False,
        camera_index: int = 0,
        orientation_gate: bool = True,
    ) -> None:
        self._server_address = server_address
        self._handedness = handedness.lower()
        self._confidence = confidence
        self._show_video = show_video
        self._camera_index = camera_index
        # Gate frames on a plausible teleop pose (palm down, facing the
        # camera, close enough) — garbage poses otherwise yank the
        # retargeter. Public: toggleable at runtime (orca_ui config message).
        self.orientation_gate = orientation_gate
        # When True, keep a copy of every captured frame for
        # latest_annotated_jpeg() (the orca_ui camera preview). Costs a frame
        # copy per capture, so it's toggled on demand.
        self.retain_frames = False
        self._stop_event = threading.Event()

        # MediaPipe setup
        mediapipe_task_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(mediapipe_task_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=self._confidence,
            min_hand_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # Latest frame data (written by callback, read by stream generator)
        self._lock = threading.Lock()
        self._latest_keypoints: np.ndarray | None = None
        self._fresh = False

        # Visualization state
        self._latest_frame: np.ndarray | None = None
        self._latest_image_landmarks = None
        self._gate_ok = False

    def _on_result(self, result, _output_image, _timestamp_ms: int) -> None:
        """MediaPipe async callback — fires on each detection."""
        if not result.hand_landmarks:
            with self._lock:
                self._latest_image_landmarks = None
                self._gate_ok = False
            return

        # Only accept the hand we care about
        detected_hand = result.handedness[0][0].category_name.lower()
        if detected_hand != self._handedness:
            return

        world_landmarks = result.hand_world_landmarks[0]
        image_landmarks = result.hand_landmarks[0]
        gate_ok = (not self.orientation_gate) or self._check_orientation(
            world_landmarks, image_landmarks)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)

        with self._lock:
            self._latest_image_landmarks = image_landmarks
            self._gate_ok = gate_ok
            if gate_ok:
                self._latest_keypoints = keypoints
                self._fresh = True

    def _check_orientation(self, world_landmarks, image_landmarks) -> bool:
        """Plausible-teleop-pose gate (ported from the legacy ingress): palm
        roughly face-down, facing the camera, and close enough that the
        landmarks aren't noise."""
        wrist = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
        index_base = np.array([world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z])
        pinky_base = np.array(
            [world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

        if self._handedness == "right":
            palm_normal = np.cross(index_base - wrist, pinky_base - wrist)
        else:
            palm_normal = np.cross(pinky_base - wrist, index_base - wrist)
        norm = np.linalg.norm(palm_normal)
        if norm < 1e-9:
            return False
        palm_normal = palm_normal / norm

        face_down_angle = np.degrees(
            np.arccos(np.clip(np.dot(palm_normal, [0, -1, 0]), -1.0, 1.0)))
        facing_camera = np.dot(palm_normal, [0, 0, -1]) > 0
        image_palm_width = np.linalg.norm([
            image_landmarks[1].x - image_landmarks[17].x,
            image_landmarks[1].y - image_landmarks[17].y,
        ])
        return 90 <= face_down_angle <= 140 and facing_camera and image_palm_width > 0.05

    def _frame_generator(self):
        """Yield HandFrame protos as fast as new data arrives."""
        while True:
            with self._lock:
                if not self._fresh:
                    kp = None
                else:
                    kp = self._latest_keypoints.copy()
                    self._fresh = False

            if kp is None:
                time.sleep(0.001)
                continue

            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness=self._handedness,
                timestamp_ns=time.time_ns(),
            )

    def stop(self) -> None:
        """Break the ``run()`` loop from another thread."""
        self._stop_event.set()

    def latest_annotated_jpeg(self, quality: int = 70,
                              width: int = 480) -> bytes | None:
        """Downscaled JPEG of the latest frame with the landmark skeleton
        overlaid (green when the orientation gate passes, gray otherwise).
        Needs ``retain_frames = True``; used for the orca_ui camera preview."""
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
            image_landmarks = self._latest_image_landmarks
            gate_ok = self._gate_ok
        if image_landmarks:
            color = (0, 255, 0) if gate_ok else (140, 140, 140)
            _draw_hand_landmarks(frame, image_landmarks, color=color)
        height = max(1, int(frame.shape[0] * width / frame.shape[1]))
        frame = cv2.resize(frame, (width, height))
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else None

    def run(self) -> None:
        """Open the webcam, connect to the server, and stream until
        interrupted (or ``stop()`` is called from another thread)."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open webcam (index {self._camera_index})")

        logger.info(
            "Connecting to %s (hand=%s, confidence=%.2f)",
            self._server_address,
            self._handedness,
            self._confidence,
        )
        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)

        # Start the gRPC stream in a background thread
        stream_future = stub.StreamHandFrames.future(self._frame_generator())

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue

                if self._show_video or self.retain_frames:
                    with self._lock:
                        self._latest_frame = frame.copy()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                self._landmarker.detect_async(mp_image, int(time.time() * 1000))

                if self._show_video:
                    self._display_frame()
                    # cv2 window pump — only safe/useful with a window open
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(1.0 / 30.0)

        except KeyboardInterrupt:
            pass
        finally:
            stream_future.cancel()
            channel.close()
            cap.release()
            self._landmarker.close()
            if self._show_video:
                cv2.destroyAllWindows()
            logger.info("Publisher shut down.")

    def _display_frame(self) -> None:
        """Show the webcam feed with landmarks overlaid."""
        with self._lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
            image_landmarks = self._latest_image_landmarks

        if image_landmarks:
            _draw_hand_landmarks(frame, image_landmarks)

        cv2.imshow("MediaPipe Publisher", frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream hand landmarks from a webcam to the orca_teleop server via gRPC.",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Which hand to track (default: right)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="MediaPipe detection confidence (default: 0.7)",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show webcam feed with landmarks overlay",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
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

    publisher = MediaPipePublisher(
        server_address=args.server,
        handedness=args.hand,
        confidence=args.confidence,
        show_video=args.show_video,
        camera_index=args.camera_index,
    )
    publisher.run()


if __name__ == "__main__":
    main()
