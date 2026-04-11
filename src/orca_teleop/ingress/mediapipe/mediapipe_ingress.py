import os
import threading
import time
from collections.abc import Callable

import cv2
import mediapipe as mp
import numpy as np
from orca_core import OrcaHand

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


class MediaPipeIngress:
    """MediaPipe hand tracking that processes hand landmarks."""

    def __init__(
        self,
        model_path: str = None,
        callback: Callable[[np.ndarray], None] | None = None,
    ):
        self.callback = callback
        mediapipe_task_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
        )

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError(
                "hand.type must be 'left' or 'right'. Update config.yaml with type field."
            )
        self.hand_type = hand.type

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(mediapipe_task_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.9,
            min_hand_presence_confidence=0.9,
            min_tracking_confidence=0.9,
            result_callback=self._result_callback,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        self.latest_frame = None
        self.latest_image_landmarks = None
        self.orientation_good = True
        self.frame_lock = threading.Lock()
        self.running = False
        self.livestream_thread = None

    def _result_callback(self, result, _output_image, _timestamp_ms: int):
        """Process MediaPipe results."""
        if not result.hand_landmarks:
            with self.frame_lock:
                self.latest_image_landmarks = None
                self.orientation_good = True
            return

        if result.handedness[0][0].category_name != self.hand_type.capitalize():
            return

        world_landmarks = result.hand_world_landmarks[0]
        image_landmarks = result.hand_landmarks[0]
        orientation_good = self._check_orientation(world_landmarks, image_landmarks)

        with self.frame_lock:
            self.latest_image_landmarks = result.hand_landmarks[0]
            self.orientation_good = orientation_good

        if orientation_good and self.callback:
            world_landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
            self.callback(world_landmarks_array)

    def _check_orientation(self, world_landmarks, image_landmarks) -> bool:
        """Check if hand orientation is suitable."""

        wrist = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
        index_base = np.array([world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z])
        pinky_base = np.array([world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

        if self.hand_type == "right":
            world_palm_normal = np.cross(index_base - wrist, pinky_base - wrist)
        else:  # left hand
            world_palm_normal = np.cross(pinky_base - wrist, index_base - wrist)
        world_palm_normal = world_palm_normal / np.linalg.norm(world_palm_normal)

        # Check if hand palm faces down and towards camera
        face_down_angle = np.degrees(
            np.arccos(np.clip(np.dot(world_palm_normal, [0, -1, 0]), -1.0, 1.0))
        )
        facing_camera = np.dot(world_palm_normal, [0, 0, -1]) > 0

        # Check if palm_width (thumb_base-pinky_base) is > 5% of image width
        image_palm_width = np.linalg.norm(
            [
                image_landmarks[1].x - image_landmarks[17].x,
                image_landmarks[1].y - image_landmarks[17].y,
            ]
        )
        distance_ok = image_palm_width > 0.05
        return 90 <= face_down_angle <= 140 and facing_camera and distance_ok

    def _process_frames(self):
        """Process webcam frames."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self.landmarker.detect_async(mp_image, int(time.time() * 1000))
            time.sleep(1.0 / 30.0)

    def start(self):
        """Start processing."""
        if not self.running:
            self.running = True
            self.livestream_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.livestream_thread.start()

    def stop(self):
        """Stop processing."""
        self.running = False
        if self.livestream_thread:
            self.livestream_thread.join()

    def display_frame(self):
        """Display frame with landmarks."""
        with self.frame_lock:
            if self.latest_frame is None:
                return

            frame = self.latest_frame.copy()

            if self.latest_image_landmarks:
                color = (0, 255, 0) if self.orientation_good else (128, 128, 128)
                _draw_hand_landmarks(frame, self.latest_image_landmarks, color=color)

            cv2.imshow("MediaPipe Hand Tracking", frame)
            cv2.waitKey(1)

    def cleanup(self):
        """Release resources."""
        self.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()


def main():
    """Standalone demo."""
    ingress = MediaPipeIngress(callback=lambda lm: print(f"Landmarks: {lm.shape}"))

    try:
        ingress.start()
        while True:
            ingress.display_frame()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        ingress.cleanup()


if __name__ == "__main__":
    main()
