"""Headset-free Quest publisher: synthetic telemetry through the real path.

No WebXR, no headset, no browser — but everything downstream of the
WebSocket is the production code. This module only fabricates the JSON
telemetry payloads the page would have sent; they are then fed to a real
:class:`QuestTelemetryState` and consumed by a real
:class:`MetaQuestPublisher`, so the basis conversion, the 25->21 joint
reduction, the wrist-angle estimator, the quaternion continuity tracker, the
pacing and the gRPC stream are all the ones that run on the day.

    # terminal 1 — arm bring-up (prints poses; no hand, no retargeter)
    python -m orca_teleop.arm --port 50051

    # terminal 2 — synthetic operator
    python -m orca_teleop.ingress.metaquest.mock_publisher --server localhost:50051

The failure modes worth rehearsing before a demo are flags, because both are
invisible until they happen on stage:

    --dropout-every 5 --dropout-for 1.5   tracking loss, i.e. SILENCE (the
                                          Quest cannot say "invalid"; the
                                          hand simply leaves the payload)
    --epoch-change-every 10               headset off and on again: WebXR
                                          re-pins its "local" space, so every
                                          pose jumps and a latched arm origin
                                          is silently wrong
    --no-arm-pose                         behave like an old publisher
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time

import numpy as np

from orca_teleop.ingress.metaquest.bridge import QuestTelemetryState
from orca_teleop.ingress.metaquest.publisher import DEFAULT_SERVER, MetaQuestPublisher

logger = logging.getLogger(__name__)

DEFAULT_MOCK_FPS = 60

# WebXR reference space: X right, Y up, -Z forward. Roughly a person's right
# wrist held out in front of them.
_WRIST_CENTRE_XR = np.array([0.15, 1.05, -0.40])
_WRIST_SWEEP_M = np.array([0.12, 0.10, 0.08])
_HEAD_HEIGHT_M = 1.60


def _hand_landmarks_local(flex: float) -> np.ndarray:
    """25 WebXR joints in a wrist-local frame, fingers curling with ``flex``.

    Laid out like the canonical MediaPipe hand (fingers along +Y, thumb off
    to +X) with the four extra metacarpals WebXR reports, so the 25->21
    reduction and the retargeter's frame estimation both see something
    non-degenerate.
    """
    curl = 0.5 * (1.0 - math.cos(flex))  # 0 open .. 1 closed

    def finger(base_x: float, base_y: float, length: float, joints: int) -> list[np.ndarray]:
        points = []
        for index in range(joints):
            fraction = (index + 1) / joints
            # Curling bends the finger forward (out of the palm plane) rather
            # than merely shortening it, so the palm normal stays well-defined.
            reach = length * fraction * (1.0 - 0.45 * curl)
            bend = length * fraction * fraction * curl * 0.8
            points.append(np.array([base_x, base_y + reach, -bend]))
        return points

    joints: list[np.ndarray] = [np.zeros(3)]  # wrist
    # Thumb: metacarpal, proximal, distal, tip (4 joints, no intermediate).
    joints += finger(base_x=0.030, base_y=0.020, length=0.070, joints=4)
    # The rest: metacarpal, proximal, intermediate, distal, tip (5 each).
    for base_x, length in ((0.020, 0.130), (0.000, 0.140), (-0.020, 0.130), (-0.040, 0.105)):
        joints += finger(base_x=base_x, base_y=0.020, length=length, joints=5)

    landmarks = np.array(joints, dtype=np.float64)
    assert landmarks.shape == (25, 3), landmarks.shape
    return landmarks


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula, so the mock has no scipy dependency either."""
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _as_column_major(matrix: np.ndarray) -> list[float]:
    """WebXR ships transforms column-major, which is what the bridge decodes."""
    return matrix.T.ravel().tolist()


def synthetic_payload(elapsed_s: float, side: str, pose_epoch: int) -> dict:
    """One telemetry packet, exactly as the WebXR page would have sent it."""
    # A slow Lissajous keeps the wrist inside a plausible reach envelope while
    # never repeating the same position and velocity together.
    position = _WRIST_CENTRE_XR + _WRIST_SWEEP_M * np.array(
        [
            math.sin(0.7 * elapsed_s),
            math.sin(0.9 * elapsed_s + 1.1),
            math.sin(0.5 * elapsed_s + 2.3),
        ]
    )
    # Rotating about a tilted axis sweeps the quaternion through the w sign
    # change, which is what exercises the continuity tracker.
    rotation = _rotation_about(np.array([0.3, 1.0, 0.2]), 0.6 * elapsed_s)

    wrist = np.eye(4)
    wrist[:3, :3] = rotation
    wrist[:3, 3] = position

    landmarks_local = _hand_landmarks_local(flex=0.8 * elapsed_s)
    landmarks_xr = landmarks_local @ rotation.T + position

    head = np.eye(4)
    head[:3, :3] = _rotation_about(np.array([0.0, 1.0, 0.0]), 0.15 * math.sin(0.4 * elapsed_s))
    head[:3, 3] = np.array([0.0, _HEAD_HEIGHT_M, 0.0])

    return {
        "type": "telemetry",
        "client_wall_ms": time.time() * 1000.0,
        "session_epoch": pose_epoch,
        "head": _as_column_major(head),
        "hands": {
            side: {
                "wrist": _as_column_major(wrist),
                "landmarks": landmarks_xr.tolist(),
            }
        },
    }


class MockQuestBridge:
    """Stands in for :class:`QuestTelemetryBridge` with no aiohttp or headset.

    Duck-typed to what ``MetaQuestPublisher`` actually uses — ``state``,
    ``start``, ``stop``, ``url``, ``ssl_context`` — so the publisher itself
    needs no mock-awareness at all.
    """

    # Truthy so the publisher skips its "expose this port over HTTPS" advice,
    # which is meaningless here. It is never used as a real SSL context.
    ssl_context = "mock"

    def __init__(
        self,
        *,
        side: str = "right",
        fps: int = DEFAULT_MOCK_FPS,
        dropout_every_s: float = 0.0,
        dropout_for_s: float = 0.0,
        epoch_change_every_s: float = 0.0,
    ) -> None:
        self.state = QuestTelemetryState()
        self._side = side
        self._period = 1.0 / max(1, fps)
        self._dropout_every_s = dropout_every_s
        self._dropout_for_s = dropout_for_s
        self._epoch_change_every_s = epoch_change_every_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pose_epoch = 1
        self._dropped_last = False

    @property
    def url(self) -> str:
        return "mock://synthetic-quest"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mock-quest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _in_dropout(self, elapsed_s: float) -> bool:
        if self._dropout_every_s <= 0.0 or self._dropout_for_s <= 0.0:
            return False
        return (elapsed_s % self._dropout_every_s) < self._dropout_for_s

    def _current_epoch(self, elapsed_s: float) -> int:
        if self._epoch_change_every_s <= 0.0:
            return self._pose_epoch
        return self._pose_epoch + int(elapsed_s // self._epoch_change_every_s)

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            elapsed_s = time.monotonic() - started

            if self._in_dropout(elapsed_s):
                # Tracking loss is silence: the hand disappears from the
                # payload, so the store invalidates the side and the publisher
                # emits nothing at all.
                if not self._dropped_last:
                    logger.info("Mock: tracking lost (hand out of view).")
                    self._dropped_last = True
                self.state.update({"type": "telemetry", "hands": {}})
            else:
                if self._dropped_last:
                    logger.info("Mock: tracking regained.")
                    self._dropped_last = False
                epoch = self._current_epoch(elapsed_s)
                self.state.update(synthetic_payload(elapsed_s, self._side, epoch))

            time.sleep(self._period)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC ingress host:port")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=int, default=DEFAULT_MOCK_FPS, help="publish rate cap")
    parser.add_argument(
        "--dropout-every",
        type=float,
        default=0.0,
        help="seconds between simulated tracking losses (0 = never)",
    )
    parser.add_argument(
        "--dropout-for", type=float, default=1.0, help="seconds each tracking loss lasts"
    )
    parser.add_argument(
        "--epoch-change-every",
        type=float,
        default=0.0,
        help="seconds between simulated headset off/on events (0 = never)",
    )
    parser.add_argument(
        "--arm-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish the 6-DoF wrist and head poses (default: enabled)",
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

    bridge = MockQuestBridge(
        side=args.hand,
        fps=args.fps,
        dropout_every_s=args.dropout_every,
        dropout_for_s=args.dropout_for,
        epoch_change_every_s=args.epoch_change_every,
    )
    logger.info(
        "Mock Quest publisher: %s hand at %d fps -> %s (no headset involved)",
        args.hand,
        args.fps,
        args.server,
    )
    MetaQuestPublisher(
        server_address=args.server,
        handedness=args.hand,
        fps=args.fps,
        arm_pose_enabled=args.arm_pose,
        bridge=bridge,
    ).run()


if __name__ == "__main__":
    main()
