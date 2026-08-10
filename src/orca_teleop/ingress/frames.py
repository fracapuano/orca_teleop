"""Pose and frame types shared by the hand and arm consumers of the ingress.

This module is deliberately dependency-light: numpy and the standard library
only. No torch, no orca_core, no grpc. An arm controller can

    from orca_teleop.ingress.frames import LatestFrame, Pose, TeleopFrame

without paying for the retargeting stack, which keeps the door open to moving
that controller into its own process behind a thin transport shim without the
contract changing.

Conventions, once, here:

* **Frame** — FLU: X forward, Y left, Z up.
* **Units** — meters.
* **Rotation** — Hamilton unit quaternion, ``w`` FIRST.
* **Reference** — the publisher's own tracking space, re-pinned whenever
  :attr:`TeleopFrame.pose_epoch` changes. Never a world frame.

That is also the LimX TRON dual-arm convention, so :meth:`Pose.as_xyz_wxyz`
produces a TRON ``left_pos``/``right_pos`` element-for-element.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from orca_teleop.constants import (
    ARM_OWNER_TAKEOVER_S,
    ARM_STALE_AFTER_S,
    QUATERNION_NORM_TOLERANCE,
)

__all__ = [
    "FrameUpdate",
    "LatestFrame",
    "Pose",
    "TeleopFrame",
]


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Shepperd's method: proper rotation matrix -> unit quaternion (w, x, y, z).

    Branching on the largest of the four candidate denominators keeps the
    division well-conditioned for every rotation, including the 180° cases
    where the trace-only formula degenerates. Hand-rolled rather than via
    scipy: scipy is present here only as an undeclared transitive dependency
    of pytorch-kinematics, and ``Rotation.as_quat(scalar_first=True)`` needs
    scipy >= 1.14, which nothing pins.
    """
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)


def _rotation_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid transform in FLU meters, with a w-first unit quaternion.

    Both arrays are read-only, so a pose handed to several consumer threads
    cannot be mutated underneath any of them.
    """

    position_m: np.ndarray  # (3,) float64, meters
    orientation_wxyz: np.ndarray  # (4,) float64, unit Hamilton quaternion

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64).reshape(-1)
        orientation = np.asarray(self.orientation_wxyz, dtype=np.float64).reshape(-1)
        if position.shape != (3,):
            raise ValueError(f"position_m must have shape (3,); got {position.shape}")
        if orientation.shape != (4,):
            raise ValueError(f"orientation_wxyz must have shape (4,); got {orientation.shape}")

        norm = float(np.linalg.norm(orientation))
        # An all-default proto Pose has qw=0, i.e. the ZERO quaternion rather
        # than identity. Raising beats normalizing: dividing by ~0 yields a
        # silent NaN pose that an arm would happily consume.
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise ValueError(f"orientation_wxyz must be a unit quaternion; |q|={norm:.6g}")
        orientation = orientation / norm

        position.setflags(write=False)
        orientation.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_wxyz", orientation)

    @property
    def matrix(self) -> np.ndarray:
        """The (4, 4) homogeneous transform T, with ``p_ref = T @ p_body``."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _rotation_from_quaternion_wxyz(self.orientation_wxyz)
        matrix[:3, 3] = self.position_m
        return matrix

    def as_xyz_wxyz(self) -> np.ndarray:
        """``[x, y, z, qw, qx, qy, qz]`` — the LimX TRON per-arm pose vector.

        This is the element order of a TRON ``request_servop`` ``left_pos`` /
        ``right_pos`` (and of each half of ``request_movep``'s ``pos``), so an
        arm sink can hand the result straight over without reordering. The
        quaternion-ordering mistake this exists to prevent is silent: xyzw
        instead of wxyz still looks like a plausible rotation.
        """
        return np.concatenate((self.position_m, self.orientation_wxyz))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Pose:
        """Build a pose from a (4, 4) homogeneous transform in FLU meters."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a (4, 4) transform; got {matrix.shape}")
        rotation = matrix[:3, :3]
        # Catches a scaled or otherwise non-rigid transform before it becomes a
        # plausible-looking quaternion. xr_matrix_to_flu_matrix does not
        # re-orthonormalize, so a corrupt WebXR matrix arrives intact.
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("Rotation block is not orthonormal")
        if float(np.linalg.det(rotation)) <= 0.0:
            raise ValueError("Rotation block is not a proper rotation (det <= 0)")
        return cls(
            position_m=matrix[:3, 3].copy(),
            orientation_wxyz=_quaternion_wxyz_from_rotation(rotation),
        )


@dataclass(frozen=True, slots=True)
class TeleopFrame:
    """One operator sample, as the arm consumer sees it.

    Produced at ingress rate (~30-60 Hz), never gated on the retargeter. The
    hand's finger angles are deliberately absent: they are produced ~100 ms
    later and several frames behind, and bundling them here would sell two
    different instants under one timestamp.
    """

    timestamp_ns: int
    """Publisher wall clock. For correlation and logging ONLY — it is
    ``Date.now()`` in a browser on another machine, so it carries NTP skew,
    can step, and can run backwards. Differentiating over it can produce a
    negative dt and an enormous commanded velocity."""

    recv_monotonic_ns: int
    """``time.monotonic_ns()`` on THIS host when the frame was decoded. The
    only stamp safe for staleness checks and finite differences."""

    stream_id: int
    """Identifies the gRPC connection. A change means a different publisher
    (or the same one reconnected) — not a continuation of the same operator."""

    pose_epoch: int
    """Generation of the reference space the poses live in. A change means
    poses either side are not comparable; a relative-mode origin must be
    re-captured. 0 => the publisher does not track it."""

    handedness: Literal["left", "right"]
    tracking_valid: bool
    wrist: Pose
    head: Pose | None
    wrist_angle_degrees: float

    @property
    def age_s(self) -> float:
        """Seconds since this frame was received, on this host's clock."""
        return (time.monotonic_ns() - self.recv_monotonic_ns) / 1e9

    @property
    def wrist_position_m(self) -> np.ndarray:
        return self.wrist.position_m

    @property
    def wrist_orientation_wxyz(self) -> np.ndarray:
        return self.wrist.orientation_wxyz


@dataclass(frozen=True, slots=True)
class FrameUpdate:
    """Result of a :meth:`LatestFrame.wait_for_next` call."""

    frame: TeleopFrame | None
    seq: int
    fresh: bool
    """False => the wait timed out; ``frame`` is whatever was last published."""
    closed: bool
    """True => the ingress is shutting down. Park the arm; do not hold."""


class LatestFrame:
    """Single-slot latest-value broadcast: one writer, N readers, no queue.

    A queue is the wrong shape for an arm target — a slow consumer would build
    a backlog of poses that are all wrong by the time they are read. The slot
    keeps exactly the newest frame, so :meth:`publish` is O(1) and can never
    backpressure the gRPC servicer thread that the hand path also runs on.

    Readers never run on a servicer thread: they poll :meth:`get_fresh` or
    block in :meth:`wait_for_next`. That keeps arbitrary consumer code (and
    its exceptions, and its blocking socket writes) off the ingress hot path.
    """

    def __init__(self, takeover_after_s: float = ARM_OWNER_TAKEOVER_S) -> None:
        # A Condition over a plain Lock, not an Event: Event.clear() races with
        # a publish between wait() returning and the reader re-arming, which
        # silently drops a frame.
        self._condition = threading.Condition(threading.Lock())
        self._frame: TeleopFrame | None = None
        self._seq = 0
        self._closed = False
        self._rejected = 0
        self._owner_stream_id: int | None = None
        self._owner_monotonic_ns = 0
        self._takeover_after_ns = int(takeover_after_s * 1e9)

    def publish(self, frame: TeleopFrame) -> bool:
        """Offer a frame. Never blocks, never raises; returns whether it won.

        The first stream to publish owns the slot until it disconnects or goes
        quiet for ``takeover_after_s``. Without this, two publishers (a second
        headset, or a left- and a right-hand publisher) would interleave two
        different wrists into one arm target.
        """
        with self._condition:
            if self._closed:
                return False
            now_ns = frame.recv_monotonic_ns
            owner = self._owner_stream_id
            if owner is not None and owner != frame.stream_id:
                if now_ns - self._owner_monotonic_ns < self._takeover_after_ns:
                    self._rejected += 1
                    return False
            self._owner_stream_id = frame.stream_id
            self._owner_monotonic_ns = now_ns
            self._frame = frame
            self._seq += 1
            self._condition.notify_all()
            return True

    def release(self, stream_id: int) -> None:
        """Give up ownership when a stream disconnects, for instant takeover."""
        with self._condition:
            if self._owner_stream_id == stream_id:
                self._owner_stream_id = None

    def get_fresh(self, max_age_s: float = ARM_STALE_AFTER_S) -> TeleopFrame | None:
        """The latest frame, or None if there is none or it is too old."""
        with self._condition:
            frame = self._frame
        if frame is None or frame.age_s > max_age_s:
            return None
        return frame

    def get_unchecked(self) -> tuple[TeleopFrame | None, int]:
        """The latest frame and its sequence number, however old. Opt-in."""
        with self._condition:
            return self._frame, self._seq

    def wait_for_next(self, last_seq: int, timeout: float) -> FrameUpdate:
        """Block until a frame newer than ``last_seq`` arrives, or we close."""
        with self._condition:
            if self._seq <= last_seq and not self._closed:
                self._condition.wait_for(
                    lambda: self._seq > last_seq or self._closed, timeout=timeout
                )
            return FrameUpdate(
                frame=self._frame,
                seq=self._seq,
                fresh=self._seq > last_seq,
                closed=self._closed,
            )

    def close(self) -> None:
        """Wake every waiter so no consumer thread can outlive the ingress."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def stats(self) -> dict:
        with self._condition:
            return {
                "seq": self._seq,
                "rejected": self._rejected,
                "owner_stream_id": self._owner_stream_id,
                "closed": self._closed,
            }
