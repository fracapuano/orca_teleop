"""Rigid transforms, matching ``orca_teleop.ingress.frames.Pose`` (commit 1b5c85e).

FLU metres, Hamilton quaternion **w first**. Fields are ``position_m`` /
``orientation_wxyz`` (``p`` / ``q_wxyz`` are aliases). ``as_xyz_wxyz()`` is a
TRON ``left_pos``/``right_pos`` element-for-element -- upstream says so, which is
why ``pos_quat`` is the default encoding.

Two behaviours are copied deliberately and must not be softened:
  * a non-unit quaternion RAISES. An all-default proto Pose has qw=0 -- the zero
    quaternion, not identity -- and normalising it yields a silent NaN pose.
  * ``from_matrix`` rejects a non-orthonormal or improper rotation block.

NO SIGN CANONICALISATION ANYWHERE. Upstream's QuaternionContinuity aligns each
quaternion to its predecessor and deliberately does not force ``w >= 0``, because
that manufactures a discontinuity at 180 deg. :func:`slerp` therefore omits the
usual ``if dot < 0: q1 = -q1``. Its tracker resets across a ``pose_epoch``
change, so drop buffers on reference change rather than interpolating across one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "QUATERNION_NORM_TOLERANCE",
    "Pose",
    "PoseLike",
    "as_pose",
    "lerp",
    "slerp",
    "pose_lerp",
    "quat_angle",
    "quat_to_matrix",
    "matrix_to_quat",
]

#: Mirrors ``orca_teleop.constants.QUATERNION_NORM_TOLERANCE``.
QUATERNION_NORM_TOLERANCE = 1e-3

#: Upstream's orthonormality tolerance in ``Pose.from_matrix``.
ORTHONORMAL_ATOL = 1e-6

# Below this |sin(theta)| slerp is degenerate and falls back to normalised lerp.
_SLERP_EPS = 1e-8
_NORM_EPS = 1e-12


@runtime_checkable
class PoseLike(Protocol):
    """Structural type for anything :func:`as_pose` accepts.

    Both :class:`Pose` and orca_teleop's ``Pose`` satisfy it.
    """

    def as_xyz_wxyz(self) -> np.ndarray: ...


def _unit(q: np.ndarray) -> np.ndarray:
    """Scale to unit length. Sign preserved; never flipped."""
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < _NORM_EPS:
        raise ValueError(f"degenerate quaternion, norm={n!r}")
    return q / n


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton wxyz quaternion -> 3x3 rotation matrix.

    Written out rather than taken from scipy, matching upstream's reasoning:
    ``Rotation.as_quat(scalar_first=True)`` needs scipy >= 1.14 and nothing pins
    it, and a scalar-last (xyzw) slip is silent -- it still looks like a
    plausible rotation.
    """
    w, x, y, z = _unit(np.asarray(q_wxyz, dtype=np.float64))
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> Hamilton wxyz quaternion (Shepperd's method).

    Branch selection on the largest candidate denominator, matching upstream's
    ``_quaternion_wxyz_from_rotation`` element for element.

    .. warning::
       ``R`` and ``-q`` describe the same rotation, so the sign returned here is
       whichever the largest matrix element implies. Round-tripping a *stream* of
       poses through matrices therefore destroys the continuity
       :func:`slerp` relies on -- exactly the failure ``QuaternionContinuity``
       exists to prevent. Interpolate quaternions directly; use matrices only for
       one-shot composition.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got {r.shape}")
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return _unit(q)


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid transform in FLU metres with a w-first unit quaternion.

    Field names, validation and read-only arrays all match upstream's ``Pose``.
    Construction is positional in both, so ``Pose(position, orientation)`` works
    against either type.

    One deliberate divergence: we **copy** the incoming arrays. Upstream builds
    its read-only arrays with ``np.asarray(...).reshape(-1)``, which returns a
    *view*, so ``setflags(write=False)`` marks only the view -- the caller's
    original array stays writable and mutations through it remain visible inside
    the pose. That undercuts the "cannot be mutated underneath any of them"
    guarantee its docstring makes, and a pose here really does cross threads
    (``arm_worker`` runs on its own thread against a one-writer/N-reader
    ``LatestFrame``). Copying is a strict superset of upstream's safety and
    changes nothing about interop, which rests on field names, accessors and
    validation semantics.
    """

    position_m: np.ndarray  # (3,) float64, metres
    orientation_wxyz: np.ndarray  # (4,) float64, unit Hamilton quaternion

    def __post_init__(self) -> None:
        # np.array, not np.asarray: copy so we never alias a caller's buffer.
        position = np.array(self.position_m, dtype=np.float64).reshape(-1)
        orientation = np.array(self.orientation_wxyz, dtype=np.float64).reshape(-1)
        if position.shape != (3,):
            raise ValueError(f"position_m must have shape (3,); got {position.shape}")
        if orientation.shape != (4,):
            raise ValueError(f"orientation_wxyz must have shape (4,); got {orientation.shape}")
        if not np.all(np.isfinite(position)):
            raise ValueError(f"position_m must be finite; got {position!r}")
        if not np.all(np.isfinite(orientation)):
            raise ValueError(f"orientation_wxyz must be finite; got {orientation!r}")

        norm = float(np.linalg.norm(orientation))
        # Raising beats normalising: an all-default proto Pose has qw=0, i.e. the
        # ZERO quaternion rather than identity, and dividing by ~0 yields a silent
        # NaN pose an arm would happily consume. (frames.py:112)
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise ValueError(f"orientation_wxyz must be a unit quaternion; |q|={norm:.6g}")
        orientation = orientation / norm  # magnitude only -- sign untouched

        position.setflags(write=False)
        orientation.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_wxyz", orientation)

    # -- aliases ---------------------------------------------------------
    @property
    def p(self) -> np.ndarray:
        """Alias for :attr:`position_m`. Read-only."""
        return self.position_m

    @property
    def q_wxyz(self) -> np.ndarray:
        """Alias for :attr:`orientation_wxyz`. Read-only."""
        return self.orientation_wxyz

    # -- accessors -------------------------------------------------------
    @property
    def matrix(self) -> np.ndarray:
        """The (4, 4) homogeneous transform T, with ``p_ref = T @ p_body``."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quat_to_matrix(self.orientation_wxyz)
        matrix[:3, 3] = self.position_m
        return matrix

    def as_xyz_wxyz(self) -> np.ndarray:
        """``[x, y, z, qw, qx, qy, qz]`` -- the LimX TRON per-arm pose vector.

        This is a ``request_servop`` ``left_pos``/``right_pos`` element for
        element, so it can go straight onto the wire without reordering.
        """
        return np.concatenate((self.position_m, self.orientation_wxyz))

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Pose":
        """Build from a (4, 4) homogeneous transform in FLU metres.

        Validates the rotation block, matching upstream: a scaled or otherwise
        non-rigid transform would otherwise become a plausible-looking
        quaternion. See :func:`matrix_to_quat` for the sign caveat.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a (4, 4) transform; got {matrix.shape}")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=ORTHONORMAL_ATOL):
            raise ValueError("Rotation block is not orthonormal")
        if float(np.linalg.det(rotation)) <= 0.0:
            raise ValueError("Rotation block is not a proper rotation (det <= 0)")
        return cls(matrix[:3, 3].copy(), matrix_to_quat(rotation))

    @classmethod
    def from_xyz_wxyz(cls, v: np.ndarray) -> "Pose":
        """Inverse of :meth:`as_xyz_wxyz`."""
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.shape != (7,):
            raise ValueError(f"expected 7 elements [x,y,z,qw,qx,qy,qz], got {v.shape}")
        return cls(v[:3], v[3:])

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    # -- misc ------------------------------------------------------------
    def inverse(self) -> "Pose":
        r_t = quat_to_matrix(self.orientation_wxyz).T
        return Pose(-r_t @ self.position_m, matrix_to_quat(r_t))


    def __repr__(self) -> str:  # pragma: no cover - debug aid
        p = ", ".join(f"{v:.4f}" for v in self.position_m)
        q = ", ".join(f"{v:.4f}" for v in self.orientation_wxyz)
        return f"Pose(position_m=[{p}], orientation_wxyz=[{q}])"


def as_pose(obj: Any) -> Pose:
    """Coerce a pose-like object (ours or upstream's) into a :class:`Pose`.

    Order matters. ``as_xyz_wxyz()`` comes first because it is the accessor
    upstream documents *and* it preserves the quaternion sign; a ``.matrix``
    round-trip does not.
    """
    if isinstance(obj, Pose):
        return obj
    getter = getattr(obj, "as_xyz_wxyz", None)
    if callable(getter):
        return Pose.from_xyz_wxyz(np.asarray(getter(), dtype=np.float64))
    position = getattr(obj, "position_m", None)
    orientation = getattr(obj, "orientation_wxyz", None)
    if position is None and orientation is None:
        position, orientation = getattr(obj, "p", None), getattr(obj, "q_wxyz", None)
    if position is not None and orientation is not None:
        return Pose(position, orientation)
    matrix = getattr(obj, "matrix", None)
    if matrix is not None:
        return Pose.from_matrix(np.asarray(matrix, dtype=np.float64))
    raise TypeError(f"cannot interpret {type(obj).__name__} as a Pose")


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Straight-line interpolation. ``t`` is not clamped by this function."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a + t * (b - a)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two **already sign-continuous**
    Hamilton wxyz quaternions.

    Deliberately omits the conventional ``if dot < 0: q1 = -q1`` shortest-arc
    fix-up. Upstream's ``QuaternionContinuity`` already aligns each quaternion to
    its predecessor and refuses to canonicalise ``w >= 0`` precisely because that
    manufactures a discontinuity whenever the wrist passes 180 deg from its
    reference -- something an operator reaches routinely. Re-canonicalising here
    would put that discontinuity back.

    Consequence: an antipodal pair takes the long way round rather than being
    silently corrected. That only arises if the continuity contract was broken
    upstream (or a buffer straddles a ``pose_epoch`` change), and going the long
    way is visible, whereas a silent flip is not.

    Neither input array is modified; a fresh array is returned.
    """
    a = np.asarray(q0, dtype=np.float64).reshape(-1)
    b = np.asarray(q1, dtype=np.float64).reshape(-1)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("slerp expects two 4-element wxyz quaternions")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    sin_theta = np.sqrt(max(0.0, 1.0 - dot * dot))
    if sin_theta < _SLERP_EPS:
        # Parallel (dot ~ +1) or antipodal (dot ~ -1). The antipodal case has no
        # unique arc; both degrade to normalised lerp, which for dot ~ -1 is
        # itself degenerate, so fall back to holding q0.
        blended = a + t * (b - a)
        if float(np.linalg.norm(blended)) < _NORM_EPS:
            return a.copy()
        return _unit(blended)
    theta = float(np.arctan2(sin_theta, dot))
    s0 = np.sin((1.0 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta
    return _unit(s0 * a + s1 * b)


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Rotation angle in radians along the arc :func:`slerp` would actually take.

    Consistent with :func:`slerp`: no sign folding, so an antipodal pair reports
    an angle approaching 2*pi rather than 0.
    """
    a = np.asarray(q0, dtype=np.float64).reshape(-1)
    b = np.asarray(q1, dtype=np.float64).reshape(-1)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 2.0 * float(np.arctan2(np.sqrt(max(0.0, 1.0 - dot * dot)), dot))


def pose_lerp(a: "PoseLike", b: "PoseLike", t: float) -> Pose:
    """Interpolate two poses: lerp the translation, slerp the rotation."""
    pa, pb = as_pose(a), as_pose(b)
    return Pose(
        lerp(pa.position_m, pb.position_m, t),
        slerp(pa.orientation_wxyz, pb.orientation_wxyz, t),
    )
