"""Clutch origins, scaling and the workspace clamp. Zero I/O.

Contract (CLAUDE.md "Mapping", guide 9-08)::

    delta = inv(T_op0) @ T_op        # operator motion in its own frame
    delta[:3, 3] *= s                # scale the TRANSLATION of the delta only
    T_target = T_robot0 @ delta

Scaling the delta rather than the final position is what makes it relative: at
engage delta is the identity, so the target is exactly ``T_robot0`` and the arm
does not jump. Scaling ``T_target[:3, 3]`` instead would multiply the robot's
absolute position and fling the arm across the workspace on the first frame.

See REFERENCE.md for the expanded derivations and the ``M = R_r0 @ R_o0.T``
ergonomics note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .config import Arm, Config
from .poses import Pose, PoseLike, as_pose

__all__ = [
    "ClutchMapper",
    "WorkspaceViolation",
    "check_inside_workspace",
    "MappingError",
    "TranslationFrame",
    "WorkspaceClampResult",
    "clamp_to_workspace",
    "compose",
    "invert_transform",
    "scale_delta_translation",
]

TranslationFrame = Literal["body", "world"]


class MappingError(RuntimeError):
    """Raised for a mapping used outside its contract (e.g. unlatched)."""


# -- small transform helpers --------------------------------------------
def invert_transform(t: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform, via the transpose (never ``np.linalg.inv``).

    For a rigid ``T = [R | p]`` the inverse is ``[R.T | -R.T p]``. Using the
    transpose keeps the result exactly orthonormal; a general matrix inverse
    accumulates error that eventually trips ``Pose.from_matrix``'s orthonormality
    check.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {t.shape}")
    rotation = t[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ t[:3, 3]
    return out


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right composition of 4x4 transforms."""
    if not transforms:
        return np.eye(4, dtype=np.float64)
    out = np.asarray(transforms[0], dtype=np.float64)
    for t in transforms[1:]:
        out = out @ np.asarray(t, dtype=np.float64)
    return out


def scale_delta_translation(delta: np.ndarray, scale: float) -> np.ndarray:
    """Return ``delta`` with its translation multiplied by ``scale``.

    Split out so the "scale the delta, not the target" rule is one testable
    thing rather than a line buried in :meth:`ClutchMapper.map`.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")
    out = np.array(delta, dtype=np.float64, copy=True)
    if out.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {out.shape}")
    out[:3, 3] *= scale
    return out


@dataclass(frozen=True)
class WorkspaceClampResult:
    """Outcome of clamping one target into an arm's reachable box."""

    pose: Pose
    was_clamped_axes: np.ndarray = field(repr=False)  # (3,) bool

    @property
    def was_clamped(self) -> bool:
        return bool(np.any(self.was_clamped_axes))

    @property
    def clamped_axis_names(self) -> tuple[str, ...]:
        """e.g. ``("x", "z")`` -- which walls the operator is pushing against."""
        return tuple(n for n, hit in zip("xyz", self.was_clamped_axes) if hit)


@dataclass(frozen=True)
class WorkspaceViolation:
    """A pose that sits outside an arm's configured box."""

    arm: str
    axes: tuple[str, ...]
    measured: tuple[float, float, float]
    clamped: tuple[float, float, float]

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(np.asarray(self.clamped) - np.asarray(self.measured)))

    def describe(self) -> str:
        parts = [
            f"{a}={self.measured[i]:+.4f} -> {self.clamped[i]:+.4f}"
            for i, a in enumerate("xyz") if a in self.axes
        ]
        return (f"{self.arm} arm is {self.distance_m * 1e3:.0f} mm outside the configured "
                f"workspace on {'/'.join(self.axes)}: " + ", ".join(parts))

    @property
    def looks_like_home(self) -> bool:
        """True when this looks like the parked/home posture.

        At home the arms hang beside the base: x near 0, z well below it. The
        workspace box starts ~0.25 m in FRONT of the base, so home is legitimately
        outside it. This is expected, not a fault -- the arms have to be brought
        forward before streaming.
        """
        return abs(self.measured[0]) < 0.10 and self.measured[2] < -0.60


def check_inside_workspace(config: Config, arm: Arm, pose: PoseLike) -> WorkspaceViolation | None:
    """Return a violation if ``pose`` is outside ``arm``'s box, else None.

    Call this on any pose that is meant to be HELD -- a measured pose, a frozen
    pose, a latch origin. The clamp is a guard against bad *targets*; applied to
    a pose the arm is already at, it stops being a guard and becomes an actuator,
    commanding the arm to the nearest legal point. On real hardware that walked
    an arm 29 cm into its stand.

    If this fires, the box and the robot disagree about reality. That is a
    question for a human, not something to clamp away.
    """
    p = as_pose(pose)
    clamped, axes = config.workspace.clamp_axes(arm, p.position_m)
    if not np.any(axes):
        return None
    return WorkspaceViolation(
        arm=arm,
        axes=tuple(a for a, hit in zip("xyz", axes) if hit),
        measured=tuple(float(v) for v in p.position_m),
        clamped=tuple(float(v) for v in clamped),
    )


def clamp_to_workspace(config: Config, arm: Arm, pose: PoseLike) -> WorkspaceClampResult:
    """Clamp a target's position into the arm's box, shrunk by the margin.

    Delegates to :meth:`tron_arm.config.Box.clamp_axes` -- the box geometry lives
    with the boxes. Orientation is never clamped: the vendor documents position
    limits only, and silently rotating a target would be a worse surprise than
    letting the robot refuse it.
    """
    p = as_pose(pose)
    clamped, axes = config.workspace.clamp_axes(arm, p.position_m)
    if not np.any(axes):
        return WorkspaceClampResult(p, axes)
    return WorkspaceClampResult(Pose(clamped, p.orientation_wxyz), axes)


# -- the mapper ----------------------------------------------------------
class ClutchMapper:
    """Maps operator wrist poses to robot targets against a latched origin pair.

    Lifecycle mirrors the clutch: :meth:`latch` on the first frame while engaged,
    :meth:`clear` on *every* exit from engaged (release, hold, reference change,
    fault). Mapping while unlatched is an error rather than a silent identity --
    a stale origin mapping a fresh frame is precisely the failure CLAUDE.md's
    lazy-latch rule exists to prevent.

    Args:
        config: supplies ``scale``, ``mapping.translation_frame`` and the
            axis-verification gate.
        tool_offset: optional ``T_FH``, flange -> palm. Rigid because the ORCA
            hand's 1-DoF wrist is commanded to 0 and is a fixed link for our
            kinematics. When set, origins and targets are handled in *palm*
            space and :meth:`map` returns the **flange** command
            ``T_BF = T_BH_target @ inv(T_FH)``.
    """

    def __init__(
        self,
        config: Config,
        *,
        tool_offset: PoseLike | np.ndarray | None = None,
    ) -> None:
        frame = config.mapping.translation_frame
        if frame == "world" and not config.mapping.world_frame_axes_verified:
            # Belt and braces: Config already refuses this combination, but the
            # mapper is constructible from a hand-built Config in tests and the
            # gate must not depend on which door you came in through.
            raise MappingError(
                "mapping.translation_frame='world' requires mapping."
                "world_frame_axes_verified: true. World mode assumes the operator's "
                "reference axes and the robot base axes coincide, which is only true "
                "once the guide 9G-09 axis check has been run on hardware -- until "
                "then it silently mirrors or swaps operator motion."
            )
        self._config = config
        self._scale = float(config.scale)
        self._frame: TranslationFrame = frame
        self._tool_offset = _as_matrix(tool_offset) if tool_offset is not None else None
        self._tool_offset_inv = (
            invert_transform(self._tool_offset) if self._tool_offset is not None else None
        )
        self._t_op0: np.ndarray | None = None
        self._t_robot0: np.ndarray | None = None

    # -- properties ------------------------------------------------------
    @property
    def latched(self) -> bool:
        return self._t_op0 is not None and self._t_robot0 is not None

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def translation_frame(self) -> TranslationFrame:
        return self._frame

    @property
    def has_tool_offset(self) -> bool:
        return self._tool_offset is not None

    @property
    def t_op0(self) -> np.ndarray | None:
        """The latched operator origin, or None. Copy; mutating it does nothing."""
        return None if self._t_op0 is None else self._t_op0.copy()

    @property
    def t_robot0(self) -> np.ndarray | None:
        """The latched robot origin in *palm* space when a tool offset is set."""
        return None if self._t_robot0 is None else self._t_robot0.copy()

    @property
    def translation_prerotation(self) -> np.ndarray | None:
        """``M = R_r0 @ R_o0.T`` -- how operator translation maps to base axes.

        None until latched. See REFERENCE.md: this is the whole of the
        "which way does my hand move the arm" question, and it is fixed at the
        instant of engage.
        """
        if not self.latched:
            return None
        return self._t_robot0[:3, :3] @ self._t_op0[:3, :3].T

    # -- lifecycle -------------------------------------------------------
    def latch(self, t_op0: PoseLike | np.ndarray, t_robot0: PoseLike | np.ndarray) -> None:
        """Capture the origin pair.

        Args:
            t_op0: ``frame.wrist.matrix`` at the moment of the latch.
            t_robot0: the robot origin -- the last COMMANDED target if already
                streaming, else ``request_get_move_pose``. Using the last command
                rather than the measured pose is what keeps the target stream
                continuous across a re-latch: the arm is still converging on the
                last command, so latching the *measurement* would bake its
                tracking error in as a step.

        With a tool offset set, ``t_robot0`` is a **flange** pose and is
        converted to palm space internally, so a re-latch is symmetric with the
        flange command :meth:`map` returns.
        """
        op0 = _as_matrix(t_op0)
        robot0 = _as_matrix(t_robot0)
        if self._tool_offset is not None:
            robot0 = robot0 @ self._tool_offset  # T_BH0 = T_BF0 @ T_FH
        self._t_op0 = op0
        self._t_robot0 = robot0

    def clear(self) -> None:
        """Drop the origins. Idempotent."""
        self._t_op0 = None
        self._t_robot0 = None

    # -- the mapping -----------------------------------------------------
    def map(self, t_op: PoseLike | np.ndarray) -> Pose:
        """Map one operator wrist pose to a robot target.

        Returns the **flange** target: with a tool offset that is
        ``T_BH_target @ inv(T_FH)``, without one it is the target directly.
        """
        if not self.latched:
            raise MappingError(
                "map() before latch(): no origin pair. Origins latch lazily on the "
                "first frame while engaged and are cleared on every exit from "
                "engaged; mapping without them would apply a stale origin to a "
                "fresh frame."
            )
        op = _as_matrix(t_op)
        target = self._map_body(op) if self._frame == "body" else self._map_world(op)
        if self._tool_offset_inv is not None:
            target = target @ self._tool_offset_inv  # T_BF = T_BH_target @ inv(T_FH)
        return Pose.from_matrix(target)

    def _map_body(self, t_op: np.ndarray) -> np.ndarray:
        """``T_robot0 @ scaled(inv(T_op0) @ T_op)`` -- the default."""
        delta = invert_transform(self._t_op0) @ t_op
        delta = scale_delta_translation(delta, self._scale)
        return self._t_robot0 @ delta

    def _map_world(self, t_op: np.ndarray) -> np.ndarray:
        """Translation taken raw in the reference frame; rotation as body mode.

        ``p_target = p_r0 + s * (p_op - p_op0)``, with the *same* rotation the
        body formula produces. The only difference is that translation skips the
        ``M = R_r0 @ R_o0.T`` pre-rotation, so operator axes drive base axes
        directly -- which is why it is gated behind the axis check.
        """
        rotation = self._t_robot0[:3, :3] @ self._t_op0[:3, :3].T @ t_op[:3, :3]
        position = self._t_robot0[:3, 3] + self._scale * (t_op[:3, 3] - self._t_op0[:3, 3])
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rotation
        out[:3, 3] = position
        return out

    def map_clamped(self, t_op: PoseLike | np.ndarray, arm: Arm) -> WorkspaceClampResult:
        """:meth:`map` followed by the workspace clamp, in the documented order."""
        return clamp_to_workspace(self._config, arm, self.map(t_op))


def _as_matrix(value: PoseLike | np.ndarray) -> np.ndarray:
    """Coerce a Pose (ours or upstream's) or a raw 4x4 into a 4x4 array."""
    if isinstance(value, np.ndarray):
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"expected a 4x4 transform, got {matrix.shape}")
        return matrix.copy()
    return as_pose(value).matrix
