"""Typed configuration for the TRON 2 arm layer, loaded from YAML.

Every value the vendor documentation leaves ambiguous is a config knob rather
than a code constant, so that a wrong guess is fixed by editing YAML instead of
editing (and re-testing) the encoder -- see CLAUDE.md "TRON 2 protocol ground
truth". Unknown keys are rejected so a typo fails loudly instead of silently
leaving the default in place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import yaml

__all__ = [
    "ARMS",
    "Arm",
    "Box",
    "Config",
    "ConfigError",
    "JOINT_LOWER",
    "JOINT_UPPER",
    "MappingConfig",
    "N_JOINTS",
    "RobotConfig",
    "ServopConfig",
    "ServopFormat",
    "VelocityConfig",
    "WorkspaceConfig",
    "default_config_path",
    "load_config",
]

Arm = Literal["left", "right"]
ServopFormat = Literal["pos_quat", "pos_rotmat"]
ARMS: tuple[Arm, Arm] = ("left", "right")

#: Joints are ordered left 0..6 then right 7..13 (LimX SDK guide V0.2).
N_JOINTS = 14

JOINT_UPPER = np.array(
    [2.6005, 3.1940, 1.4835, 0.2618, 1.3963, 0.7854, 1.5708,
     2.6005, 0.2618, 3.6652, 0.2618, 1.7453, 0.7854, 1.5708],
    dtype=np.float64,
)
JOINT_LOWER = np.array(
    [-3.1416, -0.2618, -3.6652, -2.6180, -1.7453, -0.7854, -1.5708,
     -3.1416, -3.1940, -1.4835, -2.6180, -1.3963, -0.7854, -1.5708],
    dtype=np.float64,
)

#: Element counts per arm for each servop encoding.
SERVOP_WIDTH: dict[str, int] = {"pos_quat": 7, "pos_rotmat": 12}


class ConfigError(ValueError):
    """Raised for a malformed or out-of-range configuration."""


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _reject_unknown(mapping: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    extra = sorted(set(mapping) - set(allowed))
    if extra:
        raise ConfigError(f"{where}: unknown key(s) {extra} (allowed: {sorted(allowed)})")


def _as_float(value: Any, where: str, *, positive: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: expected a number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ConfigError(f"{where}: expected a finite number, got {value!r}")
    if positive and out <= 0.0:
        raise ConfigError(f"{where}: expected a positive number, got {out!r}")
    return out


@dataclass(frozen=True)
class Box:
    """An axis-aligned workspace box in the robot base frame (FLU, metres)."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            lo, hi = getattr(self, axis)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ConfigError(f"workspace {axis}: bounds must be finite, got {(lo, hi)!r}")
            if lo >= hi:
                raise ConfigError(f"workspace {axis}: lower {lo} must be < upper {hi}")

    @property
    def bounds(self) -> np.ndarray:
        """``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]``."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str) -> "Box":
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{where}: expected a mapping with x/y/z, got {type(raw).__name__}")
        _reject_unknown(raw, ("x", "y", "z"), where)
        axes = {}
        for axis in ("x", "y", "z"):
            pair = _require(raw, axis, where)
            if not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2:
                raise ConfigError(f"{where}.{axis}: expected [lo, hi], got {pair!r}")
            axes[axis] = (
                _as_float(pair[0], f"{where}.{axis}[0]"),
                _as_float(pair[1], f"{where}.{axis}[1]"),
            )
        return cls(**axes)

    def clamp_axes(self, p: np.ndarray, margin_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Clamp ``p`` into the box shrunk by ``margin_m`` on every face.

        Returns ``(clamped, was_clamped_axes)`` where the second element is a
        3-vector of bools, one per axis. Knowing *which* axis hit the wall is
        what tells an operator they are pushing against the edge of the
        workspace rather than the arm having simply stopped.

        If the margin would collapse an axis, the clamp targets the box centre
        on that axis rather than producing an inverted interval.
        """
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if p.shape != (3,):
            raise ValueError(f"expected a 3-vector, got {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError(f"refusing to clamp a non-finite position {p!r}")
        b = self.bounds
        lo = b[:, 0] + margin_m
        hi = b[:, 1] - margin_m
        collapsed = lo >= hi
        if np.any(collapsed):
            mid = 0.5 * (b[:, 0] + b[:, 1])
            lo = np.where(collapsed, mid, lo)
            hi = np.where(collapsed, mid, hi)
        out = np.clip(p, lo, hi)
        return out, out != p

    def clamp(self, p: np.ndarray, margin_m: float = 0.0) -> tuple[np.ndarray, bool]:
        """As :meth:`clamp_axes`, collapsed to a single ``was_clamped`` bool."""
        out, axes = self.clamp_axes(p, margin_m)
        return out, bool(np.any(axes))


@dataclass(frozen=True)
class WorkspaceConfig:
    """Per-arm reachable boxes plus the safety margin held off every face.

    MoveP rejects out-of-range targets; ServoP's behaviour is undocumented, so
    we always clamp before encoding (CLAUDE.md hard rule 3).
    """

    left: Box
    right: Box
    margin_m: float

    def box(self, arm: str) -> Box:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        return self.left if arm == "left" else self.right

    def clamp(self, arm: str, p: np.ndarray) -> tuple[np.ndarray, bool]:
        return self.box(arm).clamp(p, self.margin_m)

    def clamp_axes(self, arm: str, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-axis variant of :meth:`clamp`, for diagnostics."""
        return self.box(arm).clamp_axes(p, self.margin_m)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "workspace") -> "WorkspaceConfig":
        _reject_unknown(raw, ("left", "right", "margin_m"), where)
        margin = _as_float(_require(raw, "margin_m", where), f"{where}.margin_m")
        if margin < 0.0:
            raise ConfigError(f"{where}.margin_m: must be >= 0, got {margin}")
        return cls(
            left=Box.from_mapping(_require(raw, "left", where), f"{where}.left"),
            right=Box.from_mapping(_require(raw, "right", where), f"{where}.right"),
            margin_m=margin,
        )


@dataclass(frozen=True)
class ServopConfig:
    """How to encode and pace ``request_servop``.

    ``format`` and ``send_both`` both encode genuine ambiguities in the vendor
    doc; flipping them must never require a code change.
    """

    format: ServopFormat
    rate_hz: float
    send_both: bool

    def __post_init__(self) -> None:
        if self.format not in SERVOP_WIDTH:
            raise ConfigError(
                f"servop.format: expected one of {sorted(SERVOP_WIDTH)}, got {self.format!r}"
            )
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ConfigError(f"servop.rate_hz: must be > 0, got {self.rate_hz!r}")
        if self.rate_hz > 1000.0:
            raise ConfigError(f"servop.rate_hz: {self.rate_hz} Hz is implausibly high (max 1000)")

    @property
    def width(self) -> int:
        """Number of floats per arm under the configured encoding."""
        return SERVOP_WIDTH[self.format]

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def period_ns(self) -> int:
        return int(round(1e9 / self.rate_hz))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "servop") -> "ServopConfig":
        _reject_unknown(raw, ("format", "rate_hz", "send_both"), where)
        send_both = _require(raw, "send_both", where)
        if not isinstance(send_both, bool):
            raise ConfigError(f"{where}.send_both: expected a bool, got {send_both!r}")
        return cls(
            format=str(_require(raw, "format", where)),  # type: ignore[arg-type]
            rate_hz=_as_float(_require(raw, "rate_hz", where), f"{where}.rate_hz", positive=True),
            send_both=send_both,
        )


@dataclass(frozen=True)
class VelocityConfig:
    """Cartesian velocity ceilings used to derive the per-tick step clamp."""

    lin: float  # m/s
    ang: float  # rad/s

    def __post_init__(self) -> None:
        for name in ("lin", "ang"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0.0:
                raise ConfigError(f"velocity.{name}: must be > 0, got {v!r}")

    def max_step(self, rate_hz: float) -> tuple[float, float]:
        """Largest ``(linear_m, angular_rad)`` change permitted in one tick."""
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        return self.lin / rate_hz, self.ang / rate_hz

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "velocity") -> "VelocityConfig":
        _reject_unknown(raw, ("lin", "ang"), where)
        return cls(
            lin=_as_float(_require(raw, "lin", where), f"{where}.lin", positive=True),
            ang=_as_float(_require(raw, "ang", where), f"{where}.ang", positive=True),
        )


@dataclass(frozen=True)
class MappingConfig:
    """Operator-to-robot mapping options (consumed by the sink, next layer up)."""

    translation_frame: Literal["body", "world"] = "body"
    world_frame_axes_verified: bool = False

    def __post_init__(self) -> None:
        if self.translation_frame not in ("body", "world"):
            raise ConfigError(
                f"mapping.translation_frame: expected 'body' or 'world', "
                f"got {self.translation_frame!r}"
            )
        if self.translation_frame == "world" and not self.world_frame_axes_verified:
            raise ConfigError(
                "mapping.translation_frame='world' is gated behind "
                "mapping.world_frame_axes_verified: set it to true only after "
                "confirming the operator/robot axis correspondence on hardware"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "mapping") -> "MappingConfig":
        _reject_unknown(raw, ("translation_frame", "world_frame_axes_verified"), where)
        verified = raw.get("world_frame_axes_verified", False)
        if not isinstance(verified, bool):
            raise ConfigError(f"{where}.world_frame_axes_verified: expected a bool, got {verified!r}")
        return cls(
            translation_frame=str(raw.get("translation_frame", "body")),  # type: ignore[arg-type]
            world_frame_axes_verified=verified,
        )


@dataclass(frozen=True)
class RobotConfig:
    """Connection parameters. ``url`` points at the mock by default."""

    url: str
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.startswith(("ws://", "wss://")):
            raise ConfigError(f"robot.url: expected a ws:// or wss:// URL, got {self.url!r}")
        for name in ("connect_timeout_s", "request_timeout_s"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0.0:
                raise ConfigError(f"robot.{name}: must be > 0, got {v!r}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], where: str = "robot") -> "RobotConfig":
        _reject_unknown(raw, ("url", "connect_timeout_s", "request_timeout_s"), where)
        return cls(
            url=_require(raw, "url", where),
            connect_timeout_s=_as_float(
                raw.get("connect_timeout_s", 5.0), f"{where}.connect_timeout_s", positive=True
            ),
            request_timeout_s=_as_float(
                raw.get("request_timeout_s", 2.0), f"{where}.request_timeout_s", positive=True
            ),
        )


@dataclass(frozen=True)
class Config:
    """Fully validated configuration tree."""

    robot: RobotConfig
    servop: ServopConfig
    workspace: WorkspaceConfig
    velocity: VelocityConfig
    mapping: MappingConfig
    scale: float
    home_joints: np.ndarray = field(repr=False)
    #: Optional joint posture that puts BOTH flanges inside the workspace box.
    #: Home is the documented zero configuration, which on a real TRON 2 hangs
    #: the arms beside the base -- outside the box, so streaming cannot start
    #: from there. Capture a good posture once with `tron2_cli capture-ready`.
    ready_joints: np.ndarray | None = field(default=None, repr=False)
    notify_log_path: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ConfigError(f"scale: must be > 0, got {self.scale!r}")
        j = np.asarray(self.home_joints, dtype=np.float64).reshape(-1)
        if j.shape != (N_JOINTS,):
            raise ConfigError(f"home.joints: expected {N_JOINTS} values, got {j.shape[0]}")
        if not np.all(np.isfinite(j)):
            raise ConfigError("home.joints: all values must be finite")
        bad = np.nonzero((j < JOINT_LOWER) | (j > JOINT_UPPER))[0]
        if bad.size:
            details = ", ".join(
                f"j{i}={j[i]:.4f} not in [{JOINT_LOWER[i]:.4f}, {JOINT_UPPER[i]:.4f}]" for i in bad
            )
            raise ConfigError(f"home.joints: outside the documented limits -- {details}")
        j.flags.writeable = False
        object.__setattr__(self, "home_joints", j)

        if self.ready_joints is not None:
            r = np.asarray(self.ready_joints, dtype=np.float64).reshape(-1)
            if r.shape != (N_JOINTS,):
                raise ConfigError(f"ready.joints: expected {N_JOINTS} values, got {r.shape[0]}")
            if not np.all(np.isfinite(r)):
                raise ConfigError("ready.joints: all values must be finite")
            bad = np.nonzero((r < JOINT_LOWER) | (r > JOINT_UPPER))[0]
            if bad.size:
                details = ", ".join(
                    f"j{i}={r[i]:.4f} not in [{JOINT_LOWER[i]:.4f}, {JOINT_UPPER[i]:.4f}]"
                    for i in bad)
                raise ConfigError(f"ready.joints: outside the documented limits -- {details}")
            r.flags.writeable = False
            object.__setattr__(self, "ready_joints", r)

    @property
    def max_step(self) -> tuple[float, float]:
        """Per-tick ``(linear_m, angular_rad)`` ceiling at the configured rate."""
        return self.velocity.max_step(self.servop.rate_hz)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Config":
        if not isinstance(raw, Mapping):
            raise ConfigError(f"top level: expected a mapping, got {type(raw).__name__}")
        allowed = (
            "robot", "servop", "workspace", "velocity", "mapping", "scale", "home",
            "ready", "logging",
        )
        _reject_unknown(raw, allowed, "top level")
        home = _require(raw, "home", "top level")
        _reject_unknown(home, ("joints",), "home")
        ready = raw.get("ready") or {}
        _reject_unknown(ready, ("joints",), "ready")
        logging_raw = raw.get("logging", {}) or {}
        _reject_unknown(logging_raw, ("notify_jsonl",), "logging")
        return cls(
            robot=RobotConfig.from_mapping(_require(raw, "robot", "top level")),
            servop=ServopConfig.from_mapping(_require(raw, "servop", "top level")),
            workspace=WorkspaceConfig.from_mapping(_require(raw, "workspace", "top level")),
            velocity=VelocityConfig.from_mapping(_require(raw, "velocity", "top level")),
            mapping=MappingConfig.from_mapping(raw.get("mapping", {}) or {}),
            scale=_as_float(_require(raw, "scale", "top level"), "scale", positive=True),
            home_joints=np.asarray(_require(home, "joints", "home"), dtype=np.float64),
            ready_joints=(np.asarray(ready["joints"], dtype=np.float64)
                          if ready.get("joints") is not None else None),
            notify_log_path=logging_raw.get("notify_jsonl"),
        )


def default_config_path() -> Path:
    """``configs/default.yaml`` next to the installed package."""
    return Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate a YAML config (defaults to :func:`default_config_path`)."""
    p = Path(path) if path is not None else default_config_path()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse config {p}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"config {p} is empty")
    try:
        return Config.from_mapping(raw)
    except ConfigError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
