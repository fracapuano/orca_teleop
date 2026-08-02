"""Lightweight hand-model shim: the retargeter's view of an ORCA hand config.

The retargeting stack only needs four facts from a hand model — its side,
joint ids, ROMs, and neutral pose. Constructing a full ``orca_core.OrcaHand``
for that couples the streamer to whatever orca_core revision happens to be in
this environment, while the config.yaml may come from a *different* orca_core
branch (e.g. orca_ui hands over its own model path). This shim parses the
YAML directly, so the streamer path needs no orca_core at all.

Duck-type contract (mirrors ``OrcaHand``): ``.config.type``,
``.config.joint_ids``, ``.config.joint_roms_dict``,
``.config.neutral_position``, ``.config.model_path``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class HandModelConfig:
    type: str
    joint_ids: list[str]
    joint_roms_dict: dict[str, list[float]]
    neutral_position: dict[str, float] = field(default_factory=dict)
    model_path: str | None = None


@dataclass(frozen=True)
class HandModel:
    config: HandModelConfig

    @property
    def model_name(self) -> str:
        if self.config.model_path:
            return os.path.basename(self.config.model_path.rstrip("/"))
        return "unknown"

    @classmethod
    def load(cls, config_path: str) -> HandModel:
        """Parse a hand ``config.yaml`` (or the directory containing one)."""
        if os.path.isdir(config_path):
            config_path = os.path.join(config_path, "config.yaml")
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        hand_type = raw.get("type")
        if hand_type not in ("left", "right"):
            raise ValueError(
                f"config 'type' must be 'left' or 'right', got {hand_type!r} "
                f"({config_path})"
            )
        joint_ids = list(raw.get("joint_ids") or [])
        if not joint_ids:
            raise ValueError(f"config has no joint_ids ({config_path})")
        roms_raw = raw.get("joint_roms") or {}
        joint_roms = {
            joint: [float(v) for v in rom] for joint, rom in roms_raw.items()
        }
        missing = [j for j in joint_ids if j not in joint_roms]
        if missing:
            raise ValueError(
                f"config joint_roms missing entries for {missing} ({config_path})"
            )
        neutral = {
            joint: float(v)
            for joint, v in (raw.get("neutral_position") or {}).items()
        }
        return cls(HandModelConfig(
            type=hand_type,
            joint_ids=joint_ids,
            joint_roms_dict=joint_roms,
            neutral_position=neutral,
            model_path=os.path.dirname(os.path.abspath(config_path)),
        ))
