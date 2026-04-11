"""Load per-joint reference offsets from a MuJoCo MJCF companion file.

The URDF zero pose is *not* the robot's physical zero pose: the MJCF declares a
``ref`` attribute on each joint giving the angle (radians or degrees, depending
on ``<compiler angle="..."/>``) of the physical zero in URDF space. The IK
optimizer runs in URDF space, then we add the ref offset back when emitting
``OrcaJointPositions``.

The orcahand_description repository ships an MJCF next to each URDF (e.g.
``v1/models/urdf/orcahand_right.urdf`` ↔ ``v1/models/mjcf/orcahand_right.mjcf``)
so this loader resolves it by string substitution and falls back to a sibling
search.

TODO: Migrate the retargeter to MJCF as its *sole* model source and delete
this module. This loader only exists because URDF has no concept of a rest
pose that isn't zero — MJCF does, natively, via the ``ref`` attribute on
each joint, which is why we're parsing MJCF as a sidecar in the first place.
Keeping two files in sync is the split-brain the whole module is papering
over.

The migration:

  1. In ``RetargeterConfig.from_paths`` replace
     ``pk.build_chain_from_urdf(urdf_text)`` with
     ``pk.build_chain_from_mjcf_path(mjcf_path)`` (pytorch_kinematics supports
     MJCF natively and returns the same ``Chain`` object — downstream FK code
     is format-agnostic).
  2. Read the ``ref`` attribute directly off the same parsed tree instead of
     resolving a companion file. The ``_resolve_companion_mjcf`` dance and
     the ``load_ref_offsets`` wrapper both disappear.
  3. Drop ``urdf_path`` from the public API; accept ``mjcf_path`` instead.
     The URDF can stay on disk for external consumers (viewers, ROS) but the
     retargeter no longer depends on it.

Before switching, verify on the orcahand MJCF:

  - ``pk.build_chain_from_mjcf_path`` parses the file without choking on
    ``<default>``, ``<asset>``, or ``<include>`` directives.
  - ``chain.get_joint_parameter_names()`` matches the OrcaHand config's
    prefixed joint names (same assertion as today).
  - ``<default>`` inheritance for ``ref`` is resolved correctly — in the
    orcahand file most finger joints have no explicit ``ref`` and inherit 0
    from the class default, which is the behaviour we want.
  - The ``<compiler angle="...">`` unit setting is honoured: MJCF files can
    ship in either radians or degrees, and whichever code reads ``ref`` must
    normalise accordingly.
  - MJCF joint frames are expressed relative to the parent *body*, not
    globally — ``pk`` handles this internally but hand-checked transforms
    will look different from their URDF equivalents.
"""

import logging
import math
import os
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _normalize_angle(rad: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi]``."""
    rad = rad % (2 * math.pi)
    if rad > math.pi:
        rad -= 2 * math.pi
    return rad


def _resolve_companion_mjcf(urdf_path: str) -> str | None:
    """Find the MJCF file that pairs with *urdf_path*, or return None.

    First tries the orcahand_description layout (``.../urdf/X.urdf`` →
    ``.../mjcf/X.mjcf``). Then falls back to a same-basename search in
    sibling directories of the URDF.
    """
    urdf_path = os.path.abspath(urdf_path)
    base = os.path.splitext(os.path.basename(urdf_path))[0]

    # Layout: .../urdf/X.urdf ↔ .../mjcf/X.mjcf
    parent = os.path.dirname(urdf_path)
    grandparent = os.path.dirname(parent)
    if os.path.basename(parent) == "urdf":
        candidate = os.path.join(grandparent, "mjcf", f"{base}.mjcf")
        if os.path.exists(candidate):
            return candidate

    # Fallback: any sibling .mjcf with the same basename
    for sibling in os.listdir(grandparent) if os.path.isdir(grandparent) else []:
        candidate = os.path.join(grandparent, sibling, f"{base}.mjcf")
        if os.path.exists(candidate):
            return candidate
    return None


def load_ref_offsets_from_mjcf(mjcf_path: str, hand_type: str) -> dict[str, float]:
    """Parse a MuJoCo MJCF and return ``{bare_joint_id: ref_radians}``.

    Joint names in the MJCF are prefixed with the hand type (``right_thumb_mcp``);
    the prefix is stripped so the result keys match ``OrcaHand.config.joint_ids``.
    The wrist offset is forced to zero — the wrist is a passthrough and the
    physical wrist's zero already coincides with the URDF wrist's zero.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler_el = root.find("compiler")
    angle_unit = "degree"
    if compiler_el is not None:
        angle_unit = compiler_el.get("angle", "degree")

    prefix = f"{hand_type}_"
    offsets: dict[str, float] = {}
    for joint_el in root.iter("joint"):
        name = joint_el.get("name", "")
        if not name.startswith(prefix):
            continue
        bare = name[len(prefix) :]
        ref_val = float(joint_el.get("ref", "0"))
        ref_rad = ref_val if angle_unit == "radian" else math.radians(ref_val)
        offsets[bare] = _normalize_angle(ref_rad)

    offsets["wrist"] = 0.0
    return offsets


def load_ref_offsets(urdf_path: str, hand_type: str) -> dict[str, float] | None:
    """Best-effort: locate the companion MJCF and parse its ref offsets.

    Returns ``None`` if no MJCF can be found. Callers should fall back to
    all-zero offsets in that case (and warn — the IK will then be biased by
    whatever the URDF's zero pose is).
    """
    mjcf_path = _resolve_companion_mjcf(urdf_path)
    if mjcf_path is None:
        logger.warning(
            "No companion MJCF found for %s — physical/URDF zero offsets will be "
            "all zero, which biases finger angles by the URDF rest pose.",
            urdf_path,
        )
        return None
    return load_ref_offsets_from_mjcf(mjcf_path, hand_type)
