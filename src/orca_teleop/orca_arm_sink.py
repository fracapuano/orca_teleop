"""OrcaArm visualization backends.

Renders the OrcaArm URDF in meshcat and overlays per-side target /
current-EE triads. Consumes pre-solved arm joint angles (from
:mod:`orca_teleop.orca_arm_ik`) plus optional hand joint positions from the
retargeter — no IK or kinematics math lives here.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import meshcat
import meshcat.geometry as g
import mujoco
import numpy as np
import orca_arm
import yourdfpy
from orca_core import OrcaJointPositions

from orca_teleop.orca_arm_ik import ARM_JOINTS_PER_SIDE, CARPALS_SIDE_PREFIX, SIDES

logger = logging.getLogger(__name__)

# Retargeter joint IDs -> generated orcabot URDF joint-name fragments. The
# OrcaArm URDF embeds OrcaHand joints with CAD-derived names, so we resolve
# actual indices by substring instead of depending on the full hashy names.
_HAND_JOINT_MARKERS = {
    "thumb_mcp": {"left": "T-TP-L_92b8100b_to_", "right": "T-TP-R_1c2b802d_to_"},
    "thumb_abd": {"left": "L-T-AP_58680c44_to_", "right": "R-T-AP_a9723101_to_"},
    "thumb_cmc": {"left": "T-PP_ef067304_to_", "right": "T-PP_68395e98_to_"},
    "thumb_pip": {"left": "T-PP_ef067304_to_", "right": "T-PP_68395e98_to_"},
    "thumb_dip": {"left": "T-DP_307db3cc_to_", "right": "T-DP_b7429e50_to_"},
    "index_abd": {"left": "I-AP-L_57ce92f7_to_", "right": "I-AP-R_d95d02d1_to_"},
    "index_mcp": {"left": "I-PP_3df4f91d_to_", "right": "I-PP_bacbd481_to_"},
    "index_pip": {
        "left": "I-FingerTipAssembly_ed91b18a_to_",
        "right": "I-FingerTipAssembly_ec49c16c_to_",
    },
    "middle_abd": {"left": "M-AP_e04a96f2_to_", "right": "M-AP_e04a96f2_to_"},
    "middle_mcp": {"left": "M-PP_08efa608_to_", "right": "M-PP_08efa608_to_"},
    "middle_pip": {
        "left": "M-FingerTipAssembly_34afb748_to_",
        "right": "M-FingerTipAssembly_34afb748_to_",
    },
    "ring_abd": {"left": "M-AP_6ec59111_to_", "right": "M-AP_6ec59111_to_"},
    "ring_mcp": {"left": "M-PP_8660a1eb_to_", "right": "M-PP_8660a1eb_to_"},
    "ring_pip": {
        "left": "M-FingerTipAssembly_424a8e75_to_",
        "right": "M-FingerTipAssembly_424a8e75_to_",
    },
    "pinky_abd": {"left": "P-AP_f5e42b61_to_", "right": "P-AP_f5e42b61_to_"},
    "pinky_mcp": {"left": "P-PP_1d411b9b_to_", "right": "P-PP_1d411b9b_to_"},
    "pinky_pip": {
        "left": "P-FingerTipAssembly_cd219176_to_",
        "right": "P-FingerTipAssembly_cd219176_to_",
    },
}


_TRIAD_AXIS_LEN = 0.10
_TRIAD_AXIS_R = 0.004

_AXIS_SPECS = (
    ("x", np.array([1.0, 0.0, 0.0]), 0xFF0000),
    ("y", np.array([0.0, 1.0, 0.0]), 0x00FF00),
    ("z", np.array([0.0, 0.0, 1.0]), 0x0000FF),
)


def _axis_local_transform(axis_dir: np.ndarray, length: float) -> np.ndarray:
    """4x4 transform placing a +Y cylinder along *axis_dir* for *length* m."""
    axis = np.asarray(axis_dir, dtype=float)
    y = np.array([0.0, 1.0, 0.0])
    if np.allclose(axis, y):
        R = np.eye(3)
    elif np.allclose(axis, -y):
        R = np.diag([1.0, -1.0, -1.0])
    else:
        v = np.cross(y, axis)
        s = float(np.linalg.norm(v))
        c = float(np.dot(y, axis))
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = axis * (length / 2)
    return T


_AXIS_LOCAL_T = {name: _axis_local_transform(d, _TRIAD_AXIS_LEN) for name, d, _ in _AXIS_SPECS}


def _find_mujoco_joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {name}")
    return int(joint_id)


def _find_mujoco_body_id_by_prefix(model: mujoco.MjModel, prefix: str) -> int:
    matches = [
        body_id
        for body_id in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        and name.startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one MuJoCo body with prefix {prefix!r}, got {matches}")
    return int(matches[0])


def _homogeneous_from_mujoco_body(data: mujoco.MjData, body_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(data.xpos[body_id], dtype=np.float64)
    return T


class OrcaArmMeshcatSink:
    """Meshcat-based visualizer for solved arm/hand states.

    Defaults load the bimanual OrcaArm URDF, but callers may pass an
    ``ArmIKConfig``-compatible URDF, side list, joint mapping, and EE frames to
    render other embodiments that share the same teleop result shape.
    """

    def __init__(
        self,
        *,
        urdf_path: str | Path | None = None,
        sides: Sequence[str] = SIDES,
        joint_names_by_side: Mapping[str, Sequence[str]] | None = None,
        ee_frame_by_side: Mapping[str, str] | None = None,
        home_arm_angles: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._sides = tuple(sides)
        self._robot = yourdfpy.URDF.load(str(urdf_path or orca_arm.URDF_PATH))
        self._scene = self._robot.scene

        self._actuated_names = list(self._robot.actuated_joint_names)

        if joint_names_by_side is None:
            joint_names_by_side = {
                side: [f"openarm_{side}_joint{i}" for i in range(1, ARM_JOINTS_PER_SIDE + 1)]
                for side in self._sides
            }
        ee_frame_by_side = dict(ee_frame_by_side or {})

        # Per-side: cfg indices and EE scene-graph node
        self._arm_cfg_indices: dict[str, list[int]] = {}
        self._hand_cfg_indices: dict[str, dict[str, int]] = {}
        self._ee_links: dict[str, str] = {}
        for side in self._sides:
            missing = [
                joint_name
                for joint_name in joint_names_by_side[side]
                if joint_name not in self._actuated_names
            ]
            if missing:
                raise ValueError(
                    f"URDF {urdf_path or orca_arm.URDF_PATH!r} is missing actuated "
                    f"joint(s) for {side}: {missing}"
                )
            self._arm_cfg_indices[side] = [
                self._actuated_names.index(joint_name) for joint_name in joint_names_by_side[side]
            ]
            self._hand_cfg_indices[side] = self._resolve_hand_joint_indices(side)
            self._ee_links[side] = self._resolve_ee_link(
                side,
                ee_frame_by_side.get(side),
                joint_names_by_side[side],
            )

        self._vis: meshcat.Visualizer | None = None
        self._geom_map: dict[str, str] = {}

        # Immutable neutral pose owned by the sink. Callers cannot rebind it:
        # :meth:`to_neutral_configuration` accepts an ``arm_angles`` argument as
        # a one-shot render override only.
        self._q_home: dict[str, np.ndarray] = {
            side: np.asarray(
                (
                    home_arm_angles[side]
                    if home_arm_angles is not None and side in home_arm_angles
                    else np.zeros(len(self._arm_cfg_indices[side]), dtype=np.float64)
                ),
                dtype=np.float64,
            )
            for side in self._sides
        }

    @property
    def arm_joint_names(self) -> dict[str, list[str]]:
        """Joint names at each entry of ``self._arm_cfg_indices[side]``."""
        return {
            side: [self._actuated_names[idx] for idx in indices]
            for side, indices in self._arm_cfg_indices.items()
        }

    def _resolve_ee_link(
        self,
        side: str,
        requested_frame: str | None,
        joint_names: Sequence[str],
    ) -> str:
        if requested_frame is not None and requested_frame in self._scene.graph.nodes:
            return requested_frame
        if requested_frame is not None:
            logger.warning(
                "Requested %s EE frame %r was not found in the Meshcat URDF scene; "
                "falling back to carpals auto-detection.",
                side,
                requested_frame,
            )
        return next(
            (
                n
                for n in self._scene.graph.nodes
                if f"orcahand_{side}_" in n and "Carpals" in n and "to_" not in n
            ),
            f"{joint_names[-1].rsplit('_', 1)[0]}_link{len(joint_names)}",
        )

    def _resolve_hand_joint_indices(self, side: str) -> dict[str, int]:
        prefix = f"orcahand_{side}_"
        out: dict[str, int] = {}

        wrist_matches = [
            i
            for i, name in enumerate(self._actuated_names)
            if name.startswith(f"{prefix}{CARPALS_SIDE_PREFIX[side]}-Carpals_")
            and "_to_TopTower-Model_" in name
        ]
        if len(wrist_matches) == 1:
            out["wrist"] = wrist_matches[0]
        else:
            logger.warning(
                "Could not resolve %s hand joint wrist in OrcaArm URDF (matches=%d)",
                side,
                len(wrist_matches),
            )

        for joint_id, side_markers in _HAND_JOINT_MARKERS.items():
            marker = side_markers[side]
            matches = [
                i
                for i, name in enumerate(self._actuated_names)
                if name.startswith(prefix) and marker in name
            ]
            if len(matches) != 1:
                logger.warning(
                    "Could not resolve %s hand joint %s in OrcaArm URDF (matches=%d)",
                    side,
                    joint_id,
                    len(matches),
                )
                continue
            out[joint_id] = matches[0]
        return out

    def launch(self) -> None:
        self._vis = meshcat.Visualizer()
        self._vis.open()
        self._vis.delete()
        self._load_robot_meshes()
        for side in self._sides:
            self._create_triads(side)
        logger.info("Meshcat viewer: %s", self._vis.url())

    def _load_robot_meshes(self) -> None:
        """Load all URDF visual meshes into meshcat (runs once)."""
        scene = self._scene
        for name in scene.graph.nodes_geometry:
            try:
                transform, geometry_name = scene.graph.get(name)
            except Exception:
                continue
            if geometry_name is None or geometry_name not in scene.geometry:
                continue
            mesh = scene.geometry[geometry_name]
            if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
                continue

            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)

            color = 0xCCCCCC
            if hasattr(mesh, "visual") and hasattr(mesh.visual, "main_color"):
                c = mesh.visual.main_color
                if c is not None and len(c) >= 3:
                    color = int(c[0]) << 16 | int(c[1]) << 8 | int(c[2])

            safe_name = name.replace("/", "_").replace(" ", "_")
            mpath = f"robot/{safe_name}"
            self._vis[mpath].set_object(
                g.TriangularMeshGeometry(vertices, faces),
                g.MeshPhongMaterial(color=color, reflectivity=0.5),
            )
            self._vis[mpath].set_transform(transform.astype(np.float64))
            self._geom_map[mpath] = name

    def _create_triads(self, side: str) -> None:
        """Create target and current-EE triads for one side."""
        for prefix, radius in [("target", _TRIAD_AXIS_R * 1.6), ("current", _TRIAD_AXIS_R)]:
            for axis_name, _, color in _AXIS_SPECS:
                self._vis[f"markers/{side}/{prefix}/{axis_name}"].set_object(
                    g.Cylinder(height=_TRIAD_AXIS_LEN, radius=radius),
                    g.MeshLambertMaterial(color=color, opacity=1.0),
                )

    def _set_triad(self, side: str, prefix: str, T_world: np.ndarray) -> None:
        """Position a triad at the given 4x4 world transform."""
        for axis_name in _AXIS_LOCAL_T:
            self._vis[f"markers/{side}/{prefix}/{axis_name}"].set_transform(
                T_world @ _AXIS_LOCAL_T[axis_name]
            )

    def to_neutral_configuration(self, arm_angles: dict[str, np.ndarray] | None = None) -> None:
        """Render the sink's neutral pose, or a one-shot override.

        Teleop scripts should call this once after :meth:`launch` so meshcat
        shows the starting configuration before any operator input arrives.
        Pass ``arm_angles`` to render a different pose for this call only —
        the sink's owned neutral (``self._q_home``) is not mutated. Pass
        nothing to render the sink's neutral. Target triads are placed at
        the FK of the rendered pose so target == current (no visual offset).
        """
        pose = arm_angles if arm_angles is not None else self._q_home
        # Apply cfg first so the scene graph reflects the pose, then read FK
        # back out of yourdfpy for the target triads.
        cfg = np.zeros(len(self._actuated_names))
        for side, angles in pose.items():
            for k, idx in enumerate(self._arm_cfg_indices[side]):
                cfg[idx] = angles[k]
        self._robot.update_cfg(cfg)
        target_Ts: dict[str, np.ndarray] = {}
        for side in pose:
            try:
                T_raw, _ = self._scene.graph.get(self._ee_links[side])
                target_Ts[side] = T_raw.astype(np.float64)
            except Exception:
                pass
        self.update(pose, target_Ts=target_Ts)

    def update(
        self,
        arm_angles: dict[str, np.ndarray],
        hand_positions: dict[str, OrcaJointPositions] | None = None,
        target_Ts: dict[str, np.ndarray] | None = None,
        operator_Ts: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Push new arm/hand configs + marker poses to meshcat.

        Args:
            arm_angles: ``{side: 5-element radians array}`` for each active side.
            hand_positions: ``{side: OrcaJointPositions}`` in physical degrees
                from the Retargeter. Missing joints are left at zero.
            target_Ts: ``{side: 4x4 world transform}`` for target triads.
            operator_Ts: Optional incoming operator wrist pose triads. Ignored by
                this MeshCat sink.
        """
        del operator_Ts
        cfg = np.zeros(len(self._actuated_names))
        for side, angles in arm_angles.items():
            for k, idx in enumerate(self._arm_cfg_indices[side]):
                cfg[idx] = angles[k]
        if hand_positions is not None:
            for side, positions in hand_positions.items():
                for joint_id, value_deg in positions:
                    idx = self._hand_cfg_indices.get(side, {}).get(joint_id)
                    if idx is None:
                        continue
                    cfg[idx] = np.deg2rad(value_deg)

        self._robot.update_cfg(cfg)

        # Update robot meshes
        for mpath, scene_name in self._geom_map.items():
            try:
                transform, _ = self._scene.graph.get(scene_name)
                self._vis[mpath].set_transform(transform.astype(np.float64))
            except Exception:
                pass

        # Update triads per side
        for side in arm_angles:
            if target_Ts and side in target_Ts:
                self._set_triad(side, "target", target_Ts[side])

            try:
                T_raw, _ = self._scene.graph.get(self._ee_links[side])
                self._set_triad(side, "current", T_raw.astype(np.float64))
            except Exception:
                pass

    def close(self) -> None:
        if self._vis is not None:
            self._vis.delete()
            self._vis = None


class OrcaArmMujocoSink:
    """MuJoCo viewer for the same arm/hand stream consumed by
    :class:`OrcaArmMeshcatSink`.

    The teleop pipeline still solves IK against the URDF-derived Pinocchio
    model. This sink only renders the resulting joint positions through the
    OrcaArm MJCF shipped by ``orca_arm``.
    """

    def __init__(self, mjcf_path: str | Path | None = None) -> None:
        self._mjcf_path = Path(mjcf_path or orca_arm.MJCF_PATH)
        self._model = mujoco.MjModel.from_xml_path(str(self._mjcf_path))
        self._data = mujoco.MjData(self._model)

        self._arm_qpos_indices: dict[str, list[int]] = {}
        self._hand_qpos_indices: dict[str, dict[str, int]] = {}
        self._carpals_body_ids: dict[str, int] = {}
        for side in SIDES:
            self._arm_qpos_indices[side] = [
                int(
                    self._model.jnt_qposadr[
                        _find_mujoco_joint_id(self._model, f"openarm_{side}_joint{i}")
                    ]
                )
                for i in range(1, ARM_JOINTS_PER_SIDE + 1)
            ]
            self._hand_qpos_indices[side] = self._resolve_hand_joint_indices(side)
            self._carpals_body_ids[side] = _find_mujoco_body_id_by_prefix(
                self._model,
                f"orcahand_{side}_{CARPALS_SIDE_PREFIX[side]}-Carpals_",
            )

        self._viewer = None
        mujoco.mj_forward(self._model, self._data)

    @property
    def arm_joint_names(self) -> dict[str, list[str]]:
        return {
            side: [f"openarm_{side}_joint{i}" for i in range(1, ARM_JOINTS_PER_SIDE + 1)]
            for side in SIDES
        }

    def _resolve_hand_joint_indices(self, side: str) -> dict[str, int]:
        prefix = f"orcahand_{side}_"
        out: dict[str, int] = {}

        wrist_prefix = f"{prefix}{CARPALS_SIDE_PREFIX[side]}-Carpals_"
        wrist_matches = [
            joint_id
            for joint_id in range(self._model.njnt)
            if (name := mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
            and name.startswith(wrist_prefix)
            and "_to_TopTower-Model_" in name
        ]
        if len(wrist_matches) == 1:
            out["wrist"] = int(self._model.jnt_qposadr[wrist_matches[0]])
        else:
            logger.warning(
                "Could not resolve %s hand joint wrist in OrcaArm MJCF (matches=%d)",
                side,
                len(wrist_matches),
            )

        for joint_id, side_markers in _HAND_JOINT_MARKERS.items():
            marker = side_markers[side]
            matches = [
                model_joint_id
                for model_joint_id in range(self._model.njnt)
                if (
                    name := mujoco.mj_id2name(
                        self._model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        model_joint_id,
                    )
                )
                and name.startswith(prefix)
                and marker in name
            ]
            if len(matches) != 1:
                logger.warning(
                    "Could not resolve %s hand joint %s in OrcaArm MJCF (matches=%d)",
                    side,
                    joint_id,
                    len(matches),
                )
                continue
            out[joint_id] = int(self._model.jnt_qposadr[matches[0]])
        return out

    def launch(self) -> None:
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer.cam.distance = 1.9
        self._viewer.cam.azimuth = 155
        self._viewer.cam.elevation = -18
        self._viewer.cam.lookat[:] = np.array([0.35, 0.0, 0.25], dtype=np.float64)
        self._sync_viewer()
        logger.info("MuJoCo viewer launched from %s", self._mjcf_path)

    def to_neutral_configuration(self, arm_angles: dict[str, np.ndarray] | None = None) -> None:
        pose = (
            arm_angles
            if arm_angles is not None
            else {side: np.zeros(ARM_JOINTS_PER_SIDE, dtype=np.float64) for side in SIDES}
        )
        self._write_pose(pose)
        mujoco.mj_forward(self._model, self._data)
        target_Ts = {
            side: _homogeneous_from_mujoco_body(self._data, self._carpals_body_ids[side])
            for side in pose
        }
        self.update(pose, target_Ts=target_Ts)

    def update(
        self,
        arm_angles: dict[str, np.ndarray],
        hand_positions: dict[str, OrcaJointPositions] | None = None,
        target_Ts: dict[str, np.ndarray] | None = None,
        operator_Ts: dict[str, np.ndarray] | None = None,
    ) -> None:
        del operator_Ts
        self._write_pose(arm_angles, hand_positions)
        mujoco.mj_forward(self._model, self._data)
        self._draw_triads(arm_angles, target_Ts)
        self._sync_viewer()

    def _write_pose(
        self,
        arm_angles: dict[str, np.ndarray],
        hand_positions: dict[str, OrcaJointPositions] | None = None,
    ) -> None:
        self._data.qpos[:] = 0.0
        for side, angles in arm_angles.items():
            for k, qpos_idx in enumerate(self._arm_qpos_indices[side]):
                self._data.qpos[qpos_idx] = angles[k]
        if hand_positions is not None:
            for side, positions in hand_positions.items():
                for joint_id, value_deg in positions:
                    qpos_idx = self._hand_qpos_indices.get(side, {}).get(joint_id)
                    if qpos_idx is None:
                        continue
                    self._data.qpos[qpos_idx] = np.deg2rad(value_deg)
        np.clip(
            self._data.qpos,
            self._model.jnt_range[:, 0],
            self._model.jnt_range[:, 1],
            out=self._data.qpos,
        )

    def _draw_triads(
        self,
        arm_angles: dict[str, np.ndarray],
        target_Ts: dict[str, np.ndarray] | None,
    ) -> None:
        if self._viewer is None or self._viewer.user_scn is None:
            return
        scene = self._viewer.user_scn
        scene.ngeom = 0
        for side in arm_angles:
            if target_Ts and side in target_Ts:
                self._add_triad(scene, target_Ts[side], radius=_TRIAD_AXIS_R * 1.6, alpha=1.0)
            self._add_triad(
                scene,
                _homogeneous_from_mujoco_body(self._data, self._carpals_body_ids[side]),
                radius=_TRIAD_AXIS_R,
                alpha=0.72,
            )

    def _add_triad(
        self,
        scene: mujoco.MjvScene,
        T_world: np.ndarray,
        *,
        radius: float,
        alpha: float,
    ) -> None:
        origin = np.asarray(T_world[:3, 3], dtype=np.float64)
        R = np.asarray(T_world[:3, :3], dtype=np.float64)
        for axis_index, (_, _, color) in enumerate(_AXIS_SPECS):
            rgba = np.array(
                [
                    ((color >> 16) & 0xFF) / 255.0,
                    ((color >> 8) & 0xFF) / 255.0,
                    (color & 0xFF) / 255.0,
                    alpha,
                ],
                dtype=np.float32,
            )
            self._add_connector(
                scene,
                origin,
                origin + R[:, axis_index] * _TRIAD_AXIS_LEN,
                radius,
                rgba,
            )

    def _add_connector(
        self,
        scene: mujoco.MjvScene,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: np.ndarray,
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            logger.warning("MuJoCo marker scene full; dropping triad geom")
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    def _sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
