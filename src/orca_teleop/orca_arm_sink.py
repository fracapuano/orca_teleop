"""Scrappy OrcaArm sink: pink IK + meshcat visualization.

Proof-of-concept for driving the orca_arm's 5-DOF arms via IK from
wrist-pose targets.  No pipeline integration yet — this module is
self-contained and meant to be driven by a demo script.
"""

import logging
from dataclasses import dataclass

import meshcat
import meshcat.geometry as g
import numpy as np
import orca_arm
import pink
import pinocchio as pin
import yourdfpy
from pink.limits import ConfigurationLimit

logger = logging.getLogger(__name__)

_ARM_JOINTS_PER_SIDE = 5
_SIDES = ("left", "right")

# Carpals frame naming pattern: orcahand_{side}_{L|R}-Carpals_{hash}
_CARPALS_SIDE_PREFIX = {"left": "L", "right": "R"}


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: dict[str, float]
    orientation_error: dict[str, float]
    converged: dict[str, bool]


def _find_carpals_frame_name(model: pin.Model, side: str) -> str:
    """Find the carpals frame name by pattern (avoids hardcoding hashes)."""
    prefix = f"orcahand_{side}_{_CARPALS_SIDE_PREFIX[side]}-Carpals_"
    for i in range(model.nframes):
        name = model.frames[i].name
        if name.startswith(prefix) and "to_" not in name:
            return name
    raise ValueError(f"Carpals frame not found for side={side!r}")


class BimanualIKSolver:
    """Pink-based full-pose IK for both arms of the OrcaArm.

    One pinocchio model, one config vector. Uses pink's QP-based
    differential IK with FrameTask (position + orientation) to match
    the full 6D wrist pose. Non-arm joints are locked via position
    limits (upper = lower = neutral).
    """

    def __init__(
        self,
        max_iters: int = 100,
        dt: float = 0.1,
        pos_tol: float = 1e-3,
        ori_tol: float = 0.01,
        solver: str = "quadprog",
    ) -> None:
        self._max_iters = max_iters
        self._dt = dt
        self._pos_tol = pos_tol
        self._ori_tol = ori_tol
        self._solver = solver

        self._model = pin.buildModelFromUrdf(orca_arm.URDF_PATH)
        self._data = self._model.createData()

        # Lock all non-arm joints by pinching their position limits
        arm_joint_names: set[str] = set()
        for side in _SIDES:
            for i in range(1, _ARM_JOINTS_PER_SIDE + 1):
                arm_joint_names.add(f"openarm_{side}_joint{i}")

        q_neutral = pin.neutral(self._model)
        for i in range(1, self._model.njoints):
            name = self._model.names[i]
            if name not in arm_joint_names:
                idx_q = self._model.joints[i].idx_q
                nq = self._model.joints[i].nq
                for j in range(nq):
                    self._model.lowerPositionLimit[idx_q + j] = q_neutral[idx_q + j]
                    self._model.upperPositionLimit[idx_q + j] = q_neutral[idx_q + j]

        self._limits = [ConfigurationLimit(self._model)]

        # Per-side frame names, tasks, and joint indices
        self._carpals_names: dict[str, str] = {}
        self._tasks: dict[str, pink.FrameTask] = {}
        self._arm_idx_q: dict[str, list[int]] = {}

        for side in _SIDES:
            self._carpals_names[side] = _find_carpals_frame_name(self._model, side)
            self._tasks[side] = pink.FrameTask(
                self._carpals_names[side],
                position_cost=1.0,
                orientation_cost=1.0,
            )
            joint_names = [f"openarm_{side}_joint{i}" for i in range(1, _ARM_JOINTS_PER_SIDE + 1)]
            self._arm_idx_q[side] = [
                self._model.joints[self._model.getJointId(j)].idx_q for j in joint_names
            ]

    @property
    def neutral_q(self) -> np.ndarray:
        return pin.neutral(self._model).copy()

    def forward_kinematics(self, q: np.ndarray, side: str) -> np.ndarray:
        """Return the 3-D world position of the wrist for config *q*."""
        return self.forward_kinematics_full(q, side)[:3, 3]

    def forward_kinematics_full(self, q: np.ndarray, side: str) -> np.ndarray:
        """Return the 4x4 world transform of the wrist for config *q*."""
        pin.forwardKinematics(self._model, self._data, q)
        fid = self._model.getFrameId(self._carpals_names[side])
        pin.updateFramePlacement(self._model, self._data, fid)
        return self._data.oMf[fid].homogeneous.copy()

    def sample_reachable_target(self, side: str, rng: np.random.Generator) -> pin.SE3:
        """FK at a random arm joint config → guaranteed reachable SE3 pose."""
        q = pin.neutral(self._model)
        for idx_q in self._arm_idx_q[side]:
            lo = self._model.lowerPositionLimit[idx_q]
            hi = self._model.upperPositionLimit[idx_q]
            q[idx_q] = rng.uniform(lo, hi)
        pin.forwardKinematics(self._model, self._data, q)
        fid = self._model.getFrameId(self._carpals_names[side])
        pin.updateFramePlacement(self._model, self._data, fid)
        return pin.SE3(self._data.oMf[fid])

    def solve(
        self,
        targets: dict[str, pin.SE3],
        q0: np.ndarray,
    ) -> IKResult:
        """Solve full-pose IK for one or both arms.

        Args:
            targets: ``{side: SE3 target pose}`` for each arm to solve.
            q0: full robot config to start from.

        Returns:
            IKResult with the solved config and per-side errors.
        """
        config = pink.Configuration(self._model, self._data, q0.copy())

        # Set targets on the frame tasks
        active_tasks = []
        for side, target_pose in targets.items():
            self._tasks[side].set_target(target_pose)
            active_tasks.append(self._tasks[side])

        for _ in range(self._max_iters):
            vel = pink.solve_ik(
                config,
                active_tasks,
                self._dt,
                solver=self._solver,
                limits=self._limits,
            )
            config.integrate_inplace(vel, self._dt)

            # Check convergence for all sides
            all_converged = True
            for side in targets:
                T_now = config.get_transform_frame_to_world(self._carpals_names[side])
                pos_err = np.linalg.norm(T_now.translation - targets[side].translation)
                ori_err = np.linalg.norm(pin.log3(T_now.rotation.T @ targets[side].rotation))
                if pos_err > self._pos_tol or ori_err > self._ori_tol:
                    all_converged = False
            if all_converged:
                break

        # Collect final errors
        q_result = config.q
        pos_errors: dict[str, float] = {}
        ori_errors: dict[str, float] = {}
        converged: dict[str, bool] = {}
        for side, target_pose in targets.items():
            T_now = config.get_transform_frame_to_world(self._carpals_names[side])
            pos_errors[side] = float(np.linalg.norm(T_now.translation - target_pose.translation))
            ori_errors[side] = float(
                np.linalg.norm(pin.log3(T_now.rotation.T @ target_pose.rotation))
            )
            converged[side] = pos_errors[side] < self._pos_tol and ori_errors[side] < self._ori_tol

        return IKResult(
            q=q_result, position_error=pos_errors, orientation_error=ori_errors, converged=converged
        )


# ── Meshcat visualization ────────────────────────────────────────────────────

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


class OrcaArmMeshcatSink:
    """Meshcat-based visualizer that displays the orcabot URDF with
    triads for the target pose and the current wrist pose of both arms."""

    def __init__(self) -> None:
        self._robot = yourdfpy.URDF.load(orca_arm.URDF_PATH)
        self._scene = self._robot.scene

        self._actuated_names = list(self._robot.actuated_joint_names)

        # Per-side: cfg indices and EE scene-graph node
        self._arm_cfg_indices: dict[str, list[int]] = {}
        self._ee_links: dict[str, str] = {}
        for side in _SIDES:
            self._arm_cfg_indices[side] = [
                self._actuated_names.index(f"openarm_{side}_joint{i}")
                for i in range(1, _ARM_JOINTS_PER_SIDE + 1)
            ]
            self._ee_links[side] = next(
                (
                    n
                    for n in self._scene.graph.nodes
                    if f"orcahand_{side}_" in n and "Carpals" in n and "to_" not in n
                ),
                f"openarm_{side}_link5",
            )

        self._vis: meshcat.Visualizer | None = None
        self._geom_map: dict[str, str] = {}

    def launch(self) -> None:
        self._vis = meshcat.Visualizer()
        self._vis.open()
        self._vis.delete()
        self._load_robot_meshes()
        for side in _SIDES:
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

    def update(
        self,
        arm_angles: dict[str, np.ndarray],
        target_Ts: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Push new arm configs + marker poses to meshcat.

        Args:
            arm_angles: ``{side: 5-element radians array}`` for each active side.
            target_Ts: ``{side: 4x4 world transform}`` for target triads.
        """
        cfg = np.zeros(len(self._actuated_names))
        for side, angles in arm_angles.items():
            for k, idx in enumerate(self._arm_cfg_indices[side]):
                cfg[idx] = angles[k]

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
