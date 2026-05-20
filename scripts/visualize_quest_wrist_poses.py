"""Visualize recorded Meta Quest wrist poses in an otherwise empty MuJoCo scene.

The dataset stores the raw wrist pose columns produced by
``scripts/record_quest_poses.py``. This viewer ignores all hand landmarks and
streams only the left/right wrist poses as coordinate triads against a fixed
MuJoCo world frame.

Examples:

    mjpython scripts/visualize_quest_wrist_poses.py
    mjpython scripts/visualize_quest_wrist_poses.py --basis flu
    mjpython scripts/visualize_quest_wrist_poses.py --side right --no-loop
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REPO = "fracapuano/quest-poses"
DEFAULT_FILENAME = "data.parquet"
SIDES = ("left", "right")
BasisMode = Literal["raw", "flu"]

_LEFT_RGBA = np.array([0.12, 0.54, 1.0, 1.0], dtype=np.float32)
_RIGHT_RGBA = np.array([1.0, 0.58, 0.08, 1.0], dtype=np.float32)
_AXIS_RGBA = (
    np.array([1.0, 0.08, 0.08, 1.0], dtype=np.float32),
    np.array([0.1, 0.82, 0.22, 1.0], dtype=np.float32),
    np.array([0.16, 0.35, 1.0, 1.0], dtype=np.float32),
)


def _scene_xml(axis_length: float) -> str:
    """Return a minimal MuJoCo scene with a fixed RGB world frame."""
    axis_r = 0.008
    plane_size = max(1.2, axis_length * 1.7)
    return f"""
<mujoco model="quest_wrist_world">
  <compiler angle="radian"/>
  <option timestep="0.01"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.19 0.19 0.19" rgb2="0.24 0.24 0.24"/>
    <material name="grid" texture="grid" texrepeat="20 20" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -3 3" dir="0 1 -1" diffuse="0.8 0.8 0.8"/>
    <light name="fill" pos="-3 2 2" dir="1 -1 -0.6" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="{plane_size:.3f} {plane_size:.3f} 0.01"
          material="grid" rgba="0.22 0.22 0.22 1"/>
    <geom name="world_origin" type="sphere" size="0.025" pos="0 0 0"
          rgba="1 1 1 1"/>
    <geom name="world_x" type="capsule" fromto="0 0 0 {axis_length:.6f} 0 0"
          size="{axis_r:.6f}" rgba="1 0.08 0.08 1"/>
    <geom name="world_y" type="capsule" fromto="0 0 0 0 {axis_length:.6f} 0"
          size="{axis_r:.6f}" rgba="0.1 0.82 0.22 1"/>
    <geom name="world_z" type="capsule" fromto="0 0 0 0 0 {axis_length:.6f}"
          size="{axis_r:.6f}" rgba="0.16 0.35 1 1"/>
  </worldbody>
</mujoco>
"""


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray | None:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    n = float(q @ q)
    if n < 1e-12:
        return None
    qx, qy, qz, qw = q / math.sqrt(n)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray | None:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    n = float(q @ q)
    if n < 1e-12:
        return None
    qx, qy, qz, qw = q / math.sqrt(n)
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def _basis_unity_left_to_flu() -> np.ndarray:
    try:
        from hand_tracking_sdk.convert import BASIS_UNITY_LEFT_TO_FLU

        return np.asarray(BASIS_UNITY_LEFT_TO_FLU, dtype=np.float64)
    except ImportError:
        # Unity LH: x=right, y=up, z=forward.
        # Robot FLU: x=forward, y=left, z=up.
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )


def _pose_for_row(
    cols: dict[str, list],
    row: int,
    side: str,
    basis: BasisMode,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(position, quat_wxyz)`` for a visible wrist sample."""
    if not bool(cols[f"{side}_visible"][row]):
        return None

    p = np.array(
        [
            cols[f"{side}_wrist_x"][row],
            cols[f"{side}_wrist_y"][row],
            cols[f"{side}_wrist_z"][row],
        ],
        dtype=np.float64,
    )
    q_xyzw = np.array(
        [
            cols[f"{side}_wrist_qx"][row],
            cols[f"{side}_wrist_qy"][row],
            cols[f"{side}_wrist_qz"][row],
            cols[f"{side}_wrist_qw"][row],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(p)):
        return None

    if basis == "raw":
        q = _quat_xyzw_to_wxyz(q_xyzw)
        return None if q is None else (p, q)

    R = _quat_xyzw_to_rotmat(q_xyzw)
    if R is None:
        return None
    B = _basis_unity_left_to_flu()
    return B @ p, _rotmat_to_quat_wxyz(B @ R @ B.T)


def _rotation_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    return _quat_xyzw_to_rotmat(np.array([qx, qy, qz, qw], dtype=np.float64))


def _required_columns(sides: tuple[str, ...]) -> list[str]:
    columns = ["t_ns"]
    for side in sides:
        columns.append(f"{side}_visible")
        columns.extend(f"{side}_wrist_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw"))
    return columns


def _load_columns(
    *,
    repo: str,
    filename: str,
    parquet: Path | None,
    refresh: bool,
    sides: tuple[str, ...],
) -> dict[str, list]:
    import pyarrow.parquet as pq

    if parquet is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                force_download=refresh,
            )
        )
        logger.info("Loaded %s/%s -> %s", repo, filename, path)
    else:
        path = parquet
        logger.info("Loaded local parquet -> %s", path)

    table = pq.read_table(path, columns=_required_columns(sides))
    return {name: table.column(name).to_pylist() for name in table.column_names}


def _infer_fps(t_ns: list[int], fallback: float = 30.0) -> float:
    t = np.asarray(t_ns, dtype=np.float64)
    dt = np.diff(t) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return fallback
    fps = 1.0 / float(np.median(dt))
    if not np.isfinite(fps) or fps <= 0.0:
        return fallback
    return float(np.clip(fps, 1.0, 240.0))


def _add_connector(
    scene,
    mujoco,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
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


def _add_sphere(scene, mujoco, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        rgba,
    )
    scene.ngeom += 1


def _draw_dynamic_scene(
    scene,
    mujoco,
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    trails: dict[str, deque[np.ndarray]],
    *,
    triad_length: float,
    triad_radius: float,
    trail_radius: float,
) -> None:
    scene.ngeom = 0
    for side, trail in trails.items():
        base = _LEFT_RGBA if side == "left" else _RIGHT_RGBA
        n = max(1, len(trail))
        for i, pos in enumerate(trail):
            rgba = base.copy()
            rgba[3] = 0.08 + 0.32 * (i + 1) / n
            _add_sphere(scene, mujoco, pos, trail_radius, rgba)

    for side, (pos, quat_wxyz) in poses.items():
        side_rgba = _LEFT_RGBA if side == "left" else _RIGHT_RGBA
        _add_sphere(scene, mujoco, pos, trail_radius * 2.2, side_rgba)
        R = _rotation_from_quat_wxyz(quat_wxyz)
        for axis_index, rgba in enumerate(_AXIS_RGBA):
            axis_rgba = rgba.copy()
            axis_rgba[3] = 0.95
            _add_connector(
                scene,
                mujoco,
                pos,
                pos + R[:, axis_index] * triad_length,
                triad_radius,
                axis_rgba,
            )


def _selected_sides(side_arg: str) -> tuple[str, ...]:
    if side_arg == "both":
        return SIDES
    return (side_arg,)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="parquet filename in the dataset repo",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="use a local parquet file instead of HF",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="force re-download from Hugging Face",
    )
    parser.add_argument(
        "--basis",
        choices=["raw", "flu"],
        default="raw",
        help="pose basis to visualize",
    )
    parser.add_argument(
        "--side",
        choices=["both", "left", "right"],
        default="both",
        help="which wrist stream to draw",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="playback fps; default infers from t_ns",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--start-row", type=int, default=0, help="first dataset row to replay")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="maximum number of rows to replay",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="stop when the selected row range ends",
    )
    parser.add_argument(
        "--world-axis-length",
        type=float,
        default=0.5,
        help="length of the fixed world-frame axes",
    )
    parser.add_argument(
        "--triad-length",
        type=float,
        default=0.14,
        help="length of each wrist pose axis",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=120,
        help="recent visible wrist samples to keep per side",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sides = _selected_sides(args.side)
    cols = _load_columns(
        repo=args.repo,
        filename=args.filename,
        parquet=args.parquet,
        refresh=args.refresh,
        sides=sides,
    )
    n = len(cols["t_ns"])
    if n == 0:
        raise RuntimeError("Dataset has 0 rows.")
    start = max(0, min(args.start_row, n - 1))
    stop = n if args.max_rows is None else min(n, start + max(1, args.max_rows))

    fps = float(args.fps) if args.fps is not None else _infer_fps(cols["t_ns"][start:stop])
    period = 1.0 / max(1e-6, fps * max(1e-6, float(args.speed)))
    logger.info(
        "Streaming rows [%d, %d) at %.2f fps x %.2f, basis=%s, sides=%s",
        start,
        stop,
        fps,
        args.speed,
        args.basis,
        ",".join(sides),
    )
    if args.basis == "raw":
        logger.info(
            "Raw mode interprets recorded Quest/Unity wrist quaternions directly in MuJoCo."
        )
    else:
        logger.info(
            "FLU mode applies the same Unity-left-handed -> robot-FLU basis change used by teleop."
        )

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_string(_scene_xml(args.world_axis_length))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    trails: dict[str, deque[np.ndarray]] = {
        side: deque(maxlen=max(0, int(args.trail_length))) for side in sides
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -22
        viewer.cam.lookat[:] = np.array([0.0, 0.6, 0.25], dtype=np.float64)

        row = start
        next_tick = time.perf_counter()
        while viewer.is_running():
            poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for side in sides:
                pose = _pose_for_row(cols, row, side, args.basis)
                if pose is None:
                    continue
                poses[side] = pose
                if trails[side].maxlen != 0:
                    trails[side].append(pose[0])

            if viewer.user_scn is not None:
                _draw_dynamic_scene(
                    viewer.user_scn,
                    mujoco,
                    poses,
                    trails,
                    triad_length=args.triad_length,
                    triad_radius=0.006,
                    trail_radius=0.01,
                )
            viewer.sync()

            row += 1
            if row >= stop:
                if args.no_loop:
                    logger.info("End of selected row range.")
                    break
                row = start
                for trail in trails.values():
                    trail.clear()

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
