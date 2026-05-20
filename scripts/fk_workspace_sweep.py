"""Estimate the reachable carpals-frame workspace per embodiment.

Samples N random joint configurations within the URDF's position limits,
computes FK to the carpals frame, and reports per-axis excursion from
the side's home pose.

Output informs ``orca_teleop.constants.WORKSPACE_DELTA_LIMITS_M`` — the
per-side, per-axis asymmetric (lo, hi) clip applied to the Quest delta
translation. Paste the printed ``min`` / ``max`` columns directly into the
constant: for each side, ``lo = (x_min, y_min, z_min)`` and
``hi = (x_max, y_max, z_max)``. The ``Inscribed cube half-side`` and
``sym_half`` lines remain useful sanity values but are no longer the shape
the runtime consumes.

Usage::

    python scripts/fk_workspace_sweep.py
    python scripts/fk_workspace_sweep.py --samples 500000 --seed 7
    python scripts/fk_workspace_sweep.py --embodiment orca-panda \\
        --plot plots/orcapanda_workspace.png
    python scripts/fk_workspace_sweep.py --animate

With ``--animate``, each sampled joint config is pushed live to a
viewer preview of the robot while a parallel matplotlib 3D scatter plots
the resulting carpals position in the same URDF world frame.
If the dot in matplotlib tracks the carpals triad in meshcat over many
frames, the sweep is observing what the robot actually reaches.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from orca_teleop.orca_arm_ik import (
    ArmIKConfig,
    BimanualIKSolver,
    default_orca_arm_ik_config,
    orca_panda_right_ik_config,
)
from orca_teleop.orca_arm_teleop import OrcaArmTeleopConfig, OrcaArmTeleopController

AXIS_LABELS = ("x_fwd", "y_left", "z_up")
SIDE_COLOR = {"left": "tab:blue", "right": "tab:orange"}


class SweepSink(Protocol):
    def launch(self) -> None: ...

    def update(
        self,
        arm_angles: dict[str, np.ndarray],
        hand_positions: object | None = None,
        target_Ts: dict[str, np.ndarray] | None = None,
        operator_Ts: dict[str, np.ndarray] | None = None,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass
class SideSweep:
    positions: np.ndarray
    T_home_p: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    dt: float


def _sides(ik: BimanualIKSolver) -> tuple[str, ...]:
    return tuple(ik.arm_joint_indices.keys())


def _build_q_home(ik: BimanualIKSolver) -> np.ndarray:
    controller = OrcaArmTeleopController(
        ik=ik,
        config=OrcaArmTeleopConfig(active_sides=_sides(ik)),
    )
    return controller.q_home


def _build_ik_config(args: argparse.Namespace) -> ArmIKConfig:
    if args.embodiment == "orca-panda":
        return orca_panda_right_ik_config(
            urdf_path=args.arm_urdf_path,
            ee_frame=args.right_ee_frame or "orcahand_right_R-Carpals_8d1f1041",
        )

    config = default_orca_arm_ik_config()
    if args.arm_urdf_path is None:
        return config
    return ArmIKConfig(
        urdf_path=args.arm_urdf_path,
        sides=config.sides,
        joint_names_by_side=config.joint_names_by_side,
        ee_frame_by_side=config.ee_frame_by_side,
    )


def _make_sink(
    embodiment: str,
    *,
    render_mode: str | None,
    task_seed: int | None,
) -> SweepSink:
    if embodiment == "orca-panda":
        from orca_teleop.sim import OrcaPandaCubeStackingSink

        return OrcaPandaCubeStackingSink(
            render_mode=render_mode,
            seed=task_seed,
            instant_qpos=True,
        )

    from orca_teleop.orca_arm_sink import OrcaArmMeshcatSink

    return OrcaArmMeshcatSink()


def _apply_env_home(
    ik: BimanualIKSolver,
    q_home: np.ndarray,
    sink: SweepSink,
    log: logging.Logger,
) -> np.ndarray:
    if not hasattr(sink, "home_arm_angles"):
        raise ValueError("Selected viewer/sink does not expose environment home arm angles.")

    env_home_arm_angles = sink.home_arm_angles
    q = q_home.copy()
    for side in _sides(ik):
        if side not in env_home_arm_angles:
            continue
        for idx, value in zip(
            ik.arm_joint_indices[side],
            env_home_arm_angles[side],
            strict=True,
        ):
            q[idx] = value
    q = np.clip(q, ik._model.lowerPositionLimit, ik._model.upperPositionLimit)
    log.info(
        "Using environment home arm angles: %s",
        {
            side: np.round(env_home_arm_angles[side], 3).tolist()
            for side in _sides(ik)
            if side in env_home_arm_angles
        },
    )
    return q


def _log_side_stats(
    log: logging.Logger,
    side: str,
    positions: np.ndarray,
    T_home_p: np.ndarray,
    n: int,
    dt: float,
    ik: BimanualIKSolver,
    lo: np.ndarray,
    hi: np.ndarray,
) -> None:
    rel = np.vstack([positions - T_home_p, np.zeros((1, 3), dtype=np.float64)])
    ax_lo = rel.min(axis=0)
    ax_hi = rel.max(axis=0)
    sym_half = np.minimum(np.abs(ax_lo), ax_hi)
    cube_half = float(sym_half.min())
    bounding_half = float(np.max(np.maximum(np.abs(ax_lo), np.abs(ax_hi))))

    log.info("=" * 64)
    log.info("Side: %-5s    T_home (FLU, m): %s", side, np.round(T_home_p, 3).tolist())
    log.info(
        "FK sweep: n=%d in %.2fs (%.1f us/call), joint limits per side:",
        n,
        dt,
        1e6 * dt / max(n, 1),
    )
    for j, name in enumerate(ik.arm_joint_names[side]):
        log.info("    %-22s  [%+.3f, %+.3f] rad", name, lo[j], hi[j])
    log.info("Reachable Δ from T_home (FLU x=fwd, y=left, z=up):")
    for j, label in enumerate(AXIS_LABELS):
        log.info(
            "    %-7s   min=%+.3f  max=%+.3f  sym_half=%+.3f m",
            label,
            ax_lo[j],
            ax_hi[j],
            sym_half[j],
        )
    log.info("Inscribed cube half-side : %.3f m", cube_half)
    log.info("Per-axis bounding sphere : %.3f m  (max |Δ| over all axes)", bounding_half)


def _run_fast(
    ik: BimanualIKSolver,
    q_home: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    log: logging.Logger,
) -> dict[str, SideSweep]:
    results: dict[str, SideSweep] = {}
    for side in _sides(ik):
        idx_q = ik._arm_idx_q[side]
        lo = np.array([ik._model.lowerPositionLimit[i] for i in idx_q])
        hi = np.array([ik._model.upperPositionLimit[i] for i in idx_q])

        T_home_p = ik.forward_kinematics(q_home, side)

        q = q_home.copy()
        joint_samples = rng.uniform(lo, hi, size=(samples, len(idx_q)))
        positions = np.empty((samples, 3), dtype=np.float64)

        t0 = time.monotonic()
        for k in range(samples):
            for j, i in enumerate(idx_q):
                q[i] = joint_samples[k, j]
            positions[k] = ik.forward_kinematics(q, side)
        dt = time.monotonic() - t0

        _log_side_stats(log, side, positions, T_home_p, samples, dt, ik, lo, hi)
        results[side] = SideSweep(
            positions=positions,
            T_home_p=T_home_p,
            lo=lo,
            hi=hi,
            dt=dt,
        )
    return results


def _run_animated(
    ik: BimanualIKSolver,
    q_home: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    fps: float,
    log: logging.Logger,
    sink: SweepSink | None = None,
    embodiment: str = "orca-arm",
    task_render_mode: str = "human",
    task_seed: int | None = None,
) -> None:
    # Imports kept local so the fast path stays lightweight and headless.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    if sink is None:
        sink = _make_sink(
            embodiment,
            render_mode=None if task_render_mode == "none" else task_render_mode,
            task_seed=task_seed,
        )
        sink.launch()

    # Snap meshcat to the home pose so the operator sees the anchor before the sweep starts.
    sides = _sides(ik)
    home_arm_angles = {side: np.array([q_home[i] for i in ik._arm_idx_q[side]]) for side in sides}
    home_target_Ts = {side: ik.forward_kinematics_full(q_home, side) for side in sides}
    sink.update(home_arm_angles, target_Ts=home_target_Ts)
    if hasattr(sink, "set_debug_target_frame_offsets"):
        sink.set_debug_target_frame_offsets(home_target_Ts)

    per_side: dict[str, dict] = {}
    for side in sides:
        idx_q = ik._arm_idx_q[side]
        lo = np.array([ik._model.lowerPositionLimit[i] for i in idx_q])
        hi = np.array([ik._model.upperPositionLimit[i] for i in idx_q])
        per_side[side] = {
            "idx_q": idx_q,
            "lo": lo,
            "hi": hi,
            "T_home_p": ik.forward_kinematics(q_home, side),
            "joint_samples": rng.uniform(lo, hi, size=(samples, len(idx_q))),
            "positions": np.empty((samples, 3), dtype=np.float64),
        }

    plt.ion()
    fig = plt.figure("FK workspace sweep — carpals (URDF world frame, m)", figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x_fwd (m)")
    ax.set_ylabel("y_left (m)")
    ax.set_zlabel("z_up (m)")
    ax.set_title(
        "FK workspace sweep — points are pinocchio FK output;\n"
        "compare each new dot to the carpals triad in meshcat"
    )

    scatters: dict[str, object] = {}
    current_markers: dict[str, object] = {}
    for side in sides:
        T_home_p = per_side[side]["T_home_p"]
        ax.scatter(
            [T_home_p[0]],
            [T_home_p[1]],
            [T_home_p[2]],
            c=SIDE_COLOR[side],
            marker="*",
            s=240,
            edgecolors="k",
            linewidths=1.0,
            label=f"{side} home",
        )
        scatters[side] = ax.scatter(
            [],
            [],
            [],
            c=SIDE_COLOR[side],
            s=6,
            alpha=0.45,
            label=f"{side} samples",
        )
        current_markers[side] = ax.scatter(
            [T_home_p[0]],
            [T_home_p[1]],
            [T_home_p[2]],
            c=SIDE_COLOR[side],
            marker="o",
            s=90,
            edgecolors="k",
            linewidths=1.5,
            label=f"{side} current",
        )
    ax.legend(loc="upper right", fontsize=8)

    # Pre-size axes to cover both home positions with a generous pad; rescale once
    # we know the empirical extent (so the early frames are not cramped).
    home_xyz = np.array([per_side[s]["T_home_p"] for s in sides])
    center = home_xyz.mean(axis=0)
    pad = 0.7
    ax.set_xlim(center[0] - pad, center[0] + pad)
    ax.set_ylim(center[1] - pad, center[1] + pad)
    ax.set_zlim(center[2] - pad, center[2] + pad)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    fig.canvas.draw_idle()
    plt.pause(0.001)

    frame_dt = 1.0 / max(fps, 1e-3)
    q = q_home.copy()
    last_k = -1
    t0 = time.monotonic()
    for k in range(samples):
        if not plt.fignum_exists(fig.number):
            break
        frame_start = time.monotonic()

        arm_angles: dict[str, np.ndarray] = {}
        target_Ts: dict[str, np.ndarray] = {}
        for side in sides:
            data = per_side[side]
            for j, i in enumerate(data["idx_q"]):
                q[i] = data["joint_samples"][k, j]
            T_full = ik.forward_kinematics_full(q, side)
            data["positions"][k] = T_full[:3, 3]
            arm_angles[side] = np.array([q[i] for i in data["idx_q"]])
            target_Ts[side] = T_full

        sink.update(arm_angles, target_Ts=target_Ts)

        for side in sides:
            pts = per_side[side]["positions"][: k + 1]
            scatters[side]._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            cur = pts[-1]
            current_markers[side]._offsets3d = ([cur[0]], [cur[1]], [cur[2]])
        fig.canvas.draw_idle()

        last_k = k
        elapsed = time.monotonic() - frame_start
        plt.pause(max(1e-3, frame_dt - elapsed))
    dt = time.monotonic() - t0

    n_done = last_k + 1
    if n_done > 0:
        for side in sides:
            data = per_side[side]
            _log_side_stats(
                log,
                side,
                data["positions"][:n_done],
                data["T_home_p"],
                n_done,
                dt,
                ik,
                data["lo"],
                data["hi"],
            )

    log.info("Animation done — close the matplotlib window to exit.")
    plt.ioff()
    try:
        plt.show()
    finally:
        sink.close()


def _plot_results(
    results: dict[str, SideSweep],
    *,
    output_path: Path | None,
    show: bool,
    max_points_per_side: int,
    seed: int,
    title: str,
    log: logging.Logger,
) -> None:
    if output_path is None and not show:
        return

    import matplotlib.pyplot as plt

    fig = plt.figure(title, figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x_fwd (m)")
    ax.set_ylabel("y_left (m)")
    ax.set_zlabel("z_up (m)")
    ax.set_title(title)

    rng = np.random.default_rng(seed)
    all_points = []
    for side, data in results.items():
        positions = data.positions
        if max_points_per_side > 0 and len(positions) > max_points_per_side:
            idx = rng.choice(len(positions), size=max_points_per_side, replace=False)
            positions = positions[np.sort(idx)]
        all_points.append(positions)
        color = SIDE_COLOR.get(side, None)
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=color,
            s=4,
            alpha=0.25,
            label=f"{side} samples",
        )
        home = data.T_home_p
        ax.scatter(
            [home[0]],
            [home[1]],
            [home[2]],
            c=color,
            marker="*",
            s=260,
            edgecolors="k",
            linewidths=1.0,
            label=f"{side} home",
        )

    points = np.vstack(all_points + [np.array([data.T_home_p for data in results.values()])])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        log.info("Saved workspace plot: %s", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embodiment",
        choices=["orca-arm", "orca-panda"],
        default="orca-arm",
        help="IK embodiment preset to sweep.",
    )
    parser.add_argument(
        "--arm-urdf-path",
        default=None,
        help="URDF used by the arm IK. Defaults to the selected embodiment preset.",
    )
    parser.add_argument(
        "--right-ee-frame",
        default=None,
        help="right-side end-effector frame name, mainly useful for OrcaPanda URDF variants.",
    )
    parser.add_argument(
        "--home-source",
        choices=["teleop", "neutral", "env"],
        default=None,
        help=(
            "Home pose used as the delta origin. Defaults to env for orca-panda, "
            "teleop for orca-arm."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of FK samples. Defaults: 100000 fast, 1500 with --animate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Stream each sampled joint config to meshcat and plot the resulting "
        "carpals position live in a matplotlib 3D scatter (URDF world frame).",
    )
    parser.add_argument(
        "--animate-fps",
        type=float,
        default=60.0,
        help="Target frame rate for the animated sweep (ignored without --animate).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save a static 3D scatter plot of the sampled workspace to this image path.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show the static workspace plot in a matplotlib window.",
    )
    parser.add_argument(
        "--plot-max-points-per-side",
        type=int,
        default=30_000,
        help="Maximum sampled points per side drawn in the static plot; <=0 draws all points.",
    )
    parser.add_argument(
        "--task-render-mode",
        choices=["human", "rgb_array", "none"],
        default="human",
        help="Render mode for OrcaPanda animation. Fast env-home extraction uses none.",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=None,
        help="orca_sim seed when reading the OrcaPanda environment home pose.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    ik = BimanualIKSolver(ik_config=_build_ik_config(args))
    home_source = args.home_source
    if home_source is None:
        home_source = "env" if args.embodiment == "orca-panda" else "teleop"

    if home_source == "neutral":
        q_home = ik.neutral_q.copy()
    else:
        q_home = _build_q_home(ik)

    rng = np.random.default_rng(args.seed)

    samples = args.samples
    if samples is None:
        samples = 1500 if args.animate else 100_000

    sink: SweepSink | None = None
    if home_source == "env":
        if args.embodiment != "orca-panda":
            raise ValueError(
                "--home-source env is currently implemented for --embodiment orca-panda."
            )
        sink = _make_sink(
            args.embodiment,
            render_mode=(
                None
                if (not args.animate or args.task_render_mode == "none")
                else args.task_render_mode
            ),
            task_seed=args.task_seed,
        )
        sink.launch()
        q_home = _apply_env_home(ik, q_home, sink, log)

    try:
        if args.animate:
            _run_animated(
                ik,
                q_home,
                rng,
                samples,
                args.animate_fps,
                log,
                sink=sink,
                embodiment=args.embodiment,
                task_render_mode=args.task_render_mode,
                task_seed=args.task_seed,
            )
            sink = None
        else:
            results = _run_fast(ik, q_home, rng, samples, log)
            _plot_results(
                results,
                output_path=args.plot,
                show=args.show_plot,
                max_points_per_side=args.plot_max_points_per_side,
                seed=args.seed,
                title=f"FK workspace sweep — {args.embodiment} carpals",
                log=log,
            )
    finally:
        if sink is not None:
            sink.close()


if __name__ == "__main__":
    main()
