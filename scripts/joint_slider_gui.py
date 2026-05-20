"""matplotlib sliders for the OrcaArm joints, live-driving the viewer preview.

Drag any of the 10 sliders (5 joints × 2 sides) to update the robot pose
and the wrist target triads in the selected viewer. The status line shows where each
carpals frame lands and how aligned the fingers/palm axes are with
forward/down (both in [-1, +1]; +1 = perfectly aligned).

"Print bias" dumps the current non-zero joints in the dict format used by
``scripts/teleop_arm_quest.py`` so you can paste a found pose back in.

Usage:

    .venv/bin/python scripts/joint_slider_gui.py
    .venv/bin/python scripts/joint_slider_gui.py --renderer mujoco
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from orca_teleop.orca_arm_ik import BimanualIKSolver
from orca_teleop.orca_arm_sink import OrcaArmMeshcatSink, OrcaArmMujocoSink

SIDES = ("left", "right")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer",
        choices=["meshcat", "mujoco"],
        default="meshcat",
        help="viewer backend for the slider preview (default: meshcat)",
    )
    args = parser.parse_args()

    ik = BimanualIKSolver()
    sink = OrcaArmMujocoSink() if args.renderer == "mujoco" else OrcaArmMeshcatSink()
    sink.launch()
    q = ik.neutral_q.copy()

    fig = plt.figure("OrcaArm joint sliders", figsize=(11, 7))

    sliders: dict[str, list[Slider]] = {side: [] for side in SIDES}

    SLIDER_HEIGHT = 0.04
    SLIDER_GAP = 0.07
    SLIDER_WIDTH = 0.32
    TOP_Y = 0.78

    # Side header labels + sliders
    for side in SIDES:
        x_left = 0.13 if side == "left" else 0.58
        fig.text(
            x_left + SLIDER_WIDTH / 2,
            TOP_Y + 0.08,
            side.upper(),
            ha="center",
            weight="bold",
            fontsize=12,
        )
        idx_q = ik._arm_idx_q[side]
        for k in range(5):
            i = idx_q[k]
            lo = float(ik._model.lowerPositionLimit[i])
            hi = float(ik._model.upperPositionLimit[i])
            y = TOP_Y - k * SLIDER_GAP
            ax = fig.add_axes([x_left, y, SLIDER_WIDTH, SLIDER_HEIGHT])
            s = Slider(ax, f"j{k+1}\n[{lo:+.2f},{hi:+.2f}]", lo, hi, valinit=0.0, valstep=0.01)
            sliders[side].append(s)

    # Status bar at the top
    status_ax = fig.add_axes([0.02, 0.93, 0.96, 0.05])
    status_ax.axis("off")
    status_text = status_ax.text(
        0.5, 0.5, "", ha="center", va="center", family="monospace", fontsize=9
    )

    def update(_val: object = None) -> None:
        for side in SIDES:
            idx_q = ik._arm_idx_q[side]
            for k, s in enumerate(sliders[side]):
                q[idx_q[k]] = s.val
        arm_angles = {side: np.array([q[i] for i in ik._arm_idx_q[side]]) for side in SIDES}
        target_Ts = {side: ik.forward_kinematics_full(q, side) for side in SIDES}
        sink.update(arm_angles, target_Ts=target_Ts)
        parts = []
        for side in SIDES:
            T = target_Ts[side]
            Z, Y = T[:3, 2], T[:3, 1]
            parts.append(
                f"{side:5s} pos=({T[0,3]:+.2f},{T[1,3]:+.2f},{T[2,3]:+.2f})  "
                f"fingers·fwd={Z[0]:+.2f}  palm·down={Y[2]:+.2f}"
            )
        status_text.set_text("    |    ".join(parts))
        fig.canvas.draw_idle()

    for side in SIDES:
        for s in sliders[side]:
            s.on_changed(update)

    # Buttons
    reset_ax = fig.add_axes([0.20, 0.04, 0.22, 0.06])
    print_ax = fig.add_axes([0.55, 0.04, 0.25, 0.06])
    btn_reset = Button(reset_ax, "Reset (all zero)")
    btn_print = Button(print_ax, "Print bias dict (stdout)")

    def on_reset(_e: object) -> None:
        for side in SIDES:
            for s in sliders[side]:
                s.set_val(0.0)

    def on_print(_e: object) -> None:
        lines = ["side_bias = {"]
        for side in SIDES:
            entries = [
                (k, round(s.val, 3)) for k, s in enumerate(sliders[side]) if abs(s.val) > 1e-3
            ]
            body = ", ".join(f"{k}: {v:+.3f}" for k, v in entries)
            lines.append(f'    "{side}":  {{{body}}},')
        lines.append("}")
        print("\n".join(lines), flush=True)

    btn_reset.on_clicked(on_reset)
    btn_print.on_clicked(on_print)

    # Keep button refs alive
    fig._gui_refs = (btn_reset, btn_print)  # type: ignore[attr-defined]

    update()
    try:
        plt.show()
    finally:
        sink.close()


if __name__ == "__main__":
    main()
