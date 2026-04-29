"""End-to-end: MetaQuest publisher → gRPC ingress → wrist adapter → bimanual IK → meshcat sim.

Auto-calibrates per side on the first received frame: the operator's first
wrist pose is anchored to the robot's neutral carpals.  Every subsequent pose
is multiplied by that constant offset to land in robot-world coords, then fed
straight to bimanual IK and rendered in meshcat.

In one terminal:

    python scripts/teleop_arm_quest.py

In another (replays the HF dataset):

    python -m orca_teleop.ingress.metaquest.publisher

Or, all-in-one:

    python scripts/teleop_arm_quest.py --local
"""

import argparse
import logging
import multiprocessing
import queue
import socket
import threading
import time

import numpy as np
import pinocchio as pin
from hand_tracking_sdk.convert import BASIS_UNITY_LEFT_TO_FLU

from orca_teleop.constants import DEFAULT_PORT, QUEUES_MAXSIZE
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.orca_arm_sink import BimanualIKSolver, OrcaArmMeshcatSink

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
IK_RATE_HZ = 60
CALIB_SAMPLES = 30  # ~1 second at 30 Hz publisher rate; "hold still"
# Operator motion sweeps ~60 cm; the OrcaArm 5-DOF reachable workspace is much
# smaller in 3D (axis bboxes look ~0.9 m but the reachable solid is much
# tighter). Defaults below were validated on the recorded dataset; tracking
# stays clean (median IK residual ~0 mm, p90 < 15 mm).
DEFAULT_WORKSPACE_HALF_BOX_M = 0.10
DEFAULT_AUTO_FIT_SAMPLES = 150  # 5 s at 30 Hz — observe operator natural span
DEFAULT_AUTO_FIT_MARGIN = 0.7  # use this fraction of the workspace as fit target
BOOTSTRAP_SCALE = 0.15  # scale used during span observation phase

# Unity LH → robot FLU. SDK's basis_transform_rotation_matrix takes a quaternion
# (misleading name), so we apply the basis change directly: p' = B p, R' = B R B.T.
# B has det = -1 (chirality flip), but applied on both sides it leaves det(R') = +1.
_B_UNITY_TO_FLU = np.asarray(BASIS_UNITY_LEFT_TO_FLU, dtype=np.float64)


def _wrist_pose_to_robot_se3(position: np.ndarray, rotation: np.ndarray) -> pin.SE3:
    """Quest wrist pose (Unity left-handed) → pin.SE3 in robot world (FLU) coords."""
    p = _B_UNITY_TO_FLU @ np.asarray(position, dtype=np.float64)
    R = _B_UNITY_TO_FLU @ np.asarray(rotation, dtype=np.float64) @ _B_UNITY_TO_FLU.T
    return pin.SE3(R, p)


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """SVD-based rotation average (Markley's method) over a list of 3x3 mats."""
    M = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _drain_queue(
    landmarks_q: "queue.Queue",
    calib_buf: dict[str, list[pin.SE3]],
    span_buf: dict[str, list[np.ndarray]],
    T_first: dict[str, pin.SE3],
    T_home: dict[str, pin.SE3],
    scale: dict[str, float],
    targets: dict[str, pin.SE3],
    manual_scale: float | None,
    workspace_half_box_m: float,
    auto_fit_margin: float,
    span_samples: int,
) -> None:
    """Pull everything the ingress has buffered.

    Three-phase per-side state machine:
      1. anchor (CALIB_SAMPLES of "hold still") -> T_first[side]
      2. span observation (span_samples of natural motion) -> auto-fit scale[side]
      3. teleop with locked scale

    During phase 2 we run with BOOTSTRAP_SCALE so the operator gets feedback
    while extending to demonstrate range. If --translation-scale was passed
    explicitly, phase 2 is skipped and that value is used directly.

    Mapping after calibration:
        p_target = T_home.translation + s * (p_op - p_first)   (then clamped)
        R_target = (R_op @ R_first.T) @ T_home.rotation
    """
    while True:
        try:
            item = landmarks_q.get_nowait()
        except queue.Empty:
            return
        if not isinstance(item, HandLandmarks) or item.wrist_pose is None:
            continue
        side = item.handedness
        if side not in SIDES:
            continue
        T_op = _wrist_pose_to_robot_se3(item.wrist_pose.position, item.wrist_pose.rotation)

        # Phase 1: anchor
        if side not in T_first:
            calib_buf[side].append(T_op)
            if len(calib_buf[side]) < CALIB_SAMPLES:
                continue
            mean_p = np.mean([T.translation for T in calib_buf[side]], axis=0)
            R_anchor = _mean_rotation([T.rotation for T in calib_buf[side]])
            T_first[side] = pin.SE3(R_anchor, mean_p)
            calib_buf[side].clear()
            if manual_scale is not None:
                scale[side] = manual_scale
                logger.info(
                    "Anchored %s (n=%d, op centroid=%s); using manual scale=%.3f",
                    side,
                    CALIB_SAMPLES,
                    np.round(mean_p, 3).tolist(),
                    manual_scale,
                )
            else:
                logger.info(
                    "Anchored %s (n=%d, op centroid=%s); now observing %d span"
                    " samples for auto-fit (move freely)",
                    side,
                    CALIB_SAMPLES,
                    np.round(mean_p, 3).tolist(),
                    span_samples,
                )
            continue

        # Phase 2: observe span (only if auto-fit and not yet locked)
        if side not in scale:
            span_buf[side].append(T_op.translation.copy())
            if len(span_buf[side]) >= span_samples:
                pts = np.array(span_buf[side])
                max_half = float(((pts.max(axis=0) - pts.min(axis=0)) / 2.0).max())
                if max_half > 1e-3:
                    fitted = (auto_fit_margin * workspace_half_box_m) / max_half
                else:
                    fitted = BOOTSTRAP_SCALE
                scale[side] = float(np.clip(fitted, 0.05, 1.0))
                span_buf[side].clear()
                logger.info(
                    "Auto-fit %s: op_max_half=%.3fm  ws_half=%.3fm" "  margin=%.2f  →  scale=%.3f",
                    side,
                    max_half,
                    workspace_half_box_m,
                    auto_fit_margin,
                    scale[side],
                )
            s = BOOTSTRAP_SCALE if side not in scale else scale[side]
        else:
            s = scale[side]

        # Phase 3: teleop mapping
        dR = T_op.rotation @ T_first[side].rotation.T
        dp = s * (T_op.translation - T_first[side].translation)
        if workspace_half_box_m > 0.0:
            dp = np.clip(dp, -workspace_half_box_m, workspace_half_box_m)
        R_target = dR @ T_home[side].rotation
        p_target = T_home[side].translation + dp
        targets[side] = pin.SE3(R_target, p_target)
        logger.debug(
            "%s: |dp|=%.3f m  dp=%s  target_p=%s",
            side,
            float(np.linalg.norm(dp)),
            np.round(dp, 3).tolist(),
            np.round(p_target, 3).tolist(),
        )


def _metaquest_publisher(port: int) -> None:
    """Child-process entry point for the local replay publisher."""
    from orca_teleop.ingress.metaquest.publisher import MetaQuestPublisher

    server_address = f"localhost:{port}"
    deadline = time.monotonic() + 10.0
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                break
        except OSError as err:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Ingress server on {server_address} did not become ready"
                ) from err
            time.sleep(0.1)

    MetaQuestPublisher(server_address=server_address).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="gRPC ingress port")
    parser.add_argument("--ik-rate", type=float, default=IK_RATE_HZ, help="IK/render rate (Hz)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="also spawn the MetaQuest replay publisher as a child process",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=None,
        help="manual translation scale; if unset, auto-fit from operator's"
        " observed span over the first ~5 s of post-anchor motion",
    )
    parser.add_argument(
        "--workspace-half-box",
        type=float,
        default=DEFAULT_WORKSPACE_HALF_BOX_M,
        help="per-axis clamp on the (scaled) translation delta in m; 0 disables",
    )
    parser.add_argument(
        "--auto-fit-margin",
        type=float,
        default=DEFAULT_AUTO_FIT_MARGIN,
        help="auto-fit safety factor; scale = margin * workspace_half_box / op_max_half",
    )
    parser.add_argument(
        "--span-samples",
        type=int,
        default=DEFAULT_AUTO_FIT_SAMPLES,
        help="post-anchor samples to observe operator's natural motion span",
    )
    parser.add_argument(
        "--orientation-cost",
        type=float,
        default=1.0,
        help="orientation cost in IK; 0 = position-only. Default 1.0 enables"
        " 5-DOF tracking on the two non-roll axes (see --free-roll-axis):"
        " strictly better than 3-DOF on this URDF (1/900 stuck vs 106/900)",
    )
    parser.add_argument(
        "--free-roll-axis",
        default="Y",
        choices=["X", "Y", "Z"],
        help="body-frame axis whose rotation is unconstrained when orientation-cost > 0."
        " Default Y: empirically best on the OrcaArm URDF (1/900 stuck frames vs"
        " 15/900 for Z). Z is more intuitive (free wrist roll about local +Z) but"
        " gives a less usable null space on this kinematic chain.",
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=1e-3,
        help="weight of a posture-regularization task that re-anchors to the"
        " previous joint config each frame, damping frame-to-frame change"
        " without biasing toward any specific posture. 0 disables.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.orientation_cost > 0.0:
        # Free roll about the chosen body-frame axis: zero out that axis's cost,
        # keep the other two at args.orientation_cost. This gives a 5-DOF target
        # (3 position + 2 orientation) that the 5-DOF arm can track exactly.
        ori_cost: object = np.full(3, args.orientation_cost, dtype=np.float64)
        ori_cost[ord(args.free_roll_axis) - ord("X")] = 0.0
    else:
        ori_cost = 0.0
    ik = BimanualIKSolver(orientation_cost=ori_cost, posture_cost=args.posture_cost)
    sink = OrcaArmMeshcatSink()

    # Anchor pose: lift both arms into a non-singular "shoulder-height in
    # front" config. joint2 is negated on the right side so the carpals land
    # at mirrored Cartesian positions (verified empirically: this is the
    # only sign flip needed for translation symmetry; the carpal frames'
    # rotations are 180° apart by URDF convention regardless).
    q_home = ik.neutral_q.copy()
    side_bias = {"left": {1: 0.1, 3: 1.2}, "right": {1: -0.1, 3: 1.2}}
    for side, bias in side_bias.items():
        idx_q = ik._arm_idx_q[side]
        for k, v in bias.items():
            q_home[idx_q[k]] = v
    T_home = {side: pin.SE3(ik.forward_kinematics_full(q_home, side)) for side in SIDES}

    T_first: dict[str, pin.SE3] = {}
    calib_buf: dict[str, list[pin.SE3]] = {side: [] for side in SIDES}
    span_buf: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    scale: dict[str, float] = {}
    targets: dict[str, pin.SE3] = {}
    q_prev = q_home.copy()

    landmarks_q: queue.Queue = queue.Queue(maxsize=QUEUES_MAXSIZE * 4)
    stop_event = threading.Event()
    ingress = IngressServer(landmarks_q, stop_event, port=args.port)
    ingress.start()

    sink.launch()

    publisher_process: multiprocessing.Process | None = None
    if args.local:
        publisher_process = multiprocessing.Process(
            target=_metaquest_publisher,
            args=(args.port,),
            name="metaquest-publisher",
            daemon=True,
        )
        publisher_process.start()
        logger.info("Local publisher started (pid=%d)", publisher_process.pid)

    logger.info("Ready. Waiting for publisher on :%d. Ctrl+C to stop.", args.port)

    free_pos_idx = (
        None if args.free_position_axis == "none" else ord(args.free_position_axis) - ord("X")
    )

    period = 1.0 / args.ik_rate
    next_tick = time.monotonic()
    last_log = time.monotonic()
    ik_calls = 0

    try:
        while True:
            _drain_queue(
                landmarks_q,
                calib_buf,
                span_buf,
                T_first,
                T_home,
                scale,
                targets,
                args.translation_scale,
                args.workspace_half_box,
                args.auto_fit_margin,
                args.span_samples,
            )

            if targets:
                if free_pos_idx is not None:
                    # Spoof the target's free-axis component with the current FK
                    # so the IK has zero error along that axis and won't try to
                    # move it. World-frame, since FK and target both live in world.
                    for side in targets:
                        cur_p = ik.forward_kinematics(q_prev, side)
                        T_t = targets[side]
                        new_p = T_t.translation.copy()
                        new_p[free_pos_idx] = cur_p[free_pos_idx]
                        targets[side] = pin.SE3(T_t.rotation, new_p)
                result = ik.solve(targets, q_prev)
                # No more snap-back to q_home: the posture-regularization task
                # already keeps q from drifting into bad branches, so carrying
                # q_prev gives temporally smooth output even when the target is
                # briefly unreachable.
                q_prev = result.q
                arm_angles = {
                    side: np.array([result.q[idx] for idx in ik._arm_idx_q[side]])
                    for side in targets
                }
                target_Ts = {side: targets[side].homogeneous for side in targets}
                sink.update(arm_angles, target_Ts=target_Ts)
                ik_calls += 1

            now = time.monotonic()
            if now - last_log > 5.0:
                logger.info(
                    "ik_calls=%d  active=%s  calibrated=%s",
                    ik_calls,
                    sorted(targets),
                    sorted(T_first),
                )
                last_log = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ingress.stop()
        sink.close()
        if publisher_process is not None and publisher_process.is_alive():
            publisher_process.terminate()
            publisher_process.join(timeout=3.0)
        logger.info("Done.")


if __name__ == "__main__":
    main()
