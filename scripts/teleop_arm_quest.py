"""End-to-end: MetaQuest publisher → gRPC ingress → wrist adapter → bimanual IK → meshcat sim.

Auto-calibrates per side on the first received frame: the operator's first
wrist pose is anchored to the robot's neutral carpals.  Every subsequent pose
is multiplied by that constant offset to land in robot-world coords, then fed
straight to bimanual IK and rendered in meshcat.

In one terminal:

    python scripts/teleop_arm_quest.py

In another (live Quest stream over HTS):

    python -m orca_teleop.ingress.metaquest.publisher

Or, all-in-one (spawns the live publisher as a child process so you don't
need a second terminal; the publisher still connects over localhost gRPC):

    python scripts/teleop_arm_quest.py --local

For Quest-less testing, ``--local --dummy`` spawns the dataset-replay
publisher as the child process instead::

    python scripts/teleop_arm_quest.py --local --dummy
"""

import argparse
import collections
import logging
import multiprocessing
import queue
import socket
import threading
import time

import numpy as np
import pinocchio as pin
from hand_tracking_sdk.convert import BASIS_UNITY_LEFT_TO_FLU

from orca_teleop.constants import (
    AUTO_FIT_MARGIN,
    BOOTSTRAP_SCALE,
    CLUTCH_GRACE_S,
    DEFAULT_PORT,
    INGRESS_FPS,
    MIN_SPAN_SAMPLES,
    QUEUES_MAXSIZE,
    SPAN_BUFFER_SECONDS,
    SPAN_CHANGE_THRESHOLD,
    SPAN_REFIT_PERIOD_S,
    STILL_THRESHOLD_M,
    STILL_WINDOW_SAMPLES,
    WORKSPACE_HALF_BOX_M,
)
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.orca_arm_sink import BimanualIKSolver, OrcaArmMeshcatSink

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
IK_RATE_HZ = 60

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
    pose_window: dict[str, "collections.deque"],
    span_buf: dict[str, "collections.deque"],
    last_refit_t: dict[str, float],
    clutch_start_t: dict[str, float | None],
    T_first: dict[str, pin.SE3],
    T_home: dict[str, pin.SE3],
    scale: dict[str, float],
    targets: dict[str, pin.SE3],
    ik: BimanualIKSolver,
    q_prev: np.ndarray,
    *,
    manual_scale: float | None,
    workspace_half_box_m: float,
    auto_fit_margin: float,
    min_span_samples: int,
    span_refit_period_s: float,
    span_change_threshold: float,
    still_threshold_m: float,
    still_window_samples: int,
    clutch_grace_s: float,
) -> None:
    """Per-side state machine: ``awaiting_anchor`` → ``tracking`` ⇄ ``clutched``.

    Stillness is the engagement gesture. While ``awaiting_anchor``, the side
    waits for ``still_window_samples`` of low-motion data, then anchors at the
    window mean and goes straight into ``tracking`` (no grace).

    During ``tracking``, detected stillness enters ``clutched``. While clutched
    the robot is frozen and operator motion is ignored. On the first motion
    sample after ``clutch_grace_s`` of clutch time has elapsed, the side
    exits clutch by re-anchoring ``T_first[side]`` to the operator's CURRENT
    pose and ``T_home[side]`` to ``FK(q_prev, side)`` — so dp = 0 at the
    handoff and tracking resumes from wherever the operator just repositioned.

    Span observation is a rolling background process: every visible sample is
    appended to ``span_buf[side]``. Every ``span_refit_period_s``, if the
    buffer holds at least ``min_span_samples`` points, fit a fresh translation
    scale and swap it in only if it would change by more than
    ``span_change_threshold`` (relative).

    Mutates ``T_first``, ``T_home``, ``scale``, ``targets``,
    ``clutch_start_t``, ``last_refit_t``, ``pose_window``, ``span_buf``
    in place.
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

        pose_window[side].append(T_op)

        # Stillness check needs a full window. While the deque is filling, we
        # treat the side as not-still — the awaiting-anchor branch will simply
        # keep waiting; the engaged branch (if reached) tracks normally.
        full_window = len(pose_window[side]) >= still_window_samples
        if full_window:
            pts = np.array([T.translation for T in pose_window[side]])
            still = float(np.max(pts.max(axis=0) - pts.min(axis=0))) < still_threshold_m
        else:
            still = False

        # Phase: awaiting_anchor — sit until the operator holds still. Initial
        # anchor goes straight into tracking (clutch_start_t stays None).
        if side not in T_first:
            if still:
                p_first = pts.mean(axis=0)
                R_first = _mean_rotation([T.rotation for T in pose_window[side]])
                T_first[side] = pin.SE3(R_first, p_first)
                clutch_start_t[side] = None
                # Seed an initial target at the side's home pose so the IK has
                # something to track immediately (delta = 0 → no motion).
                targets[side] = pin.SE3(T_home[side].rotation, T_home[side].translation.copy())
                if manual_scale is not None:
                    scale[side] = manual_scale
                logger.info(
                    "Anchored %s on stillness (op centroid=%s)",
                    side,
                    np.round(p_first, 3).tolist(),
                )
            continue

        # Phase: engaged.  Maintain the rolling span buffer regardless of
        # stillness — old samples drop off the deque tail, so prolonged
        # stillness doesn't stall the buffer.
        span_buf[side].append(T_op.translation.copy())
        if side not in scale:
            scale[side] = BOOTSTRAP_SCALE

        now = time.monotonic()
        if (
            manual_scale is None
            and len(span_buf[side]) >= min_span_samples
            and now - last_refit_t.get(side, 0.0) >= span_refit_period_s
        ):
            buf_pts = np.array(span_buf[side])
            max_half = float(((buf_pts.max(axis=0) - buf_pts.min(axis=0)) / 2.0).max())
            fitted = (auto_fit_margin * workspace_half_box_m) / max(max_half, 1e-3)
            fitted = float(np.clip(fitted, 0.05, 1.0))
            old = scale[side]
            if abs(fitted - old) / max(old, 1e-6) > span_change_threshold:
                scale[side] = fitted
                logger.info(
                    "Span re-fit %s: %.3f → %.3f (op_max_half=%.3fm, n=%d)",
                    side,
                    old,
                    fitted,
                    max_half,
                    len(span_buf[side]),
                )
            last_refit_t[side] = now

        # Stillness during engaged: enter clutch (or stay clutched). Robot
        # frozen; operator's current motion is irrelevant during clutch.
        if still:
            if clutch_start_t[side] is None:
                clutch_start_t[side] = now
                logger.info("Clutched %s (still detected)", side)
            continue

        # Moving: are we currently clutched?
        if clutch_start_t[side] is not None:
            elapsed = now - clutch_start_t[side]
            if elapsed < clutch_grace_s:
                # In grace: ignore motion, robot stays frozen.
                continue
            # Grace expired and operator is moving: exit clutch with a no-snap
            # re-anchor at the operator's CURRENT pose. dp = 0 this frame, so
            # the robot doesn't jump; the new anchor becomes the new origin
            # for subsequent deltas.
            T_first[side] = pin.SE3(T_op.rotation.copy(), T_op.translation.copy())
            T_home[side] = pin.SE3(ik.forward_kinematics_full(q_prev, side))
            clutch_start_t[side] = None
            logger.info("Re-anchored %s on motion resume after %.1fs clutch", side, elapsed)

        # Normal teleop mapping
        s = scale[side]
        dR = T_op.rotation @ T_first[side].rotation.T
        dp = s * (T_op.translation - T_first[side].translation)
        if workspace_half_box_m > 0.0:
            dp = np.clip(dp, -workspace_half_box_m, workspace_half_box_m)
        targets[side] = pin.SE3(
            dR @ T_home[side].rotation,
            T_home[side].translation + dp,
        )


def _metaquest_publisher(
    port: int,
    *,
    dummy: bool,
    transport: str,
    quest_host: str,
    quest_port: int,
) -> None:
    """Child-process entry: wait for the local ingress, then run the publisher.

    Mirrors ``pipeline._mediapipe_publisher``: the publisher always speaks
    gRPC to ``localhost:port``, regardless of whether the data source is a
    real Quest (``MetaQuestPublisher``) or the recorded dataset
    (``DummyMetaQuestPublisher``).
    """
    from hand_tracking_sdk import TransportMode

    from orca_teleop.ingress.metaquest.publisher import (
        DummyMetaQuestPublisher,
        MetaQuestPublisher,
    )

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

    if dummy:
        DummyMetaQuestPublisher(server_address=server_address).run()
    else:
        MetaQuestPublisher(
            server_address=server_address,
            transport_mode=TransportMode(transport),
            quest_host=quest_host,
            quest_port=quest_port,
        ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="gRPC ingress port")
    parser.add_argument("--ik-rate", type=float, default=IK_RATE_HZ, help="IK/render rate (Hz)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="spawn the publisher as a child process so a single command"
        " runs both ends. The publisher still connects to the local gRPC"
        " ingress on --port. Default child = live MetaQuestPublisher; with"
        " --dummy = DummyMetaQuestPublisher (HF dataset replay).",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="with --local, replay the HF dataset over gRPC instead of"
        " streaming live from a real Quest",
    )
    parser.add_argument(
        "--transport",
        choices=["udp", "tcp_server", "tcp_client"],
        default="udp",
        help="HTS transport mode (live --local only)",
    )
    parser.add_argument(
        "--quest-host",
        default="0.0.0.0",
        help="HTS bind/connect host (live --local only)",
    )
    parser.add_argument(
        "--quest-port",
        type=int,
        default=8765,
        help="HTS bind/connect port (live --local only)",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=None,
        help="manual translation scale; if unset, auto-fit from the operator's"
        " observed span (constants in orca_teleop.constants)",
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
        # (3 position + 2 orientation) that 5-DOF arm can track exactly.
        orientation_cost: object = np.full(3, args.orientation_cost, dtype=np.float64)
        orientation_cost[ord(args.free_roll_axis) - ord("X")] = 0.0
    else:
        orientation_cost = 0.0
    ik = BimanualIKSolver(orientation_cost=orientation_cost, posture_cost=args.posture_cost)
    sink = OrcaArmMeshcatSink()

    # Sanity: the IK uses pinocchio's q ordering, the sink uses yourdfpy's
    # actuated-joint ordering. They both look up by name, but assert the two
    # mappings actually agree before we stream q values between them.
    expected_names = {side: [f"openarm_{side}_joint{i}" for i in range(1, 6)] for side in SIDES}
    assert ik.arm_joint_names == sink.arm_joint_names == expected_names, (
        f"Arm joint index mapping mismatch:\n"
        f"  ik:       {ik.arm_joint_names}\n"
        f"  sink:     {sink.arm_joint_names}\n"
        f"  expected: {expected_names}"
    )

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
    T_home: dict[str, pin.SE3] = {
        side: pin.SE3(ik.forward_kinematics_full(q_home, side)) for side in SIDES
    }

    span_buffer_maxlen = max(int(SPAN_BUFFER_SECONDS * INGRESS_FPS), MIN_SPAN_SAMPLES)
    pose_window: dict[str, collections.deque] = {
        side: collections.deque(maxlen=STILL_WINDOW_SAMPLES) for side in SIDES
    }
    span_buf: dict[str, collections.deque] = {
        side: collections.deque(maxlen=span_buffer_maxlen) for side in SIDES
    }
    last_refit_t: dict[str, float] = {side: 0.0 for side in SIDES}
    clutch_start_t: dict[str, float | None] = {side: None for side in SIDES}
    T_first: dict[str, pin.SE3] = {}
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
            kwargs={
                "dummy": args.dummy,
                "transport": args.transport,
                "quest_host": args.quest_host,
                "quest_port": args.quest_port,
            },
            name="metaquest-publisher",
            daemon=True,
        )
        publisher_process.start()
        kind = (
            "dummy (HF dataset replay)"
            if args.dummy
            else f"live HTS ({args.transport} {args.quest_host}:{args.quest_port})"
        )
        logger.info("Local publisher started (pid=%d, %s)", publisher_process.pid, kind)

    logger.info("Ready. Waiting for publisher on :%d. Ctrl+C to stop.", args.port)

    period = 1.0 / args.ik_rate
    next_tick = time.monotonic()
    last_log = time.monotonic()
    ik_calls = 0

    try:
        while True:
            _drain_queue(
                landmarks_q,
                pose_window,
                span_buf,
                last_refit_t,
                clutch_start_t,
                T_first,
                T_home,
                scale,
                targets,
                ik,
                q_prev,
                manual_scale=args.translation_scale,
                workspace_half_box_m=WORKSPACE_HALF_BOX_M,
                auto_fit_margin=AUTO_FIT_MARGIN,
                min_span_samples=MIN_SPAN_SAMPLES,
                span_refit_period_s=SPAN_REFIT_PERIOD_S,
                span_change_threshold=SPAN_CHANGE_THRESHOLD,
                still_threshold_m=STILL_THRESHOLD_M,
                still_window_samples=STILL_WINDOW_SAMPLES,
                clutch_grace_s=CLUTCH_GRACE_S,
            )

            if targets:
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
