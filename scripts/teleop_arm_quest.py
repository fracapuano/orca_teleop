"""End-to-end: MetaQuest publisher → gRPC ingress → wrist adapter → bimanual IK → viewer sim.

Auto-calibrates per side on the first received frame: the operator's first
wrist pose is anchored to the robot's neutral carpals.  Every subsequent pose
is multiplied by that constant offset to land in robot-world coords, then fed
straight to bimanual IK and sent to the selected sink.

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

To drive the cube-stacking task env instead of a viewer-only sink::

    python scripts/teleop_arm_quest.py --local --dummy --renderer cube-stacking
"""

import argparse
import collections
import logging
import multiprocessing
import queue
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin
from orca_core import OrcaJointPositions

from orca_teleop.constants import (
    BOOTSTRAP_SCALE,
    CLUTCH_GRACE_S,
    CUTOFF_MIN,
    DEFAULT_PORT,
    MAX_JOINT_STEP_RAD,
    MIN_SPAN_SAMPLES,
    OPERATOR_WRIST_WORKSPACE_LIMITS_M,
    QUEUES_MAXSIZE,
    SPAN_CHANGE_THRESHOLD,
    SPAN_REFIT_PERIOD_S,
    STILL_THRESHOLD_M,
    STILL_WINDOW_SAMPLES,
    WORKSPACE_DELTA_LIMITS_BY_EMBODIMENT_M,
)
from orca_teleop.ingress.metaquest.landmarks import retargeter_landmarks_from_quest
from orca_teleop.ingress.server import HandLandmarks, IngressServer
from orca_teleop.orca_arm_ik import (
    ArmIKConfig,
    BimanualIKSolver,
    default_orca_arm_ik_config,
    orca_panda_right_ik_config,
)
from orca_teleop.orca_arm_sink import OrcaArmMeshcatSink, OrcaArmMujocoSink
from orca_teleop.orca_arm_teleop import (
    OrcaArmTeleopConfig,
    OrcaArmTeleopController,
    OrcaArmTeleopFrame,
    metaquest_wrist_pose_to_robot_se3,
)
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
IK_RATE_HZ = 60


@dataclass(slots=True)
class MetaQuestDrainStats:
    """Small ingress diagnostic for the wrist-pose teleop queue."""

    total_items: int = 0
    hand_landmarks: int = 0
    missing_wrist_pose: int = 0
    inactive_side: int = 0
    usable_active: int = 0
    non_hand_landmarks: int = 0
    by_side: collections.Counter[str] = field(default_factory=collections.Counter)
    usable_by_side: collections.Counter[str] = field(default_factory=collections.Counter)
    last_side: str | None = None
    last_had_wrist_pose: bool | None = None
    last_item_monotonic: float | None = None
    last_timestamp_ns: int | None = None

    def record_item(self, item: object, *, active_sides: tuple[str, ...]) -> None:
        self.total_items += 1
        self.last_item_monotonic = time.monotonic()
        if not isinstance(item, HandLandmarks):
            self.non_hand_landmarks += 1
            self.last_side = None
            self.last_had_wrist_pose = None
            self.last_timestamp_ns = None
            return

        self.hand_landmarks += 1
        self.last_side = item.handedness
        self.last_had_wrist_pose = item.wrist_pose is not None
        self.last_timestamp_ns = int(item.timestamp_ns)
        self.by_side.update((item.handedness,))

        if item.wrist_pose is None:
            self.missing_wrist_pose += 1
            return
        if item.handedness not in active_sides:
            self.inactive_side += 1
            return
        self.usable_active += 1
        self.usable_by_side.update((item.handedness,))

    def summary(self, *, active_sides: tuple[str, ...], now: float | None = None) -> str:
        if now is None:
            now = time.monotonic()
        age = "n/a"
        if self.last_item_monotonic is not None:
            age = f"{now - self.last_item_monotonic:.2f}"
        last = "none"
        if self.last_side is not None:
            wrist = "yes" if self.last_had_wrist_pose else "no"
            last = f"{self.last_side},wrist={wrist},age_s={age}"
        return (
            f"active={list(active_sides)} drained_total={self.total_items} "
            f"hand_landmarks={self.hand_landmarks} usable_active={self.usable_active} "
            f"by_side={dict(self.by_side)} usable_by_side={dict(self.usable_by_side)} "
            f"missing_wrist_pose={self.missing_wrist_pose} "
            f"inactive_side={self.inactive_side} non_hand_landmarks={self.non_hand_landmarks} "
            f"last={last}"
        )


def _speak_status(message: str, *, enabled: bool) -> None:
    """Say a short operator status message without blocking teleop."""
    if not enabled:
        return
    say_path = shutil.which("say")
    if say_path is None:
        logger.warning("Cannot speak status %r because macOS 'say' is not on PATH.", message)
        return
    try:
        subprocess.Popen(  # noqa: S603
            [say_path, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        logger.warning("Could not speak status %r: %s", message, exc)


def _wrist_pose_to_robot_se3(position: np.ndarray, rotation: np.ndarray) -> pin.SE3:
    """Quest wrist pose (Unity left-handed) → pin.SE3 in robot world (FLU) coords."""
    return metaquest_wrist_pose_to_robot_se3(position, rotation)


def _relative_flu_z_angle_degrees(T_zero: pin.SE3, T_now: pin.SE3) -> float:
    """Signed relative rotation around local FLU +Z, in degrees."""
    dR = T_zero.rotation.T @ T_now.rotation
    return float(np.rad2deg(np.arctan2(dR[1, 0], dR[0, 0])))


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """SVD-based rotation average (Markley's method) over a list of 3x3 mats."""
    M = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _default_hand_model_path() -> str:
    """Prefer the installed v2 hand model topology for finger retargeting."""
    import orca_core

    models_dir = Path(orca_core.__file__).resolve().parent / "models"
    candidates = (
        models_dir / "v2" / "orcahand_right" / "config.yaml",
        models_dir / "v1" / "orcahand_right" / "config.yaml",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    raise RuntimeError(f"No bundled OrcaHand config.yaml found under {models_dir}")


def _retargeted_action_for_side(
    action: OrcaJointPositions,
    side: str,
) -> OrcaJointPositions:
    """Adapt shared right-hand retargeter output to the mirrored OrcaArm side."""
    if side != "left":
        return action

    data = action.as_dict()
    if "wrist" in data:
        data["wrist"] = -data["wrist"]
    return OrcaJointPositions(data)


def _split_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _drain_latest_metaquest_frames(
    landmarks_q: "queue.Queue",
    active_sides: tuple[str, ...],
    stats: MetaQuestDrainStats | None = None,
) -> list[OrcaArmTeleopFrame]:
    """Drain ingress queue and return the latest controller frame per side."""
    latest_by_side: dict[str, HandLandmarks] = {}
    while True:
        try:
            item = landmarks_q.get_nowait()
        except queue.Empty:
            break
        if stats is not None:
            stats.record_item(item, active_sides=active_sides)
        if not isinstance(item, HandLandmarks) or item.wrist_pose is None:
            continue
        if item.handedness not in active_sides:
            continue
        latest_by_side[item.handedness] = item

    frames: list[OrcaArmTeleopFrame] = []
    for side in active_sides:
        item = latest_by_side.get(side)
        if item is None:
            continue
        frames.append(
            OrcaArmTeleopFrame(
                handedness=side,
                timestamp_ns=int(item.timestamp_ns),
                wrist_pose=metaquest_wrist_pose_to_robot_se3(
                    item.wrist_pose.position,
                    item.wrist_pose.rotation,
                ),
                keypoints=item.keypoints,
                keypoint_source="metaquest",
            )
        )
    return frames


def _build_ik_config(args: argparse.Namespace) -> ArmIKConfig:
    """Resolve CLI embodiment metadata into a reusable IK config."""
    if args.embodiment == "orca-panda":
        requested_sides = _split_csv(args.active_sides)
        if requested_sides and requested_sides != ("right",):
            raise ValueError("orca-panda preset only supports --active-sides right.")
        if args.left_arm_joints or args.left_ee_frame:
            raise ValueError("orca-panda is single-arm; use right-side IK arguments only.")
        right_frame = args.right_ee_frame or "orcahand_right_R-Carpals_8d1f1041"
        config = orca_panda_right_ik_config(
            urdf_path=args.arm_urdf_path,
            ee_frame=right_frame,
        )
        right_joints = _split_csv(args.right_arm_joints)
        if not right_joints:
            return config
        return ArmIKConfig(
            urdf_path=config.urdf_path,
            sides=config.sides,
            joint_names_by_side={"right": right_joints},
            ee_frame_by_side=config.ee_frame_by_side,
        )

    base = default_orca_arm_ik_config()
    active_sides = _split_csv(args.active_sides) or base.sides
    joint_names_by_side: dict[str, tuple[str, ...]] = {}
    ee_frame_by_side: dict[str, str] = {}

    for side in active_sides:
        if side not in SIDES:
            raise ValueError(f"Unsupported side {side!r}; expected one of {SIDES}")
        override = _split_csv(args.left_arm_joints if side == "left" else args.right_arm_joints)
        joint_names_by_side[side] = override or base.joint_names_by_side[side]
        frame_override = args.left_ee_frame if side == "left" else args.right_ee_frame
        if frame_override:
            ee_frame_by_side[side] = frame_override

    return ArmIKConfig(
        urdf_path=args.arm_urdf_path or base.urdf_path,
        sides=active_sides,
        joint_names_by_side=joint_names_by_side,
        ee_frame_by_side=ee_frame_by_side,
    )


def _meshcat_sink_for_ik_config(
    ik_config: ArmIKConfig,
    *,
    home_arm_angles: dict[str, np.ndarray] | None = None,
) -> OrcaArmMeshcatSink:
    """Build a Meshcat sink that mirrors the active IK embodiment metadata."""
    return OrcaArmMeshcatSink(
        urdf_path=ik_config.urdf_path,
        sides=ik_config.sides,
        joint_names_by_side=ik_config.joint_names_by_side,
        ee_frame_by_side=ik_config.ee_frame_by_side,
        home_arm_angles=home_arm_angles,
    )


def _load_orcapanda_env_home_arm_angles(
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Read the OrcaPanda reset/home pose from orca_sim without opening a viewer."""
    from orca_teleop.sim import OrcaPandaCubeStackingSink

    sink = OrcaPandaCubeStackingSink(
        render_mode=None,
        version=args.task_version,
        max_episode_steps=args.task_max_episode_steps,
        reset_on_done=not args.no_task_reset_on_done,
        seed=args.task_seed,
        instant_qpos=args.task_instant_qpos,
    )
    try:
        sink.launch()
        return sink.home_arm_angles
    finally:
        sink.close()


def _orientation_cost_for_args(args: argparse.Namespace) -> float | np.ndarray:
    """Resolve CLI orientation-cost semantics for the selected embodiment."""
    if args.orientation_cost <= 0.0:
        return 0.0

    if args.embodiment == "orca-panda" and args.ik_mode == "pose":
        # The Panda/Franka arm has enough DOF for full SE(3) wrist tracking, so
        # do not free a roll axis as we do for the 5-DOF OrcaArm preset.
        return float(args.orientation_cost)

    # Free roll about the chosen body-frame axis: zero out that axis's cost,
    # keep the other two at args.orientation_cost. This gives a 5-DOF target
    # (3 position + 2 orientation) that 5-DOF arms can track exactly.
    orientation_cost = np.full(3, args.orientation_cost, dtype=np.float64)
    orientation_cost[ord(args.free_roll_axis) - ord("X")] = 0.0
    return orientation_cost


def _workspace_delta_limits_for_args(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None:
    """Resolve translation clipping for the selected embodiment."""
    workspace_calibration_active = (
        args.translation_scale is None and args.use_translation_workspace_calibration
    )
    if args.clip_translation is None:
        clip_translation = (
            args.embodiment != "orca-panda"
            or args.translation_limit_m is not None
            or workspace_calibration_active
        )
    else:
        clip_translation = bool(args.clip_translation)

    if workspace_calibration_active and not clip_translation:
        raise ValueError(
            "--use-translation-workspace-calibration requires translation clipping limits"
        )

    if not clip_translation:
        return None

    if args.translation_limit_m is not None:
        limit = float(args.translation_limit_m)
        if limit <= 0.0:
            raise ValueError("--translation-limit-m must be positive")
        lo = (-limit, -limit, -limit)
        hi = (+limit, +limit, +limit)
        return {side: (lo, hi) for side in active_sides}

    embodiment_limits = WORKSPACE_DELTA_LIMITS_BY_EMBODIMENT_M[args.embodiment]
    missing = tuple(side for side in active_sides if side not in embodiment_limits)
    if missing:
        raise ValueError(
            f"No workspace delta limits configured for {args.embodiment} side(s): {missing}"
        )
    return {side: embodiment_limits[side] for side in active_sides}


def _drain_queue(
    landmarks_q: "queue.Queue",
    pose_window: dict[str, "collections.deque"],
    span_buffer: dict[str, "collections.deque"],
    last_refit_t: dict[str, float],
    clutch_start_t: dict[str, float | None],
    T_first: dict[str, pin.SE3],
    T_home: dict[str, pin.SE3],
    scale: dict[str, np.ndarray],
    targets: dict[str, pin.SE3],
    hand_targets: dict[str, OrcaJointPositions],
    new_target_acquire_ns: dict[str, int],
    ik: BimanualIKSolver,
    retargeters: dict[str, Retargeter],
    q_prev: np.ndarray,
    *,
    manual_scale: float | None,
    workspace_delta_limits_m: dict[
        str, tuple[tuple[float, float, float], tuple[float, float, float]]
    ],
    auto_fit_margin: float,
    min_span_samples: int,
    span_refit_period_s: float,
    span_change_threshold: float,
    still_threshold_m: float,
    still_window_samples: int,
    clutch_grace_s: float,
) -> None:
    # NOTE: yes, the signature is long. Every dict here is real per-side state
    # that we mutate in place, plus the tuning knobs the script passes through
    # from constants/CLI flags. The refactor is to bundle these into a state
    # dataclass + a config dataclass, but that's a follow-up after we ship —
    # the explicit list keeps the contract visible while the state machine
    # is still being tuned.
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
    latest_by_side: dict[str, HandLandmarks] = {}
    while True:
        try:
            item = landmarks_q.get_nowait()
        except queue.Empty:
            break
        if not isinstance(item, HandLandmarks) or item.wrist_pose is None:
            continue

        side = item.handedness
        if side not in SIDES:
            continue

        latest_by_side[side] = item

    for side in SIDES:
        item = latest_by_side.get(side)
        if item is None:
            continue

        # Per-side asymmetric reach envelope, captured once per tick so the
        # span re-fit and the clip site agree on the same numbers.
        side_lo, side_hi = (
            np.asarray(workspace_delta_limits_m[side][0], dtype=np.float64),
            np.asarray(workspace_delta_limits_m[side][1], dtype=np.float64),
        )

        # Convert the wrist pose to FLU coordinates before deriving arm targets
        # or the direct hand-wrist motor command.
        T_op = _wrist_pose_to_robot_se3(item.wrist_pose.position, item.wrist_pose.rotation)
        wrist_angle_degrees = (
            _relative_flu_z_angle_degrees(T_first[side], T_op) if side in T_first else 0.0
        )

        try:
            hand_action = retargeters[side].retarget(
                TargetPose(
                    joint_positions=retargeter_landmarks_from_quest(item.keypoints, side),
                    source="metaquest",
                    wrist_angle_degrees=wrist_angle_degrees,
                )
            )

            if hand_action is not None:
                hand_targets[side] = _retargeted_action_for_side(hand_action, side)

        except (AssertionError, ValueError):
            logger.debug("Skipping degenerate %s hand landmark frame.", side)

        pose_window[side].append(T_op)

        # Stillness-check needs a full window.
        full_window = len(pose_window[side]) == pose_window[side].maxlen
        still = False
        if full_window:
            positions = np.array([T.translation for T in pose_window[side]])
            still = float(np.max(positions.max(axis=0) - positions.min(axis=0))) < still_threshold_m

        # Phase: awaiting_anchor — sit until the operator holds still. Also, tracking starts
        if side not in T_first:
            logger.info("Awaiting anchor for %s", side)

            if still:
                p_first = positions.mean(axis=0)
                R_first = _mean_rotation([T.rotation for T in pose_window[side]])
                T_first[side] = pin.SE3(R_first, p_first)
                clutch_start_t[side] = None

                logger.info(
                    "Anchored %s on stillness (operator centroid=%s)",
                    side,
                    np.round(p_first, 3).tolist(),
                )

                # Seed an initial target at the side's home pose so the IK has
                # something to track immediately (delta = 0 is lack of motion).
                targets[side] = pin.SE3(T_home[side].rotation, T_home[side].translation.copy())
                new_target_acquire_ns[side] = int(item.timestamp_ns)
                if manual_scale is not None:
                    scale[side] = np.full(3, float(manual_scale), dtype=np.float64)

            continue

        # Phase: engaged.  Maintain the rolling span buffer regardless of
        # stillness — old samples drop off the deque tail, so prolonged
        # stillness doesn't stall the buffer.

        # Measuring the operator's ROM to estimate workspaces' ratio
        span_buffer[side].append(T_op.translation.copy())
        if side not in scale:
            scale[side] = np.full(3, BOOTSTRAP_SCALE, dtype=np.float64)

        now = time.monotonic()
        if (
            manual_scale is None
            # NOTE: keeping as arg to quickly tune, but should really be size of buffer
            and len(span_buffer[side]) >= min_span_samples
            and now - last_refit_t.get(side, 0.0) >= span_refit_period_s
        ):
            buffer_positions = np.array(span_buffer[side])
            operator_halfspace = (buffer_positions.max(axis=0) - buffer_positions.min(axis=0)) / 2.0
            # The reach envelope is asymmetric; the scale ratio only needs a
            # per-axis magnitude. (hi - lo) / 2 is the half-width of the box on
            # each axis and matches what the old symmetric constant encoded.
            robot_halfspace = (side_hi - side_lo) / 2.0

            # Per-axis gain: each axis gets its own ratio so a cramped reach on
            # one axis doesn't drag the gain down on the others. Clipped to
            # ``[CUTOFF_MIN, 1 - CUTOFF_MIN]`` to dodge the 0 / 1 singularities.
            new_ratio = np.clip(
                robot_halfspace / np.maximum(operator_halfspace, 1e-3),
                CUTOFF_MIN,
                1 - CUTOFF_MIN,
            )
            old_ratio = scale[side]
            rel_change = np.max(np.abs(new_ratio - old_ratio) / np.maximum(old_ratio, 1e-6))
            if rel_change > span_change_threshold:
                scale[side] = new_ratio
                logger.info(
                    "Span re-fit %s: %s → %s (op_half=%s m, rel_change=%.3f, n=%d)",
                    side,
                    np.round(old_ratio, 3).tolist(),
                    np.round(new_ratio, 3).tolist(),
                    np.round(operator_halfspace, 3).tolist(),
                    float(rel_change),
                    len(span_buffer[side]),
                )
            last_refit_t[side] = now

        # Stillness while engaged enters clutch mode (or stays clutched)
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

            # exit clutch with a no-snap, re-anchor at the operator's CURRENT pose
            T_first[side] = pin.SE3(T_op.rotation.copy(), T_op.translation.copy())
            T_home[side] = pin.SE3(ik.forward_kinematics_full(q_prev, side))
            clutch_start_t[side] = None

            logger.info("Re-anchored %s on motion resume after clutch", side)

        # Normal teleop mapping
        s = scale[side]
        # Apply the operator's wrist motion as a local relative rotation:
        # R_target = R_robot_home * (R_operator_first^-1 * R_operator_now).
        # Pre-multiplying by a world-frame delta would rotate the robot wrist
        # around Quest/world axes, which only works when the operator and robot
        # frames are already aligned.
        dR = T_first[side].rotation.T @ T_op.rotation
        dp = s * (T_op.translation - T_first[side].translation)

        # Asymmetric per-axis clip: the carpals reach further backward than
        # forward (left arm) / further right than left (right arm) / much further
        # up than down. Clipping to (side_lo, side_hi) directly preserves that
        # asymmetry instead of collapsing to the worst symmetric scalar.
        dp = np.clip(dp, side_lo, side_hi)

        targets[side] = pin.SE3(
            T_home[side].rotation @ dR,
            T_home[side].translation + dp,
        )
        new_target_acquire_ns[side] = int(item.timestamp_ns)


def _metaquest_publisher(
    port: int,
    *,
    dummy: bool,
    dummy_repo: str,
    dummy_filename: str,
    dummy_loop: bool,
    dummy_refresh: bool,
    transport: str,
    quest_host: str,
    quest_port: int,
    log_level: str,
) -> None:
    """Child-process entry: wait for the local ingress, then run the publisher.

    Mirrors ``pipeline._mediapipe_publisher``: the publisher always speaks
    gRPC to ``localhost:port``, regardless of whether the data source is a
    real Quest (``MetaQuestPublisher``) or the recorded dataset
    (``DummyMetaQuestPublisher``).
    """
    # On macOS, multiprocessing uses 'spawn'; the child inherits no logging
    # config from the parent. Re-init here so the publisher's handshake logs
    # ("Connecting to ...", "HTS producer started", per-side first-frame)
    # actually reach the user's terminal.
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

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
        DummyMetaQuestPublisher(
            server_address=server_address,
            repo=dummy_repo,
            filename=dummy_filename,
            loop=dummy_loop,
            refresh=dummy_refresh,
        ).run()
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
        "--dummy-repo",
        default="fracapuano/quest-calibration",
        help="Hugging Face dataset repo used by --local --dummy",
    )
    parser.add_argument(
        "--dummy-filename",
        default="data.parquet",
        help="parquet filename inside --dummy-repo",
    )
    parser.add_argument(
        "--dummy-no-loop",
        action="store_true",
        help="with --local --dummy, stop at the end of the dataset instead of looping",
    )
    parser.add_argument(
        "--dummy-refresh",
        action="store_true",
        help="with --local --dummy, force re-download of the dataset file",
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
        help="fixed scalar translation scale. If unset, use the controller"
        " workspace calibration unless disabled.",
    )
    parser.add_argument(
        "--translation-frame",
        choices=["operator_local", "world"],
        default="operator_local",
        help="frame used for startup-relative operator translation. operator_local"
        " applies the same relative-SE(3) convention as orientation.",
    )
    parser.add_argument(
        "--use-translation-workspace-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="map operator workspace endpoints onto the selected embodiment's"
        " robot workspace endpoints from constants when --translation-scale is absent.",
    )
    parser.add_argument(
        "--clip-translation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="clip startup-relative translation deltas. Defaults to enabled when"
        " workspace calibration is active, enabled for orca-arm, and disabled"
        " for raw orca-panda bootstrap/manual scale.",
    )
    parser.add_argument(
        "--translation-limit-m",
        type=float,
        default=None,
        help="optional symmetric per-axis translation cap in meters. Setting this"
        " also enables translation clipping.",
    )
    parser.add_argument(
        "--require-operator-neutral",
        action="store_true",
        help="deprecated alias for the neutral start delay: hold the robot at"
        " home, prompt the operator to move to neutral when hand frames arrive,"
        " then anchor after --neutral-start-delay-s.",
    )
    parser.add_argument(
        "--neutral-start-delay-s",
        type=float,
        default=2.0,
        help="seconds to wait after the first active hand frame before allowing"
        " teleop to anchor. Set 0 to anchor immediately.",
    )
    parser.add_argument(
        "--operator-neutral-position-tolerance-m",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--operator-neutral-orientation-tolerance-deg",
        type=float,
        default=15.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--orientation-cost",
        type=float,
        default=1.0,
        help="orientation cost in --ik-mode pose; 0 = position-only. With"
        " --embodiment orca-panda this tracks full orientation. With the"
        " 5-DOF orca-arm preset it tracks the two non-roll axes"
        " (see --free-roll-axis).",
    )
    parser.add_argument(
        "--free-roll-axis",
        default="Z",
        choices=["X", "Y", "Z"],
        help="body-frame axis whose rotation is unconstrained when orientation-cost > 0."
        " Default Z: in the home pose body-frame Z is ~antiparallel to world FLU +Z,"
        " which is exactly the axis the hand wrist motor compensates for"
        " (see _relative_flu_z_angle_degrees). Leaving Z free in the arm IK and"
        " letting the hand motor add the roll keeps the full 6-DOF operator pose"
        " tracked end-to-end. Y was the previous default — empirically slightly fewer"
        " stuck IK frames (1/900 vs 15/900) measured without the hand motor in the"
        " loop, but it drops *lateral* wrist tilt that nothing else recovers, so the"
        " operator feels the arm refusing to follow their roll. Flip back to Y if a"
        " real session shows the QP getting stuck too often.",
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
        "--embodiment",
        choices=["orca-arm", "orca-panda"],
        default="orca-arm",
        help="IK embodiment preset. orca-arm preserves the current bimanual"
        " OpenArm setup; orca-panda uses panda_joint1..7 and right-hand carpals.",
    )
    parser.add_argument(
        "--home-pose-source",
        choices=["env", "franka-rest", "ik-neutral"],
        default="env",
        help="teleop anchor/start pose source. env adopts the task/MuJoCo reset"
        " pose when the sink exposes one. franka-rest/ik-neutral use the IK"
        " model's neutral q instead, useful for testing OrcaPanda from the"
        " Franka rest configuration.",
    )
    parser.add_argument(
        "--arm-urdf-path",
        default=None,
        help="URDF used by the arm IK. Defaults to the selected embodiment preset.",
    )
    parser.add_argument(
        "--active-sides",
        default=None,
        help="comma-separated sides consumed from Quest, e.g. right or left,right."
        " Defaults to the selected embodiment preset.",
    )
    parser.add_argument(
        "--left-arm-joints",
        default=None,
        help="comma-separated left-side IK joint names, overriding the preset.",
    )
    parser.add_argument(
        "--right-arm-joints",
        default=None,
        help="comma-separated right-side IK joint names, overriding the preset.",
    )
    parser.add_argument(
        "--left-ee-frame",
        default=None,
        help="left-side end-effector frame name in the IK URDF.",
    )
    parser.add_argument(
        "--right-ee-frame",
        default=None,
        help="right-side end-effector frame name in the IK URDF.",
    )
    parser.add_argument(
        "--ik-mode",
        choices=["position", "pose"],
        default="position",
        help="arm IK mode. position uses damped least-squares carpals tracking"
        " plus --position-rotation-axes; pose uses the older Pink/QP pose task"
        " with --orientation-cost (default: position).",
    )
    parser.add_argument(
        "--ik-max-iterations",
        type=int,
        default=20,
        help="maximum IK iterations per teleop tick (default: 20).",
    )
    parser.add_argument(
        "--position-damping",
        type=float,
        default=1e-4,
        help="damping for --ik-mode position damped least-squares solve",
    )
    parser.add_argument(
        "--position-step-size",
        type=float,
        default=0.7,
        help="per-iteration step scale for --ik-mode position",
    )
    parser.add_argument(
        "--position-posture-gain",
        type=float,
        default=1e-5,
        help="small joint-space regularizer for --ik-mode position",
    )
    parser.add_argument(
        "--position-rotation-axes",
        default="XZ",
        help="world-frame rotation axes to track in --ik-mode position. Default"
        " XZ ignores Y; pass an empty string for position-only.",
    )
    parser.add_argument(
        "--position-rotation-gain",
        type=float,
        default=3e-6,
        help="relative weight for selected rotation axes in --ik-mode position."
        " This trades meters of carpals error against radians of angular error.",
    )
    parser.add_argument(
        "--workspace-position-clip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="before pose IK, project only the target position onto the arm's"
        " position-reachable workspace using a position-only IK solve. Defaults"
        " to enabled for orca-panda and disabled otherwise.",
    )
    parser.add_argument(
        "--workspace-position-clip-tolerance-m",
        type=float,
        default=0.005,
        help="position error below which workspace position clipping leaves the"
        " original target translation unchanged.",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=MAX_JOINT_STEP_RAD,
        help="maximum per-tick joint change applied after IK",
    )
    parser.add_argument(
        "--pose-filter-alpha",
        type=float,
        default=0.35,
        help="operator wrist pose low-pass alpha in [0, 1]. Lower is smoother;"
        " 1 disables smoothing.",
    )
    parser.add_argument(
        "--max-operator-translation-speed-mps",
        type=float,
        default=2.0,
        help="hold the previous operator wrist pose when a frame exceeds this"
        " translational speed, treating it as Quest tracking noise.",
    )
    parser.add_argument(
        "--max-operator-rotation-speed-radps",
        type=float,
        default=12.0,
        help="hold the previous operator wrist pose when a frame exceeds this"
        " angular speed, treating it as Quest tracking noise.",
    )
    parser.add_argument(
        "--hand-model-path",
        default=None,
        help="OrcaHand config.yaml for finger retargeting. Defaults to the"
        " installed v2 hand model topology and uses one retargeter instance"
        " per streamed side.",
    )
    parser.add_argument(
        "--hand-urdf-path",
        default=None,
        help="OrcaHand URDF for finger retargeting. Defaults to Retargeter's"
        " orcahand_description lookup.",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "meshcat", "mujoco", "cube-stacking", "panda-cube-stacking"],
        default="auto",
        help="sink for the solved arm state. auto uses meshcat for orca-arm and "
        "orca_sim.OrcaPandaCubeStacking for orca-panda.",
    )
    parser.add_argument(
        "--task-render-mode",
        choices=["human", "rgb_array", "none"],
        default="human",
        help="render mode for --renderer cube-stacking (default: human)",
    )
    parser.add_argument(
        "--task-version",
        default=None,
        help="orca_sim scene version for --renderer cube-stacking",
    )
    parser.add_argument(
        "--task-scene-file",
        default=None,
        help="orca_sim scene XML for --renderer cube-stacking. For a lighter"
        " OrcaArm scene, use orcaarm_cube_stacking.xml and pass"
        ' --task-camera-names "".',
    )
    parser.add_argument(
        "--task-camera-names",
        default=None,
        help="comma-separated camera names for task observations. Leave unset"
        " for the env default; pass an empty string to disable camera validation"
        " for lightweight scenes.",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=None,
        help="initial cube-stacking reset seed for --renderer cube-stacking",
    )
    parser.add_argument(
        "--task-max-episode-steps",
        type=int,
        default=10_000,
        help="episode horizon for --renderer cube-stacking (default: 10000)",
    )
    parser.add_argument(
        "--no-task-reset-on-done",
        action="store_true",
        help="do not auto-reset --renderer cube-stacking after success or timeout",
    )
    parser.add_argument(
        "--task-instant-qpos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="for MuJoCo task renderers, write commanded actuator positions"
        " directly into qpos instead of waiting for actuator dynamics. This is"
        " only for debugging visual command following; physics/contact teleop"
        " uses actuator dynamics by default.",
    )
    parser.add_argument(
        "--task-frame-skip",
        type=int,
        default=None,
        help="MuJoCo substeps per task env.step(). In actuator-dynamics mode,"
        " defaults to an auto value that makes simulated time roughly match"
        " --ik-rate for the current 2 ms model timestep. Larger values advance"
        " physics faster per teleop tick but update controls less frequently.",
    )
    parser.add_argument(
        "--task-debug-visuals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw MuJoCo debug target/current/operator triads for task renderers.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--speak-status",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="speak short laptop status cues when neutral anchoring changes.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    orientation_cost = _orientation_cost_for_args(args)
    ik_config = _build_ik_config(args)
    active_sides = ik_config.sides
    workspace_delta_limits_m = _workspace_delta_limits_for_args(args, active_sides)
    use_workspace_calibration = (
        args.translation_scale is None and args.use_translation_workspace_calibration
    )
    if args.translation_scale is not None:
        translation_scale: float | None = float(args.translation_scale)
        logger.info("Using fixed scalar translation scale: %.3f", translation_scale)
    elif use_workspace_calibration:
        translation_scale = None
        logger.info(
            "Using %s calibrated operator-to-robot workspace endpoint mapping.",
            args.embodiment,
        )
    else:
        translation_scale = None
        logger.info("Using controller bootstrap translation scale.")
    if workspace_delta_limits_m is None:
        logger.info("Translation target clipping disabled.")
    else:
        logger.info(
            "Translation target clipping limits: %s",
            {
                side: (
                    np.round(workspace_delta_limits_m[side][0], 3).tolist(),
                    np.round(workspace_delta_limits_m[side][1], 3).tolist(),
                )
                for side in active_sides
            },
        )

    ik = BimanualIKSolver(
        max_iterations=args.ik_max_iterations,
        orientation_cost=orientation_cost,
        posture_cost=args.posture_cost,
        ik_config=ik_config,
    )
    controller = OrcaArmTeleopController(
        config=OrcaArmTeleopConfig(
            manual_scale=translation_scale,
            workspace_delta_limits_m=workspace_delta_limits_m,
            operator_workspace_limits_m=OPERATOR_WRIST_WORKSPACE_LIMITS_M,
            min_span_samples=MIN_SPAN_SAMPLES,
            span_refit_period_s=SPAN_REFIT_PERIOD_S,
            span_change_threshold=SPAN_CHANGE_THRESHOLD,
            still_threshold_m=STILL_THRESHOLD_M,
            still_window_samples=STILL_WINDOW_SAMPLES,
            clutch_grace_s=CLUTCH_GRACE_S,
            max_joint_step_rad=args.max_joint_step_rad,
            pose_filter_alpha=args.pose_filter_alpha,
            max_operator_translation_speed_mps=args.max_operator_translation_speed_mps,
            max_operator_rotation_speed_radps=args.max_operator_rotation_speed_radps,
            hand_model_path=args.hand_model_path,
            hand_urdf_path=args.hand_urdf_path,
            ik_mode=args.ik_mode,
            orientation_cost=orientation_cost,
            posture_cost=args.posture_cost,
            position_damping=args.position_damping,
            position_step_size=args.position_step_size,
            position_posture_gain=args.position_posture_gain,
            position_rotation_axes=args.position_rotation_axes,
            position_rotation_gain=args.position_rotation_gain,
            active_sides=active_sides,
            translation_frame=args.translation_frame,
            use_workspace_calibration=use_workspace_calibration,
            workspace_position_clip=(
                args.embodiment == "orca-panda"
                if args.workspace_position_clip is None
                else args.workspace_position_clip
            ),
            workspace_position_clip_tolerance_m=args.workspace_position_clip_tolerance_m,
        ),
        ik=ik,
    )
    logger.info(
        "Arm IK embodiment=%s urdf=%s active_sides=%s joints=%s ee_frames=%s",
        args.embodiment,
        ik_config.urdf_path,
        active_sides,
        ik.arm_joint_names,
        ik_config.ee_frame_by_side or "auto-carpals",
    )
    logger.info(
        "IK loop: mode=%s max_iterations=%d position_rotation_axes=%r "
        "position_rotation_gain=%.2g",
        args.ik_mode,
        args.ik_max_iterations,
        args.position_rotation_axes,
        args.position_rotation_gain,
    )
    logger.info(
        "Workspace position clipping: enabled=%s tolerance=%.1fmm",
        controller.config.workspace_position_clip,
        1000.0 * controller.config.workspace_position_clip_tolerance_m,
    )
    logger.info(
        "Finger retargeters will initialize per side from %s",
        args.hand_model_path or _default_hand_model_path(),
    )
    renderer = args.renderer
    if renderer == "auto":
        renderer = "panda-cube-stacking" if args.embodiment == "orca-panda" else "meshcat"

    task_instant_qpos = False if args.task_instant_qpos is None else bool(args.task_instant_qpos)
    task_frame_skip = args.task_frame_skip
    if (
        task_frame_skip is None
        and renderer in ("cube-stacking", "panda-cube-stacking")
        and not task_instant_qpos
    ):
        task_frame_skip = max(1, int(round(1.0 / (float(args.ik_rate) * 0.002))))
    if renderer in ("cube-stacking", "panda-cube-stacking"):
        logger.info(
            "MuJoCo task command mode: %s; frame_skip=%s",
            "direct qpos render" if task_instant_qpos else "actuator dynamics",
            "env default" if task_frame_skip is None else task_frame_skip,
        )
    task_camera_names = (
        None if args.task_camera_names is None else _split_csv(args.task_camera_names)
    )

    meshcat_env_home_arm_angles: dict[str, np.ndarray] | None = None
    if renderer == "meshcat" and args.embodiment == "orca-panda" and args.home_pose_source == "env":
        meshcat_env_home_arm_angles = _load_orcapanda_env_home_arm_angles(args)

    if renderer == "mujoco":
        sink = OrcaArmMujocoSink()
    elif renderer == "cube-stacking":
        from orca_teleop.sim import OrcaArmCubeStackingSink

        sink = OrcaArmCubeStackingSink(
            render_mode=None if args.task_render_mode == "none" else args.task_render_mode,
            version=args.task_version,
            scene_file=args.task_scene_file,
            camera_names=task_camera_names,
            max_episode_steps=args.task_max_episode_steps,
            reset_on_done=not args.no_task_reset_on_done,
            seed=args.task_seed,
            instant_qpos=task_instant_qpos,
            frame_skip=task_frame_skip,
            debug_visuals=args.task_debug_visuals,
        )
    elif renderer == "panda-cube-stacking":
        from orca_teleop.sim import OrcaPandaCubeStackingSink

        sink = OrcaPandaCubeStackingSink(
            render_mode=None if args.task_render_mode == "none" else args.task_render_mode,
            version=args.task_version,
            scene_file=args.task_scene_file,
            camera_names=task_camera_names,
            max_episode_steps=args.task_max_episode_steps,
            reset_on_done=not args.no_task_reset_on_done,
            seed=args.task_seed,
            instant_qpos=task_instant_qpos,
            frame_skip=task_frame_skip,
            debug_visuals=args.task_debug_visuals,
        )
    else:
        sink = _meshcat_sink_for_ik_config(
            ik_config,
            home_arm_angles=meshcat_env_home_arm_angles,
        )

    # Sanity: the IK uses pinocchio's q ordering, the sink uses yourdfpy's
    # actuated-joint ordering. They both look up by name, but assert the two
    # mappings actually agree before we stream q values between them.
    expected_names = {side: ik.arm_joint_names[side] for side in active_sides}
    sink_names = {
        side: sink.arm_joint_names[side] for side in active_sides if side in sink.arm_joint_names
    }
    assert sink_names == expected_names, (
        f"Arm joint index mapping mismatch:\n"
        f"  ik:       {ik.arm_joint_names}\n"
        f"  sink:     {sink.arm_joint_names}\n"
        f"  expected: {expected_names}"
    )

    # Rolling buffer of end-to-end latencies in milliseconds. Sized for ~10s
    # of samples at INGRESS_FPS = 30, so the periodic 5s log line has a
    # stable enough population for p50/p95.
    lag_samples_ms: collections.deque = collections.deque(maxlen=300)
    carpals_error_samples_mm: collections.deque = collections.deque(maxlen=300)
    ik_error_vec_samples_m: collections.deque = collections.deque(maxlen=300)
    orientation_error_samples_rad: collections.deque = collections.deque(maxlen=300)
    joint_step_abs_samples_rad: collections.deque = collections.deque(maxlen=300)
    joint_step_l2_samples_rad: collections.deque = collections.deque(maxlen=300)
    joint_accel_abs_samples_rad: collections.deque = collections.deque(maxlen=300)
    actual_carpals_error_samples_mm: collections.deque = collections.deque(maxlen=300)
    actual_q_error_abs_samples_rad: collections.deque = collections.deque(maxlen=300)
    latest_ik_debug: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    latest_actual_debug: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    latest_motor_debug: tuple[np.ndarray, np.ndarray | None] | None = None
    active_q_indices = [idx for side in active_sides for idx in ik.arm_joint_indices[side]]
    previous_command_q: np.ndarray | None = None
    previous_command_dq: np.ndarray | None = None

    landmarks_q: queue.Queue = queue.Queue(maxsize=QUEUES_MAXSIZE * 4)
    stop_event = threading.Event()
    ingress = IngressServer(landmarks_q, stop_event, port=args.port)
    ingress.start()

    sink.launch()
    home_pose_source = args.home_pose_source
    if home_pose_source in ("franka-rest", "ik-neutral"):
        q_home = ik.neutral_q.copy()
        controller.set_home_configuration(q_home)
        logger.info(
            "Using IK neutral q as teleop home (%s): %s",
            home_pose_source,
            {
                side: np.round(
                    [q_home[idx] for idx in ik.arm_joint_indices[side]],
                    3,
                ).tolist()
                for side in active_sides
            },
        )
    else:
        q_home = controller.q_home
    if home_pose_source == "env":
        env_home_arm_angles = (
            sink.home_arm_angles
            if hasattr(sink, "home_arm_angles")
            else meshcat_env_home_arm_angles
        )
    else:
        env_home_arm_angles = None
    if env_home_arm_angles is not None:
        q_home = q_home.copy()
        for side in active_sides:
            if side not in env_home_arm_angles:
                continue
            for idx, value in zip(
                ik.arm_joint_indices[side],
                env_home_arm_angles[side],
                strict=True,
            ):
                q_home[idx] = value
        controller.set_home_configuration(q_home)
        logger.info(
            "Using %s environment home arm angles for teleop: %s",
            type(sink).__name__,
            {
                side: np.round(env_home_arm_angles[side], 3).tolist()
                for side in active_sides
                if side in env_home_arm_angles
            },
        )
    # Push the home pose immediately so the sink shows the anchor config
    # before any publisher frame arrives.
    home_arm_angles = {
        side: np.array([q_home[i] for i in ik._arm_idx_q[side]]) for side in active_sides
    }
    home_target_Ts = {side: T.homogeneous for side, T in controller.home_poses.items()}
    sink.to_neutral_configuration(home_arm_angles)
    if hasattr(sink, "set_debug_target_frame_offsets"):
        sink.set_debug_target_frame_offsets(home_target_Ts)
    sink.update(home_arm_angles, target_Ts=home_target_Ts, operator_Ts={})

    publisher_process: multiprocessing.Process | None = None
    if args.local:
        publisher_process = multiprocessing.Process(
            target=_metaquest_publisher,
            args=(args.port,),
            kwargs={
                "dummy": args.dummy,
                "dummy_repo": args.dummy_repo,
                "dummy_filename": args.dummy_filename,
                "dummy_loop": not args.dummy_no_loop,
                "dummy_refresh": args.dummy_refresh,
                "transport": args.transport,
                "quest_host": args.quest_host,
                "quest_port": args.quest_port,
                "log_level": args.log_level,
            },
            name="metaquest-publisher",
            daemon=True,
        )
        publisher_process.start()
        kind = (
            f"dummy (HF dataset replay {args.dummy_repo}/{args.dummy_filename})"
            if args.dummy
            else f"live HTS ({args.transport} {args.quest_host}:{args.quest_port})"
        )
        logger.info("Local publisher started (pid=%d, %s)", publisher_process.pid, kind)

    logger.info("Ready. Waiting for publisher on :%d. Ctrl+C to stop.", args.port)
    period = 1.0 / args.ik_rate
    next_tick = time.monotonic()
    last_log = time.monotonic()
    ik_calls = 0
    neutral_locked_sides: set[str] = set()
    neutral_all_locked_spoken = False
    neutral_prompted = False
    neutral_start_time: float | None = None
    neutral_delay_s = max(0.0, float(args.neutral_start_delay_s))
    ingress_drain_stats = MetaQuestDrainStats()

    def _current_ik_seed_from_sink() -> np.ndarray | None:
        if task_instant_qpos or not hasattr(sink, "current_arm_angles"):
            return None
        current_arm_angles = sink.current_arm_angles
        q_current = controller.q
        for side in active_sides:
            if side not in current_arm_angles:
                continue
            for idx, value in zip(
                ik.arm_joint_indices[side],
                current_arm_angles[side],
                strict=True,
            ):
                q_current[idx] = value
        return q_current

    try:
        while True:
            frames = _drain_latest_metaquest_frames(
                landmarks_q,
                active_sides,
                stats=ingress_drain_stats,
            )
            now = time.monotonic()
            if neutral_start_time is None and frames:
                neutral_start_time = now
                neutral_prompted = True
                logger.info(
                    "Hand frames received. Move to neutral pose; teleop starts in %.1fs.",
                    neutral_delay_s,
                )
                _speak_status("Move to neutral pose", enabled=args.speak_status)

            if neutral_start_time is None or now - neutral_start_time < neutral_delay_s:
                sink.update(home_arm_angles, target_Ts=home_target_Ts, operator_Ts={})
                if now - last_log > 5.0:
                    if neutral_start_time is None:
                        logger.info(
                            "Waiting for first active hand frame before neutral delay. "
                            "ingress=%s",
                            ingress_drain_stats.summary(active_sides=active_sides, now=now),
                        )
                    else:
                        remaining = max(0.0, neutral_delay_s - (now - neutral_start_time))
                        logger.info(
                            "Holding robot at neutral. Teleop starts in %.1fs.",
                            remaining,
                        )
                    last_log = now

                next_tick += period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
                continue

            if neutral_prompted:
                logger.info("Neutral delay complete. Anchoring teleop on next frame.")
                neutral_prompted = False

            result = controller.step(frames, q_current=_current_ik_seed_from_sink())

            newly_locked = {
                side
                for side, status in result.statuses.items()
                if side in active_sides
                and side not in neutral_locked_sides
                and status == "tracking"
            }
            if newly_locked:
                if len(active_sides) > 1:
                    for side in sorted(newly_locked):
                        _speak_status(f"{side} neutral locked", enabled=args.speak_status)
                neutral_locked_sides.update(newly_locked)
                if not neutral_all_locked_spoken and set(active_sides).issubset(
                    neutral_locked_sides
                ):
                    _speak_status("Neutral position locked", enabled=args.speak_status)
                    neutral_all_locked_spoken = True

            if result.arm_angles:
                command_q = result.q[active_q_indices].copy()
                if previous_command_q is not None:
                    command_dq = command_q - previous_command_q
                    joint_step_abs_samples_rad.append(np.abs(command_dq))
                    joint_step_l2_samples_rad.append(float(np.linalg.norm(command_dq)))
                    command_ddq = None
                    if previous_command_dq is not None:
                        command_ddq = command_dq - previous_command_dq
                        joint_accel_abs_samples_rad.append(np.abs(command_ddq))
                    latest_motor_debug = (
                        command_dq.copy(),
                        None if command_ddq is None else command_ddq.copy(),
                    )
                    previous_command_dq = command_dq.copy()
                previous_command_q = command_q.copy()
                carpals_error_samples_mm.extend(
                    1000.0 * error_m for error_m in result.position_error.values()
                )
                orientation_error_samples_rad.extend(result.orientation_error.values())
                for side, target_pose in result.target_poses.items():
                    solved_T = pin.SE3(ik.forward_kinematics_full(result.q, side))
                    home_T = controller.home_poses[side]
                    target_delta = target_pose.translation - home_T.translation
                    solved_delta = solved_T.translation - home_T.translation
                    error_vec = target_pose.translation - solved_T.translation
                    ik_error_vec_samples_m.append(error_vec.copy())
                    latest_ik_debug[side] = (
                        target_delta.copy(),
                        solved_delta.copy(),
                        error_vec.copy(),
                        result.orientation_error.get(side, float("nan")),
                    )
                target_Ts = {side: T.homogeneous for side, T in result.target_poses.items()}
                operator_Ts = {side: T.homogeneous for side, T in result.operator_poses.items()}
                sink.update(
                    result.arm_angles,
                    hand_positions=result.hand_positions,
                    target_Ts=target_Ts,
                    operator_Ts=operator_Ts,
                )
                ik_calls += 1

                if hasattr(sink, "current_arm_angles"):
                    current_arm_angles = sink.current_arm_angles
                    q_actual = result.q.copy()
                    for side in active_sides:
                        if side not in current_arm_angles:
                            continue
                        for idx, value in zip(
                            ik.arm_joint_indices[side],
                            current_arm_angles[side],
                            strict=True,
                        ):
                            q_actual[idx] = value
                    actual_command_error = np.abs(
                        result.q[active_q_indices] - q_actual[active_q_indices]
                    )
                    actual_q_error_abs_samples_rad.append(actual_command_error)
                    for side, target_pose in result.target_poses.items():
                        actual_T = pin.SE3(ik.forward_kinematics_full(q_actual, side))
                        home_T = controller.home_poses[side]
                        target_delta = target_pose.translation - home_T.translation
                        actual_delta = actual_T.translation - home_T.translation
                        error_vec = target_pose.translation - actual_T.translation
                        actual_carpals_error_samples_mm.append(
                            1000.0 * float(np.linalg.norm(error_vec))
                        )
                        latest_actual_debug[side] = (
                            target_delta.copy(),
                            actual_delta.copy(),
                            error_vec.copy(),
                        )

                # End-to-end latency: time.time_ns() here is "executed at sink"
                # (sink.update just returned). frame.timestamp_ns was stamped
                # by the publisher when it received the HTS frame, i.e. as
                # close to "command acquired" as we can get without device-side
                # clock cooperation. One sample per side that produced a fresh
                # target this tick.
                executed_ns = time.time_ns()
                for ts_ns in result.new_target_acquire_ns.values():
                    lag_samples_ms.append((executed_ns - ts_ns) * 1e-6)

            now = time.monotonic()
            if now - last_log > 5.0:
                if lag_samples_ms:
                    lags = np.asarray(lag_samples_ms)
                    lag_summary = (
                        f"e2e_lag_ms p50={np.percentile(lags, 50):.1f} "
                        f"p95={np.percentile(lags, 95):.1f} "
                        f"max={lags.max():.1f} n={len(lags)}"
                    )
                else:
                    lag_summary = "e2e_lag_ms=n/a"
                if carpals_error_samples_mm:
                    errors = np.asarray(carpals_error_samples_mm)
                    error_summary = (
                        f"carpals_err_mm p50={np.percentile(errors, 50):.1f} "
                        f"p95={np.percentile(errors, 95):.1f} "
                        f"max={errors.max():.1f} n={len(errors)}"
                    )
                else:
                    error_summary = "carpals_err_mm=n/a"
                if ik_error_vec_samples_m:
                    error_vecs = np.asarray(ik_error_vec_samples_m)
                    abs_vecs_mm = np.abs(error_vecs) * 1000.0
                    vec_summary = (
                        "ik_err_vec_abs_mm_p95="
                        f"{np.round(np.percentile(abs_vecs_mm, 95, axis=0), 1).tolist()}"
                    )
                else:
                    vec_summary = "ik_err_vec_abs_mm_p95=n/a"
                if orientation_error_samples_rad:
                    ori_errors = np.asarray(orientation_error_samples_rad)
                    ori_summary = (
                        f"ori_err_rad p50={np.percentile(ori_errors, 50):.3f} "
                        f"p95={np.percentile(ori_errors, 95):.3f} "
                        f"max={ori_errors.max():.3f}"
                    )
                else:
                    ori_summary = "ori_err_rad=n/a"
                if joint_step_abs_samples_rad:
                    joint_steps = np.asarray(joint_step_abs_samples_rad)
                    joint_step_l2 = np.asarray(joint_step_l2_samples_rad)
                    motor_summary = (
                        "joint_step_abs_rad_p95="
                        f"{np.round(np.percentile(joint_steps, 95, axis=0), 3).tolist()} "
                        f"joint_step_l2_rad p95={np.percentile(joint_step_l2, 95):.3f} "
                        f"max={joint_step_l2.max():.3f}"
                    )
                else:
                    motor_summary = "joint_step_abs_rad_p95=n/a"
                if joint_accel_abs_samples_rad:
                    joint_accels = np.asarray(joint_accel_abs_samples_rad)
                    accel_summary = (
                        "joint_bump_abs_rad_p95="
                        f"{np.round(np.percentile(joint_accels, 95, axis=0), 3).tolist()}"
                    )
                else:
                    accel_summary = "joint_bump_abs_rad_p95=n/a"
                if actual_q_error_abs_samples_rad:
                    actual_q_errors = np.asarray(actual_q_error_abs_samples_rad)
                    actual_q_summary = (
                        "actual_q_err_abs_rad_p95="
                        f"{np.round(np.percentile(actual_q_errors, 95, axis=0), 3).tolist()}"
                    )
                else:
                    actual_q_summary = "actual_q_err_abs_rad_p95=n/a"
                if actual_carpals_error_samples_mm:
                    actual_errors = np.asarray(actual_carpals_error_samples_mm)
                    actual_error_summary = (
                        f"actual_carpals_err_mm p50={np.percentile(actual_errors, 50):.1f} "
                        f"p95={np.percentile(actual_errors, 95):.1f} "
                        f"max={actual_errors.max():.1f} n={len(actual_errors)}"
                    )
                else:
                    actual_error_summary = "actual_carpals_err_mm=n/a"
                calibrated = [
                    side for side, status in result.statuses.items() if status == "tracking"
                ]
                status_counts = {
                    status: sum(1 for value in result.statuses.values() if value == status)
                    for status in sorted(set(result.statuses.values()))
                }
                logger.info(
                    "ik_calls=%d  mode=%s  active=%s  calibrated=%s  statuses=%s  "
                    "%s  %s  %s  %s  %s  %s  %s  %s",
                    ik_calls,
                    args.ik_mode,
                    sorted(result.target_poses),
                    sorted(calibrated),
                    status_counts,
                    lag_summary,
                    error_summary,
                    vec_summary,
                    ori_summary,
                    motor_summary,
                    accel_summary,
                    actual_q_summary,
                    actual_error_summary,
                )
                for side, (target_delta, solved_delta, error_vec, ori_error) in sorted(
                    latest_ik_debug.items()
                ):
                    logger.info(
                        "ik_debug %s target_delta_m=%s solved_delta_m=%s "
                        "err_vec_mm=%s ori_err_rad=%.3f",
                        side,
                        np.round(target_delta, 4).tolist(),
                        np.round(solved_delta, 4).tolist(),
                        np.round(1000.0 * error_vec, 1).tolist(),
                        ori_error,
                    )
                for side, (target_delta, actual_delta, error_vec) in sorted(
                    latest_actual_debug.items()
                ):
                    logger.info(
                        "actual_debug %s target_delta_m=%s actual_delta_m=%s "
                        "actual_err_vec_mm=%s",
                        side,
                        np.round(target_delta, 4).tolist(),
                        np.round(actual_delta, 4).tolist(),
                        np.round(1000.0 * error_vec, 1).tolist(),
                    )
                if latest_motor_debug is not None:
                    command_dq, command_ddq = latest_motor_debug
                    logger.info(
                        "motor_debug active_q_indices=%s dq_rad=%s ddq_rad=%s",
                        active_q_indices,
                        np.round(command_dq, 4).tolist(),
                        ("n/a" if command_ddq is None else np.round(command_ddq, 4).tolist()),
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
