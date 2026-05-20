JOIN_TIMEOUT = 3.0
HEARTBEAT_INTERVAL = 0.1
QUEUES_MAXSIZE = 8
INGRESS_FPS = 30
MOTION_NUM_STEPS = 1

WRIST_MOTOR_IDX = 16
DEFAULT_PORT = 50051
DEFAULT_HAND = "right"

# MediaPipe teleop
DEFAULT_CONFIDENCE = 0.7
_NUM_KEYPOINTS = 21
_COORDS_PER_POINT = 3
_EXPECTED_LEN = _NUM_KEYPOINTS * _COORDS_PER_POINT

# Meta Quest teleop
# Per-side, per-axis asymmetric reach envelope of the OrcaArm carpals frame,
# expressed as (lo, hi) deltas in meters relative to the side's teleop home
# pose, in URDF world frame (FLU: x_fwd, y_left, z_up). Source:
# scripts/fk_workspace_sweep.py (1500-sample joint-space sweep, seed=0). The
# numbers are bilaterally consistent — left/right mirror on y, agree on x and
# z — which is the kinematic sanity check that the FK chain is correct.
# Consumed in scripts/teleop_arm_quest.py:
#   - clip site: ``dp = np.clip(dp, lo, hi)`` element-wise (asymmetric).
#   - span re-fit: ``(hi - lo) / 2`` recovers a per-axis half-width that plays
#     the same role the old symmetric constant did in the scale ratio.
WORKSPACE_DELTA_LIMITS_M: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "left": ((-0.542, -0.272, -0.143), (+0.380, +0.465, +0.783)),
    "right": ((-0.541, -0.465, -0.142), (+0.383, +0.264, +0.783)),
}

# Same asymmetric carpals-frame reach envelope for the OrcaPanda embodiment.
# This one is right-side only, and the delta origin is the OrcaPanda task
# environment reset/home pose, because live teleop adopts that pose before
# anchoring the operator workspace. Source:
#   PYTHONPATH=src python scripts/fk_workspace_sweep.py --embodiment orca-panda
#       --samples 100000 --seed 0 --plot plots/orcapanda_workspace.png
ORCA_PANDA_WORKSPACE_DELTA_LIMITS_M: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "right": ((-1.313, -0.946, -1.019), (+0.663, +1.032, +0.755)),
}

WORKSPACE_DELTA_LIMITS_BY_EMBODIMENT_M: dict[
    str, dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
] = {
    "orca-arm": WORKSPACE_DELTA_LIMITS_M,
    "orca-panda": ORCA_PANDA_WORKSPACE_DELTA_LIMITS_M,
}

# Per-side, per-axis recorded operator wrist workspace endpoints in robot FLU
# coordinates (x_fwd, y_left, z_up), in meters. Source:
# fracapuano/quest-calibration/data.parquet, converted from raw Quest Unity
# left-handed wrist positions using the same basis transform as live teleop.
# These are absolute operator-frame positions. When no manual translation scale
# is supplied, teleop maps each startup-anchor-relative interval
# ``operator_anchor → operator_endpoint`` onto the matching robot interval
# ``robot_home → robot_endpoint`` from the selected embodiment's workspace
# delta limits.
OPERATOR_WRIST_WORKSPACE_LIMITS_M: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "left": ((-0.2996, +0.1209, +0.6857), (+0.6231, +0.6530, +1.7691)),
    "right": ((-0.2931, -0.7108, +0.6601), (+0.5784, -0.2091, +1.7704)),
}

# First visible wrist poses from fracapuano/quest-calibration/data.parquet,
# converted from raw Quest Unity coordinates to robot FLU. These are the
# operator-side home poses used by the optional neutral-lock gate.
OPERATOR_NEUTRAL_WRIST_POSES_FLU: dict[
    str,
    tuple[
        tuple[float, float, float],
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    ],
] = {
    "left": (
        (+0.2205, +0.2280, +0.9876),
        (
            (+0.965408, +0.214867, -0.147712),
            (-0.188303, +0.966387, +0.175041),
            (+0.180357, -0.141171, +0.973418),
        ),
    ),
    "right": (
        (+0.1857, -0.2866, +0.9821),
        (
            (+0.975617, +0.061277, -0.210753),
            (-0.073888, +0.995884, -0.052486),
            (+0.206670, +0.066778, +0.976129),
        ),
    ),
}
AUTO_FIT_MARGIN = 0.7
CUTOFF_MIN = 0.05

MIN_SPAN_SAMPLES = 150
SPAN_BUFFER_SECONDS = 60.0
SPAN_REFIT_PERIOD_S = 60.0
SPAN_CHANGE_THRESHOLD = 0.10
BOOTSTRAP_SCALE = 0.15

STILL_THRESHOLD_M = 0.01
STILL_WINDOW_SAMPLES = 30
CLUTCH_GRACE_S = 2.0

# Per-tick joint-step clamp applied AFTER the IK solve in the Quest teleop
# loop. Caps |Δq| = |q_new - q_prev| at this many radians per joint per IK
# tick, so a large target jump (clutch re-anchor, scale change, lost-tracking
# blip) integrates over multiple ticks instead of yanking the arm in one.
# At IK_RATE_HZ = 60, 0.05 rad/tick = 3.0 rad/s ≈ 172 deg/s — well above
# anything a Quest operator can move, but well below the URDF velocity limits.
MAX_JOINT_STEP_RAD = 0.05
