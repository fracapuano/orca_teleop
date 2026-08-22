# tron_arm — TRON 2 arm control from orca_teleop operator wrist poses
We implement the ArmSink consumer that drives a LimX TRON 2 dual-arm robot via the
request_servop WebSocket command, from TeleopFrames produced by orca_teleop
(branch feat/wrist-pose). Fingers + the ORCA hand's 1-DoF wrist are handled by the
existing hand path and are OUT OF SCOPE (wrist joint is commanded to 0 via
wrist_enabled=False; it is a rigid link for our kinematics).

## Upstream contract (orca_teleop) — READ THE SOURCE FIRST
Before writing code, read: src/orca_teleop/arm.py, src/orca_teleop/ingress/frames.py,
src/orca_teleop/ingress/metaquest/mock_publisher.py, tests/test_arm.py. If anything
below disagrees with the source, the source wins — update this file and say so.
VERIFIED against orcahand/orca_teleop @ feat/wrist-pose, commit 1b5c85e. Everything
previously in this section held; the Pose bullet below is expanded with details it
had omitted. tests/test_upstream_interop.py pins the contract — point
ORCA_TELEOP_SRC at a checkout to run it against the real frames.py.
- ArmSink ABC, five required methods: connect(), dispatch(frame), on_hold(reason),
  on_reference_change(stream_id, pose_epoch), close(). dispatch runs on the arm worker
  thread; exceptions are caught upstream -> hold. Keep dispatch non-blocking.
  RE-VERIFIED at 1b5c85e: all five signatures match this line exactly. tron_arm's
  TronArmSink duck-types rather than subclasses (the package stays importable without
  orca_teleop); sink.assert_matches_upstream_abc() checks the claim against the real
  ABC by reflection, and test_sink.py runs it whenever orca_teleop is installed.
- TeleopFrame fields: wrist: Pose (always present, validated); head: Pose|None;
  recv_monotonic_ns (ONLY clock allowed for dt/staleness/pacing); timestamp_ns
  (publisher wall clock — logging only, can run backwards, NEVER difference it);
  stream_id; pose_epoch; handedness "left"|"right"; tracking_valid (constant-true trap
  — never gate on it; absence of frames IS tracking loss); wrist_angle_degrees (not
  ours); age_s.
- Pose: FLU metres (X fwd, Y left, Z up), Hamilton wxyz. Use .matrix and
  .as_xyz_wxyz(); never hand-assemble [x,y,z,qx,qy,qz,qw]. Frozen slots dataclass;
  the FIELDS are position_m and orientation_wxyz (not p/q_wxyz — tron_arm.poses.Pose
  offers p/q_wxyz as aliases on top of the upstream names). Constructed positionally,
  Pose(position, orientation), on both sides.
- Pose REJECTS a non-unit quaternion (|q| off by > QUATERNION_NORM_TOLERANCE = 1e-3)
  rather than normalising: an all-default proto Pose carries qw=0 — the ZERO
  quaternion, not identity — and normalising it yields a silent NaN pose. Pose
  .from_matrix likewise rejects a non-orthonormal (atol 1e-6) or improper (det <= 0)
  rotation block; the WebXR path does not re-orthonormalise, so a corrupt matrix
  arrives intact. Mirror both, do not soften them.
- frames.py states outright that .as_xyz_wxyz() produces a TRON left_pos/right_pos
  ELEMENT-FOR-ELEMENT (and each half of request_movep's pos). That is upstream's
  reading of the vendor doc, not vendor confirmation — it is why pos_quat is the
  default, but servop.format stays a config knob until hardware settles it (§G-09).
- Upstream's Pose aliases the caller's array (asarray().reshape() returns a view, so
  setflags(write=False) marks only the view); its "cannot be mutated underneath any
  of them" docstring overstates it. tron_arm.poses.Pose copies instead — deliberate
  divergence, documented in the class.
- Sign continuity is produced by QuaternionContinuity (ingress/metaquest/landmarks.py):
  each quat is aligned to its PREDECESSOR, only the seed is canonicalised, and it
  RESETS on a pose_epoch change. So continuity holds within an epoch but not across
  one — drop interpolation buffers on reference change rather than slerping across it.
- on_hold reasons "no_frames"|"stale"|"tracking_invalid" all mean: hold position.
  Staleness deadline is upstream (ARM_STALE_AFTER_S=0.25, fixed via
  run_metaquest_local). Do not build a duplicate operator watchdog.
- QUATERNIONS ARE ALREADY SIGN-CONTINUOUS. Never re-canonicalize; never force w>=0
  (it reintroduces a flip at 180 deg). Slerp between successive received quats as-is.
- pose_epoch/stream_id change => our latched origins are garbage. The callback can
  arrive LATE (attached to the first frame after a dropout). Rule: origins latch
  lazily on the first frame while engaged, and are cleared on EVERY exit from
  engaged (clutch release, hold, reference change, fault).
- Mock publisher flags for tests: --dropout-every/--dropout-for,
  --epoch-change-every, --no-arm-pose, --hand left|right. Precision: the flag is
  DECLARED as --arm-pose with argparse.BooleanOptionalAction (default True), which is
  what generates --no-arm-pose; both spellings work. Also present: --server, --fps,
  --log-level. Timing gotcha for tests: _in_dropout is `elapsed % every < for`, so a
  dropout covers the FIRST `for` seconds from t=0 -- the first dropout precedes any
  latch, and only the SECOND one exercises re-latching. Epochs step at
  `elapsed // epoch_change_every`.
- run_metaquest_local(model_path=None, urdf_path=None, port=50051, handedness="right",
  quest_host="0.0.0.0", quest_port=8765, quest_fps=30, wrist_enabled=True,
  wrist_scale=1.0, sink=None, visualize_landmarks=False,
  retargeter_backend="adaptive_analytical", retargeter_config_path=None,
  arm_sink=None). NOTE wrist_enabled DEFAULTS TO TRUE, so tools/run_arm.py must pass
  wrist_enabled=False explicitly -- it refuses to run against hardware without
  --no-wrist, because a live wrist makes the rigid T_FH offset wrong.
- Tests: `uv run pytest tests/`, or `-n auto --dist loadfile` to run the files
  in parallel (needs the `dev` extra; ~3x). test_recording_ready no longer hangs
  -- its mirror-readiness test used to race the poll interval and never converge.
  `uv run` syncs the full dep set including torch and
  pytorch-kinematics (~1 GB). Only tests/test_pipeline.py, test_retargeter.py and
  test_streamer.py need those; the other 69 tests (incl. test_arm.py) run against
  grpcio + numpy + protobuf + a --no-deps orca_core, which is all the ARM path needs.

## Mapping (contract — colleague's guide §08)
T_target = T_robot0 @ inv(T_op0) @ T_op, scale s applied to the delta translation,
then workspace clamp. T_op0 = frame.wrist.matrix at lazy latch. T_robot0 = last
COMMANDED target if already streaming, else request_get_move_pose (metres + wxyz —
composes directly). Optional config mapping.translation_frame: "body" (default,
the formula above) | "world" (p_target = p_r0 + s*(p_o - p_op0), rotation unchanged;
gated behind an axis-verification flag). Optional tool offset T_FH (flange->palm,
rigid because hand wrist is at 0): command T_BF = T_BH_target @ inv(T_FH).

## TRON 2 protocol ground truth (LimX SDK guide V0.2 — do NOT invent beyond this)
- ws://<robot>:5000 (vendor default 10.192.1.2; lab config may differ). JSON envelope
  {accid, title, timestamp(ms), guid(uuid4), data}; learn accid from notify_robot_info.
- request_get_move_pose -> left/right_position [x,y,z] m + left/right_quat [w,x,y,z].
  Base frame FLU, origin at center of base lower plane; EE reference = flange (center
  of last joint).
- request_get_joint_state REPLIES with q/dq/tau of 16 entries on our unit
  (fw 2.1.24), NOT 14: [0:7] left arm, [7:14] right arm,
  [14:16] the grippers. Verified -- q[0:14] fits every documented limit while
  both alternative layouts violate them, and tau is symmetric between the arms
  (j0 -5.22/-5.11, j3 -5.34/-5.27). movej still TAKES 14. The reply also carries
  an empty `names` list, which would have settled this directly if populated.
  tron2_client.split_joint_reply re-checks the slice against the limits on every
  call and raises rather than guessing.
- request_movej: {time s, joint: [14 rad, left 7 then right 7]}, limits:
  upper [2.6005,3.1940,1.4835,0.2618,1.3963,0.7854,1.5708, 2.6005,0.2618,3.6652,0.2618,1.7453,0.7854,1.5708]
  lower [-3.1416,-0.2618,-3.6652,-2.6180,-1.7453,-0.7854,-1.5708, -3.1416,-3.1940,-1.4835,-2.6180,-1.3963,-0.7854,-1.5708]
- Workspace boxes (m): left x[0.250,0.732] y[-0.213,0.900] z[-0.673,0.5];
  right x[0.250,0.732] y[-0.900,0.213] z[-0.673,0.5]. MoveP rejects out-of-range;
  ServoP behavior UNDOCUMENTED -> always clamp ourselves (margin 0.03 m).
- request_servop: {left_pos: [...], right_pos: [...]}; no response; failures via
  notify_servop. RESOLVED by LimX (2026-08): ServoP takes the SAME parameters as
  MoveP (guide 9-3.6.3), i.e. position + wxyz quaternion -> servop.format
  "pos_quat" [x,y,z,qw,qx,qy,qz], 7/arm. The guide's "24 values" is a
  documentation error they will fix. "pos_rotmat" (12/arm) is kept only so
  --step-test can prove the format rather than assume it.
- SINGLE-ARM COMMANDS ARE NOT SUPPORTED (LimX, 2026-08): every request_servop
  must carry both left_pos and right_pos. To move one arm, hold the other at its
  current pose -- which is exactly what servop.send_both: true does. send_both:
  false is refused against non-loopback targets.
- Frames arrive 30-60 Hz. ServoP and ServoJ both require >= 50 Hz (LimX,
  2026-08); the guide's ">=500 Hz for ServoJ" is a documentation error. We stream
  at servop.rate_hz (default 100, floor 50) interpolating between the two newest
  targets, paced by recv_monotonic_ns.
- Removing the LimX grippers does NOT affect the arm: the gripper control
  interface is independent (LimX, 2026-08). Note this is about the INTERFACE --
  the gripper's mass still loads the arm, which is a separate matter.
- END-EFFECTOR LOAD COMPENSATION IS NOT AVAILABLE (LimX, 2026-08; "still under
  development"). There is no way to tell the controller what the hands weigh, so
  the ~15 mm steady-state sag cannot be corrected at the robot and will change
  when the ORCA hands are fitted. Re-measure it then.
- request_emgy_stop only works when the robot is idle — NOT a motion abort. Abort =
  freeze target + human on the hardware remote.
- notify_invalid_request echoes malformed messages back — always log all notify_*.

## Hard rules
1. Real hardware requires --robot flag AND interactive confirmation; default target
   is the local mock robot.
2. recv_monotonic_ns for all timing. timestamp_ns never appears in control math.
3. No unclamped or NaN pose can reach the encoder (guards + tests).
4. Type hints, docstrings, pytest for every module; pure math testable without I/O.
5. Python 3.10; deps numpy, scipy, websockets (+ orca_teleop's own: grpcio). No new
   heavy deps in the sink path.