# tron_arm — reference

Everything the code does. For the demo-day procedure see [README.md](README.md).

## Where this lives

`tron_arm` is a folder on the `feat/tron-arm` branch of the `orca_teleop`
repository. It is a self-contained project — its own `pyproject.toml`, its own
tests — that happens to be hosted there rather than in a repo of its own. The
`orca_teleop` code around it is untouched; nothing on `main` or
`feat/wrist-pose` changes.

## What this is

The `ArmSink` consumer for `orca_teleop`: it takes the operator's 6-DoF wrist
pose and drives a LimX TRON 2 over `request_servop`. Fingers and the ORCA hand's
1-DoF wrist are the existing hand path and are out of scope.

`tron_arm` **does not modify `orca_teleop`**. It satisfies the `ArmSink` ABC
structurally rather than subclassing it, so the package imports and its tests run
with `orca_teleop` absent. Nothing here writes into that repo.

## Layout

```
tron_arm/          the package
  poses.py         Pose + interpolation (no quaternion sign canonicalisation)
  config.py        validated YAML config, workspace boxes, joint limits
  mapping.py       clutch origins, scaling, workspace clamp
  arm_state.py     clutch/engage state machine + per-frame pipeline
  streamer.py      fixed-rate interpolating ServoP pacer
  tron2_client.py  WebSocket client, both servop encoders
  sink.py          TronArmSink — the ArmSink implementation
  clutch.py        hold-to-engage keyboard/pedal clutch
  session.py       per-session JSONL.gz logging
  step_test.py     runbook step 4, format discovery
  mock_robot.py    local stand-in for the robot
tools/
  tron2_cli.py     info, pose, joints, movej-*, capture-ready, servop-*
  run_arm.py       the teleop entry point
  report_session.py
configs/default.yaml
deploy/launch/     preflight script and tmux launchers
docs/              mapping maths, safety review, the runbook
```

## Commands

### `tools/tron2_cli.py`

| command | motion? | what |
|---|---|---|
| `info` | no | connect, learn accid, show diagnostics |
| `pose` | no | both flange poses + workspace check |
| `joints` | no | joint angles and headroom (`--raw` dumps the reply) |
| `light --effect X` | no | status light |
| `capture-ready` | no | print the current joints as a `ready:` block |
| `movej-ready` | **yes** | go to the configured ready posture |
| `movej-home` | **yes** | go to the zero configuration |
| `servop-hold --seconds N` | **yes** | stream the current pose; pacing smoke test |
| `servop-circle` | **yes** | trace a circle, report roundness |
| `servop-readback` | **yes** | command a known pose, print component-wise error |

### `tools/run_arm.py`

| flag | what |
|---|---|
| `--sim` / `--robot HOST` | mock (default) or hardware (needs confirmation) |
| `--no-wrist` | mandatory on hardware; holds the ORCA wrist at 0 |
| `--arm-only` | ingress + arm worker, no retargeter, no torch |
| `--step-test` | runbook step 4, then exit |
| `--clutch keyboard\|pedal\|none` | default keyboard, SPACE |
| `--seconds N` | stop after N seconds |
| `--log-dir` / `--session-id` / `--no-log` | session logging |

### `tools/report_session.py`

`report_session.py logs/<id>` — latency percentiles, ingress rate, hold and
reference-change timelines, notify timeline, clamp totals. `--json` for CI.

## Configuration

`configs/default.yaml`. Every ambiguity in the vendor guide is a knob here, so a
wrong guess is a config edit, never a code change.

| key | default | notes |
|---|---|---|
| `robot.url` | `ws://127.0.0.1:5000` | the mock; `--robot` overrides |
| `ready.joints` | captured | posture that puts both flanges inside the box |
| `servop.format` | `pos_quat` | **confirmed on hardware.** `pos_rotmat` is the alternative |
| `servop.rate_hz` | 100 | interpolated from 30–60 Hz ingress |
| `servop.send_both` | true | both arms every tick; frozen pose for the idle one |
| `workspace.*` | vendor boxes | margin 0.03 m held off every face |
| `velocity.lin/ang` | 0.4 m/s, 1.2 rad/s | divided by rate → per-tick step clamp |
| `scale` | 0.5 | applied to the delta translation only |
| `mapping.translation_frame` | body | `world` gated behind the §G-09 axis check |
| `home.joints` | zeros | the documented zero configuration |

## How a frame becomes a command

```
dispatch(frame)                        # upstream's arm worker thread
  clutch gate
  lazy origin latch                    # first frame while engaged
  T_target = T_robot0 · inv(T_op0) · T_op,  scale on the delta translation
  tool offset (optional)               # T_BF = T_BH_target · inv(T_FH)
  workspace clamp                      # per arm, box minus margin
  step clamp                           # per-tick velocity ceiling
  submit to the streamer               # in-memory; NO network I/O here

streamer @ servop.rate_hz              # asyncio loop thread
  interpolate between the two newest targets, one ingress interval behind
  step clamp
  encode (pos_quat | pos_rotmat)
  ws send                              # at most one in flight; drop-oldest
```

`dispatch` must stay under 1 ms — it runs on the thread that owns the staleness
deadline. Measured p95 0.38 ms with the retargeter running alongside.

### The mapping, expanded

With `delta = inv(T_op0) @ T_op` and scale `s` on its translation:

```
R_target = R_r0 · (R_o0ᵀ R_op)                      # body-frame rotation delta
p_target = p_r0 + R_r0 · ( s · R_o0ᵀ (p_op − p_o0) )
```

Two properties follow for free. **Zero jump at engage:** with `T_op == T_op0`,
`delta` is the identity, so the target is exactly `T_robot0` — no ramp needed.
And **re-clutching re-indexes the workspace**, because the origins are re-latched
wherever the operator happens to be.

Scaling the *delta* rather than the final position is what makes this relative.
Scaling `T_target[:3,3]` would multiply the robot's absolute base-frame position
and fling the arm across the workspace on the first frame.

### The pre-rotation, and why engage posture matters

Translation is pre-rotated by `M = R_r0 · R_o0ᵀ` — the orientation mismatch
between wrist and flange **at the instant of engage**. `M` is a rotation, so it
never scales or skews; it only reorients.

- Engage with the hand aligned to the flange and `M ≈ I`: forward is forward.
- Engage with the wrist yawed 40° and every translation is rotated 40°. Motion
  stays 1:1 in magnitude and feels rigid — it is simply aimed elsewhere.

There is no drift: the offset is constant for the whole engagement and vanishes
the moment you re-clutch. That is why re-clutching is the fix when the mapping
"feels rotated", and why the engage posture is worth being deliberate about.

`mapping.translation_frame: world` drops the `M` pre-rotation so operator axes
drive base axes directly. That is only correct if the two conventions genuinely
coincide, which is an empirical fact about the rig — hence the
`world_frame_axes_verified` gate.

## State machine

`DISENGAGED → ENGAGED_NO_ORIGIN → ENGAGED`, plus `HOLD` and `FAULT`.

Origins latch lazily on the first frame while engaged and are cleared on **every**
exit from engaged. `ENGAGED_NO_ORIGIN` is a distinct state because
`on_reference_change` can arrive *late* — after we have already re-latched — and a
boolean "have origin" flag would map a fresh frame against a stale origin.

`FAULT` needs an explicit reset. A dead socket faults both arms; there is no
reconnect, because resuming would stream against origins latched before the gap.

## Safety properties

Each has an adversarial test in `tests/test_safety_review.py` that fails if the
hazard is reintroduced.

- Nothing gates on `tracking_valid` — absence of frames *is* tracking loss.
- `timestamp_ns` never enters control maths — it is a browser wall clock and can
  run backwards.
- No path re-canonicalises quaternion sign — forcing `w ≥ 0` breaks continuity at
  180°, which an operator reaches routinely.
- Backpressure drops oldest; a slow socket never queues a burst of stale targets.
- A "hold" never moves the arm: an origin outside the workspace refuses rather
  than clamping. **This one cost a collision before it existed.**
- Nothing unclamped or non-finite reaches the encoder.

## Hardware findings

Measured on a TRON 2, fw `robot-tron2-r-2.1.24`.

| finding | detail |
|---|---|
| `servop.format` | **`pos_quat` accepted.** `pos_rotmat` never completed a run |
| FLU axes | confirmed — all six ±2 cm steps moved the right axis in the right direction |
| differential accuracy | spans 0.89 / 0.90 / 1.07 of a commanded 40 mm |
| steady-state sag | ~15 mm below and ~7 mm behind the commanded pose |
| joint state | **16 entries, not 14**: `[0:7]` left, `[7:14]` right, `[14:16]` grippers |
| `names` field | present but empty |
| home posture | outside the documented workspace box — expected |
| stream loss | the robot **holds** the last pose indefinitely, with torque |
| after streaming | `get_move_pose` stops answering for several seconds |
| gravity compensation | sensitive to end-effector mass; a missing gripper gave a 3:1 vertical asymmetry |
| teleop latency | dispatch→send p50 4.7 ms, p95 9.6 ms; streamer 99.99 Hz |

## Still open

1. **Payload compensation** (LimX Q7) — the ORCA hands change mass, CoM and
   moment arm. The 15 mm sag will change and there is no documented way to tell
   the controller about it.
2. **ServoP rate** (Q2) — 100 Hz is our choice; ServoP documents nothing.
3. **`notify_servop` result codes** (Q4) — undocumented; we treat any `fail*` as failure.
4. **Arm release for manual positioning** — no documented way to relax a powered arm.
5. **Operator-side axis check** (runbook step 0) — never run, so
   `translation_frame: world` stays gated.

## Rehearsing without a headset

Two publishers stand in for a live operator. Both need the robot side running —
either the mock (`--sim`) or the real robot.

**Real recorded motion** — 30 s of a real hand from a Quest, including a genuine
half-second tracking dropout at ~4 s where the hand returns 130° rotated:

```bash
uv run python -m orca_teleop.ingress.metaquest.replay_publisher \
    --server localhost:50051 --parquet <local copy>
```

`--hand left`, `--speed 2.0`, `--loop`. Without `--parquet` it downloads from the
Hub, which needs internet — so fetch it before joining the robot's Wi-Fi.

Note `--loop` ratchets the arm: each loop's epoch change re-latches the origin
where the arm currently is, so a recording with net displacement walks the arm
toward a wall. Re-run `movej-ready` between passes.

**Synthetic motion with failures on demand** — dropouts and reference changes
wherever you want them:

```bash
uv run python -m orca_teleop.ingress.metaquest.mock_publisher \
    --server localhost:50051 --dropout-every 5 --dropout-for 1.5 --epoch-change-every 10
```

## Confirming the robot accepts our commands

`--step-test` drives the robot directly — no operator, no clutch — through the
runbook's step 4: hold the current pose 5 s, a known-pose readback, ±2 cm
single-axis steps, and a 2 cm circle. It tries both servop encodings and prints
PASS/FAIL per format.

```bash
uv run python tools/run_arm.py --robot <ip> --no-wrist --step-test
```

Expect `VERDICT: use servop.format: pos_quat`. Max excursion ~3 cm. Worth running
after any firmware change or if `notify_servop` failures appear.

## Development

```bash
uv run pytest -q                 # everything
uv run pytest -q -m "not slow"   # skip the wall-clock integration runs
```

Tests need no robot. `tests/test_full_pipeline.py` additionally needs the
upstream stack and skips without it.

Against the mock, in three terminals:

```bash
uv run python -m tron_arm.mock_robot
uv run python tools/run_arm.py --sim --arm-only     # hold SPACE
uv run python -m orca_teleop.ingress.metaquest.mock_publisher --server localhost:50051
```

The mock accepts `--accept-format` and `--require-both-keys` so both protocol
ambiguities can be rehearsed without hardware.

## Other documents

| file | what |
|---|---|
| [CLAUDE.md](CLAUDE.md) | the upstream and vendor contract; hard rules |
| [deploy/launch/](deploy/launch/) | preflight script and tmux launchers |
