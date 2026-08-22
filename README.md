# orca_teleop

Repo for teleoperating the ORCA Hand consisting of an Ingress Source (for example Mediapipe, Apple Vision Pro, Rokoko Gloves, etc.) and a URDF-based Retargeter.

The repository follows a standard `src/` layout:

```text
src/orca_teleop/
tests/
```

## Development setup

Create a local virtual environment with `uv` and install the project in editable mode with the development extras used in this repository:

```bash
uv venv
source .venv/bin/activate
uv sync --extra test --extra mediapipe --extra adaptive
```

This installs the package itself plus the testing tools, MediaPipe dependencies, and the
optional solver stack used by the default adaptive analytical retargeter.

## Retargeting

The default teleop backend is `adaptive_analytical`, an Orca-native port of the Wuji-style
analytical retargeting strategy. It uses the full MediaPipe hand pose, explicit Orca frame
mappings from YAML, Pinocchio forward kinematics/Jacobians, and `nlopt` for bounded
per-frame optimization.

The legacy fingertip key-vector retargeter is still available for comparison:

```bash
python scripts/teleop_sim.py --env right --local --show-video --retargeter rmsprop
```

The adaptive backend loads `src/orca_teleop/retargeting/configs/adaptive_analytical_orca.yaml`
by default. Pass `--retarget-config path/to/config.yaml` to experiment with frame maps or
weights.

Steer your own ORCA hand using just your webcam:

```
python scripts/mediapipe_teleop_demo.py     path/to/your_orcahand_model     path/to/corresponding_urdf_file
```

Tests always run on CI. Run the regression suite from the repository root with:

```bash
pytest tests/
pytest tests/ -n auto --dist loadfile   # same suite, files in parallel
```

The parallel form needs the `test` extra (`pytest-xdist`); `--dist loadfile`
keeps each file on one worker, which is what the tests that bind fixed ports
rely on.

## Record Quest WebXR demonstrations on a physical OrcaHand

The dataset recorder includes the complete Quest ingress: it serves its own
WebXR page, receives 25 hand joints from Quest Browser over a WebSocket, maps
them to the same 21-point layout used by the original in-hand MuJoCo
demonstrations, retargets them, and drives the physical hand.

Start by installing the recording stack (cameras, Quest WebXR, adaptive
retargeter, and LeRobot):

```bash
uv sync --extra learning
```

First validate the motion with the simulated sink:

```bash
uv run python scripts/record_dataset.py \
  --repo-id "$HF_USERNAME/orca-quest-safety-check" \
  --task "check WebXR retargeting" \
  --backend sim \
  --source metaquest \
  --local \
  --hand right \
  --episode-end space \
  --num-episodes 1
```

For the real hand, run:

```bash
uv run python scripts/record_dataset.py \
  --repo-id "$HF_USERNAME/orca-quest-hardware" \
  --task "pick up the block" \
  --backend hardware \
  --source metaquest \
  --local \
  --hand right \
  --episode-end space \
  --num-episodes 20 \
  --camera front:0
```

Hold the hand still while the retargeter calibrates. Press Space to save the
current episode and begin the next; press `q` or Escape to finish. The command
records measured hardware joint positions as `observation.state`, commanded
joint positions as `action`, and each configured camera as
`observation.images.<name>`.

The command hosts the Quest page on port `8765`. WebXR requires a secure
context, so expose it in a second terminal:

```bash
ngrok http 8765
```

Open the resulting HTTPS URL in Quest Browser and tap **Start hand tracking**.
For a wired setup, use:

```bash
adb reverse tcp:8765 tcp:8765
```

Then open `http://localhost:8765` in Quest Browser; browsers treat localhost as
a secure context.

For a split-machine setup, omit `--local` on the robot and run the publisher on
the Quest-side machine:

```bash
uv run python -m orca_teleop.ingress.metaquest.publisher \
  --server ROBOT_IP:50051 \
  --hand right
```

The publisher machine hosts the WebXR page and forwards the reduced landmarks
to the robot over gRPC. Keep `--source metaquest` on the recorder so it treats
the incoming frames as the WebXR convention. The live adapter deliberately
does not apply the legacy HTS/Unity handedness flip.

## Streaming into the ORCA Hand Console (orca_ui)

`orca-teleop-streamer` runs the ingress→retargeter pipeline headlessly and
streams bare-name joint targets (degrees) into orca_ui's `/ws/teleop`
WebSocket instead of commanding a hand — orca_ui owns the hand, the
control-source arbiter, ramp-in, and the safety watchdog.

Normally you never run it by hand: the console's **Teleop tab** spawns it via
`uv run --project <this repo>` (auto-detected as a sibling checkout, or set
`--teleop-dir` / `ORCA_TELEOP_DIR` on `orca-ui`). For a remote machine
(e.g. the Linux box wired to Manus gloves), start a session in *external*
mode in the UI and launch manually with the token it shows:

```bash
orca-teleop-streamer \
    --connect ws://<console-host>:5001/ws/teleop --token <token> \
    --source mediapipe --model-path path/to/config.yaml
```

Sources: `mediapipe` (local webcam, `--camera-index`), `manus` (see
`docs/manus_glove_setup.md`; the SDK publisher is Linux-only, so on macOS run
the publisher on the Linux box against this process's gRPC port), and
`synthetic` (waveform, no hardware — wiring checks).

Notes:
- `--retargeter rmsprop` is markedly faster on CPU-only machines
  (~12 ms/frame vs ~100 ms for `adaptive_analytical`); the console defaults
  to it for interactive latency.
- The retargeter URDF comes from an `orcahand_description` checkout
  (`ORCAHAND_DESCRIPTION_DIR` or `--urdf-path`); orca_ui exports the sibling
  checkout automatically.
- `--list-cameras` probes webcams and prints JSON (also triggers the macOS
  camera-permission prompt for spawned children — run orca-ui from a
  terminal the first time).
