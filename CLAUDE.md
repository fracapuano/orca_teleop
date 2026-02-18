# ORCA Hand Teleoperation System

Real-time teleoperation for the ORCA robotic hand (DexterousDynamos). Tracks a human hand via webcam (MediaPipe) or Apple Vision Pro, retargets the pose to robot joint angles using URDF-based inverse kinematics, and streams commands to the physical hand.

## Architecture

```
Webcam/AVP → Ingress (landmarks) → Retargeter (IK optimization) → OrcaHand (motor commands) → Robot
                                                                  ↘ URDFViewer (3D browser viz)
```

**Pipeline stages:**
1. `orca_teleop/orca_ingress/mediapipe/mediapipe_ingress.py` — Webcam capture + MediaPipe hand detection (21 landmarks), orientation validation, threaded processing
2. `orca_teleop/orca_retargeter/retargeter.py` — Loads URDF via `pytorch_kinematics`, runs RMSprop optimization (2 steps/frame) to solve IK, outputs joint angle dict
3. `orca_teleop/orca_retargeter/utils/retargeter_utils.py` — Preprocessing for each input source, forward kinematics helpers, coordinate normalization
4. `scripts/mediapipe_teleop_demo.py` — End-to-end demo: multiprocessing architecture with separate robot control process, queue-based communication

## Rules

- Never add comments, docstrings, or type annotations to code you didn't write or change
- Never modify `retargeter.yaml` tuning parameters without explicit instruction — these are calibrated values
- Never hardcode absolute paths — use relative paths or construct from `model_path`/`urdf_path` arguments
- Never change the URDF coordinate frame conventions or joint naming scheme (`{hand_type}_{joint_name}`) without updating both `retargeter.py` and `retargeter_utils.py`
- Never bypass hand orientation validation in `mediapipe_ingress.py` — it prevents sending bad poses to hardware
- Never send untested joint angles to the physical robot — validate against joint ROM limits first
- Always use thread-safe access patterns (locks) when touching shared state between ingress thread and main thread

## Key Files

| File | Purpose |
|---|---|
| `scripts/mediapipe_teleop_demo.py` | Entry point — initializes ingress + retargeter, spawns robot control subprocess |
| `orca_teleop/__init__.py` | Public API — exports `MediaPipeIngress` and `Retargeter` |
| `orca_teleop/orca_ingress/mediapipe/mediapipe_ingress.py` | Webcam capture, MediaPipe hand tracking, orientation checks, landmark callbacks |
| `orca_teleop/orca_ingress/mediapipe/hand_landmarker.task` | Pre-trained MediaPipe hand landmark model (7.8 MB binary) |
| `orca_teleop/orca_retargeter/retargeter.py` | URDF loading, IK optimization (PyTorch RMSprop), joint angle computation |
| `orca_teleop/orca_retargeter/utils/retargeter_utils.py` | Source preprocessing (MediaPipe/AVP), FK helpers, key vector extraction |
| `orca_teleop/orca_retargeter/utils/retargeter.yaml` | Tuning config: learning rate, loss coefficients per finger, joint regularizers |
| `orca_teleop/viewer/urdf_viewer.py` | Live 3D URDF viewer via viser (web-based, non-blocking) |
| `pyproject.toml` | Project metadata and dependencies (poetry backend) |

## Running

```bash
source .venv/bin/activate
python scripts/mediapipe_teleop_demo.py path/to/your_orcahand_model path/to/corresponding_urdf_file [--no-display] [--no-viewer] [--no-robot]
```

- `--no-viewer` disables the 3D browser viewer
- `--no-robot` runs without robot hardware (visualization-only mode)

**Always activate the `.venv` virtual environment** (Python 3.10-3.12, not 3.13+) before running.

**Required external packages** (must be installed manually):
- `orca_core` — provides `OrcaHand` class for robot connection (separate ORCA repo, install with `pip install -e ../orca_core`)

## Directory Layout

- `scripts/` — Runnable demo scripts (entry points)
- `orca_teleop/` — Main package
  - `orca_ingress/` — Input sources (currently MediaPipe; extensible to Rokoko, etc.)
    - `mediapipe/` — Webcam-based hand tracking via MediaPipe
  - `orca_retargeter/` — URDF-based inverse kinematics retargeting
    - `utils/` — Helper functions and config
  - `viewer/` — Live 3D URDF visualization via viser (optional dependency)
- `.venv/` — Python virtual environment (not committed)

## Dependencies

**PyPI (in pyproject.toml):** `torch` (>=2.8), `numpy` (>=2.0), `pyyaml`, `mediapipe-numpy2` (custom full mediapipe build — see Known Issues), `avp-stream`, `pytorch-kinematics`, `arm-pytorch-utilities`

**Optional (viewer):** `viser` (>=0.2.0) — install with `pip install -e ".[viewer]"`

**Manual install:** `orca_core` (adjacent repo, `pip install -e ../orca_core`)

**Hardware:** Webcam (USB/built-in), ORCA robotic hand, optionally Apple Vision Pro

## Code Patterns

- Multiprocessing with daemon subprocess for robot control, `multiprocessing.Queue` for angle passing
- Threaded webcam capture in `MediaPipeIngress` with lock-protected shared state
- PyTorch-based optimization: RMSprop with 2 steps per frame, joint angle clamping after each step
- URDF forward kinematics via `pytorch_kinematics` chains for fingertip/palm position extraction
- Key vectors (palm → fingertip) used as the loss target, not raw positions
- Right-hand wrist angle negated to account for URDF coordinate convention
- MediaPipe landmarks scaled by 1.2x during preprocessing (calibration factor)
- Orientation validation: palm normal must face down (90-140°) and towards camera before callback fires
- Input-source abstraction: `retargeter_utils.py` has separate `preprocess_mediapipe_data()` and `preprocess_avp_data()` for each source

## Known Issues

- `pytorch_kinematics` and `arm-pytorch-utilities` install fine from PyPI (README claim about source install is outdated)
- `mediapipe-numpy2` (from https://github.com/cansik/mediapipe-numpy2) only has wheels for Python 3.9-3.12 — the venv must use Python 3.12 or lower. It also doesn't re-export `mediapipe.solutions` at the top level; the compatibility shim in `mediapipe_ingress.py` handles this
- Right-hand wrist angle inversion is a workaround (noted in code as future fix)
- MediaPipe landmark scale factor (1.2x) and palm offset ([0, 0, 0.015]) are hardcoded calibration values
- Abduction joint regularizers are hardcoded in `retargeter.yaml` (index, middle, ring, pinky)

## Before Finishing

After making changes, verify:
- No import errors in modified files (`source .venv/bin/activate && python -c "from orca_teleop import MediaPipeIngress, Retargeter"`)
- Changed files follow existing patterns (threading locks for shared state, joint clamping after optimization, etc.)
- Joint angles stay within ROM limits — never bypass clamping
- No hardcoded absolute paths
