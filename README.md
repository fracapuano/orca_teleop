# orca_teleop

Teleoperation system for the ORCA Hand. Tracks a human hand via an ingress source (MediaPipe webcam, Apple Vision Pro, Rokoko Gloves, etc.) and retargets the pose to robot joint angles using a URDF-based retargeter.

## Installation

**Requires Python 3.10-3.12** (the `mediapipe-numpy2` package does not have wheels for Python 3.13+).

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install orca_teleop and its dependencies

```bash
pip install -e .
```

This installs all PyPI dependencies listed in `pyproject.toml` (`torch`, `numpy`, `mediapipe-numpy2`, `pytorch-kinematics`, `arm-pytorch-utilities`, `pyyaml`, `avp-stream`).

### 2b. (Optional) Install the 3D viewer

```bash
pip install -e ".[viewer]"
```

This adds `viser`, which provides a live 3D visualization of the URDF hand model in your browser. The viewer is enabled by default when viser is installed.

### 3. Install orca_core

`orca_core` provides the `OrcaHand` class for communicating with the physical robot. Install it from the `orca_core` repository.

**For development** (if you need to modify orca_core):
```bash
pip install -e path/to/orca_core
```

**For usage only:**
```bash
pip install orca_core
```

## Usage

Steer your own ORCA Hand using just your webcam:

```bash
source .venv/bin/activate
python scripts/mediapipe_teleop_demo.py path/to/your_orcahand_model path/to/corresponding_urdf_file
```

**Options:**
- `--no-display` — Run headless (no webcam preview window)
- `--no-viewer` — Disable the 3D browser viewer
- `--no-robot` — Run without robot hardware (visualization-only mode, useful for testing the pipeline without a physical hand)
