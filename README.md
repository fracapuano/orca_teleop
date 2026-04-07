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
uv sync --extra test --extra mediapipe
```

This installs the package itself plus the testing tools and MediaPipe dependencies used by the demo and current package imports.

Steer your own ORCA hand using just your webcam:

```
python scripts/mediapipe_teleop_demo.py     path/to/your_orcahand_model     path/to/corresponding_urdf_file
```

Tests always run on CI. Run the regression suite from the repository root with:

```bash
pytest tests/
```
