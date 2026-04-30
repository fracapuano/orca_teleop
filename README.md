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

## Arm Teleop Pipeline

The arm demos all end at the same boundary: `BimanualIKSolver.solve(...)` in
`src/orca_teleop/orca_arm_sink.py` expects absolute `pin.SE3` wrist targets in
robot world coordinates, and `OrcaArmMeshcatSink` renders the solved arm state.

`scripts/teleop_arm_sim.py` is the synthetic smoke test for that boundary. It
samples reachable wrist targets from the OrcaArm URDF, solves IK, and displays
the target and current wrist triads in meshcat.

`scripts/teleop_arm_quest.py` replaces the synthetic target generator with the
Quest ingress path:

1. `src/orca_teleop/ingress/metaquest/publisher.py` publishes Quest hand poses over gRPC.
2. `orca_teleop.ingress.server.IngressServer` receives those poses as `HandLandmarks`.
3. `_wrist_pose_to_robot_se3(...)` converts Quest Unity coordinates to robot FLU.
4. `_drain_queue(...)` (1) anchors the operator's first stable wrist pose, (2) optionally auto-fits translation scale, (3) maps operator deltas onto the robot home carpals pose, to produce absolute `pin.SE3` IK targets.
5. `BimanualIKSolver` solves those targets and `OrcaArmMeshcatSink` visualizes the result.

Tests always run on CI. Run the regression suite from the repository root with:

```bash
pytest tests/
```
