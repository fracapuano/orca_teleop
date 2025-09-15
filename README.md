# orca_teleop

Repo for teleoperating the ORCA Hand consisting of an Ingress Source (for example Mediapipe, Apple Vision Pro, Rokoko Gloves, etc.) and a URDF-based Retargeter.

The Retargeter in orca_teleop requires pytorch_kinematics, but its newest pip package has no numpy>2 support yet. 
This is why you need to install pytorch_kinematics from source into your venv for the moment:

```
git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git
cd pytorch_kinematics
pip install -e . 
```

Steer your own ORCA hand using just your webcam:

```
python scripts/mediapipe_teleop_demo.py     path/to/your_orcahand_model     path/to/corresponding_urdf_file
```
