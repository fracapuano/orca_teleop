from importlib import resources


def test_package_data_files_are_shipped():
    mediapipe_dir = resources.files("orca_teleop.orca_ingress.mediapipe")
    retargeting_dir = resources.files("orca_teleop.retargeting")

    assert (mediapipe_dir / "hand_landmarker.task").is_file()
    assert (retargeting_dir / "retargeter.yaml").is_file()
