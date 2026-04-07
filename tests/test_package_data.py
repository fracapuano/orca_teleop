from importlib import resources


def test_package_data_files_are_shipped():
    mediapipe_dir = resources.files("orca_teleop.orca_ingress.mediapipe")
    retargeter_utils_dir = resources.files("orca_teleop.orca_retargeter.utils")

    assert (mediapipe_dir / "hand_landmarker.task").is_file()
    assert (retargeter_utils_dir / "retargeter.yaml").is_file()
