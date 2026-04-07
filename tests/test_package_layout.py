import sys
from pathlib import Path

import orca_teleop


def test_src_layout_exists():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "src" / "orca_teleop").is_dir()
    assert (repo_root / "tests").is_dir()


def test_top_level_package_import_is_lazy():
    assert orca_teleop.__all__ == ["MediaPipeIngress", "Retargeter"]
    assert "orca_teleop.orca_ingress.mediapipe.mediapipe_ingress" not in sys.modules
    assert "orca_teleop.orca_retargeter.retargeter" not in sys.modules
