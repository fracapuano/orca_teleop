"""Train GeoRT FK model from ORCA URDF.

Generates 100K random joint angle → fingertip position pairs via SAPIEN,
then trains an MLP to approximate forward kinematics.

Usage:
    python scripts/geort/train_fk.py --hand orca_right --urdf-path path/to/orca_right.urdf

Requires: sapien, geort (from third_party/GeoRT)
"""
import os
import sys
import argparse
import shutil

# Add GeoRT to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GEORT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "third_party", "GeoRT")
sys.path.insert(0, _GEORT_ROOT)

from geort.utils.config_utils import get_config
from geort.trainer import GeoRTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GeoRT FK model from ORCA URDF")
    parser.add_argument("--hand", type=str, required=True,
                        help="Hand config name (e.g. orca_right, orca_left)")
    parser.add_argument("--urdf-path", type=str, required=True,
                        help="Path to the ORCA URDF file")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retraining even if FK checkpoint exists")
    args = parser.parse_args()

    config = get_config(args.hand)

    # Copy URDF to GeoRT's expected assets location so SAPIEN can find it
    assets_dir = os.path.join(_GEORT_ROOT, "assets", config["name"])
    os.makedirs(assets_dir, exist_ok=True)
    urdf_dest = os.path.join(assets_dir, os.path.basename(args.urdf_path))
    if not os.path.exists(urdf_dest):
        # Copy the URDF and its sibling mesh files
        urdf_dir = os.path.dirname(os.path.abspath(args.urdf_path))
        shutil.copytree(urdf_dir, assets_dir, dirs_exist_ok=True)
        print(f"Copied URDF assets from {urdf_dir} to {assets_dir}")

    # Override urdf_path in config to point to the copied file
    config["urdf_path"] = urdf_dest

    # Change working directory to GeoRT root so relative paths work
    original_cwd = os.getcwd()
    os.chdir(_GEORT_ROOT)

    try:
        trainer = GeoRTTrainer(config)
        fk_model = trainer.get_robot_neural_fk_model(force_train=args.force_retrain)
        print(f"FK model ready. Checkpoint: {trainer.get_fk_checkpoint_path()}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
