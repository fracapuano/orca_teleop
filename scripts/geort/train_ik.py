"""Train GeoRT IK model from human data + FK model.

Learns the mapping from human fingertip positions to ORCA joint angles using
geometric losses (direction, chamfer, curvature, pinch). No paired demonstrations
needed — fully unsupervised.

Usage:
    python scripts/geort/train_ik.py --hand orca_right --urdf-path path/to/urdf \
        --human-data my_data --tag my_experiment

Requires: sapien, open3d, geort (from third_party/GeoRT)
"""
import os
import sys
import argparse

# Add GeoRT to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GEORT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "third_party", "GeoRT")
sys.path.insert(0, _GEORT_ROOT)

from geort.utils.config_utils import get_config
from geort.utils.path import get_human_data
from geort.trainer import GeoRTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GeoRT IK model for ORCA hand")
    parser.add_argument("--hand", type=str, required=True,
                        help="Hand config name (e.g. orca_right, orca_left)")
    parser.add_argument("--urdf-path", type=str, required=True,
                        help="Path to the ORCA URDF file")
    parser.add_argument("--human-data", type=str, required=True,
                        help="Name of human data (saved by collect_human_data.py)")
    parser.add_argument("--tag", type=str, default="",
                        help="Experiment tag for checkpoint naming")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--w-chamfer", type=float, default=80.0,
                        help="Chamfer loss weight")
    parser.add_argument("--w-curvature", type=float, default=0.1,
                        help="Curvature loss weight")
    parser.add_argument("--w-pinch", type=float, default=1.0,
                        help="Pinch loss weight")
    args = parser.parse_args()

    config = get_config(args.hand)

    # Override urdf_path if the asset was already copied by train_fk.py
    assets_dir = os.path.join(_GEORT_ROOT, "assets", config["name"])
    urdf_in_assets = os.path.join(assets_dir, os.path.basename(args.urdf_path))
    if os.path.exists(urdf_in_assets):
        config["urdf_path"] = urdf_in_assets
    else:
        config["urdf_path"] = os.path.abspath(args.urdf_path)

    human_data_path = get_human_data(args.human_data)
    if human_data_path is None:
        print(f"Human data '{args.human_data}' not found in {os.path.join(_GEORT_ROOT, 'data')}")
        print("Run collect_human_data.py first.")
        return

    # Change working directory to GeoRT root so relative paths work
    original_cwd = os.getcwd()
    os.chdir(_GEORT_ROOT)

    try:
        trainer = GeoRTTrainer(config)
        print(f"Training IK model with human data: {human_data_path}")
        trainer.train(
            str(human_data_path),
            tag=args.tag,
            epoch=args.epochs,
            w_chamfer=args.w_chamfer,
            w_curvature=args.w_curvature,
            w_pinch=args.w_pinch,
        )
        print("Training complete. Checkpoints saved to third_party/GeoRT/checkpoint/")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
