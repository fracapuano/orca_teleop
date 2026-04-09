import argparse
import sys

from orca_teleop import pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threaded teleop pipeline")
    parser.add_argument("model_path", help="Path to OrcaHand model directory")
    args = parser.parse_args()
    try:
        pipeline.run(args.model_path)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
