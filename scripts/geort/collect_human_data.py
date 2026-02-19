"""Collect human hand motion data for GeoRT training.

Records hand landmarks from any supported input source (mediapipe, manus)
and saves them as a numpy array in GeoRT's canonical wrist-centered frame.

Usage:
    python scripts/geort/collect_human_data.py --source mediapipe --name my_data --model-path path/to/model

Output: [N_frames, N_landmarks, 3] numpy array saved to third_party/GeoRT/data/<name>.npy
"""
import os
import sys
import time
import argparse
import numpy as np

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

_GEORT_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "GeoRT")
sys.path.insert(0, _GEORT_ROOT)

from orca_teleop.orca_retargeter.utils import retargeter_utils


def main():
    parser = argparse.ArgumentParser(description="Collect human hand data for GeoRT training")
    parser.add_argument("--source", type=str, required=True, choices=["mediapipe", "manus"],
                        help="Input source type")
    parser.add_argument("--name", type=str, default="human_data",
                        help="Dataset name (saved to third_party/GeoRT/data/<name>.npy)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to ORCA hand model (for ingress initialization)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Max recording duration in seconds (default: 5 minutes)")
    # Manus-specific args
    parser.add_argument("--glove-id", type=str, default=None,
                        help="Manus glove hex ID (required for --source manus)")
    parser.add_argument("--zmq-addr", type=str, default="tcp://localhost:8000",
                        help="ZMQ address for Manus SDK stream")
    args = parser.parse_args()

    frames = []
    recording = False

    def on_landmarks(data):
        if not recording:
            return
        if args.source == "mediapipe":
            joints, _ = retargeter_utils.preprocess_mediapipe_data({"hand_landmarks": data})
        elif args.source == "manus":
            joints, _ = retargeter_utils.preprocess_manus_data(data)
        canonical = retargeter_utils.to_geort_canonical_frame(joints, args.source)
        frames.append(canonical)

    # Initialize ingress
    if args.source == "mediapipe":
        from orca_teleop import MediaPipeIngress
        ingress = MediaPipeIngress(args.model_path, callback=on_landmarks)
    elif args.source == "manus":
        if not args.glove_id:
            parser.error("--glove-id is required for manus source")
        from orca_teleop import ManusIngress
        ingress = ManusIngress(args.model_path, args.glove_id,
                               callback=on_landmarks, zmq_addr=args.zmq_addr)

    ingress.start()
    print(f"Source: {args.source}")
    print("Press Enter to START recording, then Enter again to STOP.")
    print("Move your hand through full ROM: stretch fingers, make fists, pinch, etc.")

    try:
        input("Press Enter to start recording...")
        recording = True
        print("RECORDING... (press Enter to stop)")
        start_time = time.time()

        # Wait for stop signal or timeout
        import select
        while time.time() - start_time < args.duration:
            if select.select([sys.stdin], [], [], 0.5)[0]:
                sys.stdin.readline()
                break
            if len(frames) % 100 == 0 and len(frames) > 0:
                elapsed = time.time() - start_time
                print(f"  Collected {len(frames)} frames ({elapsed:.1f}s)")

    except KeyboardInterrupt:
        pass
    finally:
        recording = False
        ingress.cleanup()

    if len(frames) == 0:
        print("No frames collected. Exiting.")
        return

    # Save
    data = np.array(frames)
    save_dir = os.path.join(_GEORT_ROOT, "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.name}.npy")
    np.save(save_path, data)
    print(f"Saved {len(frames)} frames with shape {data.shape} to {save_path}")


if __name__ == "__main__":
    main()
