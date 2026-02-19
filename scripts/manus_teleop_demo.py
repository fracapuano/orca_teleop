import sys
import time
import multiprocessing
import numpy as np
import argparse
from orca_teleop import ManusIngress, Retargeter


def robot_control_process_worker(q, stop, ready, model_path):
    try:
        from orca_core import OrcaHand
        hand = OrcaHand(model_path)
        success, message = hand.connect()
        if not success:
            print(f"Robot process: Failed to connect: {message}")
            return
        hand.init_joints()
        ready.set()
        while not stop.is_set():
            try:
                angles = q.get(timeout=0.1)
                if angles:
                    joints = {n.split('_',1)[1] if n.startswith(('left_','right_')) else n: float(p) for n,p in angles.items()}
                    hand.set_joint_pos({n: np.rad2deg(p) for n,p in joints.items()})
            except Exception:
                continue
    except Exception as e:
        print(f"Robot process error: {e}")
    finally:
        try:
            hand.disable_torque()
            hand.disconnect()
        except Exception:
            pass


def process_skeleton(data):
    angles = retargeter.retarget(data)
    if angles_queue:
        angles_queue.put_nowait(angles)
    if viewer:
        viewer.update(angles)
        if show_mano and retargeter.mano_points is not None:
            viewer.update_mano_points(retargeter.mano_points)


def main():
    global retargeter, angles_queue, viewer, show_mano
    parser = argparse.ArgumentParser(description='Manus Glove to Orca Hand teleop demo')
    parser.add_argument('model_path')
    parser.add_argument('urdf_path')
    parser.add_argument('--glove-id', required=True, help='Manus glove hex ID (e.g. 8569617b)')
    parser.add_argument('--zmq-addr', default='tcp://localhost:8000', help='ZMQ address for Manus SDK stream')
    parser.add_argument('--no-viewer', action='store_true', help='Disable 3D URDF viewer')
    parser.add_argument('--no-robot', action='store_true', help='Run without robot hardware')
    parser.add_argument('--show-mano', action='store_true', help='Show MANO landmarks in 3D viewer')
    parser.add_argument('--retargeter', choices=['default', 'neural-geort'], default='default', help='Retargeter to use')
    parser.add_argument('--geort-checkpoint', type=str, default=None, help='Path to GeoRT IK model checkpoint (.pth)')
    parser.add_argument('--geort-config', type=str, default=None, help='Path to GeoRT config JSON (with joint limits)')
    args = parser.parse_args()
    show_mano = args.show_mano

    viewer = None
    if not args.no_viewer:
        try:
            from orca_teleop.viewer import URDFViewer
            viewer = URDFViewer(args.urdf_path)
        except ImportError:
            print("viser not installed — skipping 3D viewer (pip install -e '.[viewer]')")
        except Exception as e:
            print(f"Failed to start viewer: {e}")

    if args.retargeter == 'neural-geort':
        if not args.geort_checkpoint or not args.geort_config:
            print("Error: --geort-checkpoint and --geort-config required for neural-geort retargeter")
            return 1
        from orca_teleop import NeuralGeoRTRetargeter
        retargeter = NeuralGeoRTRetargeter(args.model_path, args.urdf_path,
                                           geort_checkpoint=args.geort_checkpoint,
                                           geort_config=args.geort_config, source="manus")
    else:
        retargeter = Retargeter(args.model_path, args.urdf_path, source="manus")
    ingress = ManusIngress(args.model_path, args.glove_id, callback=process_skeleton,
                           zmq_addr=args.zmq_addr)

    robot_control_process = None
    stop_robot_control = None
    angles_queue = None

    if not args.no_robot:
        angles_queue = multiprocessing.Queue()
        stop_robot_control = multiprocessing.Event()
        robot_ready_event = multiprocessing.Event()
        robot_control_process = multiprocessing.Process(
            target=robot_control_process_worker,
            args=(angles_queue, stop_robot_control, robot_ready_event, args.model_path), daemon=True)

        robot_control_process.start()
        if not robot_ready_event.wait(timeout=5.0):
            print("Robot initialization timeout. Exiting.")
            robot_control_process.terminate()
            return 1

    ingress.start()
    print("Manus teleop running. Ctrl+C to quit")

    try:
        while not (viewer and viewer.stopped):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Stopping demo...")
        ingress.cleanup()
        if viewer:
            viewer.close()
        if robot_control_process and robot_control_process.is_alive():
            stop_robot_control.set()
            robot_control_process.join(timeout=3.0)
            if robot_control_process.is_alive():
                robot_control_process.terminate()
                robot_control_process.join(timeout=1.0)
    return 0


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())
