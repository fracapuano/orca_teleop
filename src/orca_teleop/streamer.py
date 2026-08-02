"""orca-teleop-streamer: headless ingress→retarget→orca_ui pipeline.

The child process the ORCA Hand Console spawns (or that you launch manually,
possibly on another machine) for teleoperation. It reuses the standard
pipeline pieces — gRPC ``IngressServer`` fed by a publisher, the
``retargeter_worker`` thread — but sinks retargeted joint targets into
orca_ui's ``/ws/teleop`` socket instead of a hand. orca_ui owns the hand and
all safety; this process is a pure producer.

    orca-teleop-streamer --connect ws://127.0.0.1:5001/ws/teleop \
        --token <token> --source mediapipe --model-path path/to/config.yaml

``--source synthetic`` streams a smooth waveform with no camera, retargeter,
or torch in the loop — the end-to-end wiring check.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time

from orca_teleop.hand_model import HandModel
from orca_teleop.ui_sink import ConfigState, OrcaUiSink, StreamerStats

logger = logging.getLogger("orca_teleop.streamer")

SYNTHETIC_RATE_HZ = 30.0
SYNTHETIC_WAVE_HZ = 0.2
SYNTHETIC_AMPLITUDE = 0.35  # fraction of each joint's half-ROM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orca-teleop-streamer", description=__doc__.split("\n")[0])
    parser.add_argument("--connect", default="ws://127.0.0.1:5001/ws/teleop",
                        help="orca_ui teleop ingress URL")
    parser.add_argument("--token", default=os.environ.get("ORCA_TELEOP_TOKEN"),
                        help="session token (default: $ORCA_TELEOP_TOKEN)")
    parser.add_argument("--source", default="mediapipe",
                        choices=["mediapipe", "manus", "synthetic"])
    parser.add_argument("--model-path", default=None,
                        help="hand model config.yaml (or its directory)")
    parser.add_argument("--urdf-path", default=None,
                        help="semantic hand URDF (default: resolved from "
                             "$ORCAHAND_DESCRIPTION_DIR)")
    parser.add_argument("--retargeter", default="adaptive_analytical",
                        choices=["adaptive_analytical", "rmsprop"])
    parser.add_argument("--retarget-config", default=None,
                        help="YAML config for the adaptive backend")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="max targets/s sent to orca_ui (default 50)")
    parser.add_argument("--wrist-angle", type=float, default=0.0,
                        help="initial manual wrist angle in degrees")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="webcam index for --source mediapipe")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="MediaPipe confidence")
    parser.add_argument("--zmq-addr", default="tcp://127.0.0.1:2044",
                        help="Manus SDK client ZMQ address")
    parser.add_argument("--no-local", action="store_true",
                        help="don't spawn a local publisher — wait for a "
                             "remote one to connect to the gRPC ingress")
    parser.add_argument("--port", type=int, default=0,
                        help="gRPC ingress port (0 = ephemeral)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe webcams, print JSON, exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def list_cameras(max_index: int = 10) -> list[dict]:
    """Probe cv2 camera indices. Doubles as the macOS camera-permission
    trigger for a freshly-spawned child."""
    import cv2  # lazy: synthetic/manus paths never need it

    cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        width = height = 0
        if ok:
            read_ok, frame = cap.read()
            ok = bool(read_ok and frame is not None)
            if ok:
                height, width = frame.shape[:2]
        cap.release()
        if ok:
            cameras.append({"index": index, "width": width, "height": height})
    return cameras


# ----- synthetic source -----------------------------------------------------------


def synthetic_worker(hand_model: HandModel, actions_q: queue.Queue,
                     stop_event: threading.Event) -> None:
    """Smooth per-joint sine waves — no camera, no retargeter, no torch."""
    joints = hand_model.config.joint_ids
    roms = hand_model.config.joint_roms_dict
    t0 = time.monotonic()
    while not stop_event.is_set():
        t = time.monotonic() - t0
        angles = {}
        for index, joint in enumerate(joints):
            lo, hi = roms[joint]
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            angles[joint] = mid + SYNTHETIC_AMPLITUDE * half * math.sin(
                2 * math.pi * SYNTHETIC_WAVE_HZ * t + 0.7 * index)
        try:
            actions_q.put_nowait(angles)
        except queue.Full:
            pass
        time.sleep(1.0 / SYNTHETIC_RATE_HZ)


# ----- local publishers -------------------------------------------------------------


def _start_mediapipe_local(args, hand_model: HandModel, port: int,
                           sink: OrcaUiSink, config: ConfigState,
                           stop_event: threading.Event):
    """Run the MediaPipe publisher in-process (thread mode).

    In-process means the annotated camera frames are directly reachable for
    the orca_ui preview — a subprocess publisher can't share them. cv2 and
    mediapipe release the GIL during the heavy work, and the sink's main loop
    is light, so a thread is fine here.
    """
    from orca_teleop.ingress.mediapipe.publisher import MediaPipePublisher

    publisher = MediaPipePublisher(
        server_address=f"localhost:{port}",
        handedness=hand_model.config.type,
        confidence=args.confidence,
        camera_index=args.camera_index,
    )
    def on_config(patch: dict) -> None:
        if "orientation_gate" in patch:
            publisher.orientation_gate = bool(patch["orientation_gate"])
            logger.info("orientation gate %s",
                        "on" if publisher.orientation_gate else "off")

    config.on_update(on_config)
    if config.get("orientation_gate") is not None:
        publisher.orientation_gate = bool(config.get("orientation_gate"))

    run_thread = threading.Thread(
        target=publisher.run, name="mediapipe-publisher", daemon=True)
    run_thread.start()
    preview_thread = threading.Thread(
        target=_preview_loop, args=(publisher, sink, config, stop_event),
        name="preview", daemon=True)
    preview_thread.start()
    logger.info("local mediapipe publisher started (camera %d)",
                args.camera_index)
    return publisher, [run_thread, preview_thread]


PREVIEW_HZ = 8.0


def _preview_loop(publisher, sink: OrcaUiSink, config: ConfigState,
                  stop_event: threading.Event) -> None:
    """Ship downscaled annotated JPEG frames while orca_ui asks for them
    (``config.preview`` toggles with the browser panel's visibility)."""
    import base64

    seq = 0
    while not stop_event.is_set():
        enabled = bool(config.get("preview"))
        publisher.retain_frames = enabled
        if enabled:
            jpeg = publisher.latest_annotated_jpeg()
            if jpeg:
                seq += 1
                sink.send_preview(base64.b64encode(jpeg).decode("ascii"), seq)
        time.sleep(1.0 / PREVIEW_HZ)


def spawn_local_publisher(args, hand_model: HandModel, port: int):
    """Subprocess publishers (sources without preview support)."""
    import multiprocessing

    handedness = hand_model.config.type
    if args.source == "manus":
        # Usually remote (the Manus SDK client is Linux-only); local spawn
        # works on a Linux host with the manus extra installed.
        from orca_teleop.pipeline import _manus_publisher

        process = multiprocessing.Process(
            target=_manus_publisher, args=(port, handedness, args.zmq_addr),
            name="manus-publisher", daemon=True)
    else:
        raise SystemExit(
            f"--source {args.source} has no local publisher yet — "
            "run its publisher separately and pass --no-local")
    process.start()
    logger.info("local %s publisher started (pid=%s)", args.source, process.pid)
    return process


# ----- main --------------------------------------------------------------------------


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S")

    if args.list_cameras:
        print(json.dumps(list_cameras()))
        return

    if not args.token:
        raise SystemExit("--token (or $ORCA_TELEOP_TOKEN) is required")
    if not args.model_path:
        raise SystemExit("--model-path is required")

    hand_model = HandModel.load(args.model_path)
    config = ConfigState(manual_wrist_deg=args.wrist_angle)
    stats = StreamerStats()
    stop_event = threading.Event()
    sink = OrcaUiSink(
        args.connect, args.token, args.source, hand_model,
        config=config, stats=stats, rate_hz=args.rate, stop_event=stop_event)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())

    if args.source == "synthetic":
        _run_synthetic(sink, hand_model, stats, stop_event)
    else:
        _run_pipeline(args, sink, hand_model, config, stats, stop_event)
    logger.info("streamer shut down")


def _run_synthetic(sink: OrcaUiSink, hand_model: HandModel,
                   stats: StreamerStats, stop_event: threading.Event) -> None:
    actions_q: queue.Queue = queue.Queue(maxsize=8)
    sink.connect()
    worker = threading.Thread(
        target=synthetic_worker, args=(hand_model, actions_q, stop_event),
        name="synthetic-source", daemon=True)
    worker.start()
    # Synthetic frames double as ingress frames for the status report.
    ticker = threading.Thread(
        target=_synthetic_stats_ticker, args=(stats, stop_event),
        name="synthetic-stats", daemon=True)
    ticker.start()
    try:
        sink.run_loop(actions_q, stop_event)
    finally:
        stop_event.set()
        sink.close()


def _synthetic_stats_ticker(stats: StreamerStats,
                            stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        stats.record_frame(retarget_ms=0.0)
        time.sleep(1.0 / SYNTHETIC_RATE_HZ)


def _run_pipeline(args, sink: OrcaUiSink, hand_model: HandModel,
                  config: ConfigState, stats: StreamerStats,
                  stop_event: threading.Event) -> None:
    # Heavy imports stay inside so `--source synthetic`/`--list-cameras`
    # never pay for torch.
    from orca_teleop.ingress.server import IngressServer
    from orca_teleop.pipeline import TeleopQueues, retargeter_worker
    from orca_teleop.retargeting.retargeter import Retargeter

    logger.info("building %s retargeter…", args.retargeter)
    retargeter = Retargeter.from_paths(
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        backend=args.retargeter,
        config_path=args.retarget_config,
        hand_model=hand_model,
    )

    def on_config(patch: dict) -> None:
        if patch.get("recalibrate") and hasattr(retargeter, "reset"):
            logger.info("recalibrating retargeter")
            retargeter.reset()

    config.on_update(on_config)

    queues = TeleopQueues(
        landmarks_q=queue.Queue(maxsize=8), actions_q=queue.Queue(maxsize=8))
    server = IngressServer(queues.landmarks_q, stop_event, port=args.port)
    bound_port = server.start()
    logger.info("gRPC ingress on :%d", bound_port)

    publisher_proc = None
    local_publisher = None
    local_threads: list[threading.Thread] = []
    if not args.no_local:
        if args.source == "mediapipe":
            local_publisher, local_threads = _start_mediapipe_local(
                args, hand_model, bound_port, sink, config, stop_event)
        else:
            publisher_proc = spawn_local_publisher(args, hand_model, bound_port)
    else:
        logger.info("waiting for a remote publisher on :%d "
                    "(e.g. python -m orca_teleop.ingress.%s.publisher "
                    "--server <this-host>:%d)",
                    bound_port, args.source, bound_port)

    worker = threading.Thread(
        target=retargeter_worker,
        kwargs=dict(queues=queues, stop_event=stop_event,
                    retargeter=retargeter,
                    wrist_provider=config.wrist_deg, stats=stats),
        name="retargeter", daemon=True)
    worker.start()

    try:
        sink.connect()
        sink.run_loop(queues.actions_q, stop_event)
    finally:
        stop_event.set()
        if local_publisher is not None:
            local_publisher.stop()
        server.stop()
        worker.join(timeout=3.0)
        for thread in local_threads:
            thread.join(timeout=3.0)
        if publisher_proc is not None and publisher_proc.is_alive():
            publisher_proc.terminate()
            publisher_proc.join(timeout=3.0)
        sink.close()


if __name__ == "__main__":
    sys.exit(main())
