#!/usr/bin/env python3
"""Run the TRON 2 arm from a live orca_teleop pipeline.

    # against the local mock robot (default, safe)
    python tools/run_arm.py --sim --model-path /path/to/config.yaml

    # against real hardware -- requires --robot AND typed confirmation
    python tools/run_arm.py --robot 10.192.1.2 --model-path /path/to/config.yaml --no-wrist

``--no-wrist`` IS MANDATORY FOR HARDWARE RUNS
---------------------------------------------
It passes ``wrist_enabled=False`` to the pipeline, which commands the ORCA hand's
1-DoF wrist joint to 0 and holds it there. Our kinematics treat the hand as a
*rigid* link bolted to the flange: the tool offset ``T_FH`` is a constant. If the
wrist is left enabled it moves under us, ``T_FH`` silently stops being the
transform we are compensating for, and every palm-space target is wrong by
whatever the wrist happened to be doing. The error is smooth and plausible-
looking, which is what makes it dangerous. On hardware, pass ``--no-wrist``.

Clutch: hold SPACE to engage. Release and both arms hold position.

Hotkeys (raw terminal):
    x / ESC  force HOLD on both arms (panic; press again to release)
    o        toggle orientation freeze -- position-only mode
    q        clean exit (freeze 1 s, then drop the socket)

The clutch shares the same keyboard reader, and its own key is passed through
untouched -- so SPACE only ever means "engage".
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tron2_cli import Refused, gate_target, normalise_target  # noqa: E402

from tron_arm.clutch import KeyboardClutch  # noqa: E402
from tron_arm.config import ARMS, ConfigError, load_config  # noqa: E402
from tron_arm.session import SessionLogger, new_session_id  # noqa: E402
from tron_arm.sink import TronArmSink  # noqa: E402
from tron_arm.step_test import run_step_test  # noqa: E402

logger = logging.getLogger("run_arm")

MOCK_URL = "ws://127.0.0.1:5000"


class _AlwaysEngaged:
    """``--clutch none``: always engaged, but still reads the keyboard.

    Wraps a real keyboard reader so the hotkeys keep working; only ``engaged``
    is overridden. Intended for bench work with the mock, never for hardware.
    """

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def start(self) -> None:
        self._reader.start()

    def stop(self) -> None:
        self._reader.stop()

    @property
    def available(self) -> bool:
        return getattr(self._reader, "available", False)

    @property
    def engaged(self) -> bool:
        return True


class StatusLine:
    """One-line live status, repainted on a timer.

    Writes to stderr with a carriage return so it stays put and does not fight
    the pipeline's logging on stdout.
    """

    def __init__(self, sink: TronArmSink, *, hz: float = 5.0, stream: Any = None) -> None:
        self._sink = sink
        self._period = 1.0 / hz
        self._stream = stream if stream is not None else sys.stderr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="status", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with_newline = getattr(self._stream, "write", None)
        if with_newline is not None:
            self._stream.write("\n")
            self._stream.flush()

    def render(self) -> str:
        sink = self._sink
        parts: list[str] = []
        for arm in ARMS:
            d = sink.diagnostics(arm)
            age = f"{d.last_age_s * 1e3:.0f}ms" if d.last_age_s is not None else "--"
            parts.append(
                f"{arm[0].upper()}:{d.state[:11]:<11} age={age:>6} "
                f"ws={d.workspace_clamps} st={d.step_clamps}"
            )
        streamer = sink.streamer
        rate = f"{streamer.stats.achieved_rate_hz:6.1f}Hz" if streamer else "   --Hz"
        notify = sink.last_notify.title if sink.last_notify else "none"
        flags = []
        if sink.orientation_frozen:
            flags.append("ORI-FROZEN")
        if sink._forced_hold:  # noqa: SLF001 - status display of an operator action
            flags.append("FORCED-HOLD")
        flag_text = (" [" + ",".join(flags) + "]") if flags else ""
        return (
            f"\r{' | '.join(parts)} | {rate} | "
            f"p95={sink.stats.dispatch_percentile_ms():.2f}ms | {notify}{flag_text}   "
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream.write(self.render())
                self._stream.flush()
            except (OSError, ValueError):
                return
            self._stop.wait(self._period)


#: Force-hold. NOT space: space is the default clutch key, and a key that both
#: engages the clutch and toggles force-hold would fight itself on every key
#: repeat -- force_hold makes the clutch read as released.
FORCE_HOLD_KEYS = ("x", "\x1b")  # x, or ESC


def make_hotkey_handler(sink: TronArmSink, clutch_key: str = " ") -> Any:
    """Return an ``on_key`` callback implementing the documented hotkeys.

    The clutch shares this reader, so its own key must pass through untouched.
    """

    def handle(char: str) -> None:
        if char.lower() == clutch_key.lower():
            return  # the clutch owns this key
        if char.lower() in FORCE_HOLD_KEYS:
            new = not sink._forced_hold  # noqa: SLF001 - operator toggle
            sink.force_hold(new)
            logger.warning("forced hold %s", "ON" if new else "OFF")
        elif char.lower() == "o":
            sink.set_orientation_frozen(not sink.orientation_frozen)
            logger.warning("orientation freeze %s",
                           "ON (position-only)" if sink.orientation_frozen else "OFF")
        elif char.lower() == "q":
            logger.warning("q: shutting down")
            # Unwind the blocking pipeline call in the main thread; its own
            # handler runs the ordinary shutdown path, which calls sink.close().
            os.kill(os.getpid(), signal.SIGINT)

    return handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_arm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--sim", action="store_true",
                        help=f"drive the local mock robot at {MOCK_URL} (default)")
    target.add_argument("--robot", metavar="HOST", default=None,
                        help="drive real hardware; requires interactive confirmation")

    parser.add_argument("--config", type=Path, default=None, help="tron_arm YAML config")
    parser.add_argument("--model-path", default=None,
                        help="orca_core hand config.yaml, passed to run_metaquest_local")
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--port", type=int, default=50051, help="gRPC ingress port")
    parser.add_argument("--hand", choices=("left", "right"), default="right",
                        help="handedness reported by the publisher")
    parser.add_argument("--quest-port", type=int, default=8765)
    parser.add_argument("--no-wrist", action="store_true",
                        help="MANDATORY ON HARDWARE: wrist_enabled=False, so the ORCA "
                             "wrist holds 0 and the flange->palm offset stays rigid")
    parser.add_argument("--clutch", choices=("keyboard", "pedal", "none"), default="keyboard")
    parser.add_argument("--clutch-key", default=None,
                        help="key held to engage (default: SPACE for keyboard, 'a' for pedal)")
    parser.add_argument("--arm-only", action="store_true",
                        help="run the ingress + arm worker WITHOUT the hand retargeter. "
                             "This is runbook step 5 (single-arm teleop before hands are "
                             "in the loop) and needs no torch or hand model.")
    parser.add_argument("--step-test", action="store_true",
                        help="bypass the operator source and run the runbook step 4 "
                             "sequence: hold 5 s, G-09 readback per format, +-2 cm axis "
                             "steps, 2 cm circle. Exits when done.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop after N seconds (for CI runs)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="session log root (default: %(default)s)")
    parser.add_argument("--session-id", default=None, help="override the session id")
    parser.add_argument("--no-log", action="store_true", help="disable session logging")
    parser.add_argument("--no-status", action="store_true", help="suppress the status line")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def resolve_url(args: argparse.Namespace, config_url: str) -> str:
    """Pick the robot URL and run it past the shared hard-rule-1 gate."""
    if args.robot:
        return gate_target(normalise_target(args.robot), explicit_robot_flag=True,
                           description="run_arm (live teleoperation)")
    url = MOCK_URL if args.sim else config_url
    return gate_target(url, explicit_robot_flag=False, description="run_arm")


def _run_arm_only(sink: TronArmSink, args: argparse.Namespace,
                  ingress_cls: Any, arm_worker: Any) -> None:
    """Ingress + arm worker, without the hand retargeter.

    This is `python -m orca_teleop.arm` with our sink instead of the logging
    one: the same ingress, the same staleness deadline, the same hold and
    reference-change callbacks -- just no retargeter, no hand, and no torch.

    Runbook step 5 (single-arm teleop) happens before step 6 (hands in the
    loop), so this is a real operating mode, not a test shim.
    """
    import queue
    import threading

    # arm_worker does NOT connect the sink -- arm.py's main() and
    # pipeline.py both call connect() themselves before starting the worker.
    # Miss this and the sink silently computes targets and drops every one.
    sink.connect()

    stop = threading.Event()
    landmarks_q: queue.Queue = queue.Queue(maxsize=8)
    server = ingress_cls(landmarks_q, stop, port=args.port)
    bound = server.start()
    logger.info("arm-only ingress on :%d (no retargeter, no hand)", bound)

    # Drain and discard: with no consumer the queue fills and the servicer's
    # latest-wins drop then runs on every frame.
    def drain() -> None:
        while not stop.is_set():
            try:
                landmarks_q.get(timeout=0.1)
            except queue.Empty:
                continue

    threading.Thread(target=drain, name="drain", daemon=True).start()
    worker = threading.Thread(
        target=arm_worker, kwargs=dict(frames=server.frames, sink=sink, stop_event=stop),
        name="arm-worker", daemon=True,
    )
    worker.start()
    try:
        if args.seconds is not None:
            stop.wait(args.seconds)
        else:
            while not stop.is_set():
                stop.wait(0.2)
    finally:
        stop.set()
        worker.join(timeout=5.0)   # its finally: calls sink.close()
        with contextlib.suppress(Exception):
            server.stop()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    try:
        url = resolve_url(args, config.robot.url)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    if args.robot and not args.no_wrist:
        print(
            "refused: --no-wrist is mandatory for hardware runs. Without it the ORCA "
            "wrist stays live, the flange->palm offset T_FH stops being rigid, and "
            "every palm-space target is silently wrong.",
            file=sys.stderr,
        )
        return 2

    session = SessionLogger(
        args.log_dir,
        session_id=args.session_id or new_session_id(),
        enabled=not args.no_log,
    )
    session.open()
    if not args.no_log:
        logger.info("session log: %s", session.dir)

    # --step-test drives the robot directly: no ingress, no operator, no clutch.
    if args.step_test:
        import asyncio

        report = asyncio.run(run_step_test(config, url, arm=args.hand, session=session))
        print(report.text())
        session.close()
        return 0 if report.chosen_format else 1

    try:
        if args.arm_only:
            from orca_teleop.arm import arm_worker
            from orca_teleop.ingress.server import IngressServer
        else:
            from orca_teleop.pipeline import run_metaquest_local
    except ImportError as exc:
        print(f"orca_teleop is not importable ({exc}). Install it in this environment: "
              f"the pipeline lives upstream, not here.", file=sys.stderr)
        return 1

    sink = TronArmSink(config, url=url, session=session)
    default_key = " " if args.clutch != "pedal" else "a"
    key = args.clutch_key if args.clutch_key is not None else default_key
    hotkeys = make_hotkey_handler(sink, clutch_key=key)
    # A pedal is a keyboard with a different keycode and a faster repeat.
    timeout = 0.15 if args.clutch == "pedal" else 0.25
    clutch: Any = KeyboardClutch(key, hold_timeout_s=timeout, on_key=hotkeys)
    if args.clutch == "none":
        clutch = _AlwaysEngaged(clutch)
    sink._clutch = clutch  # noqa: SLF001 - wired after the hotkeys it shares a reader with

    status = None if args.no_status else StatusLine(sink)
    clutch.start()
    if not getattr(clutch, "available", False):
        logger.warning(
            "no tty: hotkeys and the keyboard clutch are unavailable. "
            "The clutch reads as RELEASED, so the arm will hold. Use --clutch none "
            "for headless bench runs against the mock."
        )
    if status is not None:
        status.start()

    logger.info("robot=%s  clutch=%s(%r)  wrist_enabled=%s  mode=%s", url, args.clutch, key,
                not args.no_wrist, "arm-only" if args.arm_only else "full pipeline")
    try:
        if args.arm_only:
            _run_arm_only(sink, args, IngressServer, arm_worker)
        else:
            run_metaquest_local(
                model_path=args.model_path,
                urdf_path=args.urdf_path,
                port=args.port,
                handedness=args.hand,
                quest_port=args.quest_port,
                wrist_enabled=not args.no_wrist,
                arm_sink=sink,
            )
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    finally:
        if status is not None:
            status.stop()
        clutch.stop()
        sink.close()  # idempotent: the pipeline calls it too on its own path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
