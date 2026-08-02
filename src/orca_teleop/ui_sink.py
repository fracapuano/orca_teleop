"""RobotSink that streams retargeted joint targets to the ORCA Hand Console.

Instead of commanding a hand directly, this sink dials orca_ui's teleop
ingress WebSocket (``/ws/teleop``) and speaks its wire protocol (mirrored
from ``orca_ui/hand/teleop/protocol.py``, PROTO_VERSION 1):

    out: hello {token, proto, source, hand{model_name, side}, pid, versions}
         targets {angles: {bare_joint_id: degrees}, seq, t}
         status  {ingress_fps?, tracking, retarget_ms?, calibrating?}  ~1 Hz
         log     {level, line}
    in:  hello_ok {session_id, proto, joints, roms, neutral, engaged}
         config   partial {manual_wrist_deg?, preview?, lp_alpha?, recalibrate?}
         engaged  {engaged: bool}
         stop     {}

orca_ui owns all safety (arbiter, ramp-in, watchdog, clamping); this sink
never touches a hand and simply stops streaming when told to.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from orca_teleop.constants import HEARTBEAT_INTERVAL

logger = logging.getLogger(__name__)

PROTO_VERSION = 1
HELLO_TIMEOUT_S = 10.0
STATUS_INTERVAL_S = 1.0
_SHUTDOWN_TYPES = (type(None),)


class ConfigState:
    """Thread-safe runtime config, mutated by ``config`` messages from the UI
    and read by the retargeter worker (wrist angle) and publishers."""

    def __init__(self, manual_wrist_deg: float = 0.0):
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {"manual_wrist_deg": manual_wrist_deg}
        self._listeners: list[Callable[[dict], None]] = []

    def update(self, patch: dict) -> None:
        with self._lock:
            self._values.update(patch)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(patch)
            except Exception:
                logger.exception("config listener failed")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def wrist_deg(self) -> float:
        return float(self.get("manual_wrist_deg", 0.0))

    def on_update(self, listener: Callable[[dict], None]) -> None:
        with self._lock:
            self._listeners.append(listener)


class StreamerStats:
    """Counters shared between the retargeter worker and the sink's ~1 Hz
    status reports."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_times: deque[float] = deque(maxlen=128)
        self._retarget_ms: deque[float] = deque(maxlen=128)
        self._calibrating: dict | None = None

    def record_frame(self, retarget_ms: float | None,
                     calibrating: dict | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            self._frame_times.append(now)
            if retarget_ms is not None:
                self._retarget_ms.append(retarget_ms)
            self._calibrating = calibrating

    def status_payload(self, targets_out_hz: float | None) -> dict:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._frame_times if now - t <= 2.0]
            fps = ((len(recent) - 1) / max(now - recent[0], 1e-6)
                   if len(recent) >= 2 else 0.0)
            retarget_ms = (sum(self._retarget_ms) / len(self._retarget_ms)
                           if self._retarget_ms else None)
            calibrating = self._calibrating
        payload: dict = {
            "ingress_fps": round(fps, 1),
            # Tracking = frames are flowing. When the operator's hand leaves
            # the camera / gets gated upstream, frames stop and this flips.
            "tracking": bool(recent and now - recent[-1] < 0.5),
        }
        if targets_out_hz is not None:
            payload["targets_out_hz"] = round(targets_out_hz, 1)
        if retarget_ms is not None:
            payload["retarget_ms"] = round(retarget_ms, 2)
        if calibrating is not None:
            payload["calibrating"] = calibrating
        return payload


class UiLink:
    """Minimal framing seam over the WebSocket (swappable in tests)."""

    def __init__(self, url: str):
        from websockets.sync.client import connect

        self._ws = connect(url, open_timeout=HELLO_TIMEOUT_S)

    def send_json(self, message: dict) -> None:
        self._ws.send(json.dumps(message, separators=(",", ":")))

    def recv_json(self, timeout: float | None = None) -> dict:
        return json.loads(self._ws.recv(timeout=timeout))

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


class OrcaUiSink:
    """Pluggable pipeline sink (``RobotSink`` shape) that forwards actions to
    orca_ui. ``connect()`` performs the hello handshake; ``run_loop()`` owns
    the main thread, draining ``actions_q`` latest-wins."""

    def __init__(
        self,
        url: str,
        token: str,
        source: str,
        hand_model,
        *,
        config: ConfigState | None = None,
        stats: StreamerStats | None = None,
        rate_hz: float = 50.0,
        stop_event: threading.Event | None = None,
        link_factory: Callable[[str], Any] = UiLink,
    ):
        self._url = url
        self._token = token
        self._source = source
        self._hand_model = hand_model
        self.config = config or ConfigState()
        self.stats = stats or StreamerStats()
        self._rate_hz = rate_hz
        self._stop_event = stop_event or threading.Event()
        self._link_factory = link_factory

        self._link: Any = None
        self._reader: threading.Thread | None = None
        self._engaged = False
        self._seq = 0
        self._send_times: deque[float] = deque(maxlen=128)
        self.session_id: str | None = None

    # ----- RobotSink interface ------------------------------------------------------

    def connect(self) -> None:
        logger.info("dialing %s", self._url)
        self._link = self._link_factory(self._url)
        self._link.send_json({"type": "hello", "data": {
            "token": self._token,
            "proto": PROTO_VERSION,
            "source": self._source,
            "hand": {
                "model_name": getattr(self._hand_model, "model_name", None),
                "side": self._hand_model.config.type,
            },
            "pid": os.getpid(),
            "versions": {"orca_teleop": _package_version()},
        }})
        reply = self._link.recv_json(timeout=HELLO_TIMEOUT_S)
        if reply.get("type") != "hello_ok":
            raise RuntimeError(f"expected hello_ok, got {reply.get('type')!r}")
        ack = reply.get("data") or {}
        self.session_id = ack.get("session_id")
        self._warn_on_model_mismatch(ack)
        # Config accumulated before we connected (e.g. the browser enabled
        # the camera preview while this process was starting up).
        if ack.get("config"):
            self.config.update(ack["config"])
        self._reader = threading.Thread(
            target=self._reader_loop, name="ui-sink-reader", daemon=True)
        self._reader.start()
        logger.info("connected to orca_ui (session %s)", self.session_id)

    def run_loop(self, actions_q: queue.Queue, stop_event: threading.Event) -> None:
        """Drain actions latest-wins and forward at most at ``rate_hz``."""
        min_period = 1.0 / self._rate_hz if self._rate_hz > 0 else 0.0
        last_send = 0.0
        next_status = 0.0
        while not (stop_event.is_set() or self._stop_event.is_set()):
            try:
                action = actions_q.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                action = None
            if action is not None:
                if _is_shutdown(action):
                    break
                # latest-wins: drop everything but the newest queued action.
                # A shutdown sentinel found while draining still lets the
                # final coalesced frame ship first.
                shutdown = False
                while True:
                    try:
                        newer = actions_q.get_nowait()
                    except queue.Empty:
                        break
                    if _is_shutdown(newer):
                        shutdown = True
                        break
                    action = newer
                now = time.monotonic()
                if now - last_send < min_period:
                    time.sleep(min_period - (now - last_send))
                self._send_targets(action)
                last_send = time.monotonic()
                if shutdown:
                    break
            now = time.monotonic()
            if now >= next_status:
                self._send_status()
                next_status = now + STATUS_INTERVAL_S
        stop_event.set()

    def close(self) -> None:
        self._stop_event.set()
        if self._link is not None:
            try:
                self._link.close()
            except Exception:
                pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)

    # ----- outbound ------------------------------------------------------------------

    def _send_targets(self, action) -> None:
        angles = action.as_dict() if hasattr(action, "as_dict") else dict(action)
        self._seq += 1
        self._send_times.append(time.monotonic())
        self._send_json({"type": "targets", "data": {
            "angles": {j: float(v) for j, v in angles.items()},
            "seq": self._seq,
            "t": int(time.monotonic() * 1000),
        }})

    def _send_status(self) -> None:
        now = time.monotonic()
        recent = [t for t in self._send_times if now - t <= 2.0]
        out_hz = ((len(recent) - 1) / max(now - recent[0], 1e-6)
                  if len(recent) >= 2 else None)
        self._send_json({"type": "status",
                         "data": self.stats.status_payload(out_hz)})

    def send_log(self, line: str, level: str = "info") -> None:
        self._send_json({"type": "log", "data": {"level": level, "line": line}})

    def send_preview(self, jpeg_b64: str, seq: int) -> None:
        self._send_json({"type": "preview", "data": {"jpeg": jpeg_b64,
                                                     "seq": seq}})

    def _send_json(self, message: dict) -> None:
        link = self._link
        if link is None:
            return
        try:
            link.send_json(message)
        except Exception:
            logger.info("orca_ui link send failed — stopping")
            self._stop_event.set()

    # ----- inbound -------------------------------------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._link.recv_json(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                if not self._stop_event.is_set():
                    logger.info("orca_ui link closed — stopping")
                self._stop_event.set()
                return
            kind = message.get("type")
            data = message.get("data") or {}
            if kind == "config":
                logger.info("config update: %s", data)
                self.config.update(data)
            elif kind == "engaged":
                self._engaged = bool(data.get("engaged"))
                logger.info("engaged=%s", self._engaged)
                # Smoothing/filter state must not blend across an engage flip.
                self.config.update({"_engaged": self._engaged,
                                    "recalibrate_smoothing": True})
            elif kind == "stop":
                logger.info("stop requested by orca_ui")
                self._stop_event.set()
                return

    # ----- helpers -------------------------------------------------------------------

    def _warn_on_model_mismatch(self, ack: dict) -> None:
        """The hello_ok echoes orca_ui's joint set/ROMs — a cheap split-brain
        detector between the two processes' hand models."""
        ui_joints = set(ack.get("joints") or [])
        ours = set(self._hand_model.config.joint_ids)
        if ui_joints and ui_joints != ours:
            logger.warning(
                "joint-set mismatch with orca_ui: only-here=%s only-there=%s",
                sorted(ours - ui_joints), sorted(ui_joints - ours))
            self.send_log(
                f"joint-set mismatch: streamer={sorted(ours - ui_joints)} "
                f"ui={sorted(ui_joints - ours)}", level="warning")

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event


def _is_shutdown(action) -> bool:
    from orca_teleop.pipeline import _SHUTDOWN

    return action is _SHUTDOWN


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("orca_teleop")
    except Exception:
        return "unknown"
