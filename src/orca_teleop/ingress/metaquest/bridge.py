"""Host the repository-owned Quest WebXR page and ingest its WebSocket stream."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import ssl
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from orca_teleop.ingress.metaquest.landmarks import xr_matrix_to_flu_matrix

logger = logging.getLogger(__name__)
HAND_SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class WebXRHandSample:
    sequence_id: int
    timestamp_ns: int
    landmarks: np.ndarray
    wrist_matrix: np.ndarray | None
    # The headset pose is per-packet, not per-hand, but it is copied onto every
    # hand sample so that a hand and the head it is paired with are always from
    # the same instant.
    head_matrix: np.ndarray | None = None
    pose_epoch: int = 0


class QuestTelemetryState:
    """Thread-safe latest-value store for WebXR hand frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, WebXRHandSample | None] = {side: None for side in HAND_SIDES}
        self._sequence_ids = {side: 0 for side in HAND_SIDES}
        self._last_update_monotonic = 0.0

    def update(self, payload: Mapping[str, Any], fallback_epoch: int = 0) -> None:
        """Ingest one telemetry packet.

        ``fallback_epoch`` stands in for the client's own session epoch when
        the page predates it (a cached app.js). It identifies the WebSocket
        connection, which is a coarser but fail-safe proxy: it changes more
        often than the XR reference space really does, which costs a spurious
        re-clutch, whereas missing a change lets the arm lurch.
        """
        hands = payload.get("hands", {})
        client_wall_ms = payload.get("client_wall_ms")
        timestamp_ns = (
            int(float(client_wall_ms) * 1_000_000)
            if isinstance(client_wall_ms, (int | float))
            else time.time_ns()
        )

        session_epoch = payload.get("session_epoch")
        pose_epoch = (
            int(session_epoch) & 0xFFFFFFFF
            if isinstance(session_epoch, (int | float))
            else int(fallback_epoch)
        )

        # Decoded once per packet, outside the per-hand loop: reading it twice
        # could pair one hand with head N and the other with head N+1.
        head_raw = payload.get("head")
        try:
            head_matrix = xr_matrix_to_flu_matrix(head_raw) if head_raw is not None else None
        except (TypeError, ValueError):
            head_matrix = None

        with self._lock:
            for side in HAND_SIDES:
                hand_payload = hands.get(side)
                if not isinstance(hand_payload, Mapping):
                    self._samples[side] = None
                    continue

                landmarks = np.asarray(hand_payload.get("landmarks"), dtype=np.float64)
                if landmarks.shape != (25, 3):
                    self._samples[side] = None
                    continue

                wrist_raw = hand_payload.get("wrist")
                try:
                    wrist_matrix = (
                        xr_matrix_to_flu_matrix(wrist_raw) if wrist_raw is not None else None
                    )
                except (TypeError, ValueError):
                    wrist_matrix = None

                self._sequence_ids[side] += 1
                self._samples[side] = WebXRHandSample(
                    sequence_id=self._sequence_ids[side],
                    timestamp_ns=timestamp_ns,
                    landmarks=landmarks.copy(),
                    wrist_matrix=wrist_matrix,
                    head_matrix=(None if head_matrix is None else head_matrix.copy()),
                    pose_epoch=pose_epoch,
                )
            self._last_update_monotonic = time.monotonic()

    def get_hand_sample(self, side: str) -> WebXRHandSample | None:
        if side not in HAND_SIDES:
            raise ValueError(f"Unsupported hand side: {side!r}")
        with self._lock:
            sample = self._samples[side]
            if sample is None:
                return None
            return WebXRHandSample(
                sequence_id=sample.sequence_id,
                timestamp_ns=sample.timestamp_ns,
                landmarks=sample.landmarks.copy(),
                wrist_matrix=(None if sample.wrist_matrix is None else sample.wrist_matrix.copy()),
                head_matrix=(None if sample.head_matrix is None else sample.head_matrix.copy()),
                pose_epoch=sample.pose_epoch,
            )

    @property
    def last_update_monotonic(self) -> float:
        with self._lock:
            return self._last_update_monotonic


class QuestTelemetryBridge:
    """Serve the WebXR client and receive telemetry over a same-origin WebSocket."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.ssl_context = ssl_context
        self.state = QuestTelemetryState()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: Any | None = None
        self._websockets: set[Any] = set()
        self._connected = threading.Event()
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._telemetry_packet_count = 0
        self._connection_epochs = itertools.count(1)

    @property
    def url(self) -> str:
        scheme = "https" if self.ssl_context is not None else "http"
        display_host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"{scheme}://{display_host}:{self.port}"

    def wait_until_connected(self, timeout: float | None = None) -> bool:
        return self._connected.wait(timeout)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="quest-webxr-bridge",
        )
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError("Quest WebXR bridge did not start within 10 seconds")
        if self._startup_error is not None:
            raise RuntimeError("Quest WebXR bridge failed to start") from self._startup_error

    def stop(self) -> None:
        if self._loop is None or self._thread is None:
            return
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop)
            future.result(timeout=10.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)
        self._thread = None
        self._loop = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start())
        except BaseException as exc:
            self._startup_error = exc
            logger.exception("Quest WebXR bridge startup failed")
            self._started.set()
            loop.close()
            return
        self._started.set()
        loop.run_forever()
        loop.close()

    async def _async_start(self) -> None:
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError(
                "Meta Quest WebXR input requires aiohttp; run `uv sync --extra learning` "
                "(or `--extra metaquest`)."
            ) from exc

        web_dir = Path(__file__).resolve().parent / "web"
        self._index_path = web_dir / "index.html"
        self._app_js_path = web_dir / "app.js"
        self._style_css_path = web_dir / "style.css"

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/app.js", self._handle_app_js)
        app.router.add_get("/style.css", self._handle_style_css)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_post("/debug/client-log", self._handle_debug_client_log)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner,
            self.host,
            self.port,
            ssl_context=self.ssl_context,
        )
        await site.start()

    async def _async_shutdown(self) -> None:
        for websocket in list(self._websockets):
            await websocket.close()
        self._websockets.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_index(self, _request: Any) -> Any:
        from aiohttp import web

        return self._no_store_file_response(web, self._index_path)

    async def _handle_app_js(self, _request: Any) -> Any:
        from aiohttp import web

        return self._no_store_file_response(web, self._app_js_path)

    async def _handle_style_css(self, _request: Any) -> Any:
        from aiohttp import web

        return self._no_store_file_response(web, self._style_css_path)

    @staticmethod
    def _no_store_file_response(web: Any, path: Path) -> Any:
        response = web.FileResponse(path)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _handle_debug_client_log(self, request: Any) -> Any:
        from aiohttp import web

        payload = await request.json()
        level = str(payload.get("level", "info")).upper()
        event = str(payload.get("event", "unknown"))
        message = str(payload.get("message", ""))
        logger.info("QuestClient/%s %s: %s", level, event, message)
        return web.json_response({"ok": True})

    async def _handle_ws(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        websocket = web.WebSocketResponse(heartbeat=10.0, max_msg_size=4 * 1024 * 1024)
        await websocket.prepare(request)
        self._websockets.add(websocket)
        peer = request.remote or "unknown"
        # Fallback pose epoch for a cached page that does not send its own.
        connection_epoch = next(self._connection_epochs)
        logger.info("Quest WebXR WebSocket opened from %s", peer)
        try:
            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "telemetry":
                        self.state.update(payload, fallback_epoch=connection_epoch)
                        self._connected.set()
                        self._telemetry_packet_count += 1
                        if self._telemetry_packet_count <= 3:
                            logger.info(
                                "Quest WebXR telemetry packet %d: hands=%s",
                                self._telemetry_packet_count,
                                list(payload.get("hands", {})),
                            )
                elif message.type == WSMsgType.ERROR:
                    logger.warning("Quest WebXR WebSocket error: %r", websocket.exception())
                    break
        finally:
            self._websockets.discard(websocket)
            if not self._websockets:
                self._connected.clear()
            logger.info("Quest WebXR WebSocket closed from %s", peer)
        return websocket
