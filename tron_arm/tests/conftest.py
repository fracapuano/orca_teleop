"""Shared fixtures.

Async tests are plain sync functions that call :func:`run` so the suite needs no
pytest-asyncio (CLAUDE.md hard rule 5: no new deps in the sink path).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

import numpy as np
import pytest
import websockets
from websockets.asyncio.server import ServerConnection, serve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tron_arm.config import Config, load_config
from tron_arm.mock_robot import MockTron2
from tron_arm.tron2_client import Tron2Client

T = TypeVar("T")


def run(coro: Awaitable[T]) -> T:
    """Run one coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)  # type: ignore[arg-type]


@pytest.fixture
def config() -> Config:
    """The shipped default config, with notify logging off."""
    return dataclasses.replace(load_config(), notify_log_path=None)


def tweak(config: Config, **servop: Any) -> Config:
    """Return ``config`` with ``servop`` fields replaced."""
    return dataclasses.replace(config, servop=dataclasses.replace(config.servop, **servop))


def at(config: Config, url: str) -> Config:
    """Return ``config`` pointed at ``url``."""
    return dataclasses.replace(config, robot=dataclasses.replace(config.robot, url=url))


@contextlib.asynccontextmanager
async def mock_and_client(
    config: Config, **mock_kwargs: Any
) -> AsyncIterator[tuple[MockTron2, Tron2Client]]:
    """A mock robot on an ephemeral port plus a connected client."""
    mock_kwargs.setdefault("port", 0)
    mock_kwargs.setdefault("info_period_s", 0.05)
    async with MockTron2(**mock_kwargs) as robot:
        cfg = at(config, f"ws://127.0.0.1:{robot.bound_port}")
        async with Tron2Client(cfg, notify_log_path=None) as client:
            yield robot, client


class SilentServer:
    """Accepts connections, announces ``accid`` once, then never replies.

    Used to exercise request timeouts without waiting on a real robot.
    """

    def __init__(self, accid: str = "SILENT-0001") -> None:
        self.accid = accid
        self.received: list[dict[str, Any]] = []
        self._server: Any = None

    @property
    def port(self) -> int:
        return next(iter(self._server.sockets)).getsockname()[1]

    async def __aenter__(self) -> "SilentServer":
        self._server = await serve(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, ws: ServerConnection) -> None:
        await ws.send(json.dumps({
            "accid": self.accid, "title": "notify_robot_info",
            "timestamp": 0, "guid": "boot", "data": {},
        }))
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            async for raw in ws:
                with contextlib.suppress(json.JSONDecodeError):
                    self.received.append(json.loads(raw))


class VirtualClock:
    """A monotonic-ns clock the test advances by hand."""

    def __init__(self, start_ns: int = 0) -> None:
        self.now_ns = int(start_ns)

    def __call__(self) -> int:
        return self.now_ns

    def advance_s(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)

    async def sleep(self, seconds: float) -> None:
        """Stand-in for ``asyncio.sleep`` that advances virtual time instantly."""
        self.advance_s(seconds)
        await asyncio.sleep(0)


def collector() -> tuple[list[tuple[Any, Any]], Callable[[Any, Any], Awaitable[None]]]:
    """A send callback that records ``(left, right)`` pairs."""
    sent: list[tuple[Any, Any]] = []

    async def send(left: Any, right: Any) -> None:
        sent.append((left, right))

    return sent, send


def unchecked_pose(p: Any, q: Any) -> Any:
    """Build a Pose bypassing __post_init__, to exercise downstream NaN guards.

    Pose itself rejects non-finite input, so the only way to reach the guards
    further down the chain (encoder, streamer) is to construct one behind its
    back. Those guards are defence in depth for hard rule 3, and untestable
    otherwise.
    """
    from tron_arm.poses import Pose

    pose = Pose.__new__(Pose)
    object.__setattr__(pose, "position_m", np.asarray(p, dtype=np.float64))
    object.__setattr__(pose, "orientation_wxyz", np.asarray(q, dtype=np.float64))
    return pose


def wait_until(predicate: Callable[[], Any], timeout: float = 5.0,
               interval: float = 0.001) -> bool:
    """Poll ``predicate`` until it is true, or ``timeout`` elapses.

    Tests wait on a *condition*, never on a fixed duration. A sleep long enough
    to stay reliable on a loaded CI box is orders of magnitude longer than the
    event it is waiting for, and it costs that much on every single run.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


@contextlib.contextmanager
def fast_close(seconds: float = 0.02) -> Any:
    """Shorten the sink's pre-close freeze window.

    ``TronArmSink.close`` deliberately keeps streaming for
    ``FREEZE_BEFORE_CLOSE_S`` (1 s) so a real arm settles before the socket
    drops. Every test that connects a sink used to pay that second in teardown.
    Tests that are actually *about* the freeze window set their own value.
    """
    from tron_arm import sink as sink_module

    before = sink_module.FREEZE_BEFORE_CLOSE_S
    sink_module.FREEZE_BEFORE_CLOSE_S = seconds
    try:
        yield seconds
    finally:
        sink_module.FREEZE_BEFORE_CLOSE_S = before
