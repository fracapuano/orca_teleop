"""Persistent WebSocket client for a LimX TRON 2 dual-arm robot.

Protocol (LimX SDK guide V0.2, transcribed in CLAUDE.md -- nothing here is
invented beyond it):

  * ``ws://<robot>:5000``, JSON envelope
    ``{accid, title, timestamp(ms), guid(uuid4), data}``.
  * ``accid`` is learned from ``notify_robot_info`` rather than configured.
  * Request/response calls correlate on ``guid``.
  * ``request_servop`` gets **no response**; failures arrive asynchronously as
    ``notify_servop``. It is therefore fire-and-forget here.
  * ``notify_invalid_request`` echoes malformed messages back. Every ``notify_*``
    is appended to a JSONL log with both clocks.

Titles marked ASSUMED below are not in CLAUDE.md's transcribed ground truth;
they are kept as module constants so a single edit corrects them.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection, connect as ws_connect

from .config import ARMS, Config, N_JOINTS, SERVOP_WIDTH, ServopFormat
from .poses import Pose, PoseLike, as_pose, quat_to_matrix

__all__ = [
    "NotifyRecord",
    "Tron2Client",
    "Tron2Error",
    "Tron2NotConnected",
    "Tron2ProtocolError",
    "Tron2Timeout",
    "build_envelope",
    "decode_joint_state",
    "split_joint_reply",
    "decode_move_pose",
    "encode_servop_element",
    "encode_servop_payload",
]

# -- wire titles ---------------------------------------------------------
TITLE_GET_MOVE_POSE = "request_get_move_pose"
TITLE_MOVEJ = "request_movej"
TITLE_SERVOP = "request_servop"
TITLE_EMGY_STOP = "request_emgy_stop"
TITLE_GET_JOINT_STATE = "request_get_joint_state"  # ASSUMED
TITLE_LIGHT_EFFECT = "request_light_effect"  # ASSUMED

NOTIFY_ROBOT_INFO = "notify_robot_info"
NOTIFY_SERVOP = "notify_servop"
NOTIFY_INVALID_REQUEST = "notify_invalid_request"

NotifyCallback = Callable[["NotifyRecord"], Any]


class Tron2Error(RuntimeError):
    """Base class for client errors."""


class Tron2Timeout(Tron2Error):
    """A guid-correlated request did not get a reply in time."""


class Tron2NotConnected(Tron2Error):
    """Operation attempted before :meth:`Tron2Client.connect` or after close."""


class Tron2ProtocolError(Tron2Error):
    """The robot sent something we cannot interpret."""


@dataclass(frozen=True)
class NotifyRecord:
    """One received ``notify_*`` message.

    ``t_mono_ns`` is the only clock fit for timing arithmetic;
    ``t_wall_ns`` and the robot's own ``timestamp`` are for logging only
    (CLAUDE.md hard rule 2).
    """

    title: str
    data: Mapping[str, Any]
    t_mono_ns: int
    t_wall_ns: int
    guid: str | None = None
    accid: str | None = None
    robot_timestamp_ms: int | None = None

    def as_json(self) -> str:
        return json.dumps(
            {
                "t_mono_ns": self.t_mono_ns,
                "t_wall_ns": self.t_wall_ns,
                "title": self.title,
                "guid": self.guid,
                "accid": self.accid,
                "robot_timestamp_ms": self.robot_timestamp_ms,
                "data": _jsonable(self.data),
            },
            separators=(",", ":"),
            sort_keys=False,
        )


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy scalars/arrays for the JSONL log."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# -- pure protocol helpers ----------------------------------------------
def build_envelope(
    accid: str,
    title: str,
    data: Mapping[str, Any] | None = None,
    guid: str | None = None,
    *,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """Assemble the vendor envelope.

    ``timestamp`` is wall-clock milliseconds because the wire format demands it.
    It is metadata only and never enters control maths.
    """
    return {
        "accid": accid,
        "title": title,
        "timestamp": int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms),
        "guid": str(uuid.uuid4()) if guid is None else guid,
        "data": dict(data) if data else {},
    }


def encode_servop_element(pose: PoseLike, fmt: ServopFormat) -> list[float]:
    """Encode one arm's target for ``request_servop``.

    ``pos_quat``   -> ``[x, y, z, qw, qx, qy, qz]`` (7)
    ``pos_rotmat`` -> ``[x, y, z, r11 .. r33]`` (12, row-major)

    Rejects non-finite values outright: CLAUDE.md hard rule 3 makes this the
    last line of defence before the wire.
    """
    if fmt not in SERVOP_WIDTH:
        raise ValueError(f"unknown servop format {fmt!r}, expected one of {sorted(SERVOP_WIDTH)}")
    pose = as_pose(pose)
    p = np.asarray(pose.p, dtype=np.float64)
    if fmt == "pos_quat":
        out = np.concatenate((p, np.asarray(pose.q_wxyz, dtype=np.float64)))
    else:
        out = np.concatenate((p, quat_to_matrix(pose.q_wxyz).reshape(-1)))
    if not np.all(np.isfinite(out)):
        raise ValueError(f"refusing to encode a non-finite servop element: {out!r}")
    if out.shape != (SERVOP_WIDTH[fmt],):
        raise AssertionError(f"encoder produced {out.shape[0]} elements for {fmt!r}")
    return [float(v) for v in out]


def encode_servop_payload(
    fmt: ServopFormat,
    left: PoseLike | None,
    right: PoseLike | None,
) -> dict[str, list[float]]:
    """Build the ``data`` body of ``request_servop`` from whichever sides are set.

    ``send_both`` resolution happens in :meth:`Tron2Client.send_servop`; by the
    time a side reaches here it is either a real target or ``None`` meaning
    "omit this key".
    """
    payload: dict[str, list[float]] = {}
    if left is not None:
        payload["left_pos"] = encode_servop_element(left, fmt)
    if right is not None:
        payload["right_pos"] = encode_servop_element(right, fmt)
    if not payload:
        raise ValueError("request_servop needs at least one of left/right")
    return payload


def decode_move_pose(data: Mapping[str, Any]) -> dict[str, Pose]:
    """Parse ``request_get_move_pose`` -> ``{"left": Pose, "right": Pose}``.

    Positions are metres and quaternions are wxyz, so they compose directly with
    :class:`~tron_arm.poses.Pose`.
    """
    out: dict[str, Pose] = {}
    for arm in ARMS:
        pos_key, quat_key = f"{arm}_position", f"{arm}_quat"
        if pos_key not in data or quat_key not in data:
            raise Tron2ProtocolError(
                f"get_move_pose reply missing {pos_key!r}/{quat_key!r}; got {sorted(data)}"
            )
        try:
            out[arm] = Pose(np.asarray(data[pos_key], dtype=np.float64),
                            np.asarray(data[quat_key], dtype=np.float64))
        except (ValueError, TypeError) as exc:
            raise Tron2ProtocolError(f"malformed {arm} pose in get_move_pose reply: {exc}") from exc
    return out


def split_joint_reply(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Split a joint-state reply into ``(arm_joints_14, extras)``.

    The guide documents 14 joints for ``request_movej`` -- left 7 then right 7 --
    but a real TRON 2 REPORTS more. Measured on our unit (fw 2.1.24):
    ``q``/``dq``/``tau`` all carry 16 entries, the last two being the grippers.

    The layout is verified, not assumed. On that unit ``q[0:14]`` sits inside
    every documented joint limit while both plausible alternatives (extras first,
    or one at each end) violate them, and the torques are symmetric between
    ``q[0:7]`` and ``q[7:14]`` joint for joint -- j0 at -5.22/-5.11 and j3 at
    -5.34/-5.27, the shoulder and elbow carrying gravity on each arm.

    The slice is re-checked against the limits on every call. Getting it wrong
    would make ``capture-ready`` record a shifted posture and ``movej-ready``
    drive an arm somewhere unintended at the start of every session, so a
    mismatch raises rather than guesses.
    """
    from .config import JOINT_LOWER, JOINT_UPPER

    for key in ("joint", "joints", "position", "q"):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if arr.shape[0] < N_JOINTS:
            raise Tron2ProtocolError(
                f"joint state {key!r} has {arr.shape[0]} entries, need at least {N_JOINTS}"
            )
        if not np.all(np.isfinite(arr)):
            raise Tron2ProtocolError(f"joint state {key!r} contains non-finite values")
        arms, extras = arr[:N_JOINTS], arr[N_JOINTS:]
        if extras.size:
            # Tolerance so a joint resting on a hard stop does not false-alarm.
            slack = 0.05
            bad = np.nonzero((arms < JOINT_LOWER - slack) | (arms > JOINT_UPPER + slack))[0]
            if bad.size:
                details = ", ".join(
                    f"j{i}={arms[i]:+.4f} not in [{JOINT_LOWER[i]:+.4f}, {JOINT_UPPER[i]:+.4f}]"
                    for i in bad)
                raise Tron2ProtocolError(
                    f"joint state {key!r} has {arr.shape[0]} entries and the first "
                    f"{N_JOINTS} do NOT fit the documented arm limits -- the index "
                    f"layout is not what we assume. Refusing to guess.\n  {details}\n"
                    f"  Run `tron2_cli joints --raw` and check the ordering."
                )
        return arms, extras
    raise Tron2ProtocolError(f"joint state reply has no joint vector; got {sorted(data)}")


def decode_joint_state(data: Mapping[str, Any]) -> np.ndarray:
    """The 14 arm joints in radians (left 0..6 then right 7..13).

    Extra entries a real unit reports (grippers) are dropped here; use
    :func:`split_joint_reply` if you need them.
    """
    return split_joint_reply(data)[0]


@dataclass
class ClientStats:
    """Counters for observability; all monotonically increasing."""

    servop_sent: int = 0
    servop_clamped: int = 0
    requests_sent: int = 0
    responses_received: int = 0
    timeouts: int = 0
    notifies: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))
    undecodable_messages: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "servop_sent": self.servop_sent,
            "servop_clamped": self.servop_clamped,
            "requests_sent": self.requests_sent,
            "responses_received": self.responses_received,
            "timeouts": self.timeouts,
            "undecodable_messages": self.undecodable_messages,
            "notifies": dict(self.notifies),
        }


class Tron2Client:
    """One persistent connection to a TRON 2 (or to :mod:`tron_arm.mock_robot`).

    Usage::

        async with Tron2Client(config) as client:
            poses = await client.get_move_pose()
            await client.send_servop(left=poses["left"], right=poses["right"])

    Threading: single-asyncio-task ownership. ``send_servop`` is the only method
    on the hot path and never awaits a reply.
    """

    def __init__(
        self,
        config: Config,
        *,
        url: str | None = None,
        accid: str | None = None,
        notify_log_path: str | Path | None = ...,  # type: ignore[assignment]
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._config = config
        self._url = url if url is not None else config.robot.url
        self._accid = accid or ""
        self._accid_preset = bool(accid)
        self._clock = clock
        self._ws: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[Mapping[str, Any]]] = {}
        self._notify_cbs: dict[str, list[NotifyCallback]] = defaultdict(list)
        self._accid_learned = asyncio.Event()
        self._frozen: dict[str, Pose | None] = {"left": None, "right": None}
        self.stats = ClientStats()
        self.notify_history: list[NotifyRecord] = []
        if notify_log_path is ...:
            notify_log_path = config.notify_log_path
        self._notify_log_path = Path(notify_log_path) if notify_log_path else None
        self._notify_log: Any = None

    # -- properties ------------------------------------------------------
    @property
    def url(self) -> str:
        return self._url

    @property
    def accid(self) -> str:
        """Learned (or preset) account id; empty until ``notify_robot_info``."""
        return self._accid

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def frozen_poses(self) -> dict[str, Pose | None]:
        """Last pose commanded per arm; used to fill the idle side."""
        return dict(self._frozen)

    # -- lifecycle -------------------------------------------------------
    async def __aenter__(self) -> "Tron2Client":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def connect(self, *, wait_for_accid: bool | None = None) -> None:
        """Open the socket, start the reader, and learn ``accid``.

        ``wait_for_accid`` defaults to True unless an accid was passed to
        ``__init__``. The robot broadcasts ``notify_robot_info`` at 1 Hz, so the
        wait is normally under a second.
        """
        if self._ws is not None:
            raise Tron2Error("already connected")
        if wait_for_accid is None:
            wait_for_accid = not self._accid_preset
        self._accid_learned = asyncio.Event()
        if self._accid_preset:
            self._accid_learned.set()
        try:
            self._ws = await ws_connect(
                self._url,
                open_timeout=self._config.robot.connect_timeout_s,
                ping_interval=20,
                ping_timeout=20,
                max_queue=64,
            )
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise Tron2Error(f"cannot connect to {self._url}: {exc}") from exc
        _set_tcp_nodelay(self._ws)
        self._open_notify_log()
        self._reader = asyncio.create_task(self._read_loop(), name="tron2-reader")
        if wait_for_accid:
            try:
                await asyncio.wait_for(
                    self._accid_learned.wait(), timeout=self._config.robot.connect_timeout_s
                )
            except asyncio.TimeoutError as exc:
                await self.close()
                raise Tron2Timeout(
                    f"no {NOTIFY_ROBOT_INFO} within {self._config.robot.connect_timeout_s}s; "
                    "cannot learn accid"
                ) from exc

    async def close(self) -> None:
        """Close the socket and fail any in-flight requests."""
        ws, self._ws = self._ws, None
        reader, self._reader = self._reader, None
        if ws is not None:
            await ws.close()
        if reader is not None:
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown path
                pass
        self._fail_pending(Tron2NotConnected("connection closed"))
        if self._notify_log is not None:
            self._notify_log.close()
            self._notify_log = None

    # -- notify subscriptions -------------------------------------------
    def on_notify(self, title: str, callback: NotifyCallback) -> Callable[[], None]:
        """Subscribe to a ``notify_*`` title (``"*"`` for all).

        Returns an unsubscribe callable.
        """
        self._notify_cbs[title].append(callback)

        def _unsubscribe() -> None:
            try:
                self._notify_cbs[title].remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    # -- guid-correlated requests ---------------------------------------
    async def _request(
        self,
        title: str,
        data: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        ws = self._require_ws()
        timeout = self._config.robot.request_timeout_s if timeout is None else timeout
        guid = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Mapping[str, Any]] = loop.create_future()
        self._pending[guid] = future
        envelope = build_envelope(self._accid, title, data, guid)
        try:
            await ws.send(json.dumps(envelope, separators=(",", ":")))
            self.stats.requests_sent += 1
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self.stats.timeouts += 1
            raise Tron2Timeout(f"{title} (guid {guid}) timed out after {timeout}s") from exc
        except websockets.exceptions.ConnectionClosed as exc:
            raise Tron2NotConnected(f"{title}: connection closed") from exc
        finally:
            self._pending.pop(guid, None)

    async def get_move_pose(self, *, timeout: float | None = None) -> dict[str, Pose]:
        """Current flange poses, robot base frame (FLU metres, wxyz)."""
        return decode_move_pose(await self._request(TITLE_GET_MOVE_POSE, timeout=timeout))

    async def get_joint_state(self, *, timeout: float | None = None) -> np.ndarray:
        """Current 14 joint angles in radians (left 0..6 then right 7..13)."""
        return decode_joint_state(await self._request(TITLE_GET_JOINT_STATE, timeout=timeout))

    async def movej(
        self,
        joints: Sequence[float] | np.ndarray,
        time_s: float,
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        """Joint-space move. ``joints`` is 14 radians, ``time_s`` the duration.

        Range-checked against the documented limits before sending; the robot
        rejects out-of-range targets, and we would rather fail locally.
        """
        arr = np.asarray(joints, dtype=np.float64).reshape(-1)
        if arr.shape != (N_JOINTS,):
            raise ValueError(f"movej expects {N_JOINTS} joint values, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("movej refuses non-finite joint values")
        if not np.isfinite(time_s) or time_s <= 0.0:
            raise ValueError(f"movej time must be > 0 s, got {time_s!r}")
        from .config import JOINT_LOWER, JOINT_UPPER

        bad = np.nonzero((arr < JOINT_LOWER) | (arr > JOINT_UPPER))[0]
        if bad.size:
            details = ", ".join(
                f"j{i}={arr[i]:.4f} not in [{JOINT_LOWER[i]:.4f}, {JOINT_UPPER[i]:.4f}]"
                for i in bad
            )
            raise ValueError(f"movej target outside joint limits -- {details}")
        payload = {"time": float(time_s), "joint": [float(v) for v in arr]}
        # The move itself takes time_s; allow for it plus the usual round trip.
        wait = self._config.robot.request_timeout_s + float(time_s) if timeout is None else timeout
        return await self._request(TITLE_MOVEJ, payload, timeout=wait)

    async def light_effect(
        self, effect: str = "flash", *, timeout: float | None = None, **extra: Any
    ) -> Mapping[str, Any]:
        """Set the status light. Payload keys beyond ``effect`` are passed through."""
        return await self._request(
            TITLE_LIGHT_EFFECT, {"effect": effect, **extra}, timeout=timeout
        )

    async def emgy_stop(self, *, timeout: float | None = None) -> Mapping[str, Any]:
        """Emergency stop.

        .. warning::
           Per the vendor guide this only works while the robot is **idle**. It
           is not a motion abort. Aborting motion means freezing the target and
           a human on the hardware remote (CLAUDE.md).
        """
        return await self._request(TITLE_EMGY_STOP, timeout=timeout)

    # -- servop (fire and forget) ---------------------------------------
    def set_frozen_pose(self, arm: str, pose: PoseLike | None) -> None:
        """Seed the pose used for an idle arm when ``send_both`` is true."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self._frozen[arm] = None if pose is None else as_pose(pose)

    async def prime_frozen_poses(self, *, timeout: float | None = None) -> dict[str, Pose]:
        """Read the robot's current flange poses and freeze both arms there.

        Call once before streaming so ``send_both`` has something to hold the
        idle arm at.
        """
        poses = await self.get_move_pose(timeout=timeout)
        for arm, pose in poses.items():
            self._frozen[arm] = pose
        return poses

    async def send_servop(
        self, left: PoseLike | None = None, right: PoseLike | None = None
    ) -> None:
        """Send one ``request_servop``. Fire-and-forget: there is no response.

        Failures surface asynchronously as ``notify_servop`` -- subscribe with
        :meth:`on_notify` if you need to see them.

        Poses are workspace-clamped and finiteness-checked here, so nothing
        unclamped or NaN can reach the encoder (CLAUDE.md hard rule 3). With
        ``servop.send_both`` true, a ``None`` side is filled from that arm's
        frozen pose.
        """
        ws = self._require_ws()
        resolved: dict[str, Pose | None] = {"left": None, "right": None}
        for arm, target in (("left", left), ("right", right)):
            if target is not None:
                resolved[arm] = self._prepare(arm, target)
            elif self._config.servop.send_both:
                frozen = self._frozen[arm]
                if frozen is None:
                    raise Tron2Error(
                        f"servop.send_both is true but the {arm} arm has no frozen pose; "
                        "call prime_frozen_poses() or set_frozen_pose() first"
                    )
                resolved[arm] = frozen
        payload = encode_servop_payload(self._config.servop.format, resolved["left"], resolved["right"])
        envelope = build_envelope(self._accid, TITLE_SERVOP, payload)
        try:
            await ws.send(json.dumps(envelope, separators=(",", ":")))
        except websockets.exceptions.ConnectionClosed as exc:
            raise Tron2NotConnected("servop: connection closed") from exc
        for arm, pose in resolved.items():
            if pose is not None:
                self._frozen[arm] = pose
        self.stats.servop_sent += 1

    def _prepare(self, arm: str, target: PoseLike) -> Pose:
        """Coerce, NaN-guard and workspace-clamp one side."""
        pose = as_pose(target)  # Pose.__init__ already rejects non-finite input
        clamped_p, was_clamped = self._config.workspace.clamp(arm, pose.p)
        if was_clamped:
            self.stats.servop_clamped += 1
            return Pose(clamped_p, pose.q_wxyz)
        return pose

    # -- internals -------------------------------------------------------
    def _require_ws(self) -> ClientConnection:
        if self._ws is None:
            raise Tron2NotConnected("not connected; call connect() first")
        return self._ws

    def _open_notify_log(self) -> None:
        if self._notify_log_path is None:
            return
        self._notify_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._notify_log = self._notify_log_path.open("a", encoding="utf-8")

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    self.stats.undecodable_messages += 1
                    continue
                if not isinstance(message, Mapping):
                    self.stats.undecodable_messages += 1
                    continue
                self._dispatch(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        finally:
            self._fail_pending(Tron2NotConnected("reader stopped"))

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        title = str(message.get("title", ""))
        guid = message.get("guid")
        accid = message.get("accid")
        data = message.get("data")
        if not isinstance(data, Mapping):
            data = {} if data is None else {"value": data}

        # accid auto-learn: notify_robot_info is authoritative, but accept any
        # message that carries one so a reply can bootstrap us too.
        if not self._accid and isinstance(accid, str) and accid:
            if title == NOTIFY_ROBOT_INFO or not self._accid_learned.is_set():
                self._accid = accid
                self._accid_learned.set()

        if title.startswith("notify_"):
            record = NotifyRecord(
                title=title,
                data=data,
                t_mono_ns=self._clock(),
                t_wall_ns=time.time_ns(),
                guid=str(guid) if guid else None,
                accid=str(accid) if accid else None,
                robot_timestamp_ms=(
                    int(message["timestamp"]) if isinstance(message.get("timestamp"), (int, float))
                    else None
                ),
            )
            self._record_notify(record)

        if isinstance(guid, str) and guid in self._pending:
            future = self._pending[guid]
            if not future.done():
                future.set_result(data)
                self.stats.responses_received += 1

    def _record_notify(self, record: NotifyRecord) -> None:
        self.stats.notifies[record.title] += 1
        self.notify_history.append(record)
        if self._notify_log is not None:
            self._notify_log.write(record.as_json() + "\n")
            self._notify_log.flush()
        for title in (record.title, "*"):
            for callback in tuple(self._notify_cbs.get(title, ())):
                try:
                    result = callback(record)
                    if isinstance(result, Awaitable):
                        asyncio.ensure_future(result)
                except Exception:  # noqa: BLE001 - a bad subscriber must not kill the reader
                    continue

    def _fail_pending(self, exc: BaseException) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()


def _set_tcp_nodelay(ws: ClientConnection) -> bool:
    """Disable Nagle on the underlying socket.

    ServoP sends a small frame every tick; coalescing them would add latency of
    the same order as the control period. Best-effort: a TLS or proxied
    connection may not expose a socket.
    """
    transport = getattr(ws, "transport", None)
    if transport is None:
        return False
    sock = transport.get_extra_info("socket")
    if sock is None:
        return False
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return True
    except OSError:
        return False
