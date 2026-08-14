"""A local stand-in for the TRON 2 robot, speaking the vendor WebSocket envelope.

This is the default target for everything in this package (CLAUDE.md hard rule
1: real hardware needs an explicit flag plus confirmation). It exists to make
the *ambiguous* parts of the protocol testable:

  * ``--accept-format {pos_quat,pos_rotmat}`` -- which ``request_servop``
    element layout this robot will accept.
  * ``--require-both-keys`` -- whether a single-side ``left_pos``/``right_pos``
    is legal.

Violating either produces ``notify_servop {"result": "fail_invalid_cmd"}``,
which is how a misconfigured client is meant to be diagnosed before it ever
reaches hardware.

.. warning::
   The kinematics here are **fake**. ``_flange_from_joints`` is a smooth,
   deterministic stand-in, not TRON 2 forward kinematics, and there is no IK:
   ``servop`` moves the flange pose without touching the joint vector, and a
   subsequent ``movej`` snaps the flange back onto the fake FK surface. The mock
   is for protocol and pacing tests, never for validating motion.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from .config import ARMS, JOINT_LOWER, JOINT_UPPER, N_JOINTS, SERVOP_WIDTH
from .poses import Pose, matrix_to_quat, slerp
from .tron2_client import (
    NOTIFY_INVALID_REQUEST,
    NOTIFY_ROBOT_INFO,
    NOTIFY_SERVOP,
    TITLE_EMGY_STOP,
    TITLE_GET_JOINT_STATE,
    TITLE_GET_MOVE_POSE,
    TITLE_LIGHT_EFFECT,
    TITLE_MOVEJ,
    TITLE_SERVOP,
)

__all__ = ["MockTron2", "build_parser", "main"]

DEFAULT_ACCID = "MOCK-TRON2-0001"
#: Stand-in for the real controller's `version` diagnostic string.
MOCK_VERSION = "robot-tron2-mock-0.0.1"
DEFAULT_PORT = 5000

#: Starting flange poses, comfortably inside both workspace boxes.
HOME_FLANGE = {
    "left": (np.array([0.40, 0.20, 0.00]), np.array([1.0, 0.0, 0.0, 0.0])),
    "right": (np.array([0.40, -0.20, 0.00]), np.array([1.0, 0.0, 0.0, 0.0])),
}

#: Fake FK gains: joint radians -> metres / radians. Arbitrary but fixed.
_FK_TRANSLATION = np.array(
    [
        [0.06, 0.00, 0.05, 0.03, 0.00, 0.01, 0.00],
        [0.00, 0.07, 0.00, 0.02, 0.04, 0.00, 0.01],
        [0.05, 0.00, -0.06, 0.04, 0.00, 0.02, 0.00],
    ],
    dtype=np.float64,
)
_FK_ROTATION = np.array(
    [
        [0.10, 0.00, 0.20, 0.00, 0.30, 0.00, 0.10],
        [0.00, 0.25, 0.00, 0.30, 0.00, 0.20, 0.00],
        [0.15, 0.00, 0.10, 0.00, 0.20, 0.00, 0.30],
    ],
    dtype=np.float64,
)


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> Hamilton wxyz quaternion."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / theta
    s = np.sin(theta / 2.0)
    return np.array([np.cos(theta / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ServopCheck:
    """Outcome of validating one ``request_servop`` payload."""

    ok: bool
    reason: str = ""
    targets: dict[str, Pose] = field(default_factory=dict)


class MockTron2:
    """Asyncio WebSocket server implementing enough of TRON 2 to test against."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        accid: str = DEFAULT_ACCID,
        accept_format: str = "pos_quat",
        require_both_keys: bool = False,
        servop_gain: float = 0.5,
        info_period_s: float = 1.0,
        movej_rate_hz: float = 50.0,
    ) -> None:
        if accept_format not in SERVOP_WIDTH:
            raise ValueError(
                f"accept_format must be one of {sorted(SERVOP_WIDTH)}, got {accept_format!r}"
            )
        if not 0.0 < servop_gain <= 1.0:
            raise ValueError(f"servop_gain must be in (0, 1], got {servop_gain!r}")
        self.host = host
        self.port = port
        self.accid = accid
        self.accept_format = accept_format
        self.require_both_keys = require_both_keys
        self.servop_gain = servop_gain
        self.info_period_s = info_period_s
        self.movej_rate_hz = movej_rate_hz

        self.joints = np.zeros(N_JOINTS, dtype=np.float64)
        self.flange: dict[str, Pose] = {
            arm: Pose(p, q) for arm, (p, q) in HOME_FLANGE.items()
        }
        self.light: dict[str, Any] = {"effect": "off"}
        self.servop_accepted = 0
        self.servop_rejected = 0
        self.invalid_requests = 0

        self._clients: set[ServerConnection] = set()
        self._server: Server | None = None
        self._info_task: asyncio.Task[None] | None = None
        self._movej_task: asyncio.Task[None] | None = None

    # -- lifecycle -------------------------------------------------------
    @property
    def busy(self) -> bool:
        """True while a ``movej`` is executing."""
        return self._movej_task is not None and not self._movej_task.done()

    @property
    def bound_port(self) -> int:
        """Actual port (resolves ``port=0`` after :meth:`start`)."""
        if self._server is None:
            return self.port
        return next(iter(self._server.sockets)).getsockname()[1]

    async def start(self) -> "MockTron2":
        self._server = await serve(self._handle, self.host, self.port)
        self._info_task = asyncio.create_task(self._info_loop(), name="mock-robot-info")
        return self

    async def stop(self) -> None:
        for task in (self._info_task, self._movej_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._info_task = None
        self._movej_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> "MockTron2":
        return await self.start()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()


    # -- wire helpers ----------------------------------------------------
    def _envelope(self, title: str, data: Mapping[str, Any], guid: str | None = None) -> str:
        return json.dumps(
            {
                "accid": self.accid,
                "title": title,
                "timestamp": _now_ms(),
                "guid": guid or str(uuid.uuid4()),
                "data": data,
            },
            separators=(",", ":"),
        )

    async def _send(self, ws: ServerConnection, title: str, data: Mapping[str, Any],
                    guid: str | None = None) -> None:
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await ws.send(self._envelope(title, data, guid))

    async def _broadcast(self, title: str, data: Mapping[str, Any]) -> None:
        for ws in tuple(self._clients):
            await self._send(ws, title, data)

    async def _info_loop(self) -> None:
        """Broadcast ``notify_robot_info`` at 1 Hz -- how clients learn ``accid``."""
        while True:
            await self._broadcast(
                NOTIFY_ROBOT_INFO,
                {
                    "model": "TRON2-MOCK",
                    # The runbook takes the firmware string from the `version`
                    # diagnostic; which key carries it in notify_robot_info is
                    # NOT confirmed by the vendor guide. We publish it here so
                    # the capture path is exercised, and SessionLogger tries
                    # several spellings rather than assuming this one.
                    "version": MOCK_VERSION,
                    "state": "busy" if self.busy else "idle",
                    "servop_format": self.accept_format,
                    "require_both_keys": self.require_both_keys,
                },
            )
            await asyncio.sleep(self.info_period_s)

    # -- connection handling ---------------------------------------------
    async def _handle(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        try:
            # Send info immediately so a fresh client need not wait a full second.
            await self._send(ws, NOTIFY_ROBOT_INFO, {"model": "TRON2-MOCK", "state": "idle"})
            async for raw in ws:
                await self._on_message(ws, raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    async def _on_message(self, ws: ServerConnection, raw: str | bytes) -> None:
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        try:
            message = json.loads(text)
        except json.JSONDecodeError as exc:
            await self._invalid(ws, f"not valid JSON: {exc}", text)
            return
        if not isinstance(message, dict):
            await self._invalid(ws, "envelope must be a JSON object", text)
            return

        title = message.get("title")
        if not isinstance(title, str) or not title:
            await self._invalid(ws, "envelope is missing a string 'title'", message)
            return
        sender_accid = message.get("accid", "")
        if sender_accid not in ("", None, self.accid):
            await self._invalid(ws, f"unknown accid {sender_accid!r}", message)
            return
        data = message.get("data", {})
        if data is None:
            data = {}
        if not isinstance(data, dict):
            await self._invalid(ws, "'data' must be an object", message)
            return
        guid = message.get("guid")

        if title == TITLE_SERVOP:
            await self._on_servop(ws, data)  # no reply by design
            return
        if not isinstance(guid, str) or not guid:
            await self._invalid(ws, f"{title} requires a string 'guid' to correlate", message)
            return

        handlers = {
            TITLE_GET_MOVE_POSE: self._on_get_move_pose,
            TITLE_GET_JOINT_STATE: self._on_get_joint_state,
            TITLE_MOVEJ: self._on_movej,
            TITLE_LIGHT_EFFECT: self._on_light_effect,
            TITLE_EMGY_STOP: self._on_emgy_stop,
        }
        handler = handlers.get(title)
        if handler is None:
            await self._invalid(ws, f"unknown title {title!r}", message)
            return
        await self._send(ws, title, handler(data), guid)

    async def _invalid(self, ws: ServerConnection, reason: str, echo: Any) -> None:
        """Echo a malformed message back as ``notify_invalid_request``."""
        self.invalid_requests += 1
        await self._send(ws, NOTIFY_INVALID_REQUEST, {"reason": reason, "echo": echo})

    # -- request handlers ------------------------------------------------
    def _on_get_move_pose(self, _data: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"result": "success"}
        for arm in ARMS:
            pose = self.flange[arm]
            out[f"{arm}_position"] = [float(v) for v in pose.p]
            out[f"{arm}_quat"] = [float(v) for v in pose.q_wxyz]
        return out

    def _on_get_joint_state(self, _data: Mapping[str, Any]) -> dict[str, Any]:
        return {"result": "success", "joint": [float(v) for v in self.joints]}

    def _on_movej(self, data: Mapping[str, Any]) -> dict[str, Any]:
        joint = data.get("joint")
        duration = data.get("time")
        try:
            target = np.asarray(joint, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return {"result": "fail_invalid_cmd", "reason": "'joint' must be a numeric array"}
        if target.shape != (N_JOINTS,):
            return {"result": "fail_invalid_cmd",
                    "reason": f"'joint' needs {N_JOINTS} values, got {target.shape[0]}"}
        if not np.all(np.isfinite(target)):
            return {"result": "fail_invalid_cmd", "reason": "'joint' contains non-finite values"}
        if np.any(target < JOINT_LOWER) or np.any(target > JOINT_UPPER):
            return {"result": "fail_out_of_range", "reason": "'joint' exceeds the joint limits"}
        try:
            seconds = float(duration)
        except (TypeError, ValueError):
            return {"result": "fail_invalid_cmd", "reason": "'time' must be a number"}
        if not np.isfinite(seconds) or seconds <= 0.0:
            return {"result": "fail_invalid_cmd", "reason": "'time' must be > 0"}
        if self.busy:
            return {"result": "fail_busy", "reason": "a movej is already running"}
        self._movej_task = asyncio.create_task(
            self._run_movej(target, seconds), name="mock-robot-movej"
        )
        return {"result": "success", "time": seconds}

    async def _run_movej(self, target: np.ndarray, seconds: float) -> None:
        """Linearly interpolate the joint vector, refreshing the fake flange FK."""
        start = self.joints.copy()
        steps = max(1, int(round(seconds * self.movej_rate_hz)))
        for i in range(1, steps + 1):
            await asyncio.sleep(seconds / steps)
            alpha = i / steps
            self.joints = start + alpha * (target - start)
            for arm in ARMS:
                self.flange[arm] = self._flange_from_joints(arm)

    def _flange_from_joints(self, arm: str) -> Pose:
        """FAKE forward kinematics -- deterministic, smooth, not TRON 2's."""
        idx = slice(0, 7) if arm == "left" else slice(7, 14)
        q7 = self.joints[idx]
        base_p, base_q = HOME_FLANGE[arm]
        p = base_p + _FK_TRANSLATION @ q7
        rot = _quat_from_rotvec(_FK_ROTATION @ q7)
        w0, x0, y0, z0 = base_q
        w1, x1, y1, z1 = rot
        composed = np.array(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ]
        )
        return Pose(p, composed)

    def _on_light_effect(self, data: Mapping[str, Any]) -> dict[str, Any]:
        self.light = dict(data)
        return {"result": "success", **self.light}

    def _on_emgy_stop(self, _data: Mapping[str, Any]) -> dict[str, Any]:
        # Per the vendor guide this only works while idle; it is not an abort.
        if self.busy:
            return {"result": "fail_not_idle",
                    "reason": "emgy_stop is only accepted while the robot is idle"}
        return {"result": "success"}

    # -- servop ----------------------------------------------------------
    def validate_servop(self, data: Mapping[str, Any]) -> ServopCheck:
        """Check a ``request_servop`` payload against the configured strictness.

        Pure: the round-trip tests call this directly, and ``_on_servop`` uses it
        for the wire path.
        """
        width = SERVOP_WIDTH[self.accept_format]
        extra = sorted(set(data) - {"left_pos", "right_pos"})
        if extra:
            return ServopCheck(False, f"unexpected key(s) {extra}")
        present = [k for k in ("left_pos", "right_pos") if k in data]
        if not present:
            return ServopCheck(False, "needs at least one of left_pos/right_pos")
        if self.require_both_keys and len(present) != 2:
            return ServopCheck(
                False, f"this robot requires both left_pos and right_pos; got {present}"
            )
        targets: dict[str, Pose] = {}
        for key in present:
            arm = key.split("_", 1)[0]
            value = data[key]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return ServopCheck(False, f"{key} must be an array")
            if len(value) != width:
                return ServopCheck(
                    False,
                    f"{key} has {len(value)} elements; this robot accepts "
                    f"{self.accept_format} ({width} per arm)",
                )
            try:
                arr = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return ServopCheck(False, f"{key} must be numeric")
            if not np.all(np.isfinite(arr)):
                return ServopCheck(False, f"{key} contains non-finite values")
            if self.accept_format == "pos_quat":
                norm = float(np.linalg.norm(arr[3:7]))
                if abs(norm - 1.0) > 1e-3:
                    return ServopCheck(False, f"{key} quaternion is not unit norm ({norm:.6f})")
                targets[arm] = Pose(arr[:3], arr[3:7])
            else:
                r = arr[3:12].reshape(3, 3)
                if not np.allclose(r @ r.T, np.eye(3), atol=1e-3):
                    return ServopCheck(False, f"{key} rotation block is not orthonormal")
                if abs(float(np.linalg.det(r)) - 1.0) > 1e-3:
                    return ServopCheck(False, f"{key} rotation block has determinant != 1")
                targets[arm] = Pose(arr[:3], matrix_to_quat(r))
        return ServopCheck(True, targets=targets)

    async def _on_servop(self, ws: ServerConnection, data: Mapping[str, Any]) -> None:
        check = self.validate_servop(data)
        if not check.ok:
            self.servop_rejected += 1
            await self._send(
                ws, NOTIFY_SERVOP, {"result": "fail_invalid_cmd", "reason": check.reason}
            )
            return
        self.servop_accepted += 1
        self._integrate(check.targets)

    def _integrate(self, targets: Mapping[str, Pose]) -> None:
        """First-order approach to each commanded pose.

        A pure setter would make the mock a perfect servo and hide pacing bugs;
        a first-order lag at least resembles a real follower.
        """
        g = self.servop_gain
        for arm, target in targets.items():
            current = self.flange[arm]
            self.flange[arm] = Pose(
                current.p + g * (target.p - current.p),
                slerp(current.q_wxyz, target.q_wxyz, g),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mock_robot",
        description="Local mock of a LimX TRON 2 dual-arm robot (WebSocket).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="bind port (default: %(default)s)")
    parser.add_argument("--accid", default=DEFAULT_ACCID,
                        help="account id advertised in notify_robot_info")
    parser.add_argument("--accept-format", choices=sorted(SERVOP_WIDTH), default="pos_quat",
                        help="servop element layout this robot accepts (default: %(default)s)")
    parser.add_argument("--require-both-keys", action="store_true",
                        help="reject a servop carrying only one of left_pos/right_pos")
    parser.add_argument("--servop-gain", type=float, default=0.5,
                        help="first-order follow gain per servop (default: %(default)s)")
    parser.add_argument("--info-period-s", type=float, default=1.0,
                        help="notify_robot_info broadcast period (default: %(default)s)")
    return parser


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    robot = MockTron2(
        host=args.host,
        port=args.port,
        accid=args.accid,
        accept_format=args.accept_format,
        require_both_keys=args.require_both_keys,
        servop_gain=args.servop_gain,
        info_period_s=args.info_period_s,
    )
    await robot.start()
    print(
        f"mock TRON 2 on ws://{args.host}:{robot.bound_port}  "
        f"accid={args.accid}  accept-format={args.accept_format}  "
        f"require-both-keys={args.require_both_keys}",
        flush=True,
    )
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await robot.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
