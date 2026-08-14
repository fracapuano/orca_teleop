#!/usr/bin/env python3
"""Command-line harness for the TRON 2 arm layer.

Every subcommand talks to the **local mock robot** unless ``--robot`` is given
*and* the operator confirms interactively -- CLAUDE.md hard rule 1. The gate is
enforced on the resolved target address, so pointing ``robot.url`` in the YAML
at real hardware does not sneak past it.

    python tools/tron2_cli.py info
    python tools/tron2_cli.py servop-circle --arm right --radius 0.02 --speed 0.05
    python tools/tron2_cli.py servop-readback       # guide 9G-09 format settling
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import contextlib
import dataclasses
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tron_arm.config import ARMS, Config, ConfigError, JOINT_LOWER, JOINT_UPPER, load_config
from tron_arm.mock_robot import DEFAULT_PORT
from tron_arm.mapping import check_inside_workspace
from tron_arm.poses import Pose, quat_angle
from tron_arm.streamer import PoseStreamer
from tron_arm.tron2_client import NOTIFY_ROBOT_INFO, NOTIFY_SERVOP, Tron2Client, Tron2Error

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def diagnose_link(url: str) -> str:
    """Explain a failed connect in terms of the local network.

    A WebSocket timeout says nothing about *why*. Nine times out of ten on this
    rig the answer is "the laptop is not on the robot LAN", and the machine can
    check that itself in a millisecond.
    """
    host = urlparse(url).hostname
    if host is None or is_loopback(url):
        return ""
    try:
        target = ipaddress.ip_address(host)
    except ValueError:
        return ""  # a hostname, not an address; nothing cheap to say

    local: list[str] = []
    try:
        # The hostname may not resolve (it often does not on macOS); that is not
        # an error here, just one fewer source of addresses.
        local += [info[4][0] for info in
                  socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except (socket.gaierror, OSError):
        pass
    try:  # ask the routing table which source address it would use
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.2)
        probe.connect((host, 1))
        local.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass

    same_subnet = [
        addr for addr in set(local)
        if ipaddress.ip_network(f"{addr}/24", strict=False).supernet_of(
            ipaddress.ip_network(f"{target}/32"))
    ]
    lines = ["", "link check:"]
    if same_subnet:
        lines.append(f"  you have {', '.join(same_subnet)} on the robot's subnet -- the")
        lines.append("  network looks right, so the robot itself is not answering.")
        lines.append("  Check it is powered, and that nothing else holds its control port.")
    else:
        lines.append(f"  NO local interface is on {target}'s /24 subnet.")
        lines.append(f"  Local addresses seen: {', '.join(sorted(set(local))) or 'none'}")
        lines.append("  You are not on the robot's network.")
        lines.append("  1. Turn any VPN OFF -- it takes over the default route.")
        lines.append("  2. Connect to the robot's Wi-Fi.")
        lines.append("  3. Confirm with: ifconfig | grep 'inet 10.192.1'")
    return "\n".join(lines)


class Refused(RuntimeError):
    """The safety gate declined to proceed."""


# -- target resolution + safety gate -------------------------------------
def normalise_target(value: str) -> str:
    """Accept ``host``, ``host:port`` or a full ``ws://`` URL."""
    if value.startswith(("ws://", "wss://")):
        return value
    return f"ws://{value}" if ":" in value else f"ws://{value}:{DEFAULT_PORT}"


def is_loopback(url: str) -> bool:
    host = urlparse(url).hostname
    return host is not None and host in LOOPBACK_HOSTS


def gate_target(url: str, *, explicit_robot_flag: bool, description: str) -> str:
    """Enforce hard rule 1 on a resolved target URL.

    Real hardware needs both an explicit ``--robot`` flag and a typed
    confirmation; without a TTY there is no way to confirm, so we refuse. Shared
    with ``run_arm.py`` so there is exactly one gate to audit.
    """
    if is_loopback(url):
        return url
    if not explicit_robot_flag:
        raise Refused(
            f"config points at non-loopback {url!r}. Real hardware requires an "
            f"explicit --robot flag as well (CLAUDE.md hard rule 1)."
        )
    if not sys.stdin.isatty():
        raise Refused(
            f"refusing to drive real hardware at {url} without an interactive "
            f"terminal to confirm on (CLAUDE.md hard rule 1)."
        )
    print(f"\n*** REAL HARDWARE: {url} ***")
    print(f"    command: {description}")
    print("    The arms may move. Ensure the workspace is clear and someone is")
    print("    on the hardware remote (emgy_stop only works while idle).")
    if input("    Type 'yes' to continue: ").strip() != "yes":
        raise Refused("not confirmed")
    return url


def resolve_target(args: argparse.Namespace, config: Config) -> str:
    """Pick the target URL and enforce hard rule 1."""
    url = normalise_target(args.robot) if args.robot else config.robot.url
    return gate_target(url, explicit_robot_flag=bool(args.robot), description=args.command)


def make_client(args: argparse.Namespace, config: Config) -> Tron2Client:
    url = resolve_target(args, config)
    return Tron2Client(config, url=url)


# -- formatting ----------------------------------------------------------
def fmt_vec(v: np.ndarray, width: int = 9, prec: int = 4) -> str:
    return "[" + ", ".join(f"{float(x):{width}.{prec}f}" for x in v) + "]"


def print_pose(label: str, pose: Pose) -> None:
    print(f"  {label:<6} p={fmt_vec(pose.p)}  q_wxyz={fmt_vec(pose.q_wxyz)}")


def print_headroom(config: Config, arm: str, pose: Pose) -> None:
    """How much room this pose has to each wall.

    A posture near a wall means the operator runs out of workspace on that side
    and the clamp pins the target -- which feels like the arm sticking. Worth
    seeing before a session rather than inferring it from a clamp count after.
    """
    bounds = config.workspace.box(arm).bounds
    margin = config.workspace.margin_m
    parts = []
    for i, axis in enumerate("xyz"):
        low = float(pose.position_m[i] - (bounds[i, 0] + margin))
        high = float((bounds[i, 1] - margin) - pose.position_m[i])
        parts.append(f"{axis}: {low * 100:+.0f}/{high * 100:+.0f}")
    print(f"         room to walls (cm, -/+): {'   '.join(parts)}")
    tight = [a for i, a in enumerate("xyz")
             if min(pose.position_m[i] - (bounds[i, 0] + margin),
                    (bounds[i, 1] - margin) - pose.position_m[i]) < 0.10]
    if tight:
        print(f"         NOTE: under 10 cm of room on {'/'.join(tight)} -- the operator "
              f"will run out of workspace there")


# -- subcommands ---------------------------------------------------------
async def cmd_info(args: argparse.Namespace, config: Config) -> int:
    """Connect, learn accid, and show the first notify_robot_info."""
    seen: list[Any] = []
    async with make_client(args, config) as client:
        client.on_notify(NOTIFY_ROBOT_INFO, seen.append)
        print(f"url    : {client.url}")
        print(f"accid  : {client.accid}")
        deadline = time.monotonic() + 2.0
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if seen:
            record = seen[-1]
            print(f"info   : {dict(record.data)}")
            print(f"clocks : t_mono_ns={record.t_mono_ns}  t_wall_ns={record.t_wall_ns}")
        else:
            print(f"info   : no {NOTIFY_ROBOT_INFO} within 2 s")
        print(f"stats  : {client.stats.as_dict()}")
    return 0


async def cmd_pose(args: argparse.Namespace, config: Config) -> int:
    """Print both flange poses (robot base frame, FLU metres, wxyz)."""
    async with make_client(args, config) as client:
        poses = await client.get_move_pose()
        print("flange poses (base frame, FLU m, Hamilton wxyz):")
        for arm in ARMS:
            print_pose(arm, poses[arm])
        print()
        print("workspace check (config box minus margin):")
        for arm in ARMS:
            violation = check_inside_workspace(config, arm, poses[arm])
            if violation is None:
                print(f"  {arm:<6} INSIDE")
                print_headroom(config, arm, poses[arm])
            else:
                print(f"  {arm:<6} OUTSIDE -- {violation.describe()}")
                print(f"         streaming would clamp the hold target and MOVE the arm "
                      f"{violation.distance_m * 1e3:.0f} mm")
                if violation.looks_like_home:
                    print("         This looks like the HOME posture, which is normally "
                          "outside the box:")
                    print("         the arms hang beside the base while the box starts "
                          "~0.25 m in front of it.")
                    print("         Bring the arms forward/up before streaming; this is "
                          "expected, not a fault.")
    return 0


async def cmd_joints(args: argparse.Namespace, config: Config) -> int:
    """Print the joint angles with their headroom to the documented limits."""
    async with make_client(args, config) as client:
        if getattr(args, "raw", False):
            # Bypass the decoder entirely: dump exactly what the robot sent.
            from tron_arm.tron2_client import TITLE_GET_JOINT_STATE

            reply = await client._request(TITLE_GET_JOINT_STATE)
            print("raw request_get_joint_state reply:")
            for key, value in sorted(dict(reply).items()):
                if isinstance(value, list):
                    print(f"  {key} ({len(value)} entries):")
                    for i, v in enumerate(value):
                        print(f"    [{i:2d}] {v:+.6f}"
                              if isinstance(v, (int, float)) else f"    [{i:2d}] {v!r}")
                else:
                    print(f"  {key}: {value!r}")
            return 0
        from tron_arm.tron2_client import TITLE_GET_JOINT_STATE, split_joint_reply

        reply = await client._request(TITLE_GET_JOINT_STATE)
        joints, extras = split_joint_reply(reply)
    print("joint      value      lower      upper   headroom")
    for i, value in enumerate(joints):
        side = "L" if i < 7 else "R"
        headroom = min(value - JOINT_LOWER[i], JOINT_UPPER[i] - value)
        print(f"  {side}{i % 7}   {value:9.4f}  {JOINT_LOWER[i]:9.4f}  "
              f"{JOINT_UPPER[i]:9.4f}  {headroom:9.4f}")
    if extras.size:
        print(f"\n  plus {extras.size} non-arm joint(s) this unit reports "
              f"(grippers): {', '.join(f'{v:+.4f}' for v in extras)}")
        print("  These are not sent by movej, which takes 14.")
    return 0


async def cmd_light(args: argparse.Namespace, config: Config) -> int:
    """Set the status light effect."""
    async with make_client(args, config) as client:
        print(await client.light_effect(args.effect))
    return 0


async def cmd_movej_home(args: argparse.Namespace, config: Config) -> int:
    """Move to the configured home joint vector."""
    async with make_client(args, config) as client:
        print(f"home joints: {fmt_vec(config.home_joints, 7, 3)}")
        print(f"moving over {args.time}s ...")
        print(await client.movej(config.home_joints, args.time))
        await asyncio.sleep(args.time + 0.1)
        for arm, pose in (await client.get_move_pose()).items():
            print_pose(arm, pose)
    return 0


async def _stream(
    client: Tron2Client,
    config: Config,
    target_fn,
    seconds: float,
    *,
    ingress_hz: float,
    sample_fn=None,
) -> PoseStreamer:
    """Drive the streamer from ``target_fn(t)`` for ``seconds``.

    ``target_fn`` stands in for the operator ingress: it is polled at
    ``ingress_hz`` (30-60 Hz territory) while the streamer sends at
    ``servop.rate_hz``.
    """
    failures: list[Any] = []
    client.on_notify(NOTIFY_SERVOP, failures.append)
    current = await client.prime_frozen_poses()

    streamer = PoseStreamer(config, client.send_servop)
    # Seed the step clamp with where the arms actually are, so even the first
    # tick is velocity-limited rather than a jump to the first target.
    streamer.seed_all(current)
    stop = asyncio.Event()
    task = await streamer.start(stop=stop)

    async def feed() -> None:
        period = 1.0 / ingress_hz
        t0 = time.monotonic_ns()
        while not stop.is_set():
            now = time.monotonic_ns()
            arm, pose = target_fn((now - t0) / 1e9)
            streamer.submit(arm, pose, now)
            await asyncio.sleep(period)

    async def sample() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.02)
            with contextlib.suppress(Tron2Error, asyncio.TimeoutError):
                sample_fn(await client.get_move_pose())

    tasks = [asyncio.create_task(feed())]
    if sample_fn is not None:
        tasks.append(asyncio.create_task(sample()))
    try:
        await asyncio.sleep(seconds)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        await streamer.stop()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    print(f"\nstreamer: {streamer.stats_dict()}")
    print(f"client  : {client.stats.as_dict()}")
    if failures:
        print(f"\n!! {len(failures)} {NOTIFY_SERVOP} failure(s); first: "
              f"{dict(failures[0].data)}")
        print("   Check servop.format / servop.send_both in the config.")
    else:
        print(f"\nno {NOTIFY_SERVOP} failures.")
    return streamer


async def cmd_servop_hold(args: argparse.Namespace, config: Config) -> int:
    """Stream the current pose unchanged -- the pacing smoke test."""
    async with make_client(args, config) as client:
        poses = await client.get_move_pose()
        print(f"holding both arms for {args.seconds}s at {config.servop.rate_hz} Hz "
              f"(format={config.servop.format}, send_both={config.servop.send_both})")
        for arm in ARMS:
            print_pose(arm, poses[arm])
        arms = list(ARMS)

        def target(t: float) -> tuple[str, Pose]:
            # Alternate arms so both tracks stay fed from one ingress stream.
            arm = arms[int(t * 2) % 2]
            return arm, poses[arm]

        await _stream(client, config, target, args.seconds, ingress_hz=args.ingress_hz)
    return 0


async def cmd_servop_circle(args: argparse.Namespace, config: Config) -> int:
    """Trace a circle with one arm and report how round the result was."""
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.plane]
    omega = args.speed / args.radius  # rad/s
    async with make_client(args, config) as client:
        centre = (await client.get_move_pose())[args.arm]
        print(f"circle: arm={args.arm} radius={args.radius} m speed={args.speed} m/s "
              f"plane={args.plane} omega={omega:.3f} rad/s")
        print_pose("centre", centre)

        def target(t: float) -> tuple[str, Pose]:
            p = np.array(centre.p, dtype=np.float64)
            p[axes[0]] += args.radius * math.cos(omega * t)
            p[axes[1]] += args.radius * math.sin(omega * t)
            return args.arm, Pose(p, centre.q_wxyz)

        achieved: list[np.ndarray] = []

        def collect(poses: dict[str, Pose]) -> None:
            achieved.append(np.array(poses[args.arm].p, dtype=np.float64))

        await _stream(client, config, target, args.seconds,
                      ingress_hz=args.ingress_hz, sample_fn=collect)

    if len(achieved) < 8:
        print(f"\nonly {len(achieved)} readback samples; cannot assess roundness")
        return 1
    pts = np.asarray(achieved)
    # Drop the first turn-in samples: the arm starts at the centre, not on the rim.
    pts = pts[len(pts) // 4:]
    plane_pts = pts[:, list(axes)]
    fitted_centre = plane_pts.mean(axis=0)
    radii = np.linalg.norm(plane_pts - fitted_centre, axis=1)
    off_plane = np.setdiff1d(np.arange(3), axes)[0]
    print("\ntraced path:")
    print(f"  samples            : {len(pts)}")
    print(f"  fitted centre      : {fmt_vec(fitted_centre)}  (commanded "
          f"{fmt_vec(np.asarray(centre.p)[list(axes)])})")
    print(f"  radius mean/std    : {radii.mean():.6f} / {radii.std():.6f} m "
          f"(commanded {args.radius})")
    print(f"  radius min/max     : {radii.min():.6f} / {radii.max():.6f} m")
    print(f"  off-plane drift    : {pts[:, off_plane].std():.6f} m")
    print(f"  angular coverage   : {np.ptp(np.unwrap(np.arctan2(*(plane_pts - fitted_centre).T[::-1]))):.3f} rad")
    ok = abs(radii.mean() - args.radius) < 0.15 * args.radius and radii.std() < 0.15 * args.radius
    print(f"  verdict            : {'CIRCLE' if ok else 'NOT A CIRCLE'}")
    return 0 if ok else 1


async def cmd_servop_readback(args: argparse.Namespace, config: Config) -> int:
    """Guide 9G-09 format settling: command one known pose, read it back.

    If ``servop.format`` does not match what the robot expects, either the
    command is rejected outright (``notify_servop``) or it settles somewhere
    unrelated -- both are visible in the component-wise error below.
    """
    async with make_client(args, config) as client:
        start = (await client.get_move_pose())[args.arm]
        # A deliberately asymmetric target: a wrong element layout cannot
        # coincidentally reproduce it.
        offset = np.array([0.010, -0.015, 0.020])
        angle = math.radians(10.0)
        axis = np.array([0.0, 0.0, 1.0])
        s = math.sin(angle / 2.0)
        delta_q = np.array([math.cos(angle / 2.0), *(axis * s)])
        w0, x0, y0, z0 = start.q_wxyz
        w1, x1, y1, z1 = delta_q
        q = np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ])
        known = Pose(start.p + offset, q)

        failures: list[Any] = []
        client.on_notify(NOTIFY_SERVOP, failures.append)
        await client.prime_frozen_poses()

        print(f"format={config.servop.format}  send_both={config.servop.send_both}  "
              f"arm={args.arm}")
        print_pose("start", start)
        print_pose("command", known)

        # Hold the single known pose for the settle window: ServoP is a streaming
        # command, so one packet only moves the follower part of the way.
        period = config.servop.period_s
        deadline = time.monotonic() + args.settle
        sent = 0
        while time.monotonic() < deadline:
            await client.send_servop(**{args.arm: known})
            sent += 1
            await asyncio.sleep(period)
        await asyncio.sleep(0.1)
        final = (await client.get_move_pose())[args.arm]

    print(f"\nsent {sent} servop packets over {args.settle}s")
    print_pose("readback", final)
    err_p = np.asarray(final.p) - np.asarray(known.p)
    err_q = np.asarray(final.q_wxyz) - np.asarray(known.q_wxyz)
    print("\ncomponent-wise error (readback - commanded):")
    for name, value in zip(("x", "y", "z"), err_p):
        print(f"  {name}      {value:+.6f} m")
    for name, value in zip(("qw", "qx", "qy", "qz"), err_q):
        print(f"  {name}     {value:+.6f}")
    print(f"\n  |position error| : {np.linalg.norm(err_p):.6f} m")
    print(f"  rotation error   : {math.degrees(quat_angle(known.q_wxyz, final.q_wxyz)):.4f} deg")
    if failures:
        print(f"\n!! {len(failures)} {NOTIFY_SERVOP} failure(s); first: {dict(failures[0].data)}")
        print(f"   this robot rejected servop.format={config.servop.format} / "
              f"send_both={config.servop.send_both}. Adjust the config, not the encoder.")
        return 1
    settled = np.linalg.norm(err_p) < 1e-3
    print(f"\n  verdict          : {'SETTLED' if settled else 'DID NOT SETTLE'}")
    return 0 if settled else 1


async def cmd_capture_ready(args: argparse.Namespace, config: Config) -> int:
    """Print the current joint vector as a `ready:` block to paste into the config.

    Capture this ONCE, with the arms somewhere comfortable and both flanges
    inside the workspace box. After that `movej-ready` reproduces it every
    session, so nobody has to position the arms by hand again.
    """
    async with make_client(args, config) as client:
        joints = await client.get_joint_state()
        poses = await client.get_move_pose()

    violations = [check_inside_workspace(config, arm, poses[arm]) for arm in ARMS]
    for arm, violation in zip(ARMS, violations):
        print_pose(arm, poses[arm])
        print(f"  {arm:<6} {'INSIDE' if violation is None else 'OUTSIDE -- ' + violation.describe()}")
    if any(v is not None for v in violations):
        print("\nNOT captured: move the arms so BOTH flanges are inside the box first.",
              file=sys.stderr)
        print("A ready posture outside the box is useless -- streaming would still refuse.",
              file=sys.stderr)
        return 1

    print("\nPaste this into configs/default.yaml, replacing the `ready:` block:\n")
    print("ready:")
    rows = [", ".join(f"{v:+.5f}" for v in joints[i:i + 7]) for i in (0, 7)]
    print(f"  joints: [{rows[0]},\n            {rows[1]}]")
    print("\n# left j0..j6 then right j0..j6, radians")
    return 0


async def cmd_movej_ready(args: argparse.Namespace, config: Config) -> int:
    """Move to the configured ready posture, then confirm both arms are inside."""
    if config.ready_joints is None:
        print("no ready posture configured. Capture one first:\n"
              "  tron2_cli.py --robot <ip> capture-ready", file=sys.stderr)
        return 1
    async with make_client(args, config) as client:
        print(f"moving to the ready posture over {args.time}s ...")
        print(await client.movej(config.ready_joints, args.time))
        await asyncio.sleep(args.time + 0.3)
        poses = await client.get_move_pose()
    ok = True
    for arm in ARMS:
        violation = check_inside_workspace(config, arm, poses[arm])
        print_pose(arm, poses[arm])
        if violation is None:
            print(f"  {arm:<6} INSIDE")
            print_headroom(config, arm, poses[arm])
        else:
            print(f"  {arm:<6} OUTSIDE -- {violation.describe()}")
            ok = False
    print("\nready for teleop." if ok else "\nNOT ready: recapture the posture.")
    return 0 if ok else 1


COMMANDS = {
    "info": cmd_info,
    "pose": cmd_pose,
    "joints": cmd_joints,
    "light": cmd_light,
    "movej-home": cmd_movej_home,
    "capture-ready": cmd_capture_ready,
    "movej-ready": cmd_movej_ready,
    "servop-hold": cmd_servop_hold,
    "servop-circle": cmd_servop_circle,
    "servop-readback": cmd_servop_readback,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tron2_cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--robot", default=None, metavar="HOST",
                        help="target real hardware (host, host:port or ws:// URL). "
                             "Requires interactive confirmation.")
    parser.add_argument("--format", choices=("pos_quat", "pos_rotmat"), default=None,
                        help="override servop.format for this run")
    parser.add_argument("--send-both", dest="send_both", action="store_true", default=None,
                        help="override servop.send_both to true")
    parser.add_argument("--no-send-both", dest="send_both", action="store_false",
                        help="override servop.send_both to false")
    parser.add_argument("--rate-hz", type=float, default=None,
                        help="override servop.rate_hz for this run")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="connect, learn accid, show notify_robot_info")
    sub.add_parser("pose", help="print both flange poses")
    p_joints = sub.add_parser("joints", help="print the joint angles")
    p_joints.add_argument("--raw", action="store_true",
                          help="dump the robot's reply verbatim, no decoding")

    p_light = sub.add_parser("light", help="set the status light effect")
    p_light.add_argument("--effect", default="flash")

    p_home = sub.add_parser("movej-home", help="movej to the configured home joints")
    p_home.add_argument("--time", type=float, default=3.0, help="move duration, s")

    sub.add_parser("capture-ready",
                   help="print the current joints as a ready: block for the config")
    p_ready = sub.add_parser("movej-ready", help="movej to the configured ready posture")
    p_ready.add_argument("--time", type=float, default=4.0, help="move duration, s")

    p_hold = sub.add_parser("servop-hold", help="stream the current pose unchanged")
    p_hold.add_argument("--seconds", type=float, default=3.0)
    p_hold.add_argument("--ingress-hz", type=float, default=45.0)

    p_circle = sub.add_parser("servop-circle", help="trace a circle with one arm")
    p_circle.add_argument("--arm", choices=ARMS, default="right")
    p_circle.add_argument("--radius", type=float, default=0.02, help="metres")
    p_circle.add_argument("--speed", type=float, default=0.05, help="m/s along the rim")
    p_circle.add_argument("--seconds", type=float, default=6.0)
    p_circle.add_argument("--plane", choices=("xy", "xz", "yz"), default="xy")
    p_circle.add_argument("--ingress-hz", type=float, default=45.0)

    p_read = sub.add_parser("servop-readback",
                            help="guide 9G-09: command a known pose, print the error")
    p_read.add_argument("--arm", choices=ARMS, default="right")
    p_read.add_argument("--settle", type=float, default=1.0, help="hold duration, s")
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply the ``--format`` / ``--send-both`` / ``--rate-hz`` overrides.

    These exist so the format-settling investigation is a config change, not a
    code change.
    """
    servop = config.servop
    changes: dict[str, Any] = {}
    if args.format is not None:
        changes["format"] = args.format
    if args.send_both is not None:
        changes["send_both"] = args.send_both
    if args.rate_hz is not None:
        changes["rate_hz"] = args.rate_hz
    if not changes:
        return config
    return dataclasses.replace(config, servop=dataclasses.replace(servop, **changes))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    try:
        return asyncio.run(COMMANDS[args.command](args, config))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except Tron2Error as exc:
        print(f"robot error: {exc}", file=sys.stderr)
        if "cannot connect" in str(exc):
            hint = diagnose_link(normalise_target(args.robot) if args.robot
                                 else config.robot.url)
            if hint:
                print(hint, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
