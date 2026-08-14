"""Runbook step 4: ServoP format discovery, with no operator in the loop.

The riskiest unknown in the whole stack is whether our ``request_servop``
encoding is the one this controller accepts. The vendor guide is ambiguous
(``CLAUDE.md``: 7 floats per arm vs 12), and a wrong guess does not produce
wrong motion — it produces **no motion**, either with a ``notify_servop``
failure or in complete silence. Finding that out with an operator wearing a
headset is the wrong time.

So this runs the sequence from runbook §7 step 4, driving the robot directly:

1. **Hold the current pose for 5 s.** Zero motion is the expected outcome; this
   validates format *acceptance* alone, with nothing to misread as success.
2. **§G-09 readback comparison.** Command one known, deliberately asymmetric
   pose, let it settle, read it back with ``request_get_move_pose``, and compare
   component-wise. PASS/FAIL is printed **per servop format**, so the output
   answers the question rather than requiring interpretation.
3. **±2 cm single-axis steps.** Confirms sign and axis correspondence — a format
   that passes readback can still have x and y transposed.
4. **2 cm circle at 0.05 m/s.** Continuous motion, the first thing that looks
   like teleoperation.

Every phase is workspace-clamped and step-clamped by the same code path the
operator uses, so nothing here can command something teleop could not.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import ARMS, Arm, Config, SERVOP_WIDTH
from .mapping import check_inside_workspace
from .poses import Pose, quat_angle
from .session import SessionLogger
from .streamer import PoseStreamer, apply_step_clamp
from .tron2_client import NOTIFY_SERVOP, NotifyRecord, Tron2Client, Tron2Timeout

__all__ = ["PhaseResult", "StepTestReport", "run_step_test"]

HOLD_SECONDS = 5.0
READBACK_SETTLE_S = 2.0
STEP_M = 0.02
CIRCLE_RADIUS_M = 0.02
CIRCLE_SPEED_MS = 0.05

#: Acceptance thresholds, NOT an accuracy specification.
#:
#: This test answers one question: "does this robot accept and act on this
#: encoding?" A wrong format produces rejection, silence, or metres of error --
#: never centimetres. So these only need to be tight enough to separate right
#: from wrong, and loose enough to survive a real servo follower.
#:
#: PROVISIONAL - RE-TIGHTEN AFTER RE-MEASURING. These were set from a run on
#: 2026-08-13 (fw robot-tron2-r-2.1.24) that we later established was taken with
#: a gripper fitted to only ONE arm, so the controller's gravity compensation
#: was fighting an unmodelled payload. The tell was in the axis steps: y tracked
#: at 0.94-0.97 of commanded while z ran 0.33 up / 1.56 down -- an asymmetry on
#: the gravity axis only, which no encoding error can produce.
#:
#: With both grippers fitted the arm should be far more accurate. Re-run
#: --step-test and tighten these to roughly 3x the observed error; leaving them
#: at 30 mm would wave through a real fault later.
#:
#: (The ORIGINAL 5 mm was wrong for a different reason: calibrated against the
#: mock, which converges exactly. Do not simply revert to it.)
READBACK_TOL_M = 0.030
READBACK_TOL_DEG = 10.0
#: A held pose may drift this far before we call it a failure.
HOLD_TOL_M = 0.030
#: Wait between formats: streaming leaves the controller busy, and the next
#: request_get_move_pose timed out at 2 s on hardware.
INTER_FORMAT_SETTLE_S = 6.0


@dataclass
class PhaseResult:
    """One phase's outcome. ``passed is None`` means "ran, not a pass/fail gate"."""

    name: str
    passed: bool | None
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notify_failures: int = 0

    @property
    def verdict(self) -> str:
        if self.passed is None:
            return "----"
        return "PASS" if self.passed else "FAIL"

    def line(self) -> str:
        return f"  [{self.verdict}] {self.name:<28} {self.detail}"


@dataclass
class StepTestReport:
    """Everything the run learned, per format."""

    formats: dict[str, list[PhaseResult]] = field(default_factory=dict)
    chosen_format: str | None = None
    aborted: bool = False

    def add(self, fmt: str, result: PhaseResult) -> None:
        self.formats.setdefault(fmt, []).append(result)

    def format_passed(self, fmt: str) -> bool:
        results = self.formats.get(fmt, [])
        gated = [r for r in results if r.passed is not None]
        return bool(gated) and all(r.passed for r in gated)

    def text(self) -> str:
        lines = ["", "=" * 68, "ServoP step test — runbook §7 step 4", "=" * 68]
        for fmt, results in self.formats.items():
            width = SERVOP_WIDTH[fmt]
            lines.append("")
            lines.append(f"format: {fmt}  ({width} floats per arm)")
            lines.extend(r.line() for r in results)
            lines.append(f"  => {fmt}: {'ACCEPTED' if self.format_passed(fmt) else 'REJECTED'}")
        lines.append("")
        if self.aborted:
            lines.append("ABORTED BEFORE COMMANDING ANYTHING.")
            lines.append("  The arm is parked outside the workspace box in the config, so")
            lines.append("  'hold current pose' would have been clamped into the box and the")
            lines.append("  arm driven there. Nothing was commanded.")
            lines.append("")
            lines.append("  Fix the box (configs/default.yaml workspace:) to match this")
            lines.append("  robot, or move the arm inside it with movej, then re-run.")
            lines.append("=" * 68)
            return "\n".join(lines)
        if self.chosen_format:
            lines.append(f"VERDICT: use servop.format: {self.chosen_format}")
            lines.append("  Set it in configs/default.yaml. No code change is needed --")
            lines.append("  the encoder reads this from config by design.")
        else:
            lines.append("VERDICT: NO FORMAT ACCEPTED.")
            lines.append("  Both encodings were rejected or ignored. Do not proceed to")
            lines.append("  teleop. Check notify_servop reasons above, then LimX Q1.")
        lines.append("=" * 68)
        return "\n".join(lines)


def _known_offset_pose(start: Pose) -> Pose:
    """A deliberately asymmetric target: no axis transposition reproduces it."""
    offset = np.array([0.010, -0.015, 0.020])
    angle = math.radians(10.0)
    delta = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    w0, x0, y0, z0 = start.orientation_wxyz
    w1, x1, y1, z1 = delta
    q = np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])
    return Pose(start.position_m + offset, q)


class WorkspacePrecondition(RuntimeError):
    """The arm is parked outside its configured box; nothing may be commanded."""


class _FailureWatch:
    """Counts notify_servop failures during a phase."""

    def __init__(self, client: Tron2Client) -> None:
        self.records: list[NotifyRecord] = []
        self._off = client.on_notify(NOTIFY_SERVOP, self.records.append)

    @property
    def failures(self) -> list[NotifyRecord]:
        return [r for r in self.records
                if str(r.data.get("result", "")).startswith("fail")]

    def reset(self) -> None:
        self.records.clear()

    def stop(self) -> None:
        self._off()


async def _stream_pose(
    client: Tron2Client, arm: Arm, pose_at: Any, seconds: float, config: Config,
    *, last: Pose | None = None,
) -> Pose | None:
    """Send ``pose_at(t)`` at the configured rate, step-clamped, for ``seconds``.

    Uses the same clamp as teleop, so this cannot command a faster move than an
    operator could.
    """
    period = config.servop.period_s
    max_lin, max_ang = config.max_step
    deadline = time.monotonic() + seconds
    started = time.monotonic()
    while time.monotonic() < deadline:
        target = pose_at(time.monotonic() - started)
        clamped_p, _ = config.workspace.clamp(arm, target.position_m)
        target = Pose(clamped_p, target.orientation_wxyz)
        target, _, _ = apply_step_clamp(last, target, max_lin, max_ang)
        await client.send_servop(**{arm: target})
        last = target
        await asyncio.sleep(period)
    return last


async def _run_one_format(
    config: Config, url: str, arm: Arm, fmt: str, report: StepTestReport,
    *, session: SessionLogger, quick: bool,
) -> None:
    """Run every phase against one candidate encoding."""
    cfg = dataclasses.replace(
        config,
        servop=dataclasses.replace(config.servop, format=fmt),
        # Streaming leaves the controller busy; a 2 s request timeout was not
        # enough on hardware and aborted the second format before it ran.
        robot=dataclasses.replace(config.robot, request_timeout_s=max(
            config.robot.request_timeout_s, 6.0)),
    )
    hold_s = 0.6 if quick else HOLD_SECONDS
    settle_s = 0.3 if quick else READBACK_SETTLE_S

    async with Tron2Client(cfg, url=url, notify_log_path=None) as client:
        watch = _FailureWatch(client)

        async def move_pose_with_retry(attempts: int = 4) -> dict[str, Pose]:
            """get_move_pose, retried. After a streaming burst the controller
            stops answering requests for a few seconds; a single timeout was
            aborting the second format before it ever ran."""
            for attempt in range(attempts):
                try:
                    return await client.get_move_pose()
                except Tron2Timeout:
                    if attempt == attempts - 1:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))
            raise AssertionError("unreachable")

        def emit(result: PhaseResult) -> None:
            report.add(fmt, result)
            session.event("step_test", format=fmt, phase=result.name,
                          passed=result.passed, detail=result.detail,
                          metrics=result.metrics)

        start = (await move_pose_with_retry())[arm]

        # Refuse before commanding anything. Streaming from here would clamp the
        # "hold" target into the box and drive the arm there.
        violation = check_inside_workspace(cfg, arm, start)
        if violation is not None:
            emit(PhaseResult(
                name="workspace precondition",
                passed=False,
                detail=violation.describe(),
                metrics={"measured": list(violation.measured),
                         "clamped": list(violation.clamped),
                         "distance_m": violation.distance_m},
            ))
            watch.stop()
            raise WorkspacePrecondition(violation.describe())

        await client.prime_frozen_poses()

        # -- 1. hold ----------------------------------------------------
        watch.reset()
        last = await _stream_pose(client, arm, lambda _t: start, hold_s, cfg)
        await asyncio.sleep(0.15)
        after = (await client.get_move_pose())[arm]
        drift = float(np.linalg.norm(after.position_m - start.position_m))
        failures = watch.failures
        emit(PhaseResult(
            name=f"hold current pose {hold_s:.0f}s",
            passed=not failures and drift < HOLD_TOL_M,
            detail=(f"{len(failures)} notify_servop failure(s): "
                    f"{failures[0].data.get('reason', '?') if failures else ''}"
                    if failures else
                    f"drift {drift * 1e3:.1f} mm (tol {HOLD_TOL_M * 1e3:.0f})"),
            metrics={"drift_m": drift, "notify_failures": len(failures)},
            notify_failures=len(failures),
        ))
        if failures:
            # A rejected format cannot pass anything downstream; stop early so
            # the operator reads one clear failure rather than four.
            emit(PhaseResult("readback (G-09)", False, "skipped: format rejected at hold"))
            emit(PhaseResult("+-2 cm axis steps", False, "skipped: format rejected at hold"))
            emit(PhaseResult("2 cm circle", False, "skipped: format rejected at hold"))
            watch.stop()
            return

        # -- 2. G-09 readback -------------------------------------------
        watch.reset()
        known = _known_offset_pose(start)
        last = await _stream_pose(client, arm, lambda _t: known, settle_s, cfg, last=last)
        await asyncio.sleep(0.15)
        got = (await client.get_move_pose())[arm]
        err_p = np.asarray(got.position_m) - np.asarray(known.position_m)
        err_deg = math.degrees(quat_angle(known.orientation_wxyz, got.orientation_wxyz))
        pos_err = float(np.linalg.norm(err_p))
        ok = (not watch.failures and pos_err <= READBACK_TOL_M and err_deg <= READBACK_TOL_DEG)
        emit(PhaseResult(
            name="readback (G-09)",
            passed=ok,
            detail=(f"|dp|={pos_err * 1e3:.2f} mm  dR={err_deg:.2f} deg  "
                    f"(tol {READBACK_TOL_M * 1e3:.0f} mm / {READBACK_TOL_DEG:.0f} deg)"),
            metrics={"err_xyz_m": [float(v) for v in err_p],
                     "err_pos_m": pos_err, "err_deg": err_deg},
            notify_failures=len(watch.failures),
        ))

        # -- 3. +-2 cm single-axis steps --------------------------------
        # Measured DIFFERENTIALLY: reached(+step) - reached(-step) should span
        # 2*STEP along the commanded axis. A servo with a constant steady-state
        # offset -- gravity droop, typically ~15 mm on this arm -- lands every
        # move short by the same vector, which makes absolute displacement look
        # like an axis fault. Subtracting the two ends cancels it exactly, and
        # what survives is the real question: does the commanded axis carry the
        # motion, in the right direction, at roughly the right scale?
        watch.reset()
        axis_metrics: dict[str, Any] = {}
        axis_ok = True
        for index, axis in enumerate("xyz"):
            reached: dict[float, np.ndarray] = {}
            base = (await client.get_move_pose())[arm]
            for sign in (+1.0, -1.0):
                delta = np.zeros(3)
                delta[index] = sign * STEP_M
                goal = Pose(base.position_m + delta, base.orientation_wxyz)
                last = await _stream_pose(client, arm, lambda _t, g=goal: g,
                                          settle_s, cfg, last=last)
                await asyncio.sleep(0.1)
                reached[sign] = np.asarray((await client.get_move_pose())[arm].position_m)
            span = reached[+1.0] - reached[-1.0]          # offset cancels here
            on_axis = float(span[index])
            off_axis = float(np.linalg.norm(np.delete(span, index)))
            expected = 2.0 * STEP_M
            good = (np.sign(on_axis) == 1.0                       # not mirrored
                    and abs(on_axis) > 0.6 * expected             # it really moved
                    and abs(on_axis) < 1.6 * expected             # not wildly over
                    and abs(on_axis) > off_axis)                  # it dominates
            axis_ok = axis_ok and good
            axis_metrics[axis] = {
                "span_on_axis_m": on_axis,
                "span_off_axis_m": off_axis,
                "expected_m": expected,
                "ratio": on_axis / expected,
                "reached_plus": [float(v) for v in reached[+1.0]],
                "reached_minus": [float(v) for v in reached[-1.0]],
                "ok": good,
            }
        # The constant offset itself is worth reporting: it is a real property of
        # the arm (sag under payload) and it is what LimX Q7 is about.
        emit(PhaseResult(
            name="+-2 cm axis steps",
            passed=axis_ok and not watch.failures,
            detail=("x/y/z spans " + ", ".join(
                f"{a}={axis_metrics[a]['span_on_axis_m'] * 1e3:.1f}mm"
                f"({axis_metrics[a]['ratio']:.2f})" for a in "xyz")
                + f" of {2 * STEP_M * 1e3:.0f}mm expected"
                if axis_ok else
                "AXIS MISMATCH - a commanded axis did not dominate or was "
                "mirrored; check FLU and the encoder"),
            metrics=axis_metrics,
            notify_failures=len(watch.failures),
        ))

        # -- 4. 2 cm circle ---------------------------------------------
        watch.reset()
        centre = (await client.get_move_pose())[arm]
        omega = CIRCLE_SPEED_MS / CIRCLE_RADIUS_M
        samples: list[np.ndarray] = []

        def circle_at(t: float) -> Pose:
            p = np.array(centre.position_m, dtype=np.float64)
            p[0] += CIRCLE_RADIUS_M * math.cos(omega * t)
            p[1] += CIRCLE_RADIUS_M * math.sin(omega * t)
            return Pose(p, centre.orientation_wxyz)

        async def sample_loop() -> None:
            while True:
                await asyncio.sleep(0.02)
                samples.append(np.array((await client.get_move_pose())[arm].position_m))

        sampler = asyncio.create_task(sample_loop())
        turns_s = (2 * math.pi / omega) * (1.0 if quick else 2.0) + 0.5
        await _stream_pose(client, arm, circle_at, turns_s, cfg, last=last)
        sampler.cancel()
        try:
            await sampler
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

        if len(samples) < 8:
            emit(PhaseResult("2 cm circle", False, f"only {len(samples)} readback samples"))
        else:
            pts = np.asarray(samples)[len(samples) // 4:, :2]
            fitted = pts.mean(axis=0)
            radii = np.linalg.norm(pts - fitted, axis=1)
            round_ = (abs(radii.mean() - CIRCLE_RADIUS_M) < 0.3 * CIRCLE_RADIUS_M
                      and radii.std() < 0.3 * CIRCLE_RADIUS_M)
            emit(PhaseResult(
                name="2 cm circle",
                passed=round_ and not watch.failures,
                detail=f"radius {radii.mean() * 1e3:.2f} +- {radii.std() * 1e3:.2f} mm "
                       f"(commanded {CIRCLE_RADIUS_M * 1e3:.0f})",
                metrics={"radius_mean_m": float(radii.mean()),
                         "radius_std_m": float(radii.std()),
                         "samples": len(samples)},
                notify_failures=len(watch.failures),
            ))
        watch.stop()


async def run_step_test(
    config: Config,
    url: str,
    *,
    arm: Arm = "right",
    formats: Sequence[str] | None = None,
    session: SessionLogger | None = None,
    quick: bool = False,
) -> StepTestReport:
    """Run the step test against every candidate format and report.

    ``quick`` shortens the dwell times for CI; the sequence is identical.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    session = session or SessionLogger(enabled=False)
    report = StepTestReport()
    for index, fmt in enumerate(formats or list(SERVOP_WIDTH)):
        if index:
            await asyncio.sleep(0.5 if quick else INTER_FORMAT_SETTLE_S)
        try:
            await _run_one_format(config, url, arm, fmt, report,
                                  session=session, quick=quick)
        except WorkspacePrecondition:
            # Not a format problem: the arm is somewhere the config says it
            # cannot be. Trying the other encoding would repeat the hazard.
            report.aborted = True
            break
        except Exception as exc:  # noqa: BLE001 - one bad format must not hide the other
            report.add(fmt, PhaseResult(f"format {fmt}", False, f"aborted: {exc}"))
    accepted = [f for f in report.formats if report.format_passed(f)]
    report.chosen_format = accepted[0] if accepted else None
    return report
