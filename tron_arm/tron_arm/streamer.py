"""Fixed-rate ServoP pacer: turns 30-60 Hz jittery ingress into a smooth stream.

Frames arrive from the operator at 30-60 Hz with jitter; the robot wants a
steady ServoP stream (``servop.rate_hz``, default 100). This module holds the
two newest targets per arm and interpolates between them on a drift-corrected
monotonic clock.

Why interpolate *behind* rather than extrapolate ahead: rendering at
``now - one_ingress_interval`` keeps the render time inside the segment we
already have, so a late frame degrades into a brief hold rather than an
overshoot that has to be walked back. The cost is one ingress interval
(~17-33 ms) of latency, which is the cheaper failure mode for teleoperation.

All timing uses ``recv_monotonic_ns``-style monotonic nanoseconds
(CLAUDE.md hard rule 2). The per-tick maths in :meth:`PoseStreamer.step` is pure
and testable without any I/O.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Mapping

import numpy as np

from .config import ARMS, Config
from .poses import Pose, PoseLike, as_pose, pose_lerp, quat_angle, slerp

__all__ = [
    "ArmTickInfo",
    "ArmTrack",
    "PoseStreamer",
    "Sample",
    "SampleDetail",
    "StreamerStats",
    "TickInfo",
    "apply_step_clamp",
]

SendCallback = Callable[[Pose | None, Pose | None], Awaitable[None]]

#: Cap on the interpolation delay. Without it, a long ingress dropout would set
#: a multi-second delay and the arm would crawl through stale history.
DEFAULT_MAX_DELAY_S = 0.1


@dataclass(frozen=True)
class Sample:
    """One received target and the monotonic time it arrived.

    ``meta`` rides along untouched so the caller can correlate a tick back to
    the dispatch that produced it -- the two happen on different threads, and
    without a carried token the join has to be guessed from timestamps.
    """

    pose: Pose
    t_recv_ns: int
    meta: Any = None


@dataclass(frozen=True)
class SampleDetail:
    """What :meth:`ArmTrack.sample_detail` computed, for instrumentation."""

    pose: Pose
    alpha: float
    ingress_gap_ns: int | None
    source: Sample
    """The newest sample -- the one whose ``meta`` identifies this target."""


@dataclass(frozen=True)
class ArmTickInfo:
    """Per-arm detail of one streamer tick."""

    alpha: float
    ingress_gap_ns: int | None
    src_recv_ns: int
    src_meta: Any
    lin_clamped: bool
    ang_clamped: bool
    pose: Pose | None = None
    """The pose actually commanded this tick, post-clamp."""
    held: bool = False


@dataclass(frozen=True)
class TickInfo:
    """One streamer tick, handed to ``on_tick`` after the send completes."""

    t_streamer_tick_ns: int
    t_ws_send_ns: int | None
    arms: Mapping[str, ArmTickInfo]
    sent: bool


class ArmTrack:
    """The two newest samples for one arm, plus delayed interpolation.

    Deliberately holds exactly two: a longer history would let a stalled arm
    resume from an old segment, which reads as a lurch.
    """

    __slots__ = ("_prev", "_new", "max_delay_ns", "out_of_order")

    def __init__(self, max_delay_ns: int) -> None:
        self._prev: Sample | None = None
        self._new: Sample | None = None
        self.max_delay_ns = int(max_delay_ns)
        self.out_of_order = 0

    @property
    def newest(self) -> Sample | None:
        return self._new

    @property
    def previous(self) -> Sample | None:
        return self._prev

    @property
    def ready(self) -> bool:
        """True once two distinct samples allow real interpolation."""
        return self._prev is not None and self._new is not None

    def clear(self) -> None:
        self._prev = None
        self._new = None

    def submit(self, pose: PoseLike, t_recv_ns: int, meta: Any = None) -> bool:
        """Add a target. Returns False if it was dropped as out of order.

        Non-monotonic arrivals are dropped rather than reordered: they would
        invert the interpolation segment and drive the arm backwards.
        """
        t = int(t_recv_ns)
        if self._new is not None and t <= self._new.t_recv_ns:
            self.out_of_order += 1
            return False
        self._prev = self._new
        self._new = Sample(as_pose(pose), t, meta)
        return True

    def ingress_interval_ns(self) -> int | None:
        """Gap between the two newest arrivals, or None if we only have one."""
        if self._prev is None or self._new is None:
            return None
        return self._new.t_recv_ns - self._prev.t_recv_ns

    def sample(self, now_ns: int) -> Pose | None:
        """Interpolated target for ``now_ns``; None if nothing has arrived yet."""
        detail = self.sample_detail(now_ns)
        return None if detail is None else detail.pose

    def sample_detail(self, now_ns: int) -> "SampleDetail | None":
        """As :meth:`sample`, also reporting alpha and the ingress gap.

        Renders at ``now - delay`` where ``delay`` is the measured ingress
        interval (capped by ``max_delay_ns``), so the render time normally sits
        inside ``[t_prev, t_new]``. ``alpha`` is clamped to ``[0, 1]``: if
        ingress stalls we hold at the newest target instead of extrapolating.
        """
        if self._new is None:
            return None
        if self._prev is None:
            return SampleDetail(self._new.pose, 1.0, None, self._new)
        span = self._new.t_recv_ns - self._prev.t_recv_ns
        if span <= 0:  # defensive; submit() rejects this
            return SampleDetail(self._new.pose, 1.0, span, self._new)
        delay = min(span, self.max_delay_ns)
        render_ns = int(now_ns) - delay
        alpha = (render_ns - self._prev.t_recv_ns) / span
        alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
        return SampleDetail(pose_lerp(self._prev.pose, self._new.pose, alpha),
                            alpha, span, self._new)


def apply_step_clamp(
    last: PoseLike | None,
    target: PoseLike,
    max_lin: float,
    max_ang: float,
) -> tuple[Pose, bool, bool]:
    """Limit one tick's motion to ``max_lin`` metres and ``max_ang`` radians.

    Returns ``(pose, lin_clamped, ang_clamped)``. With no previous command
    there is nothing to rate-limit against, so the target passes through.

    Uses :func:`~tron_arm.poses.quat_angle`, which does not fold sign, so an
    antipodal pair is treated as the ~2*pi rotation it literally is and gets
    clamped hard rather than silently snapping.
    """
    tgt = as_pose(target)
    if last is None:
        return tgt, False, False
    prev = as_pose(last)
    if not (np.isfinite(max_lin) and np.isfinite(max_ang)) or max_lin <= 0.0 or max_ang <= 0.0:
        raise ValueError(f"step limits must be positive and finite, got {(max_lin, max_ang)!r}")

    p = tgt.p
    lin_clamped = False
    delta = tgt.p - prev.p
    dist = float(np.linalg.norm(delta))
    if dist > max_lin:
        p = prev.p + delta * (max_lin / dist)
        lin_clamped = True

    q = tgt.q_wxyz
    ang_clamped = False
    angle = quat_angle(prev.q_wxyz, tgt.q_wxyz)
    if angle > max_ang:
        q = slerp(prev.q_wxyz, tgt.q_wxyz, max_ang / angle)
        ang_clamped = True

    if not (lin_clamped or ang_clamped):
        return tgt, False, False
    return Pose(p, q), lin_clamped, ang_clamped


@dataclass
class StreamerStats:
    """Loop health. ``jitter_p95_ms`` is the tail that matters for smoothness."""

    ticks: int = 0
    sends: int = 0
    idle_ticks: int = 0
    late_ticks: int = 0
    nan_rejected: int = 0
    lin_clamped: int = 0
    ang_clamped: int = 0
    out_of_order: int = 0
    submitted: int = 0
    dropped_backpressure: int = 0
    first_tick_ns: int | None = None
    last_tick_ns: int | None = None
    _intervals_ns: Deque[int] = field(default_factory=lambda: deque(maxlen=8192), repr=False)

    @property
    def elapsed_s(self) -> float:
        if self.first_tick_ns is None or self.last_tick_ns is None:
            return 0.0
        return (self.last_tick_ns - self.first_tick_ns) / 1e9

    @property
    def achieved_rate_hz(self) -> float:
        """Ticks per second measured across the run (0.0 before two ticks)."""
        elapsed = self.elapsed_s
        if elapsed <= 0.0 or self.ticks < 2:
            return 0.0
        return (self.ticks - 1) / elapsed

    def jitter_p95_ms(self, nominal_period_ns: int) -> float:
        """95th percentile of ``|interval - nominal|`` across observed ticks."""
        if not self._intervals_ns:
            return 0.0
        deviations = np.abs(np.asarray(self._intervals_ns, dtype=np.float64) - nominal_period_ns)
        return float(np.percentile(deviations, 95)) / 1e6

    def as_dict(self, nominal_period_ns: int) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "sends": self.sends,
            "idle_ticks": self.idle_ticks,
            "late_ticks": self.late_ticks,
            "nan_rejected": self.nan_rejected,
            "lin_clamped": self.lin_clamped,
            "ang_clamped": self.ang_clamped,
            "out_of_order": self.out_of_order,
            "submitted": self.submitted,
            "dropped_backpressure": self.dropped_backpressure,
            "elapsed_s": round(self.elapsed_s, 4),
            "achieved_rate_hz": round(self.achieved_rate_hz, 3),
            "jitter_p95_ms": round(self.jitter_p95_ms(nominal_period_ns), 4),
        }


class PoseStreamer:
    """Paces interpolated targets to a send callback at ``servop.rate_hz``.

    The loop schedules against absolute deadlines derived from a fixed start
    time (``t0 + n * period``) rather than sleeping ``period`` each time, so
    scheduling latency does not accumulate into drift. If a whole slot is
    missed, the tick index resynchronises to the present instead of trying to
    replay the backlog.
    """

    def __init__(
        self,
        config: Config,
        send: SendCallback,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_delay_s: float = DEFAULT_MAX_DELAY_S,
        on_tick: Callable[["TickInfo"], None] | None = None,
        on_fault: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._config = config
        self._send = send
        self._clock = clock
        self._sleep = sleep
        self._period_ns = config.servop.period_ns
        self._max_lin, self._max_ang = config.max_step
        max_delay_ns = int(max_delay_s * 1e9)
        self._tracks: dict[str, ArmTrack] = {arm: ArmTrack(max_delay_ns) for arm in ARMS}
        self._last_sent: dict[str, Pose | None] = {arm: None for arm in ARMS}
        self._on_tick = on_tick
        self._on_fault = on_fault
        self._sending = False
        self._last_tick_detail: dict[str, ArmTickInfo] = {}
        self.fault: BaseException | None = None
        self._frozen = False
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self.stats = StreamerStats()

    # -- properties ------------------------------------------------------
    @property
    def period_ns(self) -> int:
        return self._period_ns

    @property
    def max_step(self) -> tuple[float, float]:
        """Per-tick ``(linear_m, angular_rad)`` ceiling."""
        return self._max_lin, self._max_ang

    @property
    def running(self) -> bool:
        return self._running

    def last_sent(self, arm: str) -> Pose | None:
        return self._last_sent[arm]

    # -- ingress ---------------------------------------------------------
    def submit(self, arm: str, pose: PoseLike, t_recv_ns: int, meta: Any = None) -> bool:
        """Queue a new target for ``arm``, stamped with its arrival time.

        ``t_recv_ns`` must be the frame's ``recv_monotonic_ns``, never a
        publisher wall clock.
        """
        if arm not in self._tracks:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        self.stats.submitted += 1
        accepted = self._tracks[arm].submit(pose, t_recv_ns, meta)
        if not accepted:
            self.stats.out_of_order += 1
        else:
            self._frozen = False
        return accepted

    def seed(self, arm: str, pose: PoseLike) -> None:
        """Seed the step clamp with the arm's *current* pose.

        Without this, the first tick has no previous command to rate-limit
        against and the first target passes through unclamped -- i.e. the arm
        would be told to jump straight to it. Seed from
        :meth:`~tron_arm.tron2_client.Tron2Client.prime_frozen_poses` before
        streaming so even the first command obeys the velocity limit.
        """
        if arm not in self._tracks:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        self._last_sent[arm] = as_pose(pose)

    def seed_all(self, poses: Mapping[str, PoseLike]) -> None:
        """Seed both arms, e.g. straight from ``get_move_pose()``."""
        for arm, pose in poses.items():
            self.seed(arm, pose)

    def freeze(self) -> None:
        """Hold the last commanded pose (an upstream ``on_hold``).

        Keeps commanding, which is what "hold position" means for a servo
        stream; it does not stop the loop.
        """
        self._frozen = True

    def clear(self, arm: str | None = None) -> None:
        """Drop buffered targets, e.g. on a ``pose_epoch``/``stream_id`` change."""
        for name in (ARMS if arm is None else (arm,)):
            if name not in self._tracks:
                raise ValueError(f"unknown arm {name!r}")
            self._tracks[name].clear()

    # -- per-tick maths (pure) -------------------------------------------
    def sample(self, arm: str, now_ns: int) -> Pose | None:
        """Interpolated (unclamped) target for one arm. No side effects."""
        return self._tracks[arm].sample(now_ns)

    def step(self, now_ns: int) -> dict[str, Pose]:
        """Compute this tick's commands: interpolate, NaN-guard, step-clamp.

        Updates ``last_sent`` for every arm it returns. Arms with no data (or
        whose target failed the NaN guard) are absent from the result; a frozen
        streamer re-emits the last commanded pose.
        """
        out: dict[str, Pose] = {}
        self._last_tick_detail = {}
        for arm in ARMS:
            if self._frozen:
                held = self._last_sent[arm]
                if held is not None:
                    out[arm] = held
                continue
            detail = self._tracks[arm].sample_detail(now_ns)
            if detail is None:
                continue
            target = detail.pose
            if not (np.all(np.isfinite(target.p)) and np.all(np.isfinite(target.q_wxyz))):
                # Unreachable via Pose (its constructor rejects non-finite input),
                # but this is the guard hard rule 3 asks for and it is cheap.
                self.stats.nan_rejected += 1
                held = self._last_sent[arm]
                if held is not None:
                    out[arm] = held
                continue
            clamped, lin_hit, ang_hit = apply_step_clamp(
                self._last_sent[arm], target, self._max_lin, self._max_ang
            )
            self.stats.lin_clamped += int(lin_hit)
            self.stats.ang_clamped += int(ang_hit)
            self._last_tick_detail[arm] = ArmTickInfo(
                alpha=detail.alpha,
                ingress_gap_ns=detail.ingress_gap_ns,
                src_recv_ns=detail.source.t_recv_ns,
                src_meta=detail.source.meta,
                lin_clamped=lin_hit,
                ang_clamped=ang_hit,
                pose=clamped,
            )
            out[arm] = clamped
        for arm, pose in out.items():
            self._last_sent[arm] = pose
        return out

    async def tick(self, now_ns: int) -> dict[str, Pose]:
        """One tick: :meth:`step` then dispatch to the send callback."""
        self._note_tick(now_ns)
        commands = self.step(now_ns)
        if not commands:
            self.stats.idle_ticks += 1
            self._emit_tick(now_ns, None, sent=False)
            return commands
        if self._sending:
            # The previous send has not completed. Awaiting it would make the
            # pacing loop hostage to the socket, and queueing behind it would
            # burst a run of stale targets when the socket drained. Drop this
            # tick: the next one carries a fresher target anyway.
            self.stats.dropped_backpressure += 1
            self._emit_tick(now_ns, None, sent=False)
            return commands
        # Dispatched as a task, NOT awaited: one slow send must not stall the
        # clock. The in-flight flag is what bounds it to a single outstanding
        # frame, so this cannot pile up tasks either.
        self._sending = True
        detail = dict(self._last_tick_detail)
        asyncio.ensure_future(
            self._send_and_report(commands.get("left"), commands.get("right"), now_ns, detail)
        )
        self.stats.sends += 1
        return commands

    async def _send_and_report(self, left: Pose | None, right: Pose | None,
                               tick_ns: int, detail: dict[str, ArmTickInfo]) -> None:
        """Send one frame, then report it. Failures become a fault, not silence."""
        try:
            await self._send(left, right)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.fault = exc
            self._running = False
            if self._on_fault is not None:
                with contextlib.suppress(Exception):
                    self._on_fault(exc)
            return
        finally:
            self._sending = False
        if self._on_tick is not None:
            with contextlib.suppress(Exception):
                self._on_tick(TickInfo(tick_ns, self._clock(), detail, True))

    def _emit_tick(self, tick_ns: int, send_ns: int | None, *, sent: bool) -> None:
        if self._on_tick is None:
            return
        try:
            self._on_tick(TickInfo(tick_ns, send_ns, dict(self._last_tick_detail), sent))
        except Exception:  # noqa: BLE001 - instrumentation must never stall the loop
            pass

    def _note_tick(self, now_ns: int) -> None:
        if self.stats.first_tick_ns is None:
            self.stats.first_tick_ns = now_ns
        elif self.stats.last_tick_ns is not None:
            self.stats._intervals_ns.append(now_ns - self.stats.last_tick_ns)
        self.stats.last_tick_ns = now_ns
        self.stats.ticks += 1

    # -- the loop --------------------------------------------------------
    async def run(self, *, stop: asyncio.Event | None = None, max_ticks: int | None = None) -> None:
        """Run the paced loop until ``stop`` is set (or ``max_ticks`` elapse)."""
        self._running = True
        start_ns = self._clock()
        n = 0
        try:
            while self._running and not (stop is not None and stop.is_set()):
                if max_ticks is not None and self.stats.ticks >= max_ticks:
                    break
                n += 1
                deadline = start_ns + n * self._period_ns
                now = self._clock()
                if now < deadline:
                    await self._sleep((deadline - now) / 1e9)
                    now = self._clock()
                if now > deadline + self._period_ns:
                    # Missed at least one whole slot: count them and resync the
                    # index to the present so the deadline never accumulates lag.
                    missed = (now - deadline) // self._period_ns
                    self.stats.late_ticks += int(missed)
                    n += int(missed)
                try:
                    await self.tick(now)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # The socket died mid-motion. There is no reconnect by
                    # design: silently resuming would resume streaming against
                    # origins latched before the gap, and the robot's behaviour
                    # on stream loss is undocumented (LimX Q3). Stop, and tell
                    # the sink so it can FAULT and demand a manual re-arm.
                    self.fault = exc
                    if self._on_fault is not None:
                        with contextlib.suppress(Exception):
                            self._on_fault(exc)
                    break
        finally:
            self._running = False

    async def start(self, *, stop: asyncio.Event | None = None) -> asyncio.Task[None]:
        """Run :meth:`run` as a background task."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("streamer already running")
        self._task = asyncio.create_task(self.run(stop=stop), name="tron2-streamer")
        return self._task

    async def stop(self) -> None:
        """Stop the loop and await the background task."""
        self._running = False
        task, self._task = self._task, None
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    def stats_dict(self) -> dict[str, Any]:
        stats = self.stats.as_dict(self._period_ns)
        stats["target_rate_hz"] = self._config.servop.rate_hz
        stats["max_step_lin_m"] = self._max_lin
        stats["max_step_ang_rad"] = self._max_ang
        return stats
