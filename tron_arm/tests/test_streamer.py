"""Pacing loop, delayed interpolation, step clamping, NaN guard and stats."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from tests.conftest import VirtualClock, collector, run, tweak, unchecked_pose
from tron_arm.poses import Pose, quat_angle
from tron_arm.streamer import ArmTrack, PoseStreamer, apply_step_clamp

MS = 1_000_000


def pose_at(x: float) -> Pose:
    return Pose([x, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])


class TestArmTrack:
    def test_holds_only_the_two_newest(self):
        track = ArmTrack(100 * MS)
        for i in range(5):
            track.submit(pose_at(float(i)), i * 10 * MS)
        assert track.previous.pose.p[0] == 3.0
        assert track.newest.pose.p[0] == 4.0

    def test_returns_none_before_any_sample(self):
        assert ArmTrack(100 * MS).sample(0) is None

    def test_single_sample_is_held(self):
        track = ArmTrack(100 * MS)
        track.submit(pose_at(1.0), 0)
        np.testing.assert_allclose(track.sample(999 * MS).p, [1.0, 0.0, 0.0])

    def test_one_ingress_interval_of_delay(self):
        """At the instant a sample arrives we render the *previous* one."""
        track = ArmTrack(100 * MS)
        track.submit(pose_at(0.0), 0)
        track.submit(pose_at(1.0), 20 * MS)
        np.testing.assert_allclose(track.sample(20 * MS).p[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(track.sample(30 * MS).p[0], 0.5, atol=1e-12)
        np.testing.assert_allclose(track.sample(40 * MS).p[0], 1.0, atol=1e-12)

    def test_holds_instead_of_extrapolating_when_ingress_stalls(self):
        track = ArmTrack(100 * MS)
        track.submit(pose_at(0.0), 0)
        track.submit(pose_at(1.0), 20 * MS)
        for now in (40 * MS, 100 * MS, 5_000 * MS):
            assert track.sample(now).p[0] == pytest.approx(1.0)

    def test_ingress_interval(self):
        track = ArmTrack(100 * MS)
        assert track.ingress_interval_ns() is None
        track.submit(pose_at(0.0), 0)
        assert track.ingress_interval_ns() is None
        track.submit(pose_at(1.0), 25 * MS)
        assert track.ingress_interval_ns() == 25 * MS

    def test_delay_is_capped(self):
        """A long dropout must not set a multi-second interpolation delay."""
        track = ArmTrack(50 * MS)
        track.submit(pose_at(0.0), 0)
        track.submit(pose_at(1.0), 1_000 * MS)
        # Delay capped at 50 ms, so at t=1000 ms we are 950/1000 through.
        assert track.sample(1_000 * MS).p[0] == pytest.approx(0.95)

    def test_out_of_order_submissions_are_dropped(self):
        track = ArmTrack(100 * MS)
        track.submit(pose_at(0.0), 10 * MS)
        track.submit(pose_at(1.0), 20 * MS)
        assert track.submit(pose_at(9.0), 15 * MS) is False
        assert track.out_of_order == 1
        assert track.newest.pose.p[0] == 1.0

    def test_duplicate_timestamp_is_dropped(self):
        track = ArmTrack(100 * MS)
        track.submit(pose_at(0.0), 10 * MS)
        assert track.submit(pose_at(1.0), 10 * MS) is False

    def test_clear(self):
        track = ArmTrack(100 * MS)
        track.submit(pose_at(0.0), 0)
        track.clear()
        assert track.sample(0) is None


class TestInterpolationAccuracy:
    def test_tracks_a_synthetic_30_to_60hz_jittered_stream(self, config):
        """Ingress at a jittered 30-60 Hz; the 100 Hz samples must follow the
        ground-truth path (delayed by one ingress interval)."""
        rng = np.random.default_rng(20240812)
        omega = 2.0 * math.pi * 0.25  # a slow 0.25 Hz circle

        def truth(t_s: float) -> np.ndarray:
            return np.array([0.4 + 0.05 * math.cos(omega * t_s),
                             0.05 * math.sin(omega * t_s), 0.1])

        streamer = PoseStreamer(config, collector()[1])
        t_ns = 0
        arrivals: list[int] = []
        while t_ns < 3_000 * MS:
            streamer.submit("right", Pose(truth(t_ns / 1e9), [1, 0, 0, 0]), t_ns)
            arrivals.append(t_ns)
            t_ns += int(rng.uniform(1.0 / 60.0, 1.0 / 30.0) * 1e9)

        # Replay: at each 100 Hz tick, compare against truth at the delayed time.
        errors = []
        for tick in range(0, 3_000 * MS, 10 * MS):
            visible = [a for a in arrivals if a <= tick]
            if len(visible) < 2:
                continue
            t0, t1 = visible[-2], visible[-1]
            track = ArmTrack(int(0.1e9))
            track.submit(Pose(truth(t0 / 1e9), [1, 0, 0, 0]), t0)
            track.submit(Pose(truth(t1 / 1e9), [1, 0, 0, 0]), t1)
            got = track.sample(tick)
            # Mirror the implementation: render at tick-delay, clamped to [t0, t1].
            render_s = min(max(t0, tick - (t1 - t0)), t1) / 1e9
            errors.append(float(np.linalg.norm(got.p - truth(render_s))))

        errors = np.asarray(errors)
        # Error is pure chord-vs-arc on a 33 ms segment of a 0.05 m radius circle.
        assert errors.max() < 5e-5, f"max interpolation error {errors.max():.2e} m"

    def test_interpolates_rotation_with_slerp(self, config):
        streamer = PoseStreamer(config, collector()[1])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])
        streamer.submit("left", Pose([0.4, 0.2, 0], q0), 0)
        streamer.submit("left", Pose([0.4, 0.2, 0], q1), 20 * MS)
        mid = streamer.sample("left", 30 * MS)
        # q1's scalar is cos(pi/8), i.e. a pi/4 rotation; half of it is pi/8.
        assert quat_angle(q0, q1) == pytest.approx(math.pi / 4, abs=1e-12)
        assert quat_angle(q0, mid.q_wxyz) == pytest.approx(math.pi / 8, abs=1e-9)


class TestStepClamp:
    def test_first_command_passes_through(self):
        target = pose_at(100.0)
        got, lin, ang = apply_step_clamp(None, target, 0.004, 0.012)
        assert (lin, ang) == (False, False)
        assert got.p[0] == 100.0

    def test_linear_clamp_limits_distance_and_keeps_direction(self):
        last = Pose([0.0, 0.0, 0.0], [1, 0, 0, 0])
        target = Pose([0.3, 0.4, 0.0], [1, 0, 0, 0])  # 0.5 m away
        got, lin, ang = apply_step_clamp(last, target, 0.004, 0.012)
        assert lin and not ang
        assert float(np.linalg.norm(got.p)) == pytest.approx(0.004)
        np.testing.assert_allclose(got.p / 0.004, [0.6, 0.8, 0.0], atol=1e-12)

    def test_linear_step_just_under_the_limit_is_untouched(self):
        last = pose_at(0.0)
        got, lin, ang = apply_step_clamp(last, pose_at(0.0039), 0.004, 0.012)
        assert not lin and got.p[0] == pytest.approx(0.0039)

    def test_angular_clamp_limits_rotation(self):
        last = Pose([0, 0, 0], [1.0, 0.0, 0.0, 0.0])
        target = Pose([0, 0, 0], [math.cos(0.5), 0.0, 0.0, math.sin(0.5)])  # 1.0 rad
        got, lin, ang = apply_step_clamp(last, target, 0.004, 0.012)
        assert ang and not lin
        assert quat_angle(last.q_wxyz, got.q_wxyz) == pytest.approx(0.012, abs=1e-9)

    def test_both_axes_clamp_together(self):
        last = Pose([0, 0, 0], [1.0, 0.0, 0.0, 0.0])
        target = Pose([1.0, 0, 0], [math.cos(0.5), 0.0, 0.0, math.sin(0.5)])
        got, lin, ang = apply_step_clamp(last, target, 0.004, 0.012)
        assert lin and ang
        assert float(np.linalg.norm(got.p)) == pytest.approx(0.004)
        assert quat_angle(last.q_wxyz, got.q_wxyz) == pytest.approx(0.012, abs=1e-9)

    def test_max_step_derives_from_velocity_and_rate(self, config):
        lin, ang = config.max_step
        assert lin == pytest.approx(config.velocity.lin / config.servop.rate_hz)
        assert ang == pytest.approx(config.velocity.ang / config.servop.rate_hz)
        assert (lin, ang) == pytest.approx((0.004, 0.012))

    def test_repeated_clamped_steps_converge_at_the_velocity_limit(self, config):
        """N ticks of clamping must cover exactly N * max_step metres."""
        max_lin, max_ang = config.max_step
        current = pose_at(0.0)
        target = pose_at(1.0)
        for _ in range(10):
            current, _, _ = apply_step_clamp(current, target, max_lin, max_ang)
        assert current.p[0] == pytest.approx(10 * max_lin)

    def test_rejects_non_positive_limits(self):
        with pytest.raises(ValueError):
            apply_step_clamp(pose_at(0.0), pose_at(1.0), 0.0, 0.012)


class TestStep:
    def test_step_applies_the_clamp_and_records_stats(self, config):
        streamer = PoseStreamer(config, collector()[1])
        streamer.submit("right", pose_at(0.0), 0)
        streamer.submit("right", pose_at(10.0), 10 * MS)
        streamer.step(10 * MS)   # settles at the first sample
        out = streamer.step(30 * MS)  # far target -> clamped
        assert streamer.stats.lin_clamped >= 1
        assert float(np.linalg.norm(out["right"].p - np.array([0.0, 0, 0]))) <= config.max_step[0] + 1e-12

    def test_arms_without_data_are_absent(self, config):
        streamer = PoseStreamer(config, collector()[1])
        streamer.submit("right", pose_at(0.4), 0)
        assert sorted(streamer.step(10 * MS)) == ["right"]

    def test_nan_guard_holds_last_command_and_counts(self, config):
        streamer = PoseStreamer(config, collector()[1])
        streamer.submit("right", Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]), 0)
        first = streamer.step(1 * MS)
        # Poison the track, bypassing Pose validation, to exercise the guard.
        streamer._tracks["right"]._new = type(streamer._tracks["right"]._new)(
            unchecked_pose([np.nan, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]), 2 * MS
        )
        streamer._tracks["right"]._prev = None
        out = streamer.step(3 * MS)
        assert streamer.stats.nan_rejected == 1
        np.testing.assert_allclose(out["right"].p, first["right"].p)

    def test_freeze_re_emits_the_last_command(self, config):
        streamer = PoseStreamer(config, collector()[1])
        streamer.submit("right", pose_at(0.4), 0)
        first = streamer.step(1 * MS)
        streamer.freeze()
        for _ in range(5):
            held = streamer.step(50 * MS)
        np.testing.assert_allclose(held["right"].p, first["right"].p)

    def test_clear_drops_targets(self, config):
        streamer = PoseStreamer(config, collector()[1])
        streamer.submit("right", pose_at(0.4), 0)
        streamer.clear("right")
        assert streamer.step(10 * MS) == {}

    def test_submit_rejects_unknown_arm(self, config):
        with pytest.raises(ValueError, match="unknown arm"):
            PoseStreamer(config, collector()[1]).submit("middle", pose_at(0.0), 0)


class TestLoop:
    def test_rate_and_drift_correction_on_a_virtual_clock(self, config):
        """Deadlines come from ``t0 + n*period``, so lag must not accumulate."""
        clock = VirtualClock()
        sent, send = collector()
        streamer = PoseStreamer(config, send, clock=clock, sleep=clock.sleep)
        streamer.submit("right", pose_at(0.4), 0)
        run(streamer.run(max_ticks=200))
        assert streamer.stats.ticks == 200
        # 200 ticks at 100 Hz on a perfect clock == exactly 2.0 s of virtual time.
        assert clock.now_ns == pytest.approx(200 * 10 * MS, rel=1e-9)
        assert streamer.stats.late_ticks == 0
        assert len(sent) == 200

    def test_late_ticks_are_counted_and_the_schedule_resyncs(self, config):
        clock = VirtualClock()
        stall = {"done": False}

        async def sleep(seconds: float) -> None:
            clock.advance_s(seconds)
            if not stall["done"]:
                stall["done"] = True
                clock.advance_s(0.5)  # a 50-slot overrun
            await asyncio.sleep(0)

        streamer = PoseStreamer(config, collector()[1], clock=clock, sleep=sleep)
        streamer.submit("right", pose_at(0.4), 0)
        run(streamer.run(max_ticks=100))
        assert streamer.stats.late_ticks >= 49
        # Despite the stall, the loop ends near t0 + (ticks + missed) * period,
        # i.e. it resynced instead of trying to replay the backlog.
        assert clock.now_ns == pytest.approx(150 * 10 * MS, rel=1e-6)

    def test_idle_ticks_do_not_send(self, config):
        clock = VirtualClock()
        sent, send = collector()
        streamer = PoseStreamer(config, send, clock=clock, sleep=clock.sleep)
        run(streamer.run(max_ticks=10))  # nothing ever submitted
        assert sent == []
        assert streamer.stats.idle_ticks == 10

    def test_achieved_rate_and_jitter_stats(self, config):
        clock = VirtualClock()
        streamer = PoseStreamer(config, collector()[1], clock=clock, sleep=clock.sleep)
        streamer.submit("right", pose_at(0.4), 0)
        run(streamer.run(max_ticks=100))
        stats = streamer.stats_dict()
        assert stats["achieved_rate_hz"] == pytest.approx(100.0, rel=1e-6)
        assert stats["jitter_p95_ms"] == pytest.approx(0.0, abs=1e-6)
        assert stats["target_rate_hz"] == 100.0

    def test_jitter_p95_reflects_real_deviation(self, config):
        clock = VirtualClock()
        n = {"i": 0}

        async def sleep(seconds: float) -> None:
            n["i"] += 1
            clock.advance_s(seconds + (0.005 if n["i"] % 10 == 0 else 0.0))
            await asyncio.sleep(0)

        streamer = PoseStreamer(config, collector()[1], clock=clock, sleep=sleep)
        streamer.submit("right", pose_at(0.4), 0)
        run(streamer.run(max_ticks=100))
        assert streamer.stats.jitter_p95_ms(streamer.period_ns) > 1.0

    def test_runs_at_the_configured_rate_in_real_time(self, config):
        """Wall-clock smoke test with the real clock and real asyncio.sleep."""
        cfg = tweak(config, rate_hz=200.0)
        streamer = PoseStreamer(cfg, collector()[1])
        streamer.submit("right", pose_at(0.4), 0)

        async def body():
            stop = asyncio.Event()
            task = await streamer.start(stop=stop)
            await asyncio.sleep(0.5)
            stop.set()
            await task

        run(body())
        assert 150.0 < streamer.stats.achieved_rate_hz < 250.0
        assert streamer.stats.ticks > 50
