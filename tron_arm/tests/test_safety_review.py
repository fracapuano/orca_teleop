"""Adversarial tests for the failure modes documented in REFERENCE.md.

These are not "does it work" tests. Each one tries to *break* a property that
something else silently depends on, and most would still pass if the feature
they guard were deleted -- which is the point: they fail when a future edit
reintroduces a hazard.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import math
import pathlib
import random
import re

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import at, collector, fast_close, run, tweak
from tron_arm.arm_state import ArmController, ArmState, RobotState, TickEvents
from tron_arm.config import ARMS, load_config
from tron_arm.mock_robot import MockTron2
from tron_arm.poses import Pose, quat_angle, slerp
from tron_arm.sink import TronArmSink
from tron_arm.streamer import PoseStreamer, apply_step_clamp
from tron_arm.tron2_client import Tron2Client, Tron2NotConnected, encode_servop_element

PKG = pathlib.Path(__file__).resolve().parent.parent / "tron_arm"
TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"
SETTINGS = settings(max_examples=150, deadline=None,
                    suppress_health_check=[HealthCheck.function_scoped_fixture])


# =====================================================================
# (a) Nothing gates on tracking_valid
# =====================================================================
class TestTrackingValidIsNeverGated:
    """`tracking_valid` is a constant-true trap. Absence of frames IS tracking
    loss; upstream's worker already gates on it. If WE gated on it too, a
    publisher that ever sets it False would freeze the arm for a reason we
    cannot see."""

    def test_no_source_file_reads_tracking_valid(self):
        offenders = []
        for path in list(PKG.glob("*.py")) + list(TOOLS.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and node.attr == "tracking_valid":
                    offenders.append(f"{path.name}:{node.lineno}")
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) > 1
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value == "tracking_valid"):
                    offenders.append(f"{path.name}:{node.lineno} (getattr)")
        assert not offenders, f"code reads tracking_valid: {offenders}"

    @pytest.mark.parametrize("tracking_valid", [True, False])
    def test_frames_are_dispatched_identically_either_way(self, config, tracking_valid):
        """A False flag must change nothing about what we command."""
        controller = ArmController(config)
        robot = RobotState("right", Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]))

        @dataclasses.dataclass(frozen=True)
        class Frame:
            wrist: Pose
            handedness: str = "right"
            recv_monotonic_ns: int = 0
            tracking_valid: bool = True

        controller.tick(Frame(Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]),
                              tracking_valid=tracking_valid),
                        TickEvents(clutch=True), robot)
        got = controller.tick(Frame(Pose([0.11, 0.2, 0.3], [1, 0, 0, 0]),
                                    tracking_valid=tracking_valid), None, robot)
        assert got.target is not None, "a frame was dropped because of tracking_valid"
        assert controller.state is ArmState.ENGAGED


# =====================================================================
# (b) timestamp_ns never feeds control maths
# =====================================================================
class TestTimestampNsNeverInControlMath:
    """`timestamp_ns` is a publisher wall clock from another machine: it carries
    NTP skew, can step, and can run backwards. Differencing it can yield a
    negative dt and an enormous commanded velocity."""

    def test_no_source_file_references_timestamp_ns(self):
        """AST-level, so the docstrings explaining WHY we ignore it don't trip it."""
        offenders = []
        for path in list(PKG.glob("*.py")) + list(TOOLS.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "timestamp_ns":
                    offenders.append(f"{path.name}:{node.lineno} attribute access")
                elif isinstance(node, ast.Name) and node.id == "timestamp_ns":
                    offenders.append(f"{path.name}:{node.lineno} name")
                elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and "timestamp_ns" in node.value
                        and "\n" not in node.value):  # a string literal, not a docstring
                    offenders.append(f"{path.name}:{node.lineno} string literal")
        assert not offenders, "timestamp_ns reachable from code:\n" + "\n".join(offenders)

    def test_only_recv_monotonic_ns_is_read_from_frames(self):
        source = (PKG / "sink.py").read_text()
        assert "recv_monotonic_ns" in source
        assert 'getattr(frame, "timestamp' not in source

    def test_a_backwards_timestamp_changes_nothing(self, config):
        """The trap: a frame whose wall clock went backwards must still work."""
        controller = ArmController(config)
        robot = RobotState("right", Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]))

        @dataclasses.dataclass(frozen=True)
        class Frame:
            wrist: Pose
            recv_monotonic_ns: int
            timestamp_ns: int
            handedness: str = "right"

        controller.tick(Frame(Pose([0.10, 0.2, 0.3], [1, 0, 0, 0]), 1_000_000, 9_999_999_999),
                        TickEvents(clutch=True), robot)
        got = controller.tick(
            Frame(Pose([0.104, 0.2, 0.3], [1, 0, 0, 0]), 2_000_000, 1),  # clock went backwards
            None, robot)
        assert got.target is not None
        moved = float(np.linalg.norm(got.target.position_m - np.array([0.4, -0.2, 0.0])))
        assert moved <= config.max_step[0] + 1e-12, "a backwards wall clock produced a jump"


# =====================================================================
# (c) No path re-canonicalises quaternion sign
# =====================================================================
class TestNoSignCanonicalisation:
    """Upstream aligns each quaternion to its predecessor and refuses to force
    w >= 0, because that manufactures a discontinuity at 180 deg. If we
    re-canonicalised anywhere, the arm would snap when the wrist passes that
    point -- routinely reachable."""

    def _continuous_negative_w_sequence(self, n: int = 40) -> list[np.ndarray]:
        """A sign-continuous sweep straddling 180 deg, seeded w-negative.

        Built exactly as upstream's QuaternionContinuity does: seed, then align
        each quaternion to its predecessor. w crosses zero as the rotation passes
        180 deg -- that crossing is precisely what forcing w >= 0 would break, so
        the fixture must contain it rather than avoid it.
        """
        quats: list[np.ndarray] = []
        previous = None
        for i in range(n):
            angle = math.pi * 0.9 + i * 0.01
            q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
            if previous is None:
                q = -q if q[0] > 0 else q      # seed on the negative-w branch
            elif float(np.dot(q, previous)) < 0.0:
                q = -q                          # align to predecessor
            quats.append(q)
            previous = q
        assert any(q[0] < 0 for q in quats), "fixture must contain w-negative quats"
        assert all(float(np.dot(a, b)) > 0 for a, b in zip(quats, quats[1:])), \
            "fixture must be sign-continuous"
        return quats

    def test_slerp_receives_the_bytes_it_was_given(self):
        """Byte-identical passthrough: nothing rewrites the sign en route."""
        seen: list[tuple[bytes, bytes]] = []
        import tron_arm.poses as poses

        original = poses.slerp

        def spy(q0, q1, t):
            seen.append((np.asarray(q0, dtype=np.float64).tobytes(),
                         np.asarray(q1, dtype=np.float64).tobytes()))
            return original(q0, q1, t)

        quats = self._continuous_negative_w_sequence()
        config = load_config()
        poses.slerp = spy
        try:
            import tron_arm.streamer as streamer_mod
            streamer_mod.slerp = spy
            streamer = PoseStreamer(config, collector()[1])
            for i, q in enumerate(quats):
                streamer.submit("right", Pose([0.4, -0.2, 0.0], q), i * 20_000_000)
                streamer.step(i * 20_000_000 + 25_000_000)
        finally:
            poses.slerp = original
            streamer_mod.slerp = original

        assert seen, "slerp was never reached"
        submitted = {q.tobytes() for q in quats}
        for a, b in seen:
            for payload in (a, b):
                if payload in submitted:
                    continue
                # Anything not byte-identical must not be the sign flip.
                arr = np.frombuffer(payload, dtype=np.float64)
                assert not any(np.allclose(arr, -q) and not np.allclose(arr, q)
                               for q in quats), f"a sign-flipped quaternion reached slerp: {arr}"

    def test_pose_construction_preserves_the_sign(self):
        """Pose normalises MAGNITUDE (sign untouched); it must never flip sign."""
        for q in self._continuous_negative_w_sequence():
            got = Pose([0.4, -0.2, 0.0], q).orientation_wxyz
            np.testing.assert_allclose(got, q, atol=1e-12)
            assert np.sign(got[0]) == np.sign(q[0]) or q[0] == 0.0

    def test_encoder_emits_the_sign_it_was_given(self):
        for q in self._continuous_negative_w_sequence():
            encoded = encode_servop_element(Pose([0.4, -0.2, 0.0], q), "pos_quat")
            np.testing.assert_allclose(encoded[3:], q, atol=0.0)

    def test_no_source_file_forces_w_positive(self):
        """Catch the idiom itself, so it cannot be reintroduced."""
        patterns = [r"if\s+.*\[0\]\s*<\s*0", r"dot\s*<\s*0.*[-]\s*q", r"np\.sign\(.*\[0\]\)"]
        offenders = []
        for path in PKG.glob("*.py"):
            text = path.read_text()
            # Blank out docstrings: they DESCRIBE the forbidden idiom on purpose,
            # and a scanner that trips on its own warning is useless.
            tree = ast.parse(text)
            lines = text.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                body = getattr(node, "body", None)
                if not body or not isinstance(body[0], ast.Expr):
                    continue
                if not isinstance(body[0].value, ast.Constant) or \
                        not isinstance(body[0].value.value, str):
                    continue
                for i in range(body[0].lineno - 1, body[0].end_lineno):
                    lines[i] = ""
            code = "\n".join(lines)
            for pattern in patterns:
                for match in re.finditer(pattern, code):
                    line = code[:match.start()].count("\n") + 1
                    offenders.append(f"{path.name}:{line}: {lines[line - 1].strip()}")
        assert not offenders, "possible sign canonicalisation:\n" + "\n".join(offenders)

    def test_slerp_through_180_degrees_stays_continuous(self):
        """The exact case forcing w>=0 would break."""
        quats = self._continuous_negative_w_sequence(60)
        out = [slerp(a, b, 0.5) for a, b in zip(quats, quats[1:])]
        steps = [quat_angle(a, b) for a, b in zip(out, out[1:])]
        assert max(steps) < 0.05, f"discontinuity through 180 deg: {max(steps):.4f} rad"


# =====================================================================
# (d) epoch-during-dropout, randomised timing
# =====================================================================
class TestEpochDuringDropoutOrdering:
    """The reference-change callback can arrive at any point relative to the
    frames that resume after a dropout. Every interleaving must end with a fresh
    origin and a continuous command stream."""

    @given(
        seed=st.integers(min_value=0, max_value=10_000),
        gap_frames=st.integers(min_value=1, max_value=6),
        callback_delay=st.integers(min_value=0, max_value=5),
        jump_m=st.floats(min_value=0.0, max_value=0.8, allow_nan=False),
    )
    @SETTINGS
    def test_any_interleaving_is_safe(self, config, seed, gap_frames, callback_delay, jump_m):
        rng = random.Random(seed)
        cfg = dataclasses.replace(config, scale=1.0)
        controller = ArmController(cfg)
        robot = RobotState("right", Pose([0.40, -0.20, 0.0], [1, 0, 0, 0]))
        max_lin, max_ang = cfg.max_step

        emitted: list[Pose] = []
        cleared_pending = False
        violations: list[str] = []
        t = 0

        def frame(x: float):
            @dataclasses.dataclass(frozen=True)
            class F:
                wrist: Pose
                handedness: str = "right"
                recv_monotonic_ns: int = 0

            return F(Pose([x, 0.2, 0.3], [1, 0, 0, 0]))

        def step(f, events=None):
            nonlocal cleared_pending, t
            before = controller.machine.clear_count
            result = controller.tick(f, events, robot)
            if controller.machine.clear_count > before:
                cleared_pending = True
            if result.target is not None:
                if cleared_pending and not result.diagnostics.latched_this_tick:
                    violations.append(f"t={t}: mapped with a stale origin")
                cleared_pending = False
                emitted.append(result.target)
            t += 1

        step(frame(0.10), TickEvents(clutch=True))
        for i in range(4):
            step(frame(0.10 + 0.002 * i))

        step(None, TickEvents(hold_reason="stale"))
        for _ in range(gap_frames):
            step(None)

        # Frames resume in a NEW epoch: the operator pose has jumped.
        base = 0.10 + jump_m
        for i in range(callback_delay):
            step(frame(base + 0.002 * i))
        # ... and only now does the callback land.
        step(frame(base + 0.002 * callback_delay),
             TickEvents(reference_change=(1, 2 + rng.randint(0, 3))))
        for i in range(4):
            step(frame(base + 0.002 * (callback_delay + 1 + i)))

        assert not violations, violations
        assert controller.machine.latch_count >= 2
        jumps = [float(np.linalg.norm(b.position_m - a.position_m))
                 for a, b in zip(emitted, emitted[1:])]
        if jumps:
            assert max(jumps) <= max_lin + 1e-9, (
                f"lurch of {max(jumps):.4f} m > clamp {max_lin} (jump={jump_m:.2f} m)")
        turns = [quat_angle(a.orientation_wxyz, b.orientation_wxyz)
                 for a, b in zip(emitted, emitted[1:])]
        if turns:
            assert max(turns) <= max_ang + 1e-9


# =====================================================================
# ServoP stream integrity
# =====================================================================
class TestStreamIntegrity:
    def test_a_slow_send_drops_rather_than_queues(self, config):
        """Backpressure must be drop-oldest. Queueing would deliver a burst of
        stale targets the moment the socket drained."""
        sent: list[float] = []
        release = asyncio.Event()

        async def slow_send(left, right):
            sent.append(0.0)
            await release.wait()          # never completes during the test

        streamer = PoseStreamer(config, slow_send)
        streamer.submit("right", Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]), 0)

        async def body():
            task = asyncio.create_task(streamer.run(max_ticks=25))
            await asyncio.sleep(0.35)
            streamer._running = False
            release.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=1.0)

        run(body())
        assert len(sent) == 1, f"{len(sent)} sends queued behind a stalled socket"
        assert streamer.stats.dropped_backpressure > 0, "ticks were not dropped"

    def test_a_send_failure_faults_instead_of_stopping_silently(self, config):
        """A dead socket mid-motion must be loud. Silence is the dangerous case."""
        faults: list[BaseException] = []

        async def dead_send(left, right):
            raise Tron2NotConnected("socket closed")

        streamer = PoseStreamer(config, dead_send, on_fault=faults.append)
        streamer.submit("right", Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]), 0)
        run(streamer.run(max_ticks=50))
        assert faults, "the stream died without telling anyone"
        assert streamer.fault is not None
        assert not streamer.running

    def test_the_sink_faults_both_arms_on_a_stream_fault(self, config):
        sink = TronArmSink(config)
        sink._on_stream_fault(Tron2NotConnected("socket closed"))
        for arm in ARMS:
            assert sink.controllers[arm].state is ArmState.FAULT

    def test_fault_requires_a_manual_re_arm(self, config):
        """No silent resume: frames alone must not clear a FAULT."""
        sink = TronArmSink(config)
        sink._on_stream_fault(Tron2NotConnected("dead"))
        controller = sink.controllers["right"]
        robot = RobotState("right", Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]))
        for _ in range(20):
            got = controller.tick(Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]),
                                  TickEvents(clutch=True), robot)
            assert got.target is None
        assert controller.state is ArmState.FAULT
        controller.tick(None, TickEvents(reset=True), robot)
        assert controller.state is ArmState.DISENGAGED

    def test_no_reconnect_logic_exists(self):
        """Reconnect must stay absent until someone decides what is safe.

        AST-level: the comments explaining why there is none must not trip it.
        """
        offenders = []
        for path in (PKG / "tron2_client.py", PKG / "sink.py", PKG / "streamer.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                name = None
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = node.name
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.Name):
                    name = node.id
                if name and "reconnect" in name.lower():
                    offenders.append(f"{path.name}:{node.lineno}: {name}")
        assert not offenders, f"reconnect logic appeared: {offenders}"


# =====================================================================
# Clutch races
# =====================================================================
class TestClutchRaces:
    @pytest.fixture
    def rig(self, config):
        controller = ArmController(config)
        robot = RobotState("right", Pose([0.40, -0.20, 0.0], [1, 0, 0, 0]))
        return controller, robot

    def _frame(self, x=0.10):
        @dataclasses.dataclass(frozen=True)
        class F:
            wrist: Pose
            handedness: str = "right"
            recv_monotonic_ns: int = 0

        return F(Pose([x, 0.2, 0.3], [1, 0, 0, 0]))

    def test_press_during_hold_requires_a_fresh_latch(self, rig):
        controller, robot = rig
        controller.tick(self._frame(), TickEvents(clutch=True), robot)
        controller.tick(None, TickEvents(hold_reason="stale"), robot)
        assert not controller.mapper.latched
        controller.tick(None, TickEvents(clutch=False), robot)
        controller.tick(None, TickEvents(clutch=True), robot)   # re-press while held
        assert not controller.mapper.latched, "re-press reused a stale origin"
        got = controller.tick(self._frame(0.9), None, robot)
        assert got.diagnostics.latched_this_tick

    def test_release_on_the_latch_frame_commands_nothing(self, rig):
        """Clutch released in the same tick as the frame that would latch."""
        controller, robot = rig
        got = controller.tick(self._frame(), TickEvents(clutch=False), robot)
        assert got.target is None
        assert not controller.mapper.latched

    def test_release_immediately_after_latching_clears_the_origin(self, rig):
        controller, robot = rig
        controller.tick(self._frame(), TickEvents(clutch=True), robot)
        assert controller.mapper.latched
        controller.tick(self._frame(), TickEvents(clutch=False), robot)
        assert not controller.mapper.latched
        assert controller.state is ArmState.DISENGAGED

    def test_both_arms_engaging_in_the_same_cycle_are_independent(self, config):
        """Two arms latch against their own origins, not each other's."""
        sink = TronArmSink(config)
        for arm, x in (("left", 0.10), ("right", 0.50)):
            @dataclasses.dataclass(frozen=True)
            class F:
                wrist: Pose
                handedness: str
                recv_monotonic_ns: int = 0

            sink._dispatch_inner(F(Pose([x, 0.2, 0.3], [1, 0, 0, 0]), arm), 0)
        left = sink.controllers["left"].mapper.t_op0
        right = sink.controllers["right"].mapper.t_op0
        assert left is not None and right is not None
        assert not np.allclose(left, right), "the arms shared an origin"

    def test_force_hold_beats_a_held_clutch(self, config):
        """SPACE must win even while the operator holds the clutch."""
        from tron_arm.clutch import ScriptedClutch

        sink = TronArmSink(config, clutch=ScriptedClutch(engaged=True))
        assert sink._engaged() is True
        sink.force_hold(True)
        assert sink._engaged() is False, "force-hold lost to a held clutch"
        sink.force_hold(False)
        assert sink._engaged() is True

    def test_session_logging_never_blocks_the_dispatch_thread(self, config, tmp_path):
        """Instrumentation must not cost control latency -- it did, once.

        Measured RELATIVE to the encoding it replaced, not as an absolute time:
        an absolute bound flakes on a loaded machine, while the property under
        test is simply "json.dumps and gzip are no longer on this path".
        """
        import json
        import time as _t

        from tron_arm.session import SessionLogger

        record = {"type": "tick", "i": 1, "p": [0.1, 0.2, 0.3],
                  "arms": {"right": {"alpha": 0.5, "t_dispatch": 123}}}

        started = _t.monotonic_ns()
        for _ in range(2000):
            json.dumps(record)
        encode_ns = (_t.monotonic_ns() - started) / 2000

        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        try:
            started = _t.monotonic_ns()
            for i in range(2000):
                log.write(record)
            write_ns = (_t.monotonic_ns() - started) / 2000
        finally:
            log.close()

        assert write_ns < encode_ns, (
            f"a log write ({write_ns / 1e3:.1f} us) costs more than encoding one "
            f"({encode_ns / 1e3:.1f} us) -- is serialisation back on the hot path?"
        )

    def test_a_full_log_queue_drops_instead_of_blocking(self, tmp_path):
        from tron_arm.session import QUEUE_MAXSIZE, SessionLogger

        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log._stop.set()               # stall the writer
        if log._writer is not None:
            log._writer.join(timeout=2.0)
        try:
            for i in range(QUEUE_MAXSIZE + 500):
                log.write({"type": "tick", "i": i})   # must never block
            assert log.dropped > 0, "an overfull queue did not drop"
        finally:
            log._stop.set()
            # close() drains before stamping counts, and the writer was killed
            # on purpose above -- so leaving the queue full made it burn its
            # whole 5 s drain deadline on every run.
            while not log._queue.empty():
                log._queue.get_nowait()
            log.close()

    def test_a_clutch_that_raises_reads_as_released(self, config):
        class Exploding:
            @property
            def engaged(self):
                raise RuntimeError("pedal on fire")

        assert TronArmSink(config, clutch=Exploding())._engaged() is False


# =====================================================================
# Boundary maths
# =====================================================================
class TestBoundaryMaths:
    @pytest.mark.parametrize("arm", ARMS)
    def test_targets_exactly_on_a_box_face_are_not_clamped(self, config, arm):
        """Off-by-one at the face would jitter an arm resting against a wall."""
        bounds = config.workspace.box(arm).bounds
        margin = config.workspace.margin_m
        for axis in range(3):
            for edge in (bounds[axis, 0] + margin, bounds[axis, 1] - margin):
                p = np.array([(bounds[i, 0] + bounds[i, 1]) / 2 for i in range(3)])
                p[axis] = edge
                out, clamped = config.workspace.clamp(arm, p)
                np.testing.assert_allclose(out, p, atol=1e-12)
                assert not clamped, f"{arm} axis {axis} face {edge} was clamped"

    @pytest.mark.parametrize("arm", ARMS)
    def test_one_micron_outside_is_clamped(self, config, arm):
        bounds = config.workspace.box(arm).bounds
        margin = config.workspace.margin_m
        p = np.array([(bounds[i, 0] + bounds[i, 1]) / 2 for i in range(3)])
        p[0] = bounds[0, 1] - margin + 1e-6
        _out, clamped = config.workspace.clamp(arm, p)
        assert clamped

    @pytest.mark.parametrize("rate", [50.0, 100.0, 200.0, 500.0])
    def test_step_clamp_tracks_the_rate(self, config, rate):
        """The per-tick ceiling must scale with rate, so the m/s limit holds."""
        cfg = tweak(config, rate_hz=rate)
        max_lin, max_ang = cfg.max_step
        assert max_lin == pytest.approx(cfg.velocity.lin / rate)
        ticks = int(rate)
        current = Pose([0.0, 0.0, 0.0], [1, 0, 0, 0])
        target = Pose([10.0, 0.0, 0.0], [1, 0, 0, 0])
        for _ in range(ticks):
            current, _, _ = apply_step_clamp(current, target, max_lin, max_ang)
        # One second of clamped ticks == exactly the velocity limit, at any rate.
        assert current.position_m[0] == pytest.approx(cfg.velocity.lin, rel=1e-9)

    def test_changing_rate_mid_stream_does_not_permit_a_burst(self, config):
        slow, fast = tweak(config, rate_hz=50.0), tweak(config, rate_hz=200.0)
        assert fast.max_step[0] < slow.max_step[0]
        current = Pose([0.0, 0.0, 0.0], [1, 0, 0, 0])
        target = Pose([1.0, 0.0, 0.0], [1, 0, 0, 0])
        moved, _, _ = apply_step_clamp(current, target, *fast.max_step)
        assert float(np.linalg.norm(moved.position_m)) <= fast.max_step[0] + 1e-12

    def test_slerp_endpoints_are_exact(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([math.cos(0.4), 0.0, 0.0, math.sin(0.4)])
        np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-15)
        np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-15)

    def test_step_clamp_is_a_true_ceiling_not_an_approximation(self, config):
        rng = np.random.default_rng(11)
        max_lin, max_ang = config.max_step
        for _ in range(500):
            a = Pose(rng.uniform(-1, 1, 3), _rand_quat(rng))
            b = Pose(rng.uniform(-1, 1, 3), _rand_quat(rng))
            out, _, _ = apply_step_clamp(a, b, max_lin, max_ang)
            assert float(np.linalg.norm(out.position_m - a.position_m)) <= max_lin + 1e-9
            assert quat_angle(a.orientation_wxyz, out.orientation_wxyz) <= max_ang + 1e-9


def _rand_quat(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


# =====================================================================
# Single-key vs both-key servop, against both mock modes
# =====================================================================
class TestServopKeyModes:
    @pytest.mark.parametrize("require_both", [False, True])
    @pytest.mark.parametrize("send_both", [False, True])
    @pytest.mark.parametrize("arms", [("left",), ("right",), ("left", "right")])
    def test_matrix(self, config, require_both, send_both, arms):
        """send_both=True must satisfy a strict robot for ANY set of live arms."""
        left = Pose([0.40, 0.20, 0.0], [1, 0, 0, 0])
        right = Pose([0.40, -0.20, 0.0], [1, 0, 0, 0])

        async def body():
            async with MockTron2(port=0, info_period_s=0.05,
                                 require_both_keys=require_both) as robot:
                cfg = tweak(at(config, f"ws://127.0.0.1:{robot.bound_port}"),
                            send_both=send_both)
                async with Tron2Client(cfg, notify_log_path=None) as client:
                    await client.prime_frozen_poses()
                    kwargs = {}
                    if "left" in arms:
                        kwargs["left"] = left
                    if "right" in arms:
                        kwargs["right"] = right
                    await client.send_servop(**kwargs)
                    await asyncio.sleep(0.12)
                    return robot.servop_accepted, robot.servop_rejected

        accepted, rejected = run(body())
        single_key = not send_both and len(arms) == 1
        if require_both and single_key:
            assert (accepted, rejected) == (0, 1)
        else:
            assert (accepted, rejected) == (1, 0)


# =====================================================================
# Workspace precondition -- found on hardware, 2026-08-13
# =====================================================================
class TestHoldMustNeverMoveTheArm:
    """The clamp guards TARGETS. Applied to a pose the arm is already at, it
    becomes an actuator: it commands the nearest legal point and the arm walks
    there. On the real robot this drove an arm 29 cm into its stand, because the
    rest pose sits outside the box CLAUDE.md documents."""

    def _outside_pose(self, config, arm):
        bounds = config.workspace.box(arm).bounds
        p = np.array([(bounds[i, 0] + bounds[i, 1]) / 2 for i in range(3)])
        p[0] = bounds[0, 0] - 0.25          # well below the x minimum, as measured
        return Pose(p, [1.0, 0.0, 0.0, 0.0])

    def test_detects_an_out_of_box_pose(self, config):
        from tron_arm.mapping import check_inside_workspace

        violation = check_inside_workspace(config, "right", self._outside_pose(config, "right"))
        assert violation is not None
        assert "x" in violation.axes
        assert violation.distance_m > 0.2
        assert "outside the configured workspace" in violation.describe()

    def test_passes_a_pose_inside_the_box(self, config):
        from tron_arm.mapping import check_inside_workspace

        assert check_inside_workspace(
            config, "right", Pose([0.40, -0.20, 0.0], [1, 0, 0, 0])) is None

    def test_the_clamp_would_have_moved_it_29cm(self, config):
        """Regression: quantify what the old behaviour did."""
        pose = self._outside_pose(config, "right")
        clamped, _ = config.workspace.clamp("right", pose.position_m)
        moved = float(np.linalg.norm(clamped - pose.position_m))
        assert moved > 0.2, "fixture no longer reproduces the hazard"

    @staticmethod
    @contextlib.contextmanager
    def _mock_on_its_own_loop(config, *, park_outside: bool):
        """The sink owns a loop of its own, and connect() blocks the caller on
        it -- so the mock must not share the test's loop or they deadlock."""
        import dataclasses as dc
        import threading

        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        assert ready.wait(5.0)
        robot = MockTron2(port=0, info_period_s=0.05)
        asyncio.run_coroutine_threadsafe(robot.start(), loop).result(10.0)
        if park_outside:
            bounds = config.workspace.box("right").bounds
            for arm in ARMS:
                p = np.array(robot.flange[arm].position_m)
                p[0] = bounds[0, 0] - 0.25
                robot.flange[arm] = Pose(p, robot.flange[arm].orientation_wxyz)
        cfg = dc.replace(at(config, f"ws://127.0.0.1:{robot.bound_port}"),
                         notify_log_path=None)
        try:
            yield robot, cfg
        finally:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(robot.stop(), loop).result(5.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)

    def test_sink_refuses_to_connect_from_an_out_of_box_rest_pose(self, config):
        """The whole point: nothing is commanded, and it says why."""
        from tron_arm.tron2_client import Tron2Error

        with self._mock_on_its_own_loop(config, park_outside=True) as (robot, cfg):
            sink = TronArmSink(cfg)
            with pytest.raises(Tron2Error, match="refusing to stream"):
                sink.connect()
            sink.close()
            assert robot.servop_accepted == 0, "a servop was sent despite the refusal"

    def test_sink_connects_normally_when_inside_the_box(self, config):
        with self._mock_on_its_own_loop(config, park_outside=False) as (robot, cfg):
            sink = TronArmSink(cfg)
            sink.connect()
            try:
                assert sink.streamer is not None
            finally:
                with fast_close():   # the pre-close freeze is not under test here
                    sink.close()

    def test_step_test_aborts_before_commanding(self, config):
        """It must not try the second format either -- same hazard, twice."""
        import dataclasses as dc

        from tron_arm.step_test import run_step_test

        bounds = config.workspace.box("right").bounds

        async def body():
            async with MockTron2(port=0, info_period_s=0.05) as robot:
                for arm in ARMS:
                    p = np.array(robot.flange[arm].position_m)
                    p[0] = bounds[0, 0] - 0.25
                    robot.flange[arm] = Pose(p, robot.flange[arm].orientation_wxyz)
                cfg = dc.replace(at(config, f"ws://127.0.0.1:{robot.bound_port}"),
                                 notify_log_path=None)
                report = await run_step_test(cfg, cfg.robot.url, quick=True)
                return report, robot.servop_accepted

        report, accepted = run(body())
        assert report.aborted
        assert report.chosen_format is None
        assert accepted == 0, "the step test commanded a servop despite aborting"
        assert "ABORTED BEFORE COMMANDING ANYTHING" in report.text()
        assert len(report.formats) == 1, "it tried the second format anyway"


# =====================================================================
# Vendor-confirmed constraints (LimX, 2026-08)
# =====================================================================
class TestVendorConstraints:
    """LimX answered three ambiguities the guide left open. Each answer closes a
    configuration that used to look plausible, so each gets a test."""

    def test_rate_below_50hz_is_refused(self, config):
        """ServoJ and ServoP both require >= 50 Hz. The guide's 500 Hz for
        ServoJ, and its silence on ServoP, are documentation errors."""
        import dataclasses

        from tron_arm.config import MIN_SERVOP_RATE_HZ, ConfigError

        assert MIN_SERVOP_RATE_HZ == 50.0
        for rate in (1.0, 30.0, 49.9):
            with pytest.raises(ConfigError, match="below the 50 Hz minimum"):
                dataclasses.replace(
                    config, servop=dataclasses.replace(config.servop, rate_hz=rate))

    def test_50hz_and_above_is_accepted(self, config):
        import dataclasses

        for rate in (50.0, 100.0, 200.0):
            cfg = dataclasses.replace(
                config, servop=dataclasses.replace(config.servop, rate_hz=rate))
            assert cfg.servop.rate_hz == rate

    def test_send_both_false_is_refused_against_real_hardware(self, config):
        """LimX: the API does not accept single-arm commands. send_both=false
        would fail silently on hardware, so connect refuses it."""
        import dataclasses

        from tron_arm.tron2_client import Tron2Error

        cfg = dataclasses.replace(
            config, servop=dataclasses.replace(config.servop, send_both=False))
        sink = TronArmSink(cfg, url="ws://10.192.1.2:5000")
        with pytest.raises(Tron2Error, match="does not accept single-arm"):
            sink.connect()
        sink.close()

    def test_send_both_false_still_works_against_the_mock(self, config):
        """The mock can be told to accept single-side messages, and the tests
        that exercise that path must keep working."""

        with self._mock_on_its_own_loop_sendboth(config) as (robot, cfg):
            sink = TronArmSink(cfg)
            sink.connect()
            try:
                assert sink.streamer is not None
            finally:
                with fast_close():   # the pre-close freeze is not under test here
                    sink.close()

    @staticmethod
    @contextlib.contextmanager
    def _mock_on_its_own_loop_sendboth(config):
        import dataclasses
        import threading

        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def spin():
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        threading.Thread(target=spin, daemon=True).start()
        assert ready.wait(5.0)
        robot = MockTron2(port=0, info_period_s=0.05)
        asyncio.run_coroutine_threadsafe(robot.start(), loop).result(10.0)
        cfg = dataclasses.replace(
            at(config, f"ws://127.0.0.1:{robot.bound_port}"),
            servop=dataclasses.replace(config.servop, send_both=False),
            notify_log_path=None,
        )
        try:
            yield robot, cfg
        finally:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(robot.stop(), loop).result(5.0)
            loop.call_soon_threadsafe(loop.stop)
