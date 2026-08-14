"""Full round trip against the mock: every request type, plus the servop matrix."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from tests.conftest import at, mock_and_client, run, tweak
from tron_arm.config import JOINT_UPPER, N_JOINTS
from tron_arm.mock_robot import MockTron2
from tron_arm.poses import Pose
from tron_arm.streamer import PoseStreamer
from tron_arm.tron2_client import (
    NOTIFY_INVALID_REQUEST,
    NOTIFY_SERVOP,
    TITLE_SERVOP,
    Tron2Client,
)

POSE_L = Pose([0.40, 0.20, 0.00], [1.0, 0.0, 0.0, 0.0])
POSE_R = Pose([0.40, -0.20, 0.00], [1.0, 0.0, 0.0, 0.0])


class TestRequestTypes:
    def test_get_move_pose(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                poses = await client.get_move_pose()
                for arm in ("left", "right"):
                    np.testing.assert_allclose(poses[arm].p, robot.flange[arm].p)
                    np.testing.assert_allclose(poses[arm].q_wxyz, robot.flange[arm].q_wxyz)

        run(body())

    def test_get_joint_state(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                joints = await client.get_joint_state()
                assert joints.shape == (N_JOINTS,)
                np.testing.assert_allclose(joints, robot.joints)

        run(body())

    def test_movej_moves_the_joints_and_the_flange(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                before = await client.get_move_pose()
                target = np.full(N_JOINTS, 0.1)
                reply = await client.movej(target, 0.2)
                assert reply["result"] == "success"
                await asyncio.sleep(0.4)
                np.testing.assert_allclose(await client.get_joint_state(), target, atol=1e-9)
                after = await client.get_move_pose()
                assert not np.allclose(before["left"].p, after["left"].p)

        run(body())

    def test_movej_rejects_out_of_limit_targets_locally(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                bad = np.zeros(N_JOINTS)
                bad[3] = JOINT_UPPER[3] + 0.5
                with pytest.raises(ValueError, match="outside joint limits"):
                    await client.movej(bad, 1.0)

        run(body())

    @pytest.mark.parametrize("joints,seconds,match", [
        ([0.0] * 7, 1.0, "expects 14"),
        ([0.0] * 14, 0.0, "time must be"),
        ([0.0] * 14, -1.0, "time must be"),
    ])
    def test_movej_argument_validation(self, config, joints, seconds, match):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                with pytest.raises(ValueError, match=match):
                    await client.movej(joints, seconds)

        run(body())

    def test_movej_out_of_range_is_also_rejected_server_side(self, config):
        """Bypass the client guard to prove the mock enforces limits too."""

        async def body():
            async with mock_and_client(config) as (_robot, client):
                bad = [0.0] * N_JOINTS
                bad[3] = 3.0
                reply = await client._request("request_movej", {"time": 1.0, "joint": bad})
                assert reply["result"] == "fail_out_of_range"

        run(body())

    def test_light_effect(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                reply = await client.light_effect("blink", color="red")
                assert reply["result"] == "success"
                assert robot.light == {"effect": "blink", "color": "red"}

        run(body())

    def test_emgy_stop_when_idle(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                assert (await client.emgy_stop())["result"] == "success"

        run(body())

    def test_emgy_stop_is_refused_while_moving(self, config):
        """It is not a motion abort -- the vendor only accepts it when idle."""

        async def body():
            async with mock_and_client(config) as (_robot, client):
                await client.movej(np.full(N_JOINTS, 0.1), 0.5)
                await asyncio.sleep(0.1)
                assert (await client.emgy_stop())["result"] == "fail_not_idle"

        run(body())

    def test_servop_moves_the_flange(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                await client.prime_frozen_poses()
                target = Pose([0.45, -0.25, 0.05], [1.0, 0.0, 0.0, 0.0])
                for _ in range(40):
                    await client.send_servop(right=target)
                    await asyncio.sleep(0.002)
                np.testing.assert_allclose(robot.flange["right"].p, target.p, atol=1e-6)
                assert robot.servop_rejected == 0

        run(body())

    def test_notify_invalid_request_echoes_a_malformed_envelope(self, config):
        async def body():
            seen = []
            async with mock_and_client(config) as (robot, client):
                client.on_notify(NOTIFY_INVALID_REQUEST, seen.append)
                await client._ws.send("this is not json")
                await client._ws.send(json.dumps({"accid": client.accid, "data": {}}))
                await client._ws.send(json.dumps({
                    "accid": client.accid, "title": "request_teleport",
                    "timestamp": 0, "guid": "g", "data": {}}))
                await asyncio.sleep(0.2)
                assert robot.invalid_requests == 3
            reasons = [r.data["reason"] for r in seen]
            assert any("not valid JSON" in r for r in reasons)
            assert any("title" in r for r in reasons)
            assert any("unknown title" in r for r in reasons)
            assert all("echo" in r.data for r in seen)

        run(body())

    def test_unknown_accid_is_rejected(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                await client._ws.send(json.dumps({
                    "accid": "SOMEONE-ELSE", "title": "request_get_move_pose",
                    "timestamp": 0, "guid": "g", "data": {}}))
                await asyncio.sleep(0.15)
                assert robot.invalid_requests == 1

        run(body())

    def test_request_without_guid_is_rejected(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                await client._ws.send(json.dumps({
                    "accid": client.accid, "title": "request_get_move_pose",
                    "timestamp": 0, "data": {}}))
                await asyncio.sleep(0.15)
                assert robot.invalid_requests == 1

        run(body())


class TestServopValidationMatrix:
    """send_both x require_both_keys x format -- the two documented unknowns."""

    def _run(self, config, *, send_both, require_both_keys, fmt, accept_format, arms):
        failures: list = []

        async def body():
            async with MockTron2(port=0, info_period_s=0.05,
                                 accept_format=accept_format,
                                 require_both_keys=require_both_keys) as robot:
                cfg = tweak(at(config, f"ws://127.0.0.1:{robot.bound_port}"),
                            send_both=send_both, format=fmt)
                async with Tron2Client(cfg, notify_log_path=None) as client:
                    client.on_notify(NOTIFY_SERVOP, failures.append)
                    await client.prime_frozen_poses()
                    kwargs = {"left": POSE_L} if "left" in arms else {}
                    if "right" in arms:
                        kwargs["right"] = POSE_R
                    await client.send_servop(**kwargs)
                    await asyncio.sleep(0.15)
                    return robot.servop_accepted, robot.servop_rejected

        accepted, rejected = run(body())
        return accepted, rejected, failures

    @pytest.mark.parametrize("send_both,arms,require_both,ok", [
        (True, ("left", "right"), True, True),
        (True, ("right",), True, True),      # send_both fills the idle side
        (True, ("left",), True, True),
        (False, ("left", "right"), True, True),
        (False, ("right",), True, False),    # single key against a strict robot
        (False, ("left",), True, False),
        (False, ("right",), False, True),    # permissive robot accepts one side
        (False, ("left",), False, True),
        (True, ("right",), False, True),
    ])
    def test_send_both_permutations(self, config, send_both, arms, require_both, ok):
        accepted, rejected, failures = self._run(
            config, send_both=send_both, require_both_keys=require_both,
            fmt="pos_quat", accept_format="pos_quat", arms=arms,
        )
        if ok:
            assert (accepted, rejected) == (1, 0)
            assert failures == []
        else:
            assert (accepted, rejected) == (0, 1)
            assert failures and failures[0].data["result"] == "fail_invalid_cmd"

    @pytest.mark.parametrize("fmt", ["pos_quat", "pos_rotmat"])
    @pytest.mark.parametrize("accept", ["pos_quat", "pos_rotmat"])
    def test_format_mismatch_is_reported_as_fail_invalid_cmd(self, config, fmt, accept):
        accepted, rejected, failures = self._run(
            config, send_both=True, require_both_keys=False,
            fmt=fmt, accept_format=accept, arms=("left", "right"),
        )
        if fmt == accept:
            assert (accepted, rejected) == (1, 0) and failures == []
        else:
            assert (accepted, rejected) == (0, 1)
            assert failures[0].data["result"] == "fail_invalid_cmd"
            assert "elements" in failures[0].data["reason"]


class TestMockValidator:
    """Direct tests of the pure validator behind notify_servop."""

    @pytest.fixture
    def robot(self):
        return MockTron2(port=0)

    def test_accepts_a_well_formed_payload(self, robot):
        payload = {"left_pos": [0.4, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0]}
        assert robot.validate_servop(payload).ok

    def test_rejects_unknown_keys(self, robot):
        assert not robot.validate_servop({"middle_pos": [0.0] * 7}).ok

    def test_rejects_empty(self, robot):
        assert not robot.validate_servop({}).ok

    def test_rejects_wrong_width(self, robot):
        check = robot.validate_servop({"left_pos": [0.0] * 12})
        assert not check.ok and "12 elements" in check.reason

    def test_rejects_non_unit_quaternion(self, robot):
        check = robot.validate_servop({"left_pos": [0.4, 0.2, 0.0, 2.0, 0.0, 0.0, 0.0]})
        assert not check.ok and "unit norm" in check.reason

    def test_rejects_non_finite(self, robot):
        check = robot.validate_servop({"left_pos": [float("nan")] * 7})
        assert not check.ok and "non-finite" in check.reason

    def test_rotmat_rejects_a_non_orthonormal_block(self):
        robot = MockTron2(port=0, accept_format="pos_rotmat")
        check = robot.validate_servop({"left_pos": [0.4, 0.2, 0.0] + [1.0] * 9})
        assert not check.ok and "orthonormal" in check.reason

    def test_rotmat_rejects_a_reflection(self):
        robot = MockTron2(port=0, accept_format="pos_rotmat")
        payload = [0.4, 0.2, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        check = robot.validate_servop({"left_pos": payload})
        assert not check.ok and "determinant" in check.reason

    def test_rotmat_accepts_a_valid_rotation(self):
        robot = MockTron2(port=0, accept_format="pos_rotmat")
        payload = [0.4, 0.2, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        check = robot.validate_servop({"left_pos": payload})
        assert check.ok
        np.testing.assert_allclose(check.targets["left"].p, [0.4, 0.2, 0.0])

    def test_require_both_keys(self):
        robot = MockTron2(port=0, require_both_keys=True)
        assert not robot.validate_servop({"left_pos": [0.4, 0.2, 0.0, 1.0, 0, 0, 0]}).ok
        assert robot.validate_servop({
            "left_pos": [0.4, 0.2, 0.0, 1.0, 0, 0, 0],
            "right_pos": [0.4, -0.2, 0.0, 1.0, 0, 0, 0],
        }).ok

    def test_rejects_bad_construction(self):
        with pytest.raises(ValueError):
            MockTron2(accept_format="pos_euler")
        with pytest.raises(ValueError):
            MockTron2(servop_gain=0.0)


class TestStreamerAgainstMock:
    def test_streamed_targets_reach_the_robot_at_the_configured_rate(self, config):
        """The whole chain: submit -> interpolate -> clamp -> encode -> mock."""

        async def body():
            async with mock_and_client(config) as (robot, client):
                await client.prime_frozen_poses()
                streamer = PoseStreamer(config, client.send_servop)
                stop = asyncio.Event()
                task = await streamer.start(stop=stop)
                start = Pose([0.42, -0.22, 0.02], [1.0, 0.0, 0.0, 0.0])
                t0 = asyncio.get_running_loop().time()
                import time as _t

                for i in range(30):
                    streamer.submit("right", start, _t.monotonic_ns())
                    await asyncio.sleep(1 / 45)
                stop.set()
                await task
                await asyncio.sleep(0.05)
                return robot, streamer, client

        robot, streamer, client = run(body())
        assert streamer.stats.ticks > 40
        assert robot.servop_rejected == 0
        assert robot.servop_accepted > 40
        assert 80 < streamer.stats.achieved_rate_hz < 120

    def test_a_distant_target_is_rate_limited_not_teleported(self, config):
        """Velocity clamping must survive the whole chain."""

        async def body():
            async with mock_and_client(config) as (robot, client):
                current = await client.prime_frozen_poses()
                streamer = PoseStreamer(config, client.send_servop)
                streamer.seed_all(current)  # without this the first tick jumps
                import time as _t

                far = Pose([0.70, 0.15, 0.40], [1.0, 0.0, 0.0, 0.0])
                now = _t.monotonic_ns()
                streamer.submit("right", far, now)
                streamer.submit("right", far, now + 10_000_000)
                start = np.array(robot.flange["right"].p)
                for i in range(5):
                    await streamer.tick(now + (i + 2) * 10_000_000)
                await asyncio.sleep(0.05)
                moved = float(np.linalg.norm(np.array(robot.flange["right"].p) - start))
                return moved, streamer.stats.lin_clamped

        moved, clamped = run(body())
        assert clamped >= 4
        # 5 ticks * 4 mm is the ceiling; the mock's follower lags further behind.
        assert moved < 5 * 0.004 + 1e-9


class TestSeeding:
    """The first tick is only rate-limited if the clamp is seeded first."""

    def test_unseeded_first_tick_passes_the_target_through(self, config):
        streamer = PoseStreamer(config, lambda *_: asyncio.sleep(0))
        far = Pose([0.70, 0.15, 0.40], [1.0, 0.0, 0.0, 0.0])
        streamer.submit("right", far, 0)
        out = streamer.step(1_000_000)
        np.testing.assert_allclose(out["right"].p, far.p)
        assert streamer.stats.lin_clamped == 0

    def test_seeded_first_tick_is_clamped(self, config):
        streamer = PoseStreamer(config, lambda *_: asyncio.sleep(0))
        streamer.seed("right", POSE_R)
        far = Pose([0.70, 0.15, 0.40], [1.0, 0.0, 0.0, 0.0])
        streamer.submit("right", far, 0)
        out = streamer.step(1_000_000)
        step = float(np.linalg.norm(out["right"].p - POSE_R.p))
        assert step == pytest.approx(config.max_step[0])
        assert streamer.stats.lin_clamped == 1

    def test_seed_all_from_get_move_pose(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                streamer = PoseStreamer(config, client.send_servop)
                streamer.seed_all(await client.prime_frozen_poses())
                assert streamer.last_sent("left") is not None
                assert streamer.last_sent("right") is not None

        run(body())

    def test_seed_rejects_unknown_arm(self, config):
        streamer = PoseStreamer(config, lambda *_: asyncio.sleep(0))
        with pytest.raises(ValueError, match="unknown arm"):
            streamer.seed("middle", POSE_R)
