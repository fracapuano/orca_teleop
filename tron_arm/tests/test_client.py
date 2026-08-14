"""Client behaviour: guid correlation, timeouts, send_both, notify logging."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from tests.conftest import SilentServer, at, mock_and_client, run, tweak
from tron_arm.poses import Pose
from tron_arm.tron2_client import (
    NOTIFY_ROBOT_INFO,
    NOTIFY_SERVOP,
    TITLE_SERVOP,
    Tron2Client,
    Tron2Error,
    Tron2NotConnected,
    Tron2Timeout,
)

POSE_L = Pose([0.40, 0.20, 0.00], [1.0, 0.0, 0.0, 0.0])
POSE_R = Pose([0.40, -0.20, 0.00], [1.0, 0.0, 0.0, 0.0])


class TestConnection:
    def test_learns_accid_from_notify_robot_info(self, config):
        async def body():
            async with mock_and_client(config, accid="LEARN-ME") as (robot, client):
                assert client.accid == "LEARN-ME"
                assert client.connected

        run(body())

    def test_sends_the_learned_accid_back_on_requests(self, config):
        async def body():
            async with mock_and_client(config, accid="LEARN-ME") as (robot, client):
                # The mock rejects a wrong accid, so a successful call proves it.
                assert (await client.get_move_pose())["left"] is not None
                assert robot.invalid_requests == 0

        run(body())

    def test_connect_failure_is_reported(self, config):
        async def body():
            client = Tron2Client(at(config, "ws://127.0.0.1:1"), notify_log_path=None)
            with pytest.raises(Tron2Error, match="cannot connect"):
                await client.connect()

        run(body())

    def test_calls_before_connect_are_refused(self, config):
        async def body():
            client = Tron2Client(config, notify_log_path=None)
            with pytest.raises(Tron2NotConnected):
                await client.get_move_pose()
            with pytest.raises(Tron2NotConnected):
                await client.send_servop(POSE_L, POSE_R)

        run(body())

    def test_tcp_nodelay_is_set(self, config):
        """ServoP sends a small frame per tick; Nagle would batch them."""
        import socket

        async def body():
            async with mock_and_client(config) as (_robot, client):
                sock = client._ws.transport.get_extra_info("socket")
                # Darwin reports a truthy non-1 value, so test for "enabled".
                assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0

        run(body())

    def test_double_connect_is_refused(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                with pytest.raises(Tron2Error, match="already connected"):
                    await client.connect()

        run(body())

    def test_close_is_idempotent(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                await client.close()
                await client.close()
                assert not client.connected

        run(body())


class TestGuidCorrelation:
    def test_each_request_carries_a_distinct_guid(self, config):
        async def body():
            async with SilentServer() as server:
                client = Tron2Client(at(config, f"ws://127.0.0.1:{server.port}"),
                                     notify_log_path=None)
                await client.connect()
                for _ in range(3):
                    with pytest.raises(Tron2Timeout):
                        await client.get_move_pose(timeout=0.05)
                await client.close()
                guids = [m["guid"] for m in server.received]
                assert len(guids) == 3 and len(set(guids)) == 3

        run(body())

    def test_concurrent_requests_resolve_to_their_own_replies(self, config):
        """Interleaved replies must not cross-talk."""

        async def body():
            async with mock_and_client(config) as (_robot, client):
                poses, joints = await asyncio.gather(
                    client.get_move_pose(), client.get_joint_state()
                )
                assert sorted(poses) == ["left", "right"]
                assert joints.shape == (14,)

        run(body())

    def test_reply_with_an_unknown_guid_is_ignored(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                client._dispatch({"accid": "x", "title": "request_get_move_pose",
                                  "guid": "not-pending", "data": {}})
                assert client.stats.responses_received == 0

        run(body())

    def test_timeout_raises_and_is_counted(self, config):
        async def body():
            async with SilentServer() as server:
                client = Tron2Client(at(config, f"ws://127.0.0.1:{server.port}"),
                                     notify_log_path=None)
                await client.connect()
                with pytest.raises(Tron2Timeout, match="timed out"):
                    await client.get_move_pose(timeout=0.05)
                assert client.stats.timeouts == 1
                await client.close()

        run(body())

    def test_timeout_does_not_leak_the_pending_entry(self, config):
        async def body():
            async with SilentServer() as server:
                client = Tron2Client(at(config, f"ws://127.0.0.1:{server.port}"),
                                     notify_log_path=None)
                await client.connect()
                for _ in range(5):
                    with pytest.raises(Tron2Timeout):
                        await client.get_joint_state(timeout=0.02)
                assert client._pending == {}
                await client.close()

        run(body())

    def test_close_fails_in_flight_requests(self, config):
        async def body():
            async with SilentServer() as server:
                client = Tron2Client(at(config, f"ws://127.0.0.1:{server.port}"),
                                     notify_log_path=None)
                await client.connect()
                task = asyncio.ensure_future(client.get_move_pose(timeout=5.0))
                await asyncio.sleep(0.05)
                await client.close()
                with pytest.raises(Tron2NotConnected):
                    await task

        run(body())


class TestSendBoth:
    """Whether a single-side key is legal is UNKNOWN, so both paths must work."""

    def _sent_payloads(self, config, left, right, *, send_both, frozen=True):
        sent: list[dict] = []

        async def body():
            async with SilentServer() as server:
                cfg = tweak(at(config, f"ws://127.0.0.1:{server.port}"), send_both=send_both)
                client = Tron2Client(cfg, notify_log_path=None)
                await client.connect()
                if frozen:
                    client.set_frozen_pose("left", POSE_L)
                    client.set_frozen_pose("right", POSE_R)
                await client.send_servop(left, right)
                await asyncio.sleep(0.05)
                await client.close()
                sent.extend(m for m in server.received if m["title"] == TITLE_SERVOP)

        run(body())
        return sent

    @pytest.mark.parametrize("left,right,expected", [
        (POSE_L, POSE_R, ["left_pos", "right_pos"]),
        (POSE_L, None, ["left_pos", "right_pos"]),
        (None, POSE_R, ["left_pos", "right_pos"]),
    ])
    def test_send_both_true_always_emits_both_keys(self, config, left, right, expected):
        sent = self._sent_payloads(config, left, right, send_both=True)
        assert sorted(sent[0]["data"]) == expected

    @pytest.mark.parametrize("left,right,expected", [
        (POSE_L, POSE_R, ["left_pos", "right_pos"]),
        (POSE_L, None, ["left_pos"]),
        (None, POSE_R, ["right_pos"]),
    ])
    def test_send_both_false_emits_only_supplied_sides(self, config, left, right, expected):
        sent = self._sent_payloads(config, left, right, send_both=False)
        assert sorted(sent[0]["data"]) == expected

    def test_idle_arm_is_filled_from_the_frozen_pose(self, config):
        sent = self._sent_payloads(config, None, POSE_R, send_both=True)
        np.testing.assert_allclose(sent[0]["data"]["left_pos"][:3], POSE_L.p)

    def test_send_both_without_a_frozen_pose_is_an_error(self, config):
        with pytest.raises(Tron2Error, match="no frozen pose"):
            self._sent_payloads(config, None, POSE_R, send_both=True, frozen=False)

    def test_prime_frozen_poses_reads_from_the_robot(self, config):
        async def body():
            async with mock_and_client(config) as (robot, client):
                assert client.frozen_poses["left"] is None
                primed = await client.prime_frozen_poses()
                np.testing.assert_allclose(primed["left"].p, robot.flange["left"].p)
                np.testing.assert_allclose(client.frozen_poses["right"].p, robot.flange["right"].p)

        run(body())

    def test_frozen_pose_updates_after_each_send(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                await client.prime_frozen_poses()
                moved = Pose([0.45, -0.25, 0.05], [1, 0, 0, 0])
                await client.send_servop(right=moved)
                np.testing.assert_allclose(client.frozen_poses["right"].p, moved.p)

        run(body())

    def test_encoding_uses_the_configured_format(self, config):
        for fmt, width in (("pos_quat", 7), ("pos_rotmat", 12)):
            cfg = tweak(config, format=fmt)
            sent = self._sent_payloads(cfg, POSE_L, POSE_R, send_both=True)
            assert len(sent[0]["data"]["left_pos"]) == width


class TestServopGuards:
    def test_out_of_range_target_is_clamped_before_encoding(self, config):
        """Hard rule 3: nothing unclamped reaches the encoder."""

        async def body():
            async with mock_and_client(config) as (_robot, client):
                await client.prime_frozen_poses()
                await client.send_servop(right=Pose([9.0, -9.0, 9.0], [1, 0, 0, 0]))
                assert client.stats.servop_clamped == 1
                sent = client.frozen_poses["right"]
                bounds = config.workspace.box("right").bounds
                assert np.all(sent.p >= bounds[:, 0]) and np.all(sent.p <= bounds[:, 1])

        run(body())

    def test_in_range_target_is_not_counted_as_clamped(self, config):
        async def body():
            async with mock_and_client(config) as (_robot, client):
                await client.prime_frozen_poses()
                await client.send_servop(right=POSE_R)
                assert client.stats.servop_clamped == 0

        run(body())

    def test_servop_does_not_wait_for_a_reply(self, config):
        """Fire-and-forget: it must complete even though nothing ever answers."""

        async def body():
            async with SilentServer() as server:
                client = Tron2Client(at(config, f"ws://127.0.0.1:{server.port}"),
                                     notify_log_path=None)
                await client.connect()
                client.set_frozen_pose("left", POSE_L)
                client.set_frozen_pose("right", POSE_R)
                await asyncio.wait_for(client.send_servop(POSE_L, POSE_R), timeout=0.5)
                assert client.stats.servop_sent == 1
                await client.close()

        run(body())


class TestNotifyRegistry:
    def test_subscription_receives_records(self, config):
        async def body():
            seen = []
            async with mock_and_client(config, info_period_s=0.05) as (_robot, client):
                client.on_notify(NOTIFY_ROBOT_INFO, seen.append)
                await asyncio.sleep(0.2)
            assert seen
            assert all(r.title == NOTIFY_ROBOT_INFO for r in seen)
            assert all(r.t_mono_ns > 0 and r.t_wall_ns > 0 for r in seen)

        run(body())

    def test_wildcard_subscription(self, config):
        async def body():
            seen = []
            async with mock_and_client(config, info_period_s=0.05) as (_robot, client):
                client.on_notify("*", seen.append)
                await asyncio.sleep(0.2)
            assert seen

        run(body())

    def test_unsubscribe(self, config):
        async def body():
            seen = []
            async with mock_and_client(config, info_period_s=0.05) as (_robot, client):
                off = client.on_notify(NOTIFY_ROBOT_INFO, seen.append)
                await asyncio.sleep(0.15)
                count = len(seen)
                off()
                await asyncio.sleep(0.15)
                assert len(seen) == count

        run(body())

    def test_a_raising_subscriber_does_not_kill_the_reader(self, config):
        async def body():
            good = []

            def bad(_record):
                raise RuntimeError("boom")

            async with mock_and_client(config, info_period_s=0.05) as (_robot, client):
                client.on_notify(NOTIFY_ROBOT_INFO, bad)
                client.on_notify(NOTIFY_ROBOT_INFO, good.append)
                await asyncio.sleep(0.2)
                assert good
                assert (await client.get_move_pose()) is not None

        run(body())

    def test_notify_counters(self, config):
        async def body():
            async with mock_and_client(config, info_period_s=0.05) as (_robot, client):
                await asyncio.sleep(0.2)
                assert client.stats.notifies[NOTIFY_ROBOT_INFO] >= 2

        run(body())


class TestNotifyLog:
    def test_every_notify_is_appended_as_jsonl_with_both_clocks(self, config, tmp_path):
        path = tmp_path / "logs" / "notify.jsonl"

        async def body():
            from tron_arm.mock_robot import MockTron2

            async with MockTron2(port=0, info_period_s=0.05) as robot:
                cfg = at(config, f"ws://127.0.0.1:{robot.bound_port}")
                async with Tron2Client(cfg, notify_log_path=path) as client:
                    await asyncio.sleep(0.2)
                    # Provoke a notify_servop failure too.
                    await client.send_servop(POSE_L, POSE_R)
                    robot.require_both_keys = True
                    await client._ws.send(json.dumps({
                        "accid": client.accid, "title": TITLE_SERVOP,
                        "timestamp": 0, "guid": "g", "data": {"left_pos": [0.0] * 7}}))
                    await asyncio.sleep(0.1)

        run(body())
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert lines
        titles = {line["title"] for line in lines}
        assert NOTIFY_ROBOT_INFO in titles
        assert NOTIFY_SERVOP in titles
        for line in lines:
            assert set(line) == {"t_mono_ns", "t_wall_ns", "title", "guid", "accid",
                                 "robot_timestamp_ms", "data"}
            assert isinstance(line["t_mono_ns"], int)
            assert isinstance(line["t_wall_ns"], int)

    def test_log_is_appended_not_truncated(self, config, tmp_path):
        path = tmp_path / "notify.jsonl"
        path.write_text('{"pre-existing": true}\n')

        async def body():
            from tron_arm.mock_robot import MockTron2

            async with MockTron2(port=0, info_period_s=0.05) as robot:
                cfg = at(config, f"ws://127.0.0.1:{robot.bound_port}")
                async with Tron2Client(cfg, notify_log_path=path) as client:
                    await asyncio.sleep(0.15)

        run(body())
        assert path.read_text().startswith('{"pre-existing": true}')
        assert len(path.read_text().splitlines()) > 1

    def test_no_log_is_written_when_disabled(self, config, tmp_path):
        path = tmp_path / "never.jsonl"

        async def body():
            async with mock_and_client(config) as (_robot, _client):
                await asyncio.sleep(0.1)

        run(body())
        assert not path.exists()
