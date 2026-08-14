"""Envelope construction, both servop encoders (golden vectors), and decoders."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tron_arm.config import SERVOP_WIDTH
from tron_arm.poses import Pose
from tron_arm.tron2_client import (
    Tron2ProtocolError,
    build_envelope,
    decode_joint_state,
    decode_move_pose,
    encode_servop_element,
    encode_servop_payload,
)

SQRT_HALF = math.sqrt(0.5)

# (name, position, quaternion wxyz, expected pos_quat, expected pos_rotmat)
GOLDEN = [
    (
        "identity",
        [0.1, 0.2, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ),
    (
        "z+90deg",
        [0.4, -0.2, 0.05],
        [SQRT_HALF, 0.0, 0.0, SQRT_HALF],
        [0.4, -0.2, 0.05, SQRT_HALF, 0.0, 0.0, SQRT_HALF],
        [0.4, -0.2, 0.05, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ),
    (
        "x+90deg",
        [0.5, 0.0, -0.1],
        [SQRT_HALF, SQRT_HALF, 0.0, 0.0],
        [0.5, 0.0, -0.1, SQRT_HALF, SQRT_HALF, 0.0, 0.0],
        [0.5, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    ),
    (
        "y180deg",
        [0.3, 0.1, 0.2],
        [0.0, 0.0, 1.0, 0.0],
        [0.3, 0.1, 0.2, 0.0, 0.0, 1.0, 0.0],
        [0.3, 0.1, 0.2, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
    ),
]


class TestEnvelope:
    def test_has_exactly_the_documented_fields(self):
        env = build_envelope("ACC", "request_servop", {"left_pos": [1]}, "G")
        assert set(env) == {"accid", "title", "timestamp", "guid", "data"}
        assert env["accid"] == "ACC"
        assert env["title"] == "request_servop"
        assert env["guid"] == "G"
        assert env["data"] == {"left_pos": [1]}

    def test_generates_a_uuid4_guid_when_omitted(self):
        import uuid

        guids = {build_envelope("A", "t")["guid"] for _ in range(50)}
        assert len(guids) == 50
        assert all(uuid.UUID(g).version == 4 for g in guids)

    def test_timestamp_is_wall_clock_milliseconds(self):
        import time

        before = int(time.time() * 1000)
        ts = build_envelope("A", "t")["timestamp"]
        assert isinstance(ts, int)
        assert before - 1000 <= ts <= int(time.time() * 1000) + 1000

    def test_data_defaults_to_empty_object(self):
        assert build_envelope("A", "t")["data"] == {}
        assert build_envelope("A", "t", None)["data"] == {}

    def test_data_is_copied_not_aliased(self):
        payload = {"k": 1}
        env = build_envelope("A", "t", payload)
        payload["k"] = 2
        assert env["data"]["k"] == 1


class TestEncoders:
    @pytest.mark.parametrize("name,p,q,expected,_r", GOLDEN, ids=[g[0] for g in GOLDEN])
    def test_pos_quat_golden(self, name, p, q, expected, _r):
        got = encode_servop_element(Pose(p, q), "pos_quat")
        assert len(got) == 7
        np.testing.assert_allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize("name,p,q,_v,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
    def test_pos_rotmat_golden(self, name, p, q, _v, expected):
        got = encode_servop_element(Pose(p, q), "pos_rotmat")
        assert len(got) == 12
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_rotmat_is_row_major(self):
        """r11 r12 r13 r21 ... -- a column-major bug swaps [4] and [6]."""
        got = encode_servop_element(Pose([0, 0, 0], [SQRT_HALF, 0.0, 0.0, SQRT_HALF]), "pos_rotmat")
        r = np.asarray(got[3:]).reshape(3, 3)
        np.testing.assert_allclose(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
        assert got[4] == pytest.approx(-1.0)
        assert got[6] == pytest.approx(1.0)

    @pytest.mark.parametrize("fmt,width", sorted(SERVOP_WIDTH.items()))
    def test_width_matches_the_config_table(self, fmt, width):
        assert len(encode_servop_element(Pose([0, 0, 0], [1, 0, 0, 0]), fmt)) == width

    def test_returns_plain_floats_so_json_can_serialise(self):
        got = encode_servop_element(Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]), "pos_quat")
        assert all(type(v) is float for v in got)

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="unknown servop format"):
            encode_servop_element(Pose([0, 0, 0], [1, 0, 0, 0]), "pos_euler")

    def test_rejects_non_finite_pose(self, ):
        """Hard rule 3: no NaN reaches the encoder, even bypassing Pose validation."""
        from tests.conftest import unchecked_pose

        bad = unchecked_pose([np.nan, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="non-finite"):
            encode_servop_element(bad, "pos_quat")

    def test_accepts_a_duck_typed_pose(self):
        class Upstream:
            def as_xyz_wxyz(self):
                return np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])

        assert encode_servop_element(Upstream(), "pos_quat")[:3] == [0.1, 0.2, 0.3]


class TestPayload:
    def test_both_sides(self):
        pose = Pose([0.4, 0.2, 0.0], [1, 0, 0, 0])
        payload = encode_servop_payload("pos_quat", pose, pose)
        assert sorted(payload) == ["left_pos", "right_pos"]

    @pytest.mark.parametrize("left,right,expected", [
        (True, False, ["left_pos"]),
        (False, True, ["right_pos"]),
        (True, True, ["left_pos", "right_pos"]),
    ])
    def test_only_supplied_sides_produce_keys(self, left, right, expected):
        pose = Pose([0.4, 0.0, 0.0], [1, 0, 0, 0])
        payload = encode_servop_payload("pos_quat", pose if left else None, pose if right else None)
        assert sorted(payload) == expected

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            encode_servop_payload("pos_quat", None, None)


class TestDecoders:
    def test_decode_move_pose(self):
        data = {
            "left_position": [0.4, 0.2, 0.0], "left_quat": [1.0, 0.0, 0.0, 0.0],
            "right_position": [0.4, -0.2, 0.1], "right_quat": [SQRT_HALF, 0.0, 0.0, SQRT_HALF],
        }
        poses = decode_move_pose(data)
        assert sorted(poses) == ["left", "right"]
        np.testing.assert_allclose(poses["right"].p, [0.4, -0.2, 0.1])
        np.testing.assert_allclose(poses["right"].q_wxyz, [SQRT_HALF, 0.0, 0.0, SQRT_HALF])

    def test_decode_move_pose_reads_quat_as_wxyz(self):
        """A scalar-last misread would put 1.0 in qz instead of qw."""
        poses = decode_move_pose({
            "left_position": [0, 0, 0], "left_quat": [1.0, 0.0, 0.0, 0.0],
            "right_position": [0, 0, 0], "right_quat": [1.0, 0.0, 0.0, 0.0],
        })
        assert poses["left"].q_wxyz[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("missing", ["left_position", "left_quat", "right_quat"])
    def test_decode_move_pose_missing_key(self, missing):
        data = {
            "left_position": [0, 0, 0], "left_quat": [1, 0, 0, 0],
            "right_position": [0, 0, 0], "right_quat": [1, 0, 0, 0],
        }
        del data[missing]
        with pytest.raises(Tron2ProtocolError):
            decode_move_pose(data)

    def test_decode_joint_state(self):
        np.testing.assert_allclose(decode_joint_state({"joint": list(range(14))}), np.arange(14))

    @pytest.mark.parametrize("key", ["joint", "joints", "position", "q"])
    def test_decode_joint_state_key_aliases(self, key):
        assert decode_joint_state({key: [0.0] * 14}).shape == (14,)

    def test_decode_joint_state_too_short(self):
        with pytest.raises(Tron2ProtocolError, match="at least 14"):
            decode_joint_state({"joint": [0.0] * 7})

    def test_decode_joint_state_accepts_a_longer_reply(self):
        """Measured on our unit: q/dq/tau carry 16, the last two being
        grippers. The first 14 are the arms."""
        from tron_arm.tron2_client import split_joint_reply

        real = [-0.0071, 0.0161, -0.0188, -1.6466, -0.0068, -0.0650, 0.0171,
                0.0123, 0.0177, -0.009285, -1.689615, 0.0761, 0.0013, -0.0022,
                0.0340, -0.0155]
        arms, extras = split_joint_reply({"q": real})
        assert arms.shape == (14,) and extras.shape == (2,)
        np.testing.assert_allclose(arms, real[:14])
        np.testing.assert_allclose(extras, real[14:])
        assert decode_joint_state({"q": real}).shape == (14,)

    def test_decode_joint_state_refuses_a_layout_that_breaks_the_limits(self):
        """If the first 14 do not fit the arm limits our slice is wrong, and a
        wrong slice would make capture-ready record a shifted posture."""
        from tron_arm.config import JOINT_UPPER

        bad = [0.0] * 16
        bad[3] = float(JOINT_UPPER[3]) + 1.0
        with pytest.raises(Tron2ProtocolError, match="do NOT fit the documented"):
            decode_joint_state({"q": bad})

    def test_exactly_14_is_unchanged(self):
        assert decode_joint_state({"joint": [0.0] * 14}).shape == (14,)

    def test_decode_joint_state_non_finite(self):
        with pytest.raises(Tron2ProtocolError, match="non-finite"):
            decode_joint_state({"joint": [float("nan")] * 14})

    def test_decode_joint_state_missing(self):
        with pytest.raises(Tron2ProtocolError, match="no joint vector"):
            decode_joint_state({"result": "success"})
