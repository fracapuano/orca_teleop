import numpy as np
import pytest
from orca_core import OrcaJointPositions
from orca_core.hardware_hand import MockOrcaHand

from orca_teleop.policies import (
    _JOINT_NAME_ALIASES,
    LeRobotACTPolicyAdapter,
    LeRobotPolicyAdapter,
    action_to_joint_positions,
)


def _joint_ids(n: int = 3) -> list[str]:
    return list(MockOrcaHand().config.joint_ids)[:n]


def test_action_to_joint_positions_accepts_vector():
    joint_ids = _joint_ids(3)

    action = action_to_joint_positions(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), joint_ids)

    np.testing.assert_allclose(action.as_array(joint_ids), [1.0, 2.0, 3.0])


def test_action_to_joint_positions_accepts_chunked_batch():
    joint_ids = _joint_ids(2)
    chunk = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    action = action_to_joint_positions(chunk, joint_ids)

    np.testing.assert_allclose(action.as_array(joint_ids), [1.0, 2.0])


def test_action_to_joint_positions_accepts_joint_mapping():
    joint_ids = _joint_ids(2)

    action = action_to_joint_positions({joint_ids[1]: 4.0, joint_ids[0]: 2.0}, joint_ids)

    np.testing.assert_allclose(action.as_array(joint_ids), [2.0, 4.0])


def test_action_to_joint_positions_accepts_unprefixed_joint_mapping():
    # v1 configs name this joint thumb_pip, v2 thumb_cmc. Target the hand's own
    # name while the policy emits the other, so the alias is what makes it match.
    thumb = "thumb_cmc" if "thumb_cmc" in MockOrcaHand().config.joint_ids else "thumb_pip"
    joint_ids = ["wrist", thumb]

    action = action_to_joint_positions(
        {"wrist": 1.0, _JOINT_NAME_ALIASES[thumb]: 2.0}, joint_ids
    )

    np.testing.assert_allclose(action.as_array(joint_ids), [1.0, 2.0])


def test_action_to_joint_positions_recurses_action_key():
    joint_ids = _joint_ids(2)

    action = action_to_joint_positions({"action": [5.0, 6.0]}, joint_ids)

    np.testing.assert_allclose(action.as_array(joint_ids), [5.0, 6.0])


def test_action_to_joint_positions_returns_orca_action():
    joint_ids = _joint_ids(1)
    existing = OrcaJointPositions.from_dict({joint_ids[0]: 1.5})

    action = action_to_joint_positions(existing, joint_ids)

    assert action is existing


def test_action_to_joint_positions_rejects_wrong_width():
    with pytest.raises(ValueError, match="expects 3 joints"):
        action_to_joint_positions([1.0, 2.0], _joint_ids(3))


def test_lerobot_act_adapter_alias_is_generic_adapter():
    assert LeRobotACTPolicyAdapter is LeRobotPolicyAdapter


def test_lerobot_adapter_aligns_named_state_subset():
    adapter = LeRobotPolicyAdapter(
        policy=None,
        preprocess=None,
        postprocess=None,
        dataset_features={
            "observation.state": {
                "shape": (4,),
                "names": ["panda_joint1", "wrist", "thumb_cmc", "index_mcp"],
            }
        },
        device=None,
    )

    observation = adapter._align_observation(
        {
            "observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "observation.state.names": ["right_wrist", "right_thumb_pip", "right_index_mcp"],
        }
    )

    np.testing.assert_allclose(observation["observation.state"], [0.0, 1.0, 2.0, 3.0])
