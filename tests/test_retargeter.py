import numpy as np
import pytest
import torch

from orca_teleop.retargeting.retargeter import (
    RetargeterConfig,
    TargetPose,
    _normalize_regularizer_weights,
    weighted_vector_loss,
)


def _mediapipe_pose() -> np.ndarray:
    """A geometrically plausible 21-point MediaPipe hand layout.
    TODO: Define this as the correspondant mediapipe pose of the hand's neutral pose."""
    kp = np.zeros((21, 3), dtype=np.float32)
    kp[0] = [0.0, 0.0, 0.0]
    kp[1] = [0.03, 0.02, 0.0]
    kp[2] = [0.05, 0.04, 0.0]
    kp[3] = [0.06, 0.06, 0.0]
    kp[4] = [0.07, 0.08, 0.0]
    kp[5] = [0.02, 0.06, 0.0]
    kp[6] = [0.02, 0.09, 0.0]
    kp[7] = [0.02, 0.11, 0.0]
    kp[8] = [0.02, 0.13, 0.0]
    kp[9] = [0.00, 0.07, 0.0]
    kp[10] = [0.00, 0.10, 0.0]
    kp[11] = [0.00, 0.12, 0.0]
    kp[12] = [0.00, 0.14, 0.0]
    kp[13] = [-0.02, 0.06, 0.0]
    kp[14] = [-0.02, 0.09, 0.0]
    kp[15] = [-0.02, 0.11, 0.0]
    kp[16] = [-0.02, 0.13, 0.0]
    kp[17] = [-0.04, 0.05, 0.0]
    kp[18] = [-0.04, 0.07, 0.0]
    kp[19] = [-0.04, 0.09, 0.0]
    kp[20] = [-0.04, 0.10, 0.0]
    return kp


def test_target_pose_valid_construction():
    pose = TargetPose(joint_positions=_mediapipe_pose())
    assert pose.joint_positions.shape == (21, 3)
    assert pose.source == "mediapipe"


def test_target_pose_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        TargetPose(joint_positions=np.zeros((21,)))
    with pytest.raises(ValueError, match="shape"):
        TargetPose(joint_positions=np.zeros((21, 4)))


def test_target_pose_joint_positions_are_immutable():
    pose = TargetPose(joint_positions=_mediapipe_pose())
    with pytest.raises(ValueError, match="read-only"):
        pose.joint_positions[0, 0] = 99.0


def test_target_pose_wrist_angle_coerced_to_float():
    pose = TargetPose(joint_positions=_mediapipe_pose(), wrist_angle_degrees=10)
    assert isinstance(pose.wrist_angle_degrees, float)


def test_target_pose_wrist_angle_defaults_to_zero():
    pose = TargetPose(joint_positions=_mediapipe_pose())
    assert pose.wrist_angle_degrees == 0.0


def test_target_pose_input_array_is_copied():
    arr = _mediapipe_pose()
    pose = TargetPose(joint_positions=arr)
    arr[0, 0] = 999.0
    assert pose.joint_positions[0, 0] == 0.0


def test_weighted_vector_loss_zero_when_identical():
    loss_fn = weighted_vector_loss()
    kvs = torch.rand(5, 3)
    assert loss_fn(kvs, kvs).item() == pytest.approx(0.0, abs=1e-6)


def test_weighted_vector_loss_positive_when_different():
    loss_fn = weighted_vector_loss()
    target = torch.zeros(5, 3)
    robot = torch.ones(5, 3)
    assert loss_fn(target, robot).item() > 0.0


def test_weighted_vector_loss_increases_with_distance():
    loss_fn = weighted_vector_loss()
    target = torch.zeros(5, 3)
    small_error = loss_fn(target, torch.full((5, 3), 0.1))
    large_error = loss_fn(target, torch.full((5, 3), 1.0))
    assert small_error.item() < large_error.item()


def test_weighted_vector_loss_gradients_flow():
    loss_fn = weighted_vector_loss()
    robot = torch.zeros(5, 3, requires_grad=True)
    loss = loss_fn(torch.ones(5, 3), robot)
    loss.backward()
    assert robot.grad is not None
    assert not torch.all(robot.grad == 0)


def test_weighted_vector_loss_zero_coefficient_silences_finger():
    loss_fn = weighted_vector_loss(coeffs=(1.0, 0.0, 0.0, 0.0, 0.0))
    target = torch.zeros(5, 3)
    robot = torch.zeros(5, 3)
    robot[1:] = 100.0
    assert loss_fn(target, robot).item() == pytest.approx(0.0, abs=1e-6)


def test_weighted_vector_loss_higher_coefficient_increases_loss():
    target = torch.zeros(5, 3)
    robot = torch.ones(5, 3)
    low = weighted_vector_loss(coeffs=(1.0, 1.0, 1.0, 1.0, 1.0))(target, robot)
    high = weighted_vector_loss(coeffs=(10.0, 1.0, 1.0, 1.0, 1.0))(target, robot)
    assert high.item() > low.item()


def test_retargeter_config_default_ik_loss_produces_scalar_tensor():
    default_factory = RetargeterConfig.__dataclass_fields__["ik_loss"].default_factory
    loss_fn = default_factory()
    result = loss_fn(torch.zeros(5, 3), torch.zeros(5, 3))
    assert isinstance(result, torch.Tensor)
    assert result.shape == ()


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([2.0, 3.0, 5.0], [0.2, 0.3, 0.5]),
        ([1.0, 1.0, 1.0, 1.0], [0.25, 0.25, 0.25, 0.25]),
        ([1.0, 3.0], [0.25, 0.75]),
    ],
)
def test_normalize_regularizer_weights_sums_to_one(weights, expected):
    weights = torch.tensor(weights)
    normalized = _normalize_regularizer_weights(weights)
    assert normalized.sum().item() == pytest.approx(1.0)
    assert normalized.tolist() == pytest.approx(expected)


def test_normalize_regularizer_weights_keeps_all_zero_vector():
    weights = torch.zeros(4)
    normalized = _normalize_regularizer_weights(weights)
    assert torch.equal(normalized, weights)


def test_normalize_regularizer_weights_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_regularizer_weights(torch.tensor([1.0, -1.0]))
