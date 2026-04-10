import numpy as np
import pytest
import torch

from orca_teleop.retargeting.constants import KEY_VECTORS_SHAPE
from orca_teleop.retargeting.retargeter import (
    RetargeterConfig,
    TargetPose,
    weighted_vector_loss,
)


def test_target_pose_valid_construction():
    pose = TargetPose(key_vectors=np.zeros(KEY_VECTORS_SHAPE))
    assert pose.key_vectors.shape == KEY_VECTORS_SHAPE


def test_target_pose_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        TargetPose(key_vectors=np.zeros((3, 3)))


def test_target_pose_key_vectors_are_immutable():
    pose = TargetPose(key_vectors=np.ones(KEY_VECTORS_SHAPE))
    with pytest.raises(ValueError, match="read-only"):
        pose.key_vectors[0, 0] = 99.0


def test_target_pose_wrist_angle_coerced_to_float():
    pose = TargetPose(key_vectors=np.zeros(KEY_VECTORS_SHAPE), wrist_angle_degrees=10)
    assert isinstance(pose.wrist_angle_degrees, float)


def test_target_pose_wrist_angle_defaults_to_zero():
    pose = TargetPose(key_vectors=np.zeros(KEY_VECTORS_SHAPE))
    assert pose.wrist_angle_degrees == 0.0


def test_target_pose_input_array_is_copied():
    arr = np.ones(KEY_VECTORS_SHAPE)
    pose = TargetPose(key_vectors=arr)
    arr[0, 0] = 999.0
    assert pose.key_vectors[0, 0] == 1.0


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
    # Only thumb (index 0) has weight; all others are zero
    loss_fn = weighted_vector_loss(coeffs=(1.0, 0.0, 0.0, 0.0, 0.0))
    target = torch.zeros(5, 3)
    robot = torch.zeros(5, 3)
    robot[1:] = 100.0  # large error on non-thumb fingers — should not count
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
    assert result.shape == ()  # scalar
