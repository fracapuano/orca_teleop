import numpy as np
import pinocchio as pin
import pytest

from orca_teleop.orca_arm_ik import BimanualIKSolver, orca_panda_right_ik_config


@pytest.mark.parametrize("side", ["left", "right"])
def test_position_ik_converges_to_reachable_carpals_target(side: str) -> None:
    ik = BimanualIKSolver(max_iterations=100)
    q0 = ik.neutral_q.copy()
    q_target = q0.copy()

    for k, idx in enumerate(ik.arm_joint_indices[side]):
        q_target[idx] = np.clip(
            0.05 * (k + 1),
            ik._model.lowerPositionLimit[idx],
            ik._model.upperPositionLimit[idx],
        )

    target = {side: pin.SE3(ik.forward_kinematics_full(q_target, side))}
    result = ik.solve_position(target, q0)

    assert result.position_error[side] < 1e-3
    assert result.converged[side]


@pytest.mark.parametrize("side", ["left", "right"])
def test_position_ik_can_track_xz_rotation_axes(side: str) -> None:
    ik = BimanualIKSolver(max_iterations=150)
    q0 = ik.neutral_q.copy()
    q_target = q0.copy()

    for k, idx in enumerate(ik.arm_joint_indices[side]):
        q_target[idx] = np.clip(
            0.04 * (k + 1),
            ik._model.lowerPositionLimit[idx],
            ik._model.upperPositionLimit[idx],
        )

    target = {side: pin.SE3(ik.forward_kinematics_full(q_target, side))}
    result = ik.solve_position(target, q0, rotation_axes="XZ", rotation_gain=0.2)
    solved_T = ik.forward_kinematics_full(result.q, side)
    tracked_rotation_error = pin.log3(target[side].rotation @ solved_T[:3, :3].T)[[0, 2]]

    assert result.position_error[side] < 1e-3
    assert np.linalg.norm(tracked_rotation_error) < 5e-3


def test_pose_convergence_ignores_masked_orientation_axis() -> None:
    ik = BimanualIKSolver(orientation_cost=np.array([1.0, 1.0, 0.0]))
    q0 = ik.neutral_q.copy()
    current = pin.SE3(ik.forward_kinematics_full(q0, "right"))
    target = pin.SE3(current.rotation @ pin.exp3(np.array([0.0, 0.0, 0.5])), current.translation)

    result = ik.evaluate({"right": target}, q0)

    assert result.orientation_error["right"] < 1e-12
    assert result.converged["right"]


def test_pose_convergence_checks_unmasked_orientation_axis() -> None:
    ik = BimanualIKSolver(orientation_cost=np.array([1.0, 1.0, 1.0]))
    q0 = ik.neutral_q.copy()
    current = pin.SE3(ik.forward_kinematics_full(q0, "right"))
    target = pin.SE3(current.rotation @ pin.exp3(np.array([0.0, 0.0, 0.5])), current.translation)

    result = ik.evaluate({"right": target}, q0)

    assert result.orientation_error["right"] > ik._orientation_tolerance
    assert not result.converged["right"]


def test_orca_panda_config_builds_right_only_solver() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)

    assert ik.arm_joint_names == {
        "right": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
    }
    assert list(ik.arm_joint_indices) == ["right"]
    assert np.all(ik.neutral_q >= ik._model.lowerPositionLimit)
    assert np.all(ik.neutral_q <= ik._model.upperPositionLimit)
