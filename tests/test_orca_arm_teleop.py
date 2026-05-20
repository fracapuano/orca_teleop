from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from orca_teleop.orca_arm_ik import BimanualIKSolver, orca_panda_right_ik_config
from orca_teleop.orca_arm_teleop import (
    OrcaArmTeleopConfig,
    OrcaArmTeleopController,
    OrcaArmTeleopFrame,
    _retargeted_action_for_side,
    metaquest_wrist_pose_to_robot_se3,
)


def _rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_metaquest_wrist_pose_to_robot_se3_uses_sdk_flu_conversion(monkeypatch) -> None:
    hts_convert = pytest.importorskip("hand_tracking_sdk.convert")
    calls = []

    def fake_position(pos, basis):
        calls.append(("position", pos, basis))
        return (1.0, 2.0, 3.0)

    def fake_rotation_matrix(qx, qy, qz, qw, basis):
        calls.append(("rotation", (qx, qy, qz, qw), basis))
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    monkeypatch.setattr(hts_convert, "basis_transform_position", fake_position)
    monkeypatch.setattr(hts_convert, "basis_transform_rotation_matrix", fake_rotation_matrix)

    T = metaquest_wrist_pose_to_robot_se3(
        np.array([0.1, 0.2, 0.3]),
        _rot_z(0.4),
    )

    assert calls[0] == (
        "position",
        (0.1, 0.2, 0.3),
        hts_convert.BASIS_UNITY_LEFT_TO_FLU,
    )
    assert calls[1][0] == "rotation"
    assert calls[1][2] == hts_convert.BASIS_UNITY_LEFT_TO_FLU
    np.testing.assert_allclose(T.translation, np.array([1.0, 2.0, 3.0]), atol=1e-12)
    np.testing.assert_allclose(T.rotation, np.eye(3), atol=1e-12)


def test_retargeters_use_shared_right_hand_convention(monkeypatch, tmp_path: Path) -> None:
    right_urdf = tmp_path / "orcahand_right.urdf"
    left_urdf = tmp_path / "orcahand_left.urdf"
    right_urdf.write_text("<robot/>")
    left_urdf.write_text("<robot/>")
    calls = []

    class StubRetargeter:
        @staticmethod
        def from_paths(model_path, urdf_path, **kwargs):
            calls.append((model_path, urdf_path, kwargs))
            return object()

    monkeypatch.setattr("orca_teleop.orca_arm_teleop.Retargeter", StubRetargeter)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            hand_model_path="/models/orcahand_right/config.yaml",
            hand_urdf_path=str(right_urdf),
        )
    )

    controller._retargeter_for("left")
    controller._retargeter_for("right")

    assert calls == [
        (
            "/models/orcahand_right/config.yaml",
            str(right_urdf),
            {"hand_type_override": "right"},
        ),
        (
            "/models/orcahand_right/config.yaml",
            str(right_urdf),
            {"hand_type_override": "right"},
        ),
    ]


def test_left_side_only_flips_wrist_for_shared_right_retargeter() -> None:
    from orca_core import OrcaJointPositions

    action = OrcaJointPositions({"index_mcp": 30.0, "index_pip": 45.0, "wrist": 10.0})

    out = _retargeted_action_for_side(action, "left").as_dict()

    assert out["index_mcp"] == 30.0
    assert out["index_pip"] == 45.0
    assert out["wrist"] == -10.0


def test_initial_anchor_does_not_build_retargeter(monkeypatch) -> None:
    calls = []

    class StubRetargeter:
        @staticmethod
        def from_paths(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

    monkeypatch.setattr("orca_teleop.orca_arm_teleop.Retargeter", StubRetargeter)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            still_window_samples=3,
            still_threshold_m=0.01,
            manual_scale=0.5,
            ik_mode="position",
        )
    )
    frame = OrcaArmTeleopFrame(
        handedness="right",
        timestamp_ns=1,
        wrist_pose=pin.SE3(np.eye(3), np.zeros(3)),
        keypoints=np.zeros((21, 3), dtype=np.float64),
        keypoint_source="canonical",
    )

    result = controller.step([frame])

    assert result.statuses["right"] == "tracking"
    assert result.arm_angles
    assert calls == []


def test_stationary_frames_keep_tracking_from_startup_anchor() -> None:
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            manual_scale=0.5,
            ik_mode="position",
        )
    )
    frame = OrcaArmTeleopFrame(
        handedness="right",
        timestamp_ns=1,
        wrist_pose=pin.SE3(np.eye(3), np.zeros(3)),
    )

    statuses = [controller.step([frame]).statuses["right"] for _ in range(5)]

    assert statuses == ["tracking"] * 5


def test_controller_accepts_per_side_per_axis_translation_scale() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            manual_scale={"right": np.array([2.0, 3.0, 4.0])},
            translation_frame="world",
        ),
        ik=ik,
    )

    first_p = np.array([0.1, 0.2, 0.3])
    now_p = first_p + np.array([0.01, 0.02, 0.03])

    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(np.eye(3), first_p),
            )
        ]
    )
    result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(np.eye(3), now_p),
            )
        ]
    )

    expected_dp = np.array([0.02, 0.06, 0.12])
    np.testing.assert_allclose(
        result.target_poses["right"].translation,
        controller.home_poses["right"].translation + expected_dp,
        atol=1e-12,
    )


def test_controller_can_leave_translation_unclipped() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            manual_scale=1.0,
            workspace_delta_limits_m=None,
            translation_frame="world",
        ),
        ik=ik,
    )

    first_p = np.zeros(3)
    large_dp = np.array([2.0, -2.0, 2.0])

    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(np.eye(3), first_p),
            )
        ]
    )
    result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(np.eye(3), first_p + large_dp),
            )
        ]
    )

    np.testing.assert_allclose(
        result.target_poses["right"].translation,
        controller.home_poses["right"].translation + large_dp,
        atol=1e-12,
    )


def test_controller_maps_operator_workspace_endpoints_to_robot_endpoints() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    robot_lo = np.array([-0.5, -0.25, -1.0])
    robot_hi = np.array([1.0, 0.5, 2.0])
    operator_lo = np.array([-1.0, -2.0, -3.0])
    operator_hi = np.array([2.0, 4.0, 6.0])
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            workspace_delta_limits_m={"right": (tuple(robot_lo), tuple(robot_hi))},
            operator_workspace_limits_m={"right": (tuple(operator_lo), tuple(operator_hi))},
        ),
        ik=ik,
    )

    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(np.eye(3), np.zeros(3)),
            )
        ]
    )

    result_hi = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(np.eye(3), operator_hi),
            )
        ]
    )
    np.testing.assert_allclose(
        result_hi.target_poses["right"].translation,
        controller.home_poses["right"].translation + robot_hi,
        atol=1e-12,
    )

    result_lo = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=3,
                wrist_pose=pin.SE3(np.eye(3), operator_lo),
            )
        ]
    )
    np.testing.assert_allclose(
        result_lo.target_poses["right"].translation,
        controller.home_poses["right"].translation + robot_lo,
        atol=1e-12,
    )


def test_controller_maps_operator_x_endpoint_to_robot_x_endpoint_by_default() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    robot_lo = np.array([-0.5, -0.25, -1.0])
    robot_hi = np.array([1.0, 0.5, 2.0])
    operator_lo = np.array([-1.0, -2.0, -3.0])
    operator_hi = np.array([2.0, 4.0, 6.0])
    operator_anchor = np.array([0.25, 1.0, 1.5])
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            workspace_delta_limits_m={"right": (tuple(robot_lo), tuple(robot_hi))},
            operator_workspace_limits_m={"right": (tuple(operator_lo), tuple(operator_hi))},
        ),
        ik=ik,
    )

    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(_rot_z(np.pi / 2.0), operator_anchor),
            )
        ]
    )

    x_hi_pose = operator_anchor.copy()
    x_hi_pose[0] = operator_hi[0]
    result_hi = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(_rot_z(np.pi / 2.0), x_hi_pose),
            )
        ]
    )

    expected_hi = np.array([robot_hi[0], 0.0, 0.0])
    np.testing.assert_allclose(
        result_hi.target_poses["right"].translation,
        controller.home_poses["right"].translation + expected_hi,
        atol=1e-12,
    )

    x_lo_pose = operator_anchor.copy()
    x_lo_pose[0] = operator_lo[0]
    result_lo = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=3,
                wrist_pose=pin.SE3(_rot_z(np.pi / 2.0), x_lo_pose),
            )
        ]
    )

    expected_lo = np.array([robot_lo[0], 0.0, 0.0])
    np.testing.assert_allclose(
        result_lo.target_poses["right"].translation,
        controller.home_poses["right"].translation + expected_lo,
        atol=1e-12,
    )


def test_wrist_translation_target_uses_operator_local_delta() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            manual_scale=1.0,
        ),
        ik=ik,
    )

    first_R = _rot_z(np.pi / 2.0)
    local_forward = np.array([0.1, 0.0, 0.0])
    now_p = first_R @ local_forward

    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(first_R, np.zeros(3)),
            )
        ]
    )
    result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(first_R, now_p),
            )
        ]
    )

    expected_dp = controller.home_poses["right"].rotation @ local_forward
    np.testing.assert_allclose(
        result.target_poses["right"].translation,
        controller.home_poses["right"].translation + expected_dp,
        atol=1e-12,
    )


def test_wrist_rotation_target_uses_operator_local_delta() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            manual_scale=0.0,
        ),
        ik=ik,
    )

    first_R = _rot_x(0.7)
    local_delta_R = _rot_y(0.35)
    now_R = first_R @ local_delta_R

    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(first_R, np.zeros(3)),
            )
        ]
    )
    result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(now_R, np.zeros(3)),
            )
        ]
    )

    expected_R = controller.home_poses["right"].rotation @ local_delta_R

    np.testing.assert_allclose(
        result.target_poses["right"].rotation,
        expected_R,
        atol=1e-12,
    )


def test_controller_uses_solver_sides_by_default() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            still_window_samples=2,
            still_threshold_m=0.01,
            manual_scale=0.5,
            ik_mode="position",
        ),
        ik=ik,
    )

    right_frame = OrcaArmTeleopFrame(
        handedness="right",
        timestamp_ns=1,
        wrist_pose=pin.SE3(np.eye(3), np.zeros(3)),
    )
    left_frame = OrcaArmTeleopFrame(
        handedness="left",
        timestamp_ns=2,
        wrist_pose=pin.SE3(np.eye(3), np.ones(3)),
    )

    for _ in range(2):
        result = controller.step([left_frame, right_frame])

    assert result.statuses == {"right": "tracking"}
    assert set(result.arm_angles) == {"right"}


def test_workspace_position_clip_projects_translation_not_orientation() -> None:
    ik = BimanualIKSolver(
        ik_config=orca_panda_right_ik_config(),
        max_iterations=30,
        orientation_cost=0.0,
    )
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="pose",
            orientation_cost=0.0,
            manual_scale=1.0,
            workspace_delta_limits_m=None,
            translation_frame="world",
            workspace_position_clip=True,
            workspace_position_clip_tolerance_m=1e-6,
            max_joint_step_rad=10.0,
        ),
        ik=ik,
    )

    first = pin.SE3(np.eye(3), np.zeros(3))
    controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=first,
            )
        ]
    )

    target_rotation = _rot_z(0.3)
    result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(target_rotation, np.array([10.0, 0.0, 0.0])),
            )
        ]
    )

    target = result.target_poses["right"]
    unprojected_translation = controller.home_poses["right"].translation + np.array(
        [10.0, 0.0, 0.0]
    )
    assert np.linalg.norm(target.translation - unprojected_translation) > 1.0
    np.testing.assert_allclose(
        target.rotation,
        controller.home_poses["right"].rotation @ target_rotation,
        atol=1e-12,
    )


def test_operator_neutral_gate_holds_home_until_recorded_neutral_matches() -> None:
    ik = BimanualIKSolver(ik_config=orca_panda_right_ik_config(), max_iterations=1)
    neutral_pose = (
        (0.0, 0.0, 0.0),
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    controller = OrcaArmTeleopController(
        OrcaArmTeleopConfig(
            active_sides=("right",),
            ik_mode="position",
            manual_scale=1.0,
            workspace_delta_limits_m=None,
            translation_frame="world",
            require_operator_neutral=True,
            operator_neutral_position_tolerance_m=0.05,
            operator_neutral_orientation_tolerance_rad=0.1,
            operator_neutral_poses_flu={"right": neutral_pose},
        ),
        ik=ik,
    )

    far_result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=1,
                wrist_pose=pin.SE3(np.eye(3), np.array([0.2, 0.0, 0.0])),
            )
        ]
    )

    assert far_result.statuses["right"] == "awaiting_neutral"
    np.testing.assert_allclose(
        far_result.target_poses["right"].translation,
        controller.home_poses["right"].translation,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        far_result.operator_poses["right"].translation,
        controller.home_poses["right"].translation + np.array([0.2, 0.0, 0.0]),
        atol=1e-12,
    )

    neutral_result = controller.step(
        [
            OrcaArmTeleopFrame(
                handedness="right",
                timestamp_ns=2,
                wrist_pose=pin.SE3(np.eye(3), np.zeros(3)),
            )
        ]
    )

    assert neutral_result.statuses["right"] == "tracking"
