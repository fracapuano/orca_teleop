"""Pose accessors and interpolation, with the sign-continuity contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tron_arm.poses import (
    Pose,
    as_pose,
    lerp,
    matrix_to_quat,
    pose_lerp,
    quat_angle,
    quat_to_matrix,
    slerp,
)

Q_Z90 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
R_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestPose:
    def test_accessor_names_match_upstream(self):
        pose = Pose([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        assert hasattr(pose, "matrix") and callable(getattr(pose, "as_xyz_wxyz"))
        assert pose.as_xyz_wxyz().tolist() == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]

    def test_as_xyz_wxyz_is_position_then_scalar_first_quat(self):
        pose = Pose([0.1, 0.2, 0.3], Q_Z90)
        v = pose.as_xyz_wxyz()
        assert v.shape == (7,)
        np.testing.assert_allclose(v[:3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(v[3:], Q_Z90)
        # The scalar leads; this is NOT [x,y,z,qx,qy,qz,qw].
        assert v[3] == pytest.approx(math.cos(math.pi / 4))

    def test_matrix_round_trip(self):
        pose = Pose([0.4, -0.2, 0.1], Q_Z90)
        back = Pose.from_matrix(pose.matrix)
        np.testing.assert_allclose(back.p, pose.p, atol=1e-12)
        np.testing.assert_allclose(back.q_wxyz, pose.q_wxyz, atol=1e-12)

    def test_matrix_is_homogeneous_with_expected_rotation(self):
        m = Pose([1.0, 2.0, 3.0], Q_Z90).matrix
        assert m.shape == (4, 4)
        np.testing.assert_allclose(m[3], [0, 0, 0, 1])
        np.testing.assert_allclose(m[:3, :3], R_Z90, atol=1e-12)
        np.testing.assert_allclose(m[:3, 3], [1, 2, 3])

    def test_arrays_are_copied_and_read_only(self):
        """We copy where upstream aliases -- see the Pose docstring."""
        p = np.array([1.0, 2.0, 3.0])
        pose = Pose(p, [1.0, 0.0, 0.0, 0.0])
        assert not np.shares_memory(p, pose.position_m)
        p[0] = 99.0
        assert pose.position_m[0] == 1.0
        with pytest.raises(ValueError):
            pose.position_m[0] = 5.0

    def test_field_names_mirror_upstream(self):
        """orca_teleop's Pose fields are position_m / orientation_wxyz."""
        pose = Pose([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(pose.position_m, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pose.orientation_wxyz, [1.0, 0.0, 0.0, 0.0])
        # p / q_wxyz are aliases onto the same arrays, not copies.
        assert pose.p is pose.position_m
        assert pose.q_wxyz is pose.orientation_wxyz

    @pytest.mark.parametrize("q", [
        [0.0, 0.0, 0.0, 0.0],       # an all-default proto Pose: qw=0, NOT identity
        [-2.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
    ])
    def test_non_unit_quaternion_raises_rather_than_being_normalised(self, q):
        """Upstream raises here; normalising ~0 would yield a silent NaN pose."""
        with pytest.raises(ValueError, match="unit quaternion"):
            Pose([0, 0, 0], q)

    def test_quaternion_within_tolerance_is_normalised_sign_intact(self):
        from tron_arm.poses import QUATERNION_NORM_TOLERANCE

        assert QUATERNION_NORM_TOLERANCE == 1e-3
        pose = Pose([0, 0, 0], [-(1.0 + 5e-4), 0.0, 0.0, 0.0])
        assert pose.orientation_wxyz[0] == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "p,q",
        [([np.nan, 0, 0], [1, 0, 0, 0]), ([0, 0, 0], [np.nan, 0, 0, 0]),
         ([np.inf, 0, 0], [1, 0, 0, 0])],
    )
    def test_rejects_non_finite(self, p, q):
        with pytest.raises(ValueError):
            Pose(p, q)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            Pose([1, 2], [1, 0, 0, 0])
        with pytest.raises(ValueError):
            Pose([1, 2, 3], [1, 0, 0])

    @pytest.mark.parametrize("bad,match", [
        (np.diag([2.0, 2.0, 2.0, 1.0]), "orthonormal"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "proper rotation"),
    ])
    def test_from_matrix_validates_the_rotation_block(self, bad, match):
        """A scaled or reflected transform must not become a plausible quat."""
        with pytest.raises(ValueError, match=match):
            Pose.from_matrix(bad)

    def test_inverse(self):
        pose = Pose([0.4, -0.2, 0.1], Q_Z90)
        np.testing.assert_allclose(pose.matrix @ pose.inverse().matrix, np.eye(4), atol=1e-12)


class TestQuatMatrix:
    def test_known_rotation(self):
        np.testing.assert_allclose(quat_to_matrix(Q_Z90), R_Z90, atol=1e-12)
        np.testing.assert_allclose(matrix_to_quat(R_Z90), Q_Z90, atol=1e-12)

    @pytest.mark.parametrize(
        "q",
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
         [0.5, 0.5, 0.5, 0.5], [0.2, -0.3, 0.6, 0.7]],
    )
    def test_round_trip_all_shepperd_branches(self, q):
        q = np.asarray(q, dtype=np.float64)
        q = q / np.linalg.norm(q)
        back = matrix_to_quat(quat_to_matrix(q))
        # A matrix cannot carry sign, so compare up to sign here (and only here).
        assert np.allclose(back, q, atol=1e-9) or np.allclose(back, -q, atol=1e-9)


class TestSignContinuity:
    """CLAUDE.md: quaternions arrive sign-continuous; never re-canonicalise."""

    def test_slerp_does_not_mutate_input_sign(self):
        q0 = np.array([-0.7071067811865476, 0.0, 0.0, -0.7071067811865476])
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q0_before, q1_before = q0.copy(), q1.copy()
        slerp(q0, q1, 0.5)
        np.testing.assert_array_equal(q0, q0_before)
        np.testing.assert_array_equal(q1, q1_before)
        assert q0[0] < 0.0  # still negative: nothing forced w >= 0

    def test_slerp_takes_the_long_way_when_dot_is_negative(self):
        """The forbidden ``if dot < 0: q1 = -q1`` fix-up would give the short arc."""
        q0 = np.array([-1.0, 0.0, 0.0, 0.0])
        q1 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
        assert float(np.dot(q0, q1)) < 0.0
        got = slerp(q0, q1, 0.5)
        shortest = slerp(-q0, q1, 0.5)  # what sign folding would have produced
        assert not np.allclose(got, shortest, atol=1e-6)
        assert not np.allclose(got, -shortest, atol=1e-6)

    def test_quat_angle_reports_the_arc_slerp_actually_takes(self):
        q0 = np.array([-1.0, 0.0, 0.0, 0.0])
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        # Antipodal: a sign-folding implementation would report 0.
        assert quat_angle(q0, q1) == pytest.approx(2 * math.pi, abs=1e-9)

    def test_pose_lerp_preserves_a_negative_scalar_component(self):
        a = Pose([0, 0, 0], [-1.0, 0.0, 0.0, 0.0])
        b = Pose([1, 0, 0], [-0.9987502603949663, 0.0, 0.0, -0.04997916927067833])
        out = pose_lerp(a, b, 0.5)
        assert out.q_wxyz[0] < 0.0


class TestInterpolation:
    def test_lerp_endpoints_and_midpoint(self):
        a, b = np.array([0.0, 0.0, 0.0]), np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(lerp(a, b, 0.0), a)
        np.testing.assert_allclose(lerp(a, b, 1.0), b)
        np.testing.assert_allclose(lerp(a, b, 0.25), [0.5, 1.0, 1.5])

    def test_lerp_does_not_mutate_inputs(self):
        a, b = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
        lerp(a, b, 0.5)
        np.testing.assert_array_equal(a, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(b, [2.0, 2.0, 2.0])

    def test_slerp_endpoints(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(slerp(q0, Q_Z90, 0.0), q0, atol=1e-12)
        np.testing.assert_allclose(slerp(q0, Q_Z90, 1.0), Q_Z90, atol=1e-12)

    def test_slerp_midpoint_halves_the_angle(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        mid = slerp(q0, Q_Z90, 0.5)
        assert quat_angle(q0, mid) == pytest.approx(math.pi / 4, abs=1e-9)
        assert quat_angle(mid, Q_Z90) == pytest.approx(math.pi / 4, abs=1e-9)

    def test_slerp_output_is_unit_norm(self):
        q1 = np.array([0.2, -0.3, 0.6, 0.7])
        q1 = q1 / np.linalg.norm(q1)
        for t in np.linspace(0.0, 1.0, 11):
            assert np.linalg.norm(slerp(np.array([1.0, 0, 0, 0]), q1, t)) == pytest.approx(1.0)

    def test_slerp_handles_identical_quaternions(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(slerp(q, q, 0.37), q, atol=1e-12)

    def test_slerp_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            slerp(np.zeros(3), np.zeros(4), 0.5)


class TestAsPose:
    def test_passthrough(self):
        pose = Pose([1, 2, 3], [1, 0, 0, 0])
        assert as_pose(pose) is pose

    def test_accepts_a_duck_typed_upstream_pose(self):
        """Anything exposing ``as_xyz_wxyz()`` works -- e.g. orca_teleop's Pose."""

        class Upstream:
            def as_xyz_wxyz(self):
                return np.array([1.0, 2.0, 3.0, -1.0, 0.0, 0.0, 0.0])

            @property
            def matrix(self):
                raise AssertionError("must prefer as_xyz_wxyz(): matrix loses quaternion sign")

        got = as_pose(Upstream())
        np.testing.assert_allclose(got.p, [1, 2, 3])
        assert got.q_wxyz[0] == pytest.approx(-1.0)  # sign survived

    def test_accepts_matrix_only_object(self):
        class MatrixOnly:
            matrix = Pose([1, 2, 3], Q_Z90).matrix

        np.testing.assert_allclose(as_pose(MatrixOnly()).p, [1, 2, 3])

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError):
            as_pose(object())
