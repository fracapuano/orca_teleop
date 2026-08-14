"""Mapping geometry: property tests (hypothesis) plus the constructed cases.

The properties are the ones an operator would notice if they broke:
zero jump at engage, rotation composing in the body frame, and translation
arriving on the axes the pre-rotation M = R_r0 @ R_o0.T says it should.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tron_arm.config import Config, ConfigError, MappingConfig, load_config
from tron_arm.mapping import (
    ClutchMapper,
    MappingError,
    clamp_to_workspace,
    compose,
    invert_transform,
    scale_delta_translation,
)
from tron_arm.poses import Pose, quat_angle

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# -- strategies ----------------------------------------------------------
def finite(lo: float, hi: float):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


@st.composite
def unit_vectors(draw) -> np.ndarray:
    v = np.array([draw(finite(-1.0, 1.0)) for _ in range(3)])
    norm = float(np.linalg.norm(v))
    assume(norm > 1e-6)
    return v / norm


@st.composite
def rotations(draw) -> np.ndarray:
    """A proper rotation matrix, via axis-angle (no scipy, no gimbal games)."""
    axis = draw(unit_vectors())
    angle = draw(finite(-math.pi, math.pi))
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


@st.composite
def transforms(draw, reach: float = 1.0) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = draw(rotations())
    t[:3, 3] = [draw(finite(-reach, reach)) for _ in range(3)]
    return t


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = position
    return t


@pytest.fixture
def cfg() -> Config:
    return load_config()


def mapper(cfg: Config, *, scale: float | None = None, **kwargs) -> ClutchMapper:
    if scale is not None:
        cfg = dataclasses.replace(cfg, scale=scale)
    return ClutchMapper(cfg, **kwargs)


# -- helpers under test --------------------------------------------------
class TestTransformHelpers:
    @given(t=transforms())
    @SETTINGS
    def test_invert_is_a_true_inverse(self, t):
        np.testing.assert_allclose(t @ invert_transform(t), np.eye(4), atol=1e-9)
        np.testing.assert_allclose(invert_transform(t) @ t, np.eye(4), atol=1e-9)

    @given(t=transforms())
    @SETTINGS
    def test_invert_stays_exactly_orthonormal(self, t):
        """np.linalg.inv drifts; the transpose form does not."""
        r = invert_transform(t)[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)

    def test_invert_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="4x4"):
            invert_transform(np.eye(3))

    @given(a=transforms(), b=transforms(), c=transforms())
    @SETTINGS
    def test_compose_is_left_to_right(self, a, b, c):
        np.testing.assert_allclose(compose(a, b, c), a @ b @ c, atol=1e-12)

    def test_compose_of_nothing_is_identity(self):
        np.testing.assert_allclose(compose(), np.eye(4))

    @given(t=transforms(), s=finite(0.01, 5.0))
    @SETTINGS
    def test_scale_delta_touches_translation_only(self, t, s):
        out = scale_delta_translation(t, s)
        np.testing.assert_allclose(out[:3, :3], t[:3, :3], atol=1e-12)
        np.testing.assert_allclose(out[:3, 3], s * t[:3, 3], atol=1e-12)

    def test_scale_delta_does_not_mutate_input(self):
        t = np.eye(4)
        t[:3, 3] = [1.0, 2.0, 3.0]
        scale_delta_translation(t, 0.5)
        np.testing.assert_array_equal(t[:3, 3], [1.0, 2.0, 3.0])

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_scale_must_be_positive_and_finite(self, bad):
        with pytest.raises(ValueError, match="scale"):
            scale_delta_translation(np.eye(4), bad)


# -- the three headline properties --------------------------------------
class TestZeroJumpAtEngage:
    @given(op0=transforms(), robot0=transforms(reach=0.5), s=finite(0.01, 2.0))
    @SETTINGS
    def test_mapping_the_latch_pose_returns_the_robot_origin(self, cfg, op0, robot0, s):
        """T_op == T_op0 => target == T_robot0, for any scale and any origins."""
        m = mapper(cfg, scale=s)
        m.latch(op0, robot0)
        np.testing.assert_allclose(m.map(op0).matrix, robot0, atol=1e-9)

    @given(op0=transforms(), robot0=transforms(reach=0.5), s=finite(0.01, 2.0))
    @SETTINGS
    def test_zero_jump_holds_in_world_mode_too(self, cfg, op0, robot0, s):
        world = dataclasses.replace(
            cfg,
            scale=s,
            mapping=MappingConfig(translation_frame="world", world_frame_axes_verified=True),
        )
        m = ClutchMapper(world)
        m.latch(op0, robot0)
        np.testing.assert_allclose(m.map(op0).matrix, robot0, atol=1e-9)

    def test_re_latching_mid_stream_does_not_move_the_target(self, cfg):
        """The property the whole lazy-latch dance depends on."""
        m = mapper(cfg)
        op0 = transform(rotation_about([0, 0, 1], 0.3), np.array([0.1, 0.2, 0.3]))
        robot0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        m.latch(op0, robot0)
        op = transform(rotation_about([0, 1, 0], 0.2), np.array([0.15, 0.25, 0.35]))
        first = m.map(op)
        # Re-latch here (new epoch), against the last commanded target.
        m.clear()
        m.latch(op, first)
        np.testing.assert_allclose(m.map(op).matrix, first.matrix, atol=1e-9)


class TestBodyFrameRotation:
    @given(u=unit_vectors(), a=finite(-3.0, 3.0), op0=transforms(), robot0=transforms(reach=0.5))
    @SETTINGS
    def test_wrist_rotation_about_its_own_axis_composes_on_the_right(
        self, cfg, u, a, op0, robot0
    ):
        """R_target == R_r0 @ R(u, a) for a body-frame operator rotation."""
        m = mapper(cfg)
        m.latch(op0, robot0)
        # Rotate the wrist about its own axis u: post-multiply in its own frame.
        r_delta = rotation_about(u, a)
        t_op = op0 @ transform(r_delta, np.zeros(3))
        got = m.map(t_op).matrix[:3, :3]
        np.testing.assert_allclose(got, robot0[:3, :3] @ r_delta, atol=1e-9)

    @given(u=unit_vectors(), a=finite(-3.0, 3.0), op0=transforms(), robot0=transforms(reach=0.5))
    @SETTINGS
    def test_a_pure_wrist_rotation_does_not_translate_the_target(self, cfg, u, a, op0, robot0):
        m = mapper(cfg)
        m.latch(op0, robot0)
        t_op = op0 @ transform(rotation_about(u, a), np.zeros(3))
        np.testing.assert_allclose(m.map(t_op).position_m, robot0[:3, 3], atol=1e-9)

    def test_rotation_is_unaffected_by_scale(self, cfg):
        """Scale is a translation knob only; rotating slower is not a thing."""
        op0 = transform(rotation_about([1, 0, 0], 0.4), np.array([0.1, 0.0, 0.0]))
        robot0 = transform(np.eye(3), np.array([0.4, -0.2, 0.0]))
        t_op = op0 @ transform(rotation_about([0, 0, 1], 0.7), np.zeros(3))
        rotations_out = []
        for s in (0.25, 1.0, 4.0):
            m = mapper(cfg, scale=s)
            m.latch(op0, robot0)
            rotations_out.append(m.map(t_op).matrix[:3, :3])
        np.testing.assert_allclose(rotations_out[0], rotations_out[1], atol=1e-12)
        np.testing.assert_allclose(rotations_out[1], rotations_out[2], atol=1e-12)


class TestTranslationPreRotation:
    """Operator translation reaches the base rotated by M = R_r0 @ R_o0.T."""

    @given(op0=transforms(), robot0=transforms(reach=0.5), d=unit_vectors(), s=finite(0.05, 2.0))
    @SETTINGS
    def test_translation_maps_through_m(self, cfg, op0, robot0, d, s):
        m = mapper(cfg, scale=s)
        m.latch(op0, robot0)
        step = 0.05 * d
        t_op = transform(op0[:3, :3], op0[:3, 3] + step)  # pure reference-frame translation
        expected_m = robot0[:3, :3] @ op0[:3, :3].T
        np.testing.assert_allclose(expected_m, m.translation_prerotation, atol=1e-12)
        np.testing.assert_allclose(
            m.map(t_op).position_m, robot0[:3, 3] + s * (expected_m @ step), atol=1e-9
        )

    def test_thirty_degree_engage_mismatch_rotates_the_mapped_direction_exactly(self, cfg):
        """The constructed case: engage 30 deg off and the arm goes 30 deg off.

        Operator frame yawed +30 deg from the robot frame at engage, so
        M = R_r0 @ R_o0.T is a -30 deg yaw. Pushing the hand along the operator's
        +X must move the arm along a direction rotated by exactly that.
        """
        thirty = math.radians(30.0)
        r_o0 = rotation_about([0, 0, 1], thirty)
        r_r0 = np.eye(3)
        op0 = transform(r_o0, np.array([0.1, 0.2, 0.3]))
        robot0 = transform(r_r0, np.array([0.40, -0.20, 0.0]))
        m = mapper(cfg, scale=1.0)
        m.latch(op0, robot0)

        step = np.array([0.05, 0.0, 0.0])  # along the REFERENCE frame's +X
        t_op = transform(r_o0, op0[:3, 3] + step)
        moved = m.map(t_op).position_m - robot0[:3, 3]

        expected_m = r_r0 @ r_o0.T
        np.testing.assert_allclose(expected_m, m.translation_prerotation, atol=1e-12)
        np.testing.assert_allclose(moved, expected_m @ step, atol=1e-12)

        # The angle between commanded and mapped direction is exactly 30 deg.
        cos = float(np.dot(moved, step) / (np.linalg.norm(moved) * np.linalg.norm(step)))
        assert math.degrees(math.acos(np.clip(cos, -1, 1))) == pytest.approx(30.0, abs=1e-9)
        # Length is preserved: M is a rotation, so it reorients without scaling.
        assert float(np.linalg.norm(moved)) == pytest.approx(float(np.linalg.norm(step)), abs=1e-12)

    def test_m_is_identity_when_the_frames_agree_at_engage(self, cfg):
        m = mapper(cfg)
        m.latch(transform(np.eye(3), np.array([0.1, 0.0, 0.0])),
                transform(np.eye(3), np.array([0.4, -0.2, 0.0])))
        np.testing.assert_allclose(m.translation_prerotation, np.eye(3), atol=1e-12)

    @given(op0=transforms(), robot0=transforms(reach=0.5))
    @SETTINGS
    def test_m_is_always_a_proper_rotation(self, cfg, op0, robot0):
        m = mapper(cfg)
        m.latch(op0, robot0)
        got = m.translation_prerotation
        np.testing.assert_allclose(got @ got.T, np.eye(3), atol=1e-9)
        assert float(np.linalg.det(got)) == pytest.approx(1.0, abs=1e-9)


# -- scale ---------------------------------------------------------------
class TestScale:
    @given(op0=transforms(), robot0=transforms(reach=0.5), d=unit_vectors(), s=finite(0.05, 3.0))
    @SETTINGS
    def test_translation_is_linear_in_scale(self, cfg, op0, robot0, d, s):
        step = 0.1 * d
        t_op = transform(op0[:3, :3], op0[:3, 3] + step)
        base = mapper(cfg, scale=1.0)
        base.latch(op0, robot0)
        scaled = mapper(cfg, scale=s)
        scaled.latch(op0, robot0)
        moved_base = base.map(t_op).position_m - robot0[:3, 3]
        moved_scaled = scaled.map(t_op).position_m - robot0[:3, 3]
        np.testing.assert_allclose(moved_scaled, s * moved_base, atol=1e-9)

    def test_scale_applies_to_the_delta_not_the_absolute_target(self, cfg):
        """The bug this ordering prevents: scaling T_target[:3,3] instead."""
        m = mapper(cfg, scale=0.5)
        robot0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        op0 = transform(np.eye(3), np.zeros(3))
        m.latch(op0, robot0)
        got = m.map(op0).position_m
        np.testing.assert_allclose(got, [0.40, -0.20, 0.0], atol=1e-12)
        assert not np.allclose(got, 0.5 * np.array([0.40, -0.20, 0.0]))


# -- world mode ----------------------------------------------------------
class TestWorldMode:
    def _world_cfg(self, cfg: Config, *, verified: bool = True, scale: float = 0.5) -> Config:
        return dataclasses.replace(
            cfg,
            scale=scale,
            mapping=MappingConfig(
                translation_frame="world", world_frame_axes_verified=verified
            ),
        )

    def test_unverified_axes_refuse_world_mode_at_config_level(self, cfg):
        with pytest.raises(ConfigError, match="world_frame_axes_verified"):
            MappingConfig(translation_frame="world", world_frame_axes_verified=False)

    def test_unverified_axes_refuse_world_mode_at_mapper_level(self, cfg):
        """Belt and braces: a hand-built Config must not sneak past the gate."""
        sneaky = dataclasses.replace(cfg, mapping=MappingConfig(translation_frame="body"))
        object.__setattr__(sneaky.mapping, "translation_frame", "world")
        with pytest.raises(MappingError, match="9G-09"):
            ClutchMapper(sneaky)

    def test_error_names_the_axis_check(self, cfg):
        sneaky = dataclasses.replace(cfg, mapping=MappingConfig(translation_frame="body"))
        object.__setattr__(sneaky.mapping, "translation_frame", "world")
        with pytest.raises(MappingError) as excinfo:
            ClutchMapper(sneaky)
        message = str(excinfo.value)
        assert "9G-09" in message and "world_frame_axes_verified" in message

    @given(op0=transforms(), robot0=transforms(reach=0.5), d=unit_vectors(), s=finite(0.05, 2.0))
    @SETTINGS
    def test_world_translation_is_the_raw_reference_delta(self, cfg, op0, robot0, d, s):
        """p_target = p_r0 + s * (p_op - p_op0), with no M pre-rotation."""
        m = ClutchMapper(self._world_cfg(cfg, scale=s))
        m.latch(op0, robot0)
        step = 0.1 * d
        t_op = transform(op0[:3, :3], op0[:3, 3] + step)
        np.testing.assert_allclose(m.map(t_op).position_m, robot0[:3, 3] + s * step, atol=1e-9)

    @given(op0=transforms(), robot0=transforms(reach=0.5), t_op=transforms())
    @SETTINGS
    def test_world_rotation_is_identical_to_body_mode(self, cfg, op0, robot0, t_op):
        body = ClutchMapper(cfg)
        body.latch(op0, robot0)
        world = ClutchMapper(self._world_cfg(cfg, scale=cfg.scale))
        world.latch(op0, robot0)
        np.testing.assert_allclose(
            world.map(t_op).matrix[:3, :3], body.map(t_op).matrix[:3, :3], atol=1e-9
        )

    def test_the_two_modes_differ_when_frames_are_misaligned(self, cfg):
        """If they never differed the config knob would be pointless."""
        r_o0 = rotation_about([0, 0, 1], math.radians(30.0))
        op0 = transform(r_o0, np.array([0.1, 0.2, 0.3]))
        robot0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        step = np.array([0.05, 0.0, 0.0])
        t_op = transform(r_o0, op0[:3, 3] + step)
        body = ClutchMapper(cfg)
        body.latch(op0, robot0)
        world = ClutchMapper(self._world_cfg(cfg, scale=cfg.scale))
        world.latch(op0, robot0)
        assert not np.allclose(body.map(t_op).position_m, world.map(t_op).position_m, atol=1e-6)


# -- tool offset ---------------------------------------------------------
class TestToolOffset:
    OFFSET_M = 0.15

    def _offset(self) -> np.ndarray:
        """T_FH: palm sits 15 cm along the flange's +X."""
        return transform(np.eye(3), np.array([self.OFFSET_M, 0.0, 0.0]))

    def test_zero_jump_at_engage_with_a_tool_offset(self, cfg):
        """The flange command at engage is still exactly the latched flange pose."""
        m = mapper(cfg, tool_offset=self._offset())
        op0 = transform(rotation_about([0, 0, 1], 0.4), np.array([0.1, 0.2, 0.3]))
        flange0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        m.latch(op0, flange0)
        np.testing.assert_allclose(m.map(op0).matrix, flange0, atol=1e-9)

    def test_palm_fixed_rotation_holds_the_palm_and_translates_the_flange(self, cfg):
        """Rotate about the palm: palm position constant, flange swings on a 15 cm arc."""
        offset = self._offset()
        m = mapper(cfg, scale=1.0, tool_offset=offset)
        op0 = transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
        flange0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        m.latch(op0, flange0)

        palm0 = flange0 @ offset
        yaw = math.radians(90.0)
        t_op = op0 @ transform(rotation_about([0, 0, 1], yaw), np.zeros(3))

        flange_target = m.map(t_op).matrix
        palm_target = flange_target @ offset

        # The palm is what the operator is holding: it must not move.
        np.testing.assert_allclose(palm_target[:3, 3], palm0[:3, 3], atol=1e-9)
        # The flange, 15 cm behind it, must.
        moved = float(np.linalg.norm(flange_target[:3, 3] - flange0[:3, 3]))
        assert moved > 0.1
        # A 90 deg swing about a 15 cm arm displaces the flange by r*sqrt(2).
        assert moved == pytest.approx(self.OFFSET_M * math.sqrt(2.0), abs=1e-9)

    def test_without_the_offset_the_same_rotation_keeps_the_flange_still(self, cfg):
        """Contrast case: no offset means rotation is about the flange itself."""
        m = mapper(cfg, scale=1.0)
        op0 = transform(np.eye(3), np.zeros(3))
        flange0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        m.latch(op0, flange0)
        t_op = op0 @ transform(rotation_about([0, 0, 1], math.radians(90.0)), np.zeros(3))
        np.testing.assert_allclose(m.map(t_op).position_m, flange0[:3, 3], atol=1e-12)

    @given(op0=transforms(), t_op=transforms(), d=unit_vectors())
    @SETTINGS
    def test_tool_offset_commutes_with_pure_translation(self, cfg, op0, t_op, d):
        """A pure operator translation moves palm and flange identically."""
        offset = transform(np.eye(3), 0.15 * d)
        flange0 = transform(np.eye(3), np.array([0.40, -0.20, 0.0]))
        with_tool = mapper(cfg, scale=1.0, tool_offset=offset)
        without = mapper(cfg, scale=1.0)
        with_tool.latch(op0, flange0)
        without.latch(op0, flange0)
        step = 0.05 * d
        moved_op = transform(op0[:3, :3], op0[:3, 3] + step)
        np.testing.assert_allclose(
            with_tool.map(moved_op).position_m, without.map(moved_op).position_m, atol=1e-9
        )

    def test_has_tool_offset_flag(self, cfg):
        assert mapper(cfg, tool_offset=self._offset()).has_tool_offset
        assert not mapper(cfg).has_tool_offset


# -- lifecycle -----------------------------------------------------------
class TestLifecycle:
    def test_map_before_latch_raises(self, cfg):
        with pytest.raises(MappingError, match="before latch"):
            mapper(cfg).map(np.eye(4))

    def test_clear_unlatches(self, cfg):
        m = mapper(cfg)
        m.latch(np.eye(4), np.eye(4))
        assert m.latched
        m.clear()
        assert not m.latched
        with pytest.raises(MappingError):
            m.map(np.eye(4))

    def test_clear_is_idempotent(self, cfg):
        m = mapper(cfg)
        m.clear()
        m.clear()
        assert not m.latched

    def test_origins_are_copies(self, cfg):
        m = mapper(cfg)
        op0 = np.eye(4)
        m.latch(op0, np.eye(4))
        op0[0, 3] = 99.0
        assert m.t_op0[0, 3] == 0.0
        m.t_op0[0, 3] = 42.0  # mutating the returned copy must not stick
        assert m.t_op0[0, 3] == 0.0

    def test_prerotation_is_none_before_latch(self, cfg):
        assert mapper(cfg).translation_prerotation is None

    def test_accepts_poses_as_well_as_matrices(self, cfg):
        m = mapper(cfg)
        m.latch(Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]), Pose([0.4, -0.2, 0.0], [1, 0, 0, 0]))
        assert m.latched
        out = m.map(Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]))
        np.testing.assert_allclose(out.position_m, [0.4, -0.2, 0.0], atol=1e-12)

    def test_accepts_an_upstream_shaped_pose(self, cfg):
        from tests.test_upstream_interop import TranscribedPose

        m = mapper(cfg)
        m.latch(TranscribedPose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
                TranscribedPose(np.array([0.4, -0.2, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])))
        assert m.latched

    @given(t=transforms(), robot0=transforms(reach=0.5))
    @SETTINGS
    def test_output_is_always_a_valid_pose(self, cfg, t, robot0):
        """Pose.from_matrix validates orthonormality; the mapper must never trip it."""
        m = mapper(cfg)
        m.latch(t, robot0)
        out = m.map(t @ transform(rotation_about([0, 0, 1], 0.3), np.array([0.01, 0.0, 0.0])))
        assert isinstance(out, Pose)
        assert np.isfinite(out.as_xyz_wxyz()).all()


# -- workspace clamp -----------------------------------------------------
class TestWorkspaceClamp:
    def test_inside_is_untouched_and_reports_no_axes(self, cfg):
        pose = Pose([0.40, -0.20, 0.0], [1, 0, 0, 0])
        got = clamp_to_workspace(cfg, "right", pose)
        assert not got.was_clamped
        assert got.clamped_axis_names == ()
        assert got.pose is pose

    def test_reports_which_axes_hit_the_wall(self, cfg):
        got = clamp_to_workspace(cfg, "right", Pose([9.0, -0.2, 9.0], [1, 0, 0, 0]))
        assert got.was_clamped
        assert got.clamped_axis_names == ("x", "z")
        np.testing.assert_allclose(got.pose.position_m, [0.732 - 0.03, -0.2, 0.5 - 0.03])

    def test_orientation_is_never_clamped(self, cfg):
        q = np.array([math.cos(0.3), 0.0, 0.0, math.sin(0.3)])
        got = clamp_to_workspace(cfg, "right", Pose([9.0, 9.0, 9.0], q))
        np.testing.assert_allclose(got.pose.orientation_wxyz, q, atol=1e-12)

    @pytest.mark.parametrize("arm", ["left", "right"])
    @given(p=st.tuples(finite(-3, 3), finite(-3, 3), finite(-3, 3)))
    @SETTINGS
    def test_output_is_always_inside_the_box(self, cfg, arm, p):
        got = clamp_to_workspace(cfg, arm, Pose(np.array(p), [1, 0, 0, 0]))
        bounds = cfg.workspace.box(arm).bounds
        margin = cfg.workspace.margin_m
        assert np.all(got.pose.position_m >= bounds[:, 0] + margin - 1e-12)
        assert np.all(got.pose.position_m <= bounds[:, 1] - margin + 1e-12)

    def test_left_and_right_boxes_differ(self, cfg):
        left = clamp_to_workspace(cfg, "left", Pose([0.4, -5.0, 0.0], [1, 0, 0, 0]))
        right = clamp_to_workspace(cfg, "right", Pose([0.4, -5.0, 0.0], [1, 0, 0, 0]))
        assert left.pose.position_m[1] != right.pose.position_m[1]

    def test_map_clamped_applies_both_steps(self, cfg):
        m = mapper(cfg, scale=1.0)
        m.latch(np.eye(4), transform(np.eye(3), np.array([0.7, -0.2, 0.0])))
        far = transform(np.eye(3), np.array([5.0, 0.0, 0.0]))
        got = m.map_clamped(far, "right")
        assert got.was_clamped and "x" in got.clamped_axis_names
        assert got.pose.position_m[0] == pytest.approx(0.732 - 0.03)
