"""Config loading, validation and the workspace clamp."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from tron_arm.config import (
    ARMS,
    JOINT_LOWER,
    JOINT_UPPER,
    N_JOINTS,
    Box,
    Config,
    ConfigError,
    ServopConfig,
    VelocityConfig,
    default_config_path,
    load_config,
)


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(default_config_path().read_text())


def build(raw: dict, **_) -> Config:
    return Config.from_mapping(raw)


class TestShippedDefault:
    def test_loads(self, config):
        assert config.servop.format == "pos_quat"
        assert config.servop.rate_hz == 100.0
        assert config.servop.send_both is True
        assert config.workspace.margin_m == 0.03
        assert (config.velocity.lin, config.velocity.ang) == (0.4, 1.2)
        assert config.scale == 0.5
        assert config.mapping.translation_frame == "body"
        assert config.home_joints.shape == (N_JOINTS,)

    def test_default_target_is_the_mock_not_hardware(self, config):
        """CLAUDE.md hard rule 1: the shipped default must be local."""
        assert "127.0.0.1" in config.robot.url

    def test_workspace_boxes_match_the_vendor_guide(self, config):
        np.testing.assert_allclose(config.workspace.left.bounds,
                                   [[0.250, 0.732], [-0.213, 0.900], [-0.673, 0.5]])
        np.testing.assert_allclose(config.workspace.right.bounds,
                                   [[0.250, 0.732], [-0.900, 0.213], [-0.673, 0.5]])

    def test_home_joints_are_inside_the_documented_limits(self, config):
        assert np.all(config.home_joints >= JOINT_LOWER)
        assert np.all(config.home_joints <= JOINT_UPPER)

    def test_home_joints_are_read_only(self, config):
        with pytest.raises(ValueError):
            config.home_joints[0] = 1.0


class TestValidation:
    def test_unknown_top_level_key_is_rejected(self, raw):
        raw["speed"] = 1
        with pytest.raises(ConfigError, match="unknown key"):
            build(raw)

    def test_unknown_servop_key_is_rejected(self, raw):
        raw["servop"]["formt"] = "pos_quat"
        with pytest.raises(ConfigError, match="unknown key"):
            build(raw)

    def test_bad_servop_format(self, raw):
        raw["servop"]["format"] = "pos_euler"
        with pytest.raises(ConfigError, match="servop.format"):
            build(raw)

    @pytest.mark.parametrize("rate", [0, -1, 5000])
    def test_bad_rate(self, raw, rate):
        raw["servop"]["rate_hz"] = rate
        with pytest.raises(ConfigError):
            build(raw)

    def test_send_both_must_be_a_bool(self, raw):
        raw["servop"]["send_both"] = "true"
        with pytest.raises(ConfigError, match="send_both"):
            build(raw)

    def test_negative_margin(self, raw):
        raw["workspace"]["margin_m"] = -0.01
        with pytest.raises(ConfigError, match="margin_m"):
            build(raw)

    def test_inverted_box(self, raw):
        raw["workspace"]["left"]["x"] = [0.9, 0.1]
        with pytest.raises(ConfigError, match="must be <"):
            build(raw)

    @pytest.mark.parametrize("field", ["lin", "ang"])
    def test_non_positive_velocity(self, raw, field):
        raw["velocity"][field] = 0
        with pytest.raises(ConfigError, match=field):
            build(raw)

    def test_non_positive_scale(self, raw):
        raw["scale"] = 0
        with pytest.raises(ConfigError, match="scale"):
            build(raw)

    def test_wrong_joint_count(self, raw):
        raw["home"]["joints"] = [0.0] * 7
        with pytest.raises(ConfigError, match="expected 14"):
            build(raw)

    def test_joint_outside_limits(self, raw):
        raw["home"]["joints"][3] = 1.0  # upper limit for j3 is 0.2618
        with pytest.raises(ConfigError, match="outside the documented limits"):
            build(raw)

    def test_non_ws_url(self, raw):
        raw["robot"]["url"] = "http://10.192.1.2:5000"
        with pytest.raises(ConfigError, match="ws://"):
            build(raw)

    def test_world_frame_is_gated_behind_axis_verification(self, raw):
        raw["mapping"]["translation_frame"] = "world"
        with pytest.raises(ConfigError, match="world_frame_axes_verified"):
            build(raw)

    def test_world_frame_allowed_once_verified(self, raw):
        raw["mapping"]["translation_frame"] = "world"
        raw["mapping"]["world_frame_axes_verified"] = True
        assert build(raw).mapping.translation_frame == "world"

    def test_missing_required_section(self, raw):
        del raw["servop"]
        with pytest.raises(ConfigError, match="missing required key"):
            build(raw)

    def test_load_reports_the_path(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("servop: {}\n")
        with pytest.raises(ConfigError, match=str(bad)):
            load_config(bad)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot read"):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ConfigError, match="empty"):
            load_config(empty)


class TestWorkspaceClamp:
    def test_inside_is_untouched(self, config):
        p = np.array([0.4, -0.2, 0.0])
        out, clamped = config.workspace.clamp("right", p)
        assert not clamped
        np.testing.assert_array_equal(out, p)

    def test_margin_is_held_off_every_face(self, config):
        out, clamped = config.workspace.clamp("right", [10.0, -10.0, 10.0])
        assert clamped
        np.testing.assert_allclose(out, [0.732 - 0.03, -0.900 + 0.03, 0.5 - 0.03])

    def test_left_and_right_boxes_differ_in_y(self, config):
        left, _ = config.workspace.clamp("left", [0.4, -5.0, 0.0])
        right, _ = config.workspace.clamp("right", [0.4, -5.0, 0.0])
        assert left[1] == pytest.approx(-0.213 + 0.03)
        assert right[1] == pytest.approx(-0.900 + 0.03)

    def test_rejects_unknown_arm(self, config):
        with pytest.raises(ValueError, match="unknown arm"):
            config.workspace.clamp("middle", [0.4, 0.0, 0.0])

    def test_rejects_non_finite(self, config):
        with pytest.raises(ValueError, match="non-finite"):
            config.workspace.clamp("left", [np.nan, 0.0, 0.0])

    def test_collapsed_axis_targets_the_centre(self):
        box = Box(x=(0.0, 0.01), y=(0.0, 1.0), z=(0.0, 1.0))
        out, clamped = box.clamp([5.0, 0.5, 0.5], margin_m=0.5)
        assert clamped
        assert out[0] == pytest.approx(0.005)

    @pytest.mark.parametrize("arm", ARMS)
    def test_clamped_output_is_always_inside(self, config, arm):
        rng = np.random.default_rng(7)
        for _ in range(200):
            p = rng.uniform(-2.0, 2.0, size=3)
            out, _ = config.workspace.clamp(arm, p)
            bounds = config.workspace.box(arm).bounds
            assert np.all(out >= bounds[:, 0] - 1e-12)
            assert np.all(out <= bounds[:, 1] + 1e-12)


class TestDerived:
    def test_max_step(self):
        assert VelocityConfig(0.4, 1.2).max_step(100.0) == pytest.approx((0.004, 0.012))

    def test_period(self):
        servop = ServopConfig("pos_quat", 100.0, True)
        assert servop.period_s == pytest.approx(0.01)
        assert servop.period_ns == 10_000_000

    @pytest.mark.parametrize("fmt,width", [("pos_quat", 7), ("pos_rotmat", 12)])
    def test_width(self, fmt, width):
        assert ServopConfig(fmt, 100.0, True).width == width
