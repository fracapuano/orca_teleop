"""CLI argument handling and the hard-rule-1 real-hardware gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import tron2_cli
from tests.conftest import at
from tron2_cli import Refused, apply_overrides, build_parser, is_loopback, normalise_target, resolve_target


def args(**kwargs) -> argparse.Namespace:
    base = {"robot": None, "command": "pose", "format": None, "send_both": None, "rate_hz": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestTargetParsing:
    @pytest.mark.parametrize("value,expected", [
        ("10.192.1.2", "ws://10.192.1.2:5000"),
        ("10.192.1.2:6000", "ws://10.192.1.2:6000"),
        ("ws://robot.local:5000", "ws://robot.local:5000"),
        ("wss://robot.local:5000", "wss://robot.local:5000"),
    ])
    def test_normalise_target(self, value, expected):
        assert normalise_target(value) == expected

    @pytest.mark.parametrize("url,expected", [
        ("ws://127.0.0.1:5000", True),
        ("ws://localhost:5000", True),
        ("ws://[::1]:5000", True),
        ("ws://10.192.1.2:5000", False),
        ("ws://robot.local:5000", False),
    ])
    def test_is_loopback(self, url, expected):
        assert is_loopback(url) is expected


class TestSafetyGate:
    """Real hardware needs --robot AND interactive confirmation."""

    def test_default_target_is_the_local_mock(self, config):
        assert resolve_target(args(), config) == config.robot.url
        assert is_loopback(resolve_target(args(), config))

    def test_loopback_never_prompts(self, config, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not prompt for the mock"))
        assert resolve_target(args(robot="127.0.0.1"), config) == "ws://127.0.0.1:5000"

    def test_non_loopback_config_url_alone_is_refused(self, config):
        """Editing robot.url in YAML must not bypass the gate."""
        hardware = at(config, "ws://10.192.1.2:5000")
        with pytest.raises(Refused, match="requires an explicit --robot"):
            resolve_target(args(), hardware)

    def test_non_tty_is_refused_even_with_the_flag(self, config, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(Refused, match="without an interactive terminal"):
            resolve_target(args(robot="10.192.1.2"), config)

    def test_declining_the_prompt_refuses(self, config, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "no")
        with pytest.raises(Refused, match="not confirmed"):
            resolve_target(args(robot="10.192.1.2"), config)

    @pytest.mark.parametrize("answer", ["", "y", "YES", "yes ", " yes", "yep"])
    def test_only_an_exact_yes_is_accepted(self, config, monkeypatch, answer):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: answer)
        if answer.strip() == "yes":
            assert resolve_target(args(robot="10.192.1.2"), config) == "ws://10.192.1.2:5000"
        else:
            with pytest.raises(Refused):
                resolve_target(args(robot="10.192.1.2"), config)

    def test_confirmed_hardware_is_allowed(self, config, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "yes")
        assert resolve_target(args(robot="10.192.1.2"), config) == "ws://10.192.1.2:5000"
        assert "REAL HARDWARE" in capsys.readouterr().out

    def test_refusal_exit_code_is_2(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert tron2_cli.main(["--robot", "10.192.1.2", "pose"]) == 2


class TestOverrides:
    def test_no_overrides_returns_the_same_config(self, config):
        assert apply_overrides(config, args()) is config

    def test_format_override(self, config):
        assert apply_overrides(config, args(format="pos_rotmat")).servop.format == "pos_rotmat"

    def test_send_both_override(self, config):
        assert apply_overrides(config, args(send_both=False)).servop.send_both is False
        assert apply_overrides(config, args(send_both=True)).servop.send_both is True

    def test_rate_override(self, config):
        assert apply_overrides(config, args(rate_hz=250.0)).servop.rate_hz == 250.0

    def test_override_does_not_mutate_the_original(self, config):
        before = config.servop.format
        apply_overrides(config, args(format="pos_rotmat"))
        assert config.servop.format == before

    def test_invalid_override_is_still_validated(self, config):
        from tron_arm.config import ConfigError

        with pytest.raises(ConfigError):
            apply_overrides(config, args(rate_hz=-5.0))


class TestParser:
    def test_every_documented_subcommand_exists(self):
        parser = build_parser()
        for command in ("info", "pose", "joints", "light", "movej-home",
                        "servop-hold", "servop-circle", "servop-readback"):
            assert parser.parse_args([command]).command == command

    def test_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_circle_defaults_match_the_documented_invocation(self):
        parsed = build_parser().parse_args(
            ["servop-circle", "--arm", "right", "--radius", "0.02", "--speed", "0.05"])
        assert (parsed.arm, parsed.radius, parsed.speed) == ("right", 0.02, 0.05)

    def test_hold_takes_seconds(self):
        assert build_parser().parse_args(["servop-hold", "--seconds", "5"]).seconds == 5.0

    def test_send_both_flags_are_mutually_expressive(self):
        assert build_parser().parse_args(["--send-both", "pose"]).send_both is True
        assert build_parser().parse_args(["--no-send-both", "pose"]).send_both is False
        assert build_parser().parse_args(["pose"]).send_both is None


class TestReadyPosture:
    """Capture a joint posture once, replay it before every session."""

    def test_a_config_without_a_ready_posture_is_valid(self, config):
        """Built explicitly: the shipped default is null, but a user who has
        captured their robot's posture must not fail the test suite."""
        import dataclasses

        assert dataclasses.replace(config, ready_joints=None).ready_joints is None

    def test_ready_joints_are_validated_against_the_limits(self, config):
        import dataclasses

        import numpy as np

        from tron_arm.config import JOINT_UPPER, ConfigError

        bad = np.zeros(14)
        bad[3] = JOINT_UPPER[3] + 0.5
        with pytest.raises(ConfigError, match="ready.joints"):
            dataclasses.replace(config, ready_joints=bad)

    def test_ready_joints_wrong_length(self, config):
        import dataclasses

        import numpy as np

        from tron_arm.config import ConfigError

        with pytest.raises(ConfigError, match="expected 14"):
            dataclasses.replace(config, ready_joints=np.zeros(7))

    def test_a_valid_ready_posture_is_accepted_and_read_only(self, config):
        import dataclasses

        import numpy as np

        cfg = dataclasses.replace(config, ready_joints=np.full(14, 0.1))
        assert cfg.ready_joints.shape == (14,)
        with pytest.raises(ValueError):
            cfg.ready_joints[0] = 1.0

    def test_movej_ready_refuses_without_a_configured_posture(self, config, monkeypatch):
        import asyncio
        import dataclasses

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        # Explicitly unset, so this holds whether or not the local config has one.
        rc = asyncio.run(run_arm_ready(dataclasses.replace(config, ready_joints=None)))
        assert rc == 1

    def test_a_captured_posture_is_loaded_and_validated(self, config):
        """If the local config has one (captured from a real robot), it must be
        the right shape and inside the documented joint limits."""
        if config.ready_joints is None:
            pytest.skip("no ready posture captured in this config")
        import numpy as np

        from tron_arm.config import JOINT_LOWER, JOINT_UPPER

        assert config.ready_joints.shape == (14,)
        assert np.all(config.ready_joints >= JOINT_LOWER)
        assert np.all(config.ready_joints <= JOINT_UPPER)

    def test_both_commands_are_exposed(self):
        parser = build_parser()
        for command in ("capture-ready", "movej-ready"):
            assert parser.parse_args([command]).command == command


async def run_arm_ready(config):
    """Helper: movej-ready with no posture configured must refuse, not connect."""
    import tron2_cli

    args = argparse.Namespace(robot=None, sim=False, command="movej-ready", time=1.0)
    return await tron2_cli.cmd_movej_ready(args, config)
