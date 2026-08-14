"""run_arm.py: target gating, the mandatory --no-wrist rule, and the status line."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import run_arm
from tron2_cli import Refused


def args(**kw) -> argparse.Namespace:
    base = {"robot": None, "sim": False}
    base.update(kw)
    return argparse.Namespace(**base)


class TestTargetGate:
    def test_sim_uses_the_mock(self):
        assert run_arm.resolve_url(args(sim=True), "ws://10.0.0.1:5000") == run_arm.MOCK_URL

    def test_default_uses_the_config_url(self, config):
        assert run_arm.resolve_url(args(), config.robot.url) == config.robot.url

    def test_non_loopback_config_without_the_flag_is_refused(self):
        with pytest.raises(Refused, match="requires an explicit --robot"):
            run_arm.resolve_url(args(), "ws://10.192.1.2:5000")

    def test_robot_flag_without_a_tty_is_refused(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(Refused, match="interactive terminal"):
            run_arm.resolve_url(args(robot="10.192.1.2"), "ws://127.0.0.1:5000")

    def test_confirmed_robot_flag_is_allowed(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "yes")
        assert run_arm.resolve_url(args(robot="10.192.1.2"), "") == "ws://10.192.1.2:5000"

    def test_shares_one_gate_with_tron2_cli(self):
        """One implementation to audit, not two."""
        import tron2_cli

        assert run_arm.gate_target is tron2_cli.gate_target


class TestWristRule:
    def test_hardware_without_no_wrist_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "yes")
        assert run_arm.main(["--robot", "10.192.1.2"]) == 2
        assert "--no-wrist is mandatory" in capsys.readouterr().err

    def test_the_refusal_explains_why(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "yes")
        run_arm.main(["--robot", "10.192.1.2"])
        err = capsys.readouterr().err
        assert "T_FH" in err and "rigid" in err

    def test_sim_does_not_require_no_wrist(self):
        """Bench runs against the mock may leave the wrist enabled."""
        parsed = run_arm.build_parser().parse_args(["--sim"])
        assert not parsed.no_wrist and parsed.sim

    def test_help_documents_the_rule(self):
        assert "MANDATORY FOR HARDWARE" in run_arm.__doc__
        action = next(a for a in run_arm.build_parser()._actions if a.dest == "no_wrist")
        assert "MANDATORY" in action.help


class TestParser:
    def test_sim_and_robot_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            run_arm.build_parser().parse_args(["--sim", "--robot", "10.0.0.1"])

    def test_clutch_choices(self):
        for choice in ("keyboard", "pedal", "none"):
            assert run_arm.build_parser().parse_args(["--clutch", choice]).clutch == choice

    def test_defaults(self):
        parsed = run_arm.build_parser().parse_args([])
        assert parsed.clutch == "keyboard" and parsed.hand == "right" and parsed.port == 50051


class TestStatusLine:
    def _sink(self, config):
        from tron_arm.sink import TronArmSink

        return TronArmSink(config)

    def test_renders_every_documented_field(self, config):
        sink = self._sink(config)
        sink.diagnostics("right").state = "engaged"
        sink.diagnostics("right").last_age_s = 0.012
        sink.diagnostics("right").workspace_clamps = 3
        sink.diagnostics("right").step_clamps = 7
        line = run_arm.StatusLine(sink, stream=io.StringIO()).render()
        assert "R:engaged" in line          # per-arm state
        assert "L:" in line                 # both arms
        assert "age=  12ms" in line         # staleness
        assert "ws=3" in line and "st=7" in line  # clamp counts
        assert "Hz" in line                 # achieved streamer rate
        assert "none" in line               # last notify

    def test_shows_operator_flags(self, config):
        sink = self._sink(config)
        sink.set_orientation_frozen(True)
        sink.force_hold(True)
        line = run_arm.StatusLine(sink, stream=io.StringIO()).render()
        assert "ORI-FROZEN" in line and "FORCED-HOLD" in line

    def test_starts_with_a_carriage_return_so_it_stays_on_one_line(self, config):
        assert run_arm.StatusLine(self._sink(config), stream=io.StringIO()).render()[0] == "\r"

    def test_start_stop_is_clean(self, config):
        stream = io.StringIO()
        status = run_arm.StatusLine(self._sink(config), hz=100.0, stream=stream)
        status.start()
        status.stop()
        assert stream.getvalue().endswith("\n")


class TestHotkeys:
    def _sink(self, config):
        from tron_arm.sink import TronArmSink

        return TronArmSink(config)

    def test_space_is_the_clutch_and_never_toggles_force_hold(self, config):
        """The bug this guards: SPACE engaging the clutch AND force-holding
        would fight itself on every key repeat."""
        sink = self._sink(config)
        handle = run_arm.make_hotkey_handler(sink, clutch_key=" ")
        for _ in range(10):
            handle(" ")
        assert not sink._forced_hold

    @pytest.mark.parametrize("key", ["x", "\x1b"])
    def test_force_hold_keys(self, config, key):
        sink = self._sink(config)
        handle = run_arm.make_hotkey_handler(sink, clutch_key=" ")
        handle(key)
        assert sink._forced_hold
        handle(key)
        assert not sink._forced_hold

    def test_a_pedal_clutch_key_is_also_passed_through(self, config):
        sink = self._sink(config)
        handle = run_arm.make_hotkey_handler(sink, clutch_key="a")
        handle("a")
        assert not sink._forced_hold

    def test_o_toggles_orientation_freeze(self, config):
        sink = self._sink(config)
        handle = run_arm.make_hotkey_handler(sink)
        handle("o")
        assert sink.orientation_frozen
        handle("O")
        assert not sink.orientation_frozen

    def test_q_raises_sigint_in_this_process(self, config, monkeypatch):
        import signal

        killed = []
        monkeypatch.setattr(run_arm.os, "kill", lambda pid, sig: killed.append((pid, sig)))
        run_arm.make_hotkey_handler(self._sink(config))("q")
        assert killed and killed[0][1] == signal.SIGINT

    def test_other_keys_do_nothing(self, config):
        sink = self._sink(config)
        run_arm.make_hotkey_handler(sink)("z")
        assert not sink._forced_hold and not sink.orientation_frozen


class TestAlwaysEngaged:
    def test_reports_engaged_but_still_forwards_the_reader(self):
        """--clutch none still needs the reader running, for the hotkeys."""
        calls = []

        class Reader:
            engaged = False
            available = True

            def start(self):
                calls.append("start")

            def stop(self):
                calls.append("stop")

        wrapper = run_arm._AlwaysEngaged(Reader())
        wrapper.start()
        assert wrapper.engaged is True
        wrapper.stop()
        assert calls == ["start", "stop"]
