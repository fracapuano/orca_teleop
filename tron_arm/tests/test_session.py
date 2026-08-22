"""Session logging, step test and the report tool."""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import report_session

from tests.conftest import at, run
from tron_arm.config import load_config
from tron_arm.mock_robot import MockTron2
from tron_arm.session import SessionLogger, collect_meta, git_describe, new_session_id
from tron_arm.step_test import run_step_test


class TestSessionId:
    def test_is_sortable_and_timestamped(self):
        assert len(new_session_id()) == 15 and new_session_id()[8] == "-"

    def test_prefix(self):
        assert new_session_id("ci-").startswith("ci-")


class TestGitDescribe:
    def test_missing_path(self):
        got = git_describe("/nonexistent/repo")
        assert got["commit"] is None and "does not exist" in got["note"]

    def test_none(self):
        assert git_describe(None)["note"] == "path not known"

    def test_non_repo_is_distinguished_from_failure(self, tmp_path):
        got = git_describe(tmp_path)
        assert got["commit"] is None
        assert "not a git repository" in got["note"]

    def test_real_repo(self):
        pytest.importorskip("orca_teleop")
        from tron_arm.session import _find_orca_teleop

        repo = _find_orca_teleop()
        if repo is None:
            pytest.skip("orca_teleop not a checkout")
        got = git_describe(repo)
        assert got["commit"] and len(got["commit"]) == 40
        assert got["branch"]


class TestSessionLogger:
    def test_writes_gzipped_jsonl(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log.event("dispatch", arm="right", value=1)
        log.event("tick", value=2)
        log.close()
        lines = gzip.open(log.records_path, "rt").read().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["type"] == "dispatch" and first["arm"] == "right"
        assert isinstance(first["t_mono_ns"], int)

    def test_uncompressed_option(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s", compress=False)
        log.open()
        log.event("x")
        log.close()
        assert log.records_path.suffix == ".jsonl"
        assert log.records_path.read_text().strip()

    def test_disabled_logger_is_inert(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s", enabled=False)
        log.open()
        log.event("dispatch")
        log.write_meta({"a": 1})
        log.close()
        assert not (tmp_path / "s").exists()

    def test_numpy_values_are_serialised(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log.event("tick", p=np.array([1.0, 2.0, 3.0]), n=np.float64(4.0))
        log.close()
        got = json.loads(gzip.open(log.records_path, "rt").readline())
        assert got["p"] == [1.0, 2.0, 3.0] and got["n"] == 4.0

    def test_unserialisable_record_is_dropped_not_raised(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log.write({"type": "x", "bad": object()})
        log.close()
        assert log.dropped == 1 and log.records_written == 0

    def test_write_after_close_is_a_no_op(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log.close()
        log.event("late")  # must not raise
        assert log.records_written == 0

    def test_concurrent_writers(self, tmp_path):
        """dispatch writes from the arm thread, tick from the loop thread."""
        import threading

        log = SessionLogger(tmp_path, session_id="s")
        log.open()

        def spam(kind: str) -> None:
            for i in range(200):
                log.event(kind, i=i)

        threads = [threading.Thread(target=spam, args=(k,)) for k in ("dispatch", "tick")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log.close()
        lines = gzip.open(log.records_path, "rt").read().strip().splitlines()
        assert len(lines) == 400
        for line in lines:
            json.loads(line)  # every line must be intact, not interleaved

    def test_context_manager(self, tmp_path):
        with SessionLogger(tmp_path, session_id="s") as log:
            log.event("x")
        assert log.records_path.exists()


class TestMeta:
    def test_records_the_url_actually_connected_to_not_the_config_default(self, config, tmp_path):
        """--robot overrides the config, and provenance must follow it: a real
        hardware session once logged itself as loopback."""
        meta = collect_meta(config, session_id="s", url="ws://10.192.1.2:5000")
        assert meta["robot"]["url"] == "ws://10.192.1.2:5000"
        assert config.robot.url != "ws://10.192.1.2:5000"

    def test_url_falls_back_to_the_config_when_not_overridden(self, config, tmp_path):
        assert collect_meta(config, session_id="s")["robot"]["url"] == config.robot.url

    def test_collects_both_repos_and_config(self, config, tmp_path):
        meta = collect_meta(config, session_id="s", tron_arm_repo=tmp_path, accid="ACC")
        assert set(meta["repos"]) == {"tron_arm", "orca_teleop"}
        assert meta["robot"]["accid"] == "ACC"
        assert meta["servop"]["format"] == config.servop.format
        assert meta["servop"]["send_both"] == config.servop.send_both
        assert meta["config"]["scale"] == config.scale
        assert meta["config"]["max_step"]["lin_m"] == config.max_step[0]
        assert len(meta["config"]["home_joints"]) == 14

    def test_written_to_disk_and_updated(self, config, tmp_path):
        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log.write_meta(collect_meta(config, session_id="s"))
        log.close(achieved_rate_hz=99.5, stats={"a": 1})
        meta = json.loads(log.meta_path.read_text())
        assert meta["achieved_rate_hz"] == 99.5 and meta["final_stats"] == {"a": 1}
        assert meta["records_written"] == 0

    @pytest.mark.parametrize("key", ["version", "firmware_version", "firmware", "sw_version"])
    def test_firmware_version_key_variants(self, tmp_path, key):
        log = SessionLogger(tmp_path, session_id="s")
        log.note_robot_info({key: "robot-tron2-r-1.2.3"})
        assert log.firmware_version == "robot-tron2-r-1.2.3"

    def test_unknown_firmware_key_is_not_guessed(self, tmp_path):
        log = SessionLogger(tmp_path, session_id="s")
        log.note_robot_info({"model": "TRON2", "state": "idle"})
        assert log.firmware_version is None


# (robot's accepted format, formats to attempt) -- one entry per distinct run.
_STEP_TEST_SCENARIOS = (
    ("pos_quat", None),
    ("pos_rotmat", None),
    ("pos_rotmat", ("pos_quat",)),
)


@pytest.fixture(scope="module")
def step_reports() -> dict[tuple, Any]:
    """Every --step-test scenario, run once and concurrently.

    Even in ``quick`` mode a step test streams real poses through four phases,
    so each run costs seconds. Two things follow. Seven assertions about three
    scenarios do not need seven runs -- the report is immutable, so the runs are
    shared. And the scenarios are independent (each has its own mock on its own
    ephemeral port) and spend nearly all of that time in ``asyncio.sleep``, so
    running them together costs about what the slowest one costs alone.
    """
    async def one(accept_format: str, formats: tuple[str, ...] | None):
        kwargs = {} if formats is None else {"formats": list(formats)}
        async with MockTron2(port=0, info_period_s=0.05,
                             accept_format=accept_format) as robot:
            cfg = at(load_config(), f"ws://127.0.0.1:{robot.bound_port}")
            cfg = dataclasses.replace(cfg, notify_log_path=None)
            return await run_step_test(cfg, cfg.robot.url, quick=True, **kwargs)

    async def all_of_them():
        return await asyncio.gather(*(one(fmt, f) for fmt, f in _STEP_TEST_SCENARIOS))

    return dict(zip(_STEP_TEST_SCENARIOS, run(all_of_them()), strict=True))


class TestStepTest:
    @staticmethod
    def _run(reports, accept_format: str, formats: list[str] | None = None):
        return reports[(accept_format, tuple(formats) if formats else None)]

    def test_identifies_pos_quat(self, step_reports):
        report = self._run(step_reports, "pos_quat")
        assert report.chosen_format == "pos_quat"
        assert report.format_passed("pos_quat")
        assert not report.format_passed("pos_rotmat")

    def test_identifies_pos_rotmat(self, step_reports):
        report = self._run(step_reports, "pos_rotmat")
        assert report.chosen_format == "pos_rotmat"
        assert not report.format_passed("pos_quat")

    def test_runs_all_four_phases_for_the_accepted_format(self, step_reports):
        report = self._run(step_reports, "pos_quat")
        names = [r.name for r in report.formats["pos_quat"]]
        assert any("hold" in n for n in names)
        assert any("readback" in n for n in names)
        assert any("axis steps" in n for n in names)
        assert any("circle" in n for n in names)

    def test_rejected_format_short_circuits(self, step_reports):
        """A rejected format reports one clear failure, not four confusing ones."""
        report = self._run(step_reports, "pos_rotmat")
        rejected = report.formats["pos_quat"]
        assert rejected[0].notify_failures > 0
        assert all("skipped" in r.detail for r in rejected[1:])

    def test_report_text_prints_pass_fail_per_format(self, step_reports):
        text = self._run(step_reports, "pos_quat").text()
        assert "[PASS]" in text and "[FAIL]" in text
        assert "pos_quat: ACCEPTED" in text and "pos_rotmat: REJECTED" in text
        assert "VERDICT: use servop.format: pos_quat" in text

    def test_single_format_selection(self, step_reports):
        """Only the requested format is attempted.

        Paired with a robot that rejects it, so the run short-circuits after the
        hold phase: what is under test is which formats get iterated, not how
        far any one of them gets.
        """
        report = self._run(step_reports, "pos_rotmat", formats=["pos_quat"])
        assert list(report.formats) == ["pos_quat"]

    def test_rejects_unknown_arm(self):
        with pytest.raises(ValueError, match="unknown arm"):
            run(run_step_test(load_config(), "ws://127.0.0.1:1", arm="middle"))


class TestReportSession:
    @pytest.fixture
    def session_dir(self, tmp_path, config):
        log = SessionLogger(tmp_path, session_id="s")
        log.open()
        log.write_meta(collect_meta(config, session_id="s", accid="ACC"))
        base = 1_000_000_000
        for i in range(50):
            log.write({"type": "dispatch", "t_mono_ns": base + i * 20_000_000,
                       "arm": "right", "t_dispatch": base + i * 20_000_000,
                       "recv_monotonic_ns": base + i * 20_000_000,
                       "state": "engaged", "latched": i == 0, "commanded": True,
                       "ws_clamped_axes": ["x"] if i > 45 else [],
                       "step_clamped_lin": i % 10 == 0, "step_clamped_ang": False})
            log.write({"type": "tick", "t_mono_ns": base + i * 20_000_000 + 5_000_000,
                       "t_streamer_tick": base + i * 20_000_000 + 5_000_000,
                       "t_ws_send": base + i * 20_000_000 + 6_000_000, "sent": True,
                       "arms": {"right": {"alpha": 0.5, "interpolation_gap_ns": 20_000_000,
                                          "src_recv_ns": base + i * 20_000_000,
                                          "t_dispatch": base + i * 20_000_000,
                                          "lin_clamped": False, "ang_clamped": False,
                                          "p": [0.4 + i * 0.001, -0.2, 0.0]}}})
        log.write({"type": "hold", "t_mono_ns": base + 10**9, "arm": "right",
                   "reason": "stale", "state": "hold"})
        log.write({"type": "reference_change", "t_mono_ns": base + 2 * 10**9,
                   "stream_id": 1, "pose_epoch": 2})
        log.write({"type": "notify", "t_mono_ns": base, "title": "notify_robot_info",
                   "data": {"state": "idle"}})
        log.write({"type": "notify", "t_mono_ns": base + 3 * 10**9, "title": "notify_servop",
                   "data": {"result": "fail_invalid_cmd", "reason": "7 elements"}})
        log.close(achieved_rate_hz=100.0, stats={"sink": {"dispatch_p50_ms": 0.4,
                                                          "dispatch_p95_ms": 0.8}})
        return log.dir

    def test_all_sections_present(self, session_dir):
        text = report_session.render(report_session.summarise(session_dir))
        for section in ("Provenance", "Dispatch -> ws send latency", "Ingress",
                        "Holds", "Reference changes", "notify_*", "Clamps"):
            assert section in text, section

    def test_latency_percentiles(self, session_dir):
        summary = report_session.summarise(session_dir)
        assert summary["latency_ms"]["p50"] == pytest.approx(6.0, abs=0.5)
        assert summary["latency_ms"]["p95"] == pytest.approx(6.0, abs=0.5)

    def test_counts(self, session_dir):
        summary = report_session.summarise(session_dir)
        assert summary["dispatches"] == 50 and summary["ticks"] == 50
        assert len(summary["holds"]) == 1 and len(summary["reference_changes"]) == 1
        assert summary["latches"] == 1
        assert len(summary["notify_failures"]) == 1

    def test_failure_notifies_are_called_out(self, session_dir):
        text = report_session.render(report_session.summarise(session_dir))
        assert "FAILURE(s)" in text and "fail_invalid_cmd" in text

    def test_clamp_totals(self, session_dir):
        text = report_session.render(report_session.summarise(session_dir))
        assert "workspace  4 dispatch(es)" in text and "'x': 4" in text

    def test_ingress_rate(self, session_dir):
        assert report_session.summarise(session_dir)["ingress"]["mean_hz"] == pytest.approx(50.0, abs=1)

    def test_truncated_final_line_is_tolerated(self, session_dir):
        """A killed session leaves a partial line; the report must still work."""
        path = session_dir / "records.jsonl.gz"
        data = gzip.open(path, "rt").read()
        with gzip.open(path, "wt") as handle:
            handle.write(data + '{"type": "tick", "t_mo')
        assert report_session.render(report_session.summarise(session_dir))

    def test_missing_meta_is_tolerated(self, session_dir):
        (session_dir / "session_meta.json").unlink()
        assert "SESSION REPORT" in report_session.render(report_session.summarise(session_dir))

    def test_missing_records_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert report_session.main([str(tmp_path / "empty")]) == 2

    def test_not_a_directory(self, tmp_path):
        assert report_session.main([str(tmp_path / "nope")]) == 2

    def test_output_width_is_paste_safe(self, session_dir):
        for line in report_session.render(report_session.summarise(session_dir)).splitlines():
            assert len(line) <= 120, f"line too wide to paste: {line!r}"
