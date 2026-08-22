"""CI end-to-end: mock robot + mock publisher + `run_arm --sim`, for real.

This launches ``tools/run_arm.py`` as a **subprocess**, exactly as an operator
would, and then judges the run from the session log it leaves behind. Nothing is
inspected in-process: if the log cannot answer the question, neither can anyone
reading it after a real run.

Assertions (runbook §7 step 7 / prompt 5.4):
  * achieved streamer rate within 5% of the configured rate;
  * zero unexpected ``notify_*`` (robot_info is expected; a servop failure is not);
  * no commanded discontinuity beyond the per-tick step clamp;
  * in-sink dispatch p95 < 1 ms.

Skipped without orca_teleop (needs grpcio).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("orca_teleop.ingress.server", reason="orca_teleop not installed")

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import report_session  # noqa: E402

RUN_SECONDS = 8.0
pytestmark = pytest.mark.slow


def _python() -> str:
    return sys.executable


def _await_port(port: int, proc: subprocess.Popen, what: str, timeout: float = 30.0) -> None:
    """Block until ``port`` accepts a connection, or ``proc`` dies.

    Waiting for the socket beats sleeping a guessed number of seconds twice
    over: it returns as soon as the process is actually up, and it reports the
    child's own output when it never comes up at all.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"{what} exited {proc.returncode} during startup:\n"
                f"{proc.communicate()[0][-3000:]}"
            )
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)
    raise AssertionError(f"{what} never bound port {port} within {timeout:.0f}s")


def _orca_teleop_python() -> str:
    """The interpreter that can import orca_teleop -- normally this one."""
    return sys.executable


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> dict:
    """Run the whole stack once; every test reads the resulting log."""
    log_root = tmp_path_factory.mktemp("logs")
    session_id = "ci-e2e"
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    robot = subprocess.Popen(
        [_python(), "-m", "tron_arm.mock_robot", "--host", "127.0.0.1", "--port", "5057"],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    publisher = None
    try:
        _await_port(5057, robot, "mock robot")

        arm = subprocess.Popen(
            [_python(), str(TOOLS / "run_arm.py"),
             "--robot", "127.0.0.1:5057",   # loopback: the hard-rule-1 gate lets it through
             "--arm-only", "--clutch", "none", "--no-status", "--no-wrist",
             "--port", "50077",
             "--seconds", str(RUN_SECONDS),
             "--log-dir", str(log_root), "--session-id", session_id],
            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        _await_port(50077, arm, "run_arm ingress")

        publisher = subprocess.Popen(
            [_orca_teleop_python(), "-m", "orca_teleop.ingress.metaquest.mock_publisher",
             "--server", "localhost:50077", "--hand", "right", "--fps", "60"],
            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        out, _ = arm.communicate(timeout=RUN_SECONDS + 40)
        assert arm.returncode == 0, f"run_arm exited {arm.returncode}:\n{out[-3000:]}"
    finally:
        for proc in (publisher, robot):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    session_dir = log_root / session_id
    assert session_dir.is_dir(), f"no session log at {session_dir}"
    return {
        "dir": session_dir,
        "summary": report_session.summarise(session_dir),
        "records": list(report_session.load_records(session_dir)),
        "meta": report_session.load_meta(session_dir),
    }


class TestRunCompleted:
    def test_session_log_exists_with_both_files(self, session):
        assert (session["dir"] / "records.jsonl.gz").exists()
        assert (session["dir"] / "session_meta.json").exists()

    def test_frames_actually_flowed(self, session):
        summary = session["summary"]
        assert summary["dispatches"] > 100, f"only {summary['dispatches']} dispatches"
        assert summary["ticks"] > 500, f"only {summary['ticks']} ticks"

    def test_meta_records_provenance(self, session):
        meta = session["meta"]
        assert meta["repos"]["orca_teleop"]["commit"], "orca_teleop commit not captured"
        assert meta["robot"]["accid"], "robot accid not captured"
        assert meta["servop"]["format"] in ("pos_quat", "pos_rotmat")
        assert meta["config"]["scale"] == 0.5
        # The mock reports a version string; the capture path must find it.
        assert meta["robot"]["firmware_version"], "firmware version not captured"


class TestAchievedRate:
    def test_within_5_percent_of_target(self, session):
        meta = session["meta"]
        target = meta["servop"]["rate_hz"]
        achieved = meta["achieved_rate_hz"]
        assert achieved is not None, "achieved rate not recorded"
        error = abs(achieved - target) / target
        assert error <= 0.05, f"achieved {achieved:.2f} Hz vs target {target} Hz ({error:.1%})"

    def test_no_late_ticks_to_speak_of(self, session):
        streamer = session["meta"]["final_stats"]["streamer"]
        # A handful across the run is scheduler noise; a flood means we cannot keep up.
        assert streamer["late_ticks"] < 0.02 * streamer["ticks"], streamer


class TestNotifies:
    def test_zero_unexpected_notifies(self, session):
        titles = session["summary"]["notify_titles"]
        expected = {"notify_robot_info"}
        unexpected = {t: n for t, n in titles.items() if t not in expected}
        assert not unexpected, f"unexpected notify_*: {unexpected}"

    def test_zero_failure_notifies(self, session):
        assert session["summary"]["notify_failures"] == []

    def test_robot_info_was_seen(self, session):
        assert session["summary"]["notify_titles"].get("notify_robot_info", 0) >= 5


class TestContinuity:
    """No commanded discontinuity beyond the per-tick step clamp."""

    def _tracks(self, session) -> dict[str, list]:
        tracks: dict[str, list] = {}
        for record in session["records"]:
            if record.get("type") != "tick" or not record.get("sent"):
                continue
            for arm, detail in (record.get("arms") or {}).items():
                if detail.get("p"):
                    tracks.setdefault(arm, []).append(np.asarray(detail["p"], dtype=float))
        return tracks

    def test_commanded_stream_respects_the_step_clamp(self, session):
        max_lin = session["meta"]["config"]["max_step"]["lin_m"]
        tracks = self._tracks(session)
        assert tracks, "no commanded positions logged"
        for arm, track in tracks.items():
            assert len(track) > 100, f"{arm}: only {len(track)} commands"
            steps = np.linalg.norm(np.diff(np.asarray(track), axis=0), axis=1)
            worst = float(steps.max())
            # Equality is the norm: a moving arm sits exactly at the ceiling.
            assert worst <= max_lin + 1e-9, (
                f"{arm}: largest commanded step {worst:.6f} m > clamp {max_lin} m"
            )

    def test_the_arm_actually_moved(self, session):
        """Guard the guard: a frozen arm would satisfy continuity trivially."""
        tracks = self._tracks(session)
        moved = {
            arm: float(np.max(np.linalg.norm(np.asarray(track) - track[0], axis=1)))
            for arm, track in tracks.items()
        }
        assert max(moved.values()) > 0.01, f"nothing moved: {moved}"


class TestLatency:
    def test_in_sink_dispatch_p95_under_1ms(self, session):
        sink = session["meta"]["final_stats"]["sink"]
        assert sink["dispatch_p95_ms"] < 1.0, (
            f"dispatch p95 {sink['dispatch_p95_ms']:.3f} ms -- the arm worker thread is blocking"
        )

    def test_dispatch_to_send_latency_is_sane(self, session):
        latency = session["summary"]["latency_ms"]
        assert latency["p95"] is not None
        # Interpolation deliberately renders one ingress interval behind, so
        # ~2 frame times is expected; 80 ms is the runbook's end-to-end budget.
        assert latency["p95"] < 80.0, f"p95 {latency['p95']:.1f} ms"

    def test_each_dispatch_counted_once(self, session):
        """The latency series must not double-count re-sent held targets."""
        assert session["summary"]["ticks"] > session["summary"]["dispatches"]
        latency_n = len(
            [v for values in report_session.latencies_ms(
                [r for r in session["records"] if r.get("type") == "tick"]).values()
             for v in values]
        )
        assert latency_n <= session["summary"]["dispatches"]


class TestReport:
    def test_text_report_renders(self, session):
        text = report_session.render(report_session.summarise(session["dir"]))
        for heading in ("SESSION REPORT", "Provenance", "Dispatch -> ws send latency",
                        "Ingress", "Holds", "Reference changes", "notify_*", "Clamps"):
            assert heading in text, f"missing section: {heading}"

    def test_report_cli_exits_zero(self, session):
        assert report_session.main([str(session["dir"])]) == 0

    def test_json_report_is_valid(self, session, capsys):
        assert report_session.main([str(session["dir"]), "--json"]) == 0
        json.loads(capsys.readouterr().out)
