"""Per-session logging: gzipped JSONL records plus a metadata sidecar.

Layout::

    logs/<session-id>/
      records.jsonl.gz     one JSON object per line, `type` discriminates
      session_meta.json    provenance: git hashes, config, robot identity

Why gzip: at 100 Hz a tick record per tick is ~360 k records/hour. Uncompressed
that is a few hundred MB per session of highly repetitive JSON, which compresses
roughly 10:1. Writing it compressed costs a little CPU on the loop thread and
saves having to remember to clean up.

Record types (`type` field):

===================  =============================================
``dispatch``         one operator frame consumed on the arm worker
``tick``             one streamer tick, incl. the ws send timestamp
``hold``             upstream ``on_hold(reason)``
``reference_change`` upstream ``on_reference_change``
``notify``           any ``notify_*`` from the robot
``step_test``        a ``--step-test`` phase result
===================  =============================================

All timestamps are monotonic nanoseconds (``CLAUDE.md`` hard rule 2). Wall clock
appears only in ``session_meta.json``, for correlating with other logs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["SessionLogger", "git_describe", "new_session_id", "collect_meta"]


def new_session_id(prefix: str = "") -> str:
    """A sortable, human-readable session id: ``20260813-142530``."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{prefix}{stamp}" if prefix else stamp


def git_describe(repo: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Best-effort ``{path, branch, commit, dirty}`` for a checkout.

    Never raises: a missing repo, a missing git, or a detached HEAD all degrade
    to nulls. Provenance that crashes the run it is describing is worse than
    provenance that is partially unknown.
    """
    out: dict[str, Any] = {"path": str(repo) if repo else None,
                           "branch": None, "commit": None, "dirty": None, "note": None}
    if repo is None:
        out["note"] = "path not known"
        return out
    if not Path(repo).exists():
        out["note"] = "path does not exist"
        return out
    if not (Path(repo) / ".git").exists():
        # Distinguish "not under version control" from "git failed": the first
        # means the log cannot be tied to a commit at all, which a reader of
        # this file needs to know before comparing two sessions.
        out["note"] = "not a git repository -- this session cannot be tied to a commit"
        return out

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    out["commit"] = run("rev-parse", "HEAD")
    out["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    # A dirty tree means the commit hash does NOT describe what ran. Worth
    # knowing when a log is being compared against another session.
    out["dirty"] = None if status is None else bool(status)
    return out


def _find_orca_teleop() -> Path | None:
    """Locate the installed orca_teleop checkout, for its git hash."""
    try:
        import orca_teleop  # noqa: PLC0415 - optional
    except Exception:  # noqa: BLE001
        return None
    src = Path(orca_teleop.__file__).resolve().parent
    # .../<repo>/src/orca_teleop/__init__.py -> <repo>
    for parent in list(src.parents)[:4]:
        if (parent / ".git").exists():
            return parent
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1)]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and type(value).__name__.endswith(("State", "Enum")):
        return value.value
    return value


def collect_meta(
    config: Any,
    *,
    session_id: str,
    tron_arm_repo: str | os.PathLike[str] | None = None,
    accid: str | None = None,
    firmware_version: str | None = None,
    url: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble ``session_meta.json``.

    Records the git state of **both** repos: a log is only comparable with
    another if you know which code produced each. ``firmware_version`` is the
    robot's ``version`` string when it supplies one -- see
    :meth:`SessionLogger.note_robot_info`.
    """
    repo = Path(tron_arm_repo) if tron_arm_repo else Path(__file__).resolve().parent.parent
    return {
        "session_id": session_id,
        "created_wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "created_wall_ns": time.time_ns(),
        "created_monotonic_ns": time.monotonic_ns(),
        "repos": {
            "tron_arm": git_describe(repo),
            "orca_teleop": git_describe(_find_orca_teleop()),
        },
        "robot": {
            # The URL actually connected to, which is NOT config.robot.url when
            # --robot overrides it. Recording the config value made a real
            # hardware session look like a loopback one.
            "url": url or getattr(getattr(config, "robot", None), "url", None),
            "accid": accid,
            "firmware_version": firmware_version,
        },
        "servop": {
            "format": config.servop.format,
            "send_both": config.servop.send_both,
            "rate_hz": config.servop.rate_hz,
        },
        "config": _config_snapshot(config),
        "achieved_rate_hz": None,  # filled in at close()
        **(dict(extra) if extra else {}),
    }


def _config_snapshot(config: Any) -> dict[str, Any]:
    """A full, replayable copy of the config that produced this session."""
    workspace = config.workspace
    return {
        "scale": config.scale,
        "mapping": {
            "translation_frame": config.mapping.translation_frame,
            "world_frame_axes_verified": config.mapping.world_frame_axes_verified,
        },
        "velocity": {"lin": config.velocity.lin, "ang": config.velocity.ang},
        "max_step": {"lin_m": config.max_step[0], "ang_rad": config.max_step[1]},
        "workspace": {
            "margin_m": workspace.margin_m,
            "left": _jsonable(workspace.left.bounds),
            "right": _jsonable(workspace.right.bounds),
        },
        "home_joints": _jsonable(config.home_joints),
    }


#: Bounded so a slow disk cannot grow memory without limit. Oldest records are
#: dropped first: recent history is what a post-mortem needs.
QUEUE_MAXSIZE = 20_000


class SessionLogger:
    """Gzipped JSONL writer for one session, off the control path.

    ``write()`` only appends to a queue; a background thread does the JSON
    encoding, gzip and disk I/O. That matters because ``dispatch`` records are
    written from upstream's arm worker thread, which must stay under 1 ms --
    doing ``json.dumps`` plus gzip there measurably blew that budget.

    The queue is bounded and drops oldest on overflow, so a stalled disk costs
    log fidelity, never control latency.

    Disabled loggers (``enabled=False``) accept every call and do nothing.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = "logs",
        *,
        session_id: str | None = None,
        enabled: bool = True,
        compress: bool = True,
    ) -> None:
        self.session_id = session_id or new_session_id()
        self.root = Path(root)
        self.dir = self.root / self.session_id
        self.enabled = enabled
        self._compress = compress
        self._fh: Any = None
        self._meta: dict[str, Any] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._writer: threading.Thread | None = None
        self._stop = threading.Event()
        self.records_written = 0
        self.dropped = 0
        self.firmware_version: str | None = None

    # -- lifecycle -------------------------------------------------------
    @property
    def records_path(self) -> Path:
        return self.dir / ("records.jsonl.gz" if self._compress else "records.jsonl")

    @property
    def meta_path(self) -> Path:
        return self.dir / "session_meta.json"

    def open(self) -> Path:
        if not self.enabled:
            return self.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._compress:
            self._fh = gzip.open(self.records_path, "at", encoding="utf-8")
        else:
            self._fh = self.records_path.open("a", encoding="utf-8")
        self._stop.clear()
        self._writer = threading.Thread(target=self._drain, name="session-log", daemon=True)
        self._writer.start()
        return self.dir

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        self._meta = dict(meta)
        self._flush_meta()


    def _flush_meta(self) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(_jsonable(self._meta), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def note_robot_info(self, data: Mapping[str, Any]) -> None:
        """Extract a firmware version from a ``notify_robot_info`` payload.

        The vendor guide does not document which key carries it -- the runbook
        gets it from the ``version`` diagnostic -- so this tries the plausible
        spellings and records nothing rather than guessing wrong.
        """
        for key in ("version", "firmware_version", "firmware", "sw_version"):
            value = data.get(key)
            if isinstance(value, str) and value:
                self.firmware_version = value
                if self._meta:
                    self._meta.setdefault("robot", {})["firmware_version"] = value
                    self._flush_meta()
                return

    def close(self, *, achieved_rate_hz: float | None = None,
              stats: Mapping[str, Any] | None = None) -> None:
        # Drain before stamping counts, so records_written is the real total.
        deadline = time.monotonic() + 5.0
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        if achieved_rate_hz is not None or stats is not None:
            self._meta["achieved_rate_hz"] = achieved_rate_hz
            if stats is not None:
                self._meta["final_stats"] = _jsonable(stats)
            self._meta["records_written"] = self.records_written
            self._meta["records_dropped"] = self.dropped
            self._flush_meta()
        # Let the writer finish what is queued before the handle closes.
        self._stop.set()
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.join(timeout=5.0)
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "SessionLogger":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- writing ---------------------------------------------------------
    def write(self, record: Mapping[str, Any]) -> None:
        """Queue one record. Never blocks, never raises.

        This is called from the arm worker thread on every dispatch, so it must
        stay trivial: no encoding, no I/O, no lock held across either.
        """
        if not self.enabled or self._writer is None:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop the oldest, keep the newest: a post-mortem cares about what
            # happened just before the end.
            self.dropped += 1
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(record)

    def _drain(self) -> None:
        """Encode and write, off the control path."""
        while True:
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                line = json.dumps(_jsonable(record), separators=(",", ":"))
            except (TypeError, ValueError):
                self.dropped += 1
                continue
            try:
                if self._fh is not None:
                    self._fh.write(line + "\n")
                    self.records_written += 1
            except (OSError, ValueError):
                self.dropped += 1

    def event(self, type_: str, **fields: Any) -> None:
        """Write a record of ``type_`` stamped with the monotonic clock."""
        self.write({"type": type_, "t_mono_ns": time.monotonic_ns(), **fields})
