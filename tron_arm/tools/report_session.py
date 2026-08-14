#!/usr/bin/env python3
"""Summarise a session log as plain text, for pasting into an issue.

    python tools/report_session.py logs/<session-id>
    python tools/report_session.py logs/<session-id> --json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


def load_records(session_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield records, skipping unparseable lines.

    A killed session leaves a truncated final line; refusing to report on an
    interrupted run would fail exactly when it matters.
    """
    for name in ("records.jsonl.gz", "records.jsonl"):
        path = session_dir / name
        if not path.exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return
    raise FileNotFoundError(f"no records.jsonl(.gz) in {session_dir}")


def load_meta(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "session_meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def latencies_ms(ticks: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    """dispatch -> FIRST ws send carrying it, per arm.

    Counting every tick would be wrong: a held target is re-sent every tick, so
    the tail would measure how long the operator paused, not pipeline latency.
    """
    seen: set[tuple[str, int]] = set()
    per_arm: dict[str, list[float]] = defaultdict(list)
    for tick in ticks:
        send = tick.get("t_ws_send")
        if send is None:
            continue
        for arm, detail in (tick.get("arms") or {}).items():
            dispatch = detail.get("t_dispatch")
            if not dispatch or (arm, int(dispatch)) in seen:
                continue
            seen.add((arm, int(dispatch)))
            per_arm[arm].append((send - dispatch) / 1e6)
    return per_arm


def summarise(session_dir: Path) -> dict[str, Any]:
    """Machine-readable summary; the text report is rendered from this."""
    meta = load_meta(session_dir)
    records = list(load_records(session_dir))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[record.get("type", "?")].append(record)

    dispatches, ticks = by_type["dispatch"], by_type["tick"]
    per_arm = latencies_ms(ticks)
    flat = [v for values in per_arm.values() for v in values]
    recv = sorted(d["recv_monotonic_ns"] for d in dispatches if d.get("recv_monotonic_ns"))
    gaps = np.diff(recv) / 1e9 if len(recv) > 2 else np.array([])
    gaps = gaps[gaps > 0]
    notifies = by_type["notify"]

    return {
        "session_id": session_dir.name,
        "meta": meta,
        "records": len(records),
        "dispatches": len(dispatches),
        "ticks": len(ticks),
        "latency_ms": {
            "per_arm": {a: {"n": len(v), "p50": float(np.percentile(v, 50)),
                            "p95": float(np.percentile(v, 95)),
                            "max": float(np.max(v))} for a, v in per_arm.items()},
            "p50": float(np.percentile(flat, 50)) if flat else None,
            "p95": float(np.percentile(flat, 95)) if flat else None,
            "max": float(np.max(flat)) if flat else None,
        },
        "ingress": {
            "frames": len(recv),
            "mean_hz": float(np.mean(1.0 / gaps)) if len(gaps) else None,
            "gap_p95_ms": float(np.percentile(gaps, 95) * 1e3) if len(gaps) else None,
            "gap_max_ms": float(gaps.max() * 1e3) if len(gaps) else None,
            "gaps_over_stale_deadline": int(np.sum(gaps > 0.25)) if len(gaps) else 0,
        },
        "holds": [{"t": h.get("t_mono_ns"), "arm": h.get("arm"),
                   "reason": h.get("reason"), "state": h.get("state")}
                  for h in by_type["hold"]],
        "reference_changes": [{"t": r.get("t_mono_ns"), "stream_id": r.get("stream_id"),
                               "pose_epoch": r.get("pose_epoch")}
                              for r in by_type["reference_change"]],
        "latches": sum(1 for d in dispatches if d.get("latched")),
        "notify_titles": dict(Counter(n.get("title") for n in notifies)),
        "notify_failures": [
            {"t": n.get("t_mono_ns"), "title": n.get("title"), "data": n.get("data")}
            for n in notifies
            if str((n.get("data") or {}).get("result", "")).startswith("fail")
        ],
        "clamps": {
            "workspace_dispatches": sum(1 for d in dispatches if d.get("ws_clamped_axes")),
            "workspace_axes": dict(Counter(
                a for d in dispatches for a in (d.get("ws_clamped_axes") or []))),
            "step_lin": sum(1 for t in ticks for a in (t.get("arms") or {}).values()
                            if a.get("lin_clamped")),
            "step_ang": sum(1 for t in ticks for a in (t.get("arms") or {}).values()
                            if a.get("ang_clamped")),
        },
        "achieved_rate_hz": meta.get("achieved_rate_hz"),
        "step_test": [{"format": s.get("format"), "phase": s.get("phase"),
                       "passed": s.get("passed"), "detail": s.get("detail")}
                      for s in by_type["step_test"]],
    }


def render(s: dict[str, Any]) -> str:
    """Plain text, <=78 cols, no colour -- must survive being pasted anywhere."""
    meta, out = s["meta"], []
    bar = "-" * 78
    add = out.append

    add("=" * 78)
    add(f"SESSION REPORT  {s['session_id']}")
    add("=" * 78)

    add("\nProvenance\n" + bar)
    for name, info in (meta.get("repos") or {}).items():
        commit = (info.get("commit") or "")[:12] or "unknown"
        note = f"  [{info['note']}]" if info.get("note") else ""
        add(f"  {name:<14} {info.get('branch') or '?'} @ {commit}"
            f"{' +dirty' if info.get('dirty') else ''}{note}")
    robot, servop = meta.get("robot", {}), meta.get("servop", {})
    add(f"  robot          {robot.get('url')}  accid={robot.get('accid')}")
    add(f"  firmware       {robot.get('firmware_version') or 'not reported'}")
    add(f"  servop         format={servop.get('format')} send_both={servop.get('send_both')} "
        f"rate={servop.get('rate_hz')} Hz")
    config = meta.get("config", {})
    add(f"  scale          {config.get('scale')}  "
        f"frame={config.get('mapping', {}).get('translation_frame')}  "
        f"step clamp={config.get('max_step', {}).get('lin_m')} m/tick")
    add(f"  records        {s['records']} ({s['dispatches']} dispatch, {s['ticks']} tick)")

    add("\nDispatch -> ws send latency (each dispatch counted once)\n" + bar)
    if s["latency_ms"]["p50"] is not None:
        add("  arm         n      p50     p95     max   (ms)")
        for arm, v in sorted(s["latency_ms"]["per_arm"].items()):
            add(f"  {arm:<8} {v['n']:6d}  {v['p50']:6.2f}  {v['p95']:6.2f}  {v['max']:6.2f}")
        add(f"  {'all':<8} {'':6}  {s['latency_ms']['p50']:6.2f}  "
            f"{s['latency_ms']['p95']:6.2f}  {s['latency_ms']['max']:6.2f}")
        add("  Frame received -> bytes to the socket. Excludes robot execution.")
        add("  Runbook step 7 targets < 80 ms motion-to-motion end to end.")
    else:
        add("  (no ticks carried a dispatch stamp -- was anything engaged?)")
    sink = (meta.get("final_stats") or {}).get("sink") or {}
    if sink:
        add(f"  in-sink dispatch cost: p50 {sink.get('dispatch_p50_ms', float('nan')):.3f} ms  "
            f"p95 {sink.get('dispatch_p95_ms', float('nan')):.3f} ms  (must stay < 1 ms)")

    ingress = s["ingress"]
    add("\nIngress\n" + bar)
    if ingress["mean_hz"]:
        add(f"  {ingress['frames']} frames, mean {ingress['mean_hz']:.1f} Hz")
        add(f"  gap p95 {ingress['gap_p95_ms']:.1f} ms   max {ingress['gap_max_ms']:.1f} ms")
        add(f"  gaps past ARM_STALE_AFTER_S (0.25 s): {ingress['gaps_over_stale_deadline']}")
    else:
        add("  (too few frames)")

    add("\nHolds\n" + bar)
    if s["holds"]:
        counts = Counter(h["reason"] for h in s["holds"])
        add(f"  {len(s['holds'])} event(s): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        for hold in s["holds"][:20]:
            add(f"    {hold['arm']:<6} {hold['reason']:<18} -> {hold['state']}")
    else:
        add("  none -- frames never stopped past the deadline")

    add("\nReference changes\n" + bar)
    if s["reference_changes"]:
        for ref in s["reference_changes"]:
            add(f"    stream={ref['stream_id']} epoch={ref['pose_epoch']} -> origins cleared")
        add(f"  {len(s['reference_changes'])} change(s), {s['latches']} latch(es) total.")
        add("  Each change should be followed by exactly one re-latch.")
    else:
        add("  none")

    add("\nnotify_*\n" + bar)
    for title, count in sorted(s["notify_titles"].items()):
        add(f"  {title:<28} {count}")
    if s["notify_failures"]:
        add(f"  !! {len(s['notify_failures'])} FAILURE(s):")
        for note in s["notify_failures"][:10]:
            data = note.get("data") or {}
            add(f"     {note['title']}: {data.get('result')} {data.get('reason', '')}")
    else:
        add("  no failure notifies")

    clamps = s["clamps"]
    add("\nClamps\n" + bar)
    add(f"  workspace  {clamps['workspace_dispatches']} dispatch(es) {clamps['workspace_axes']}")
    add(f"  step       {clamps['step_lin']} linear, {clamps['step_ang']} angular")

    streamer = (meta.get("final_stats") or {}).get("streamer") or {}
    if streamer:
        add("\nStreamer\n" + bar)
        add(f"  achieved {streamer.get('achieved_rate_hz')} Hz "
            f"(target {streamer.get('target_rate_hz')})  "
            f"jitter p95 {streamer.get('jitter_p95_ms')} ms")
        add(f"  ticks {streamer.get('ticks')}  late {streamer.get('late_ticks')}  "
            f"idle {streamer.get('idle_ticks')}  NaN {streamer.get('nan_rejected')}")

    if s["step_test"]:
        add("\nStep test\n" + bar)
        for phase in s["step_test"]:
            verdict = {True: "PASS", False: "FAIL", None: "----"}[phase["passed"]]
            add(f"  [{verdict}] {phase['format']:<11} {phase['phase']:<26} {phase['detail']}")

    add("\n" + "=" * 78)
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.session.is_dir():
        print(f"not a directory: {args.session}", file=sys.stderr)
        return 2
    try:
        summary = summarise(args.session)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, default=str) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
