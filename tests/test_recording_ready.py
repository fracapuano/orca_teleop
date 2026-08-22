"""Tests for recording readiness gating."""

from __future__ import annotations

import queue
import threading

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.recording import (
    TeleopActionMirror,
    drain_actions_queue,
    poll_latest_action,
    teleop_consumer_loop,
    wait_for_recording_ready,
    wait_for_teleop_mirror_ready,
)
from tests.conftest import wait_until

_SHUTDOWN = object()
_JOINT_IDS = [f"j{i}" for i in range(3)]


def _action(values: list[float] | None = None) -> OrcaJointPositions:
    vals = values if values is not None else [0.0, 0.0, 0.0]
    return OrcaJointPositions.from_dict(dict(zip(_JOINT_IDS, vals, strict=True)))


def test_wait_for_recording_ready_requires_action_and_obs_streaks():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    obs_calls = {"n": 0}
    dispatched: list[OrcaJointPositions] = []

    def get_observation():
        obs_calls["n"] += 1
        return {"ok": True}

    for i in range(5):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    ready = wait_for_recording_ready(
        get_observation=get_observation,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=dispatched.append,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )

    assert ready is True
    assert obs_calls["n"] == 5
    assert len(dispatched) == 5


def test_wait_for_recording_ready_resets_when_observation_fails():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    obs_calls = {"n": 0}

    def get_observation():
        obs_calls["n"] += 1
        if obs_calls["n"] < 4:
            raise RuntimeError("camera dead")
        return {"ok": True}

    # 3 failing probes, then 5 healthy ones — must not declare ready early.
    for i in range(8):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    ready = wait_for_recording_ready(
        get_observation=get_observation,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=None,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )

    assert ready is True
    assert obs_calls["n"] == 8


def test_wait_for_recording_ready_returns_false_on_stop():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def get_observation():
        stop_event.set()   # the operator hits ctrl-c mid-warm-up
        return None

    # get_observation only runs once an action has been dequeued, so the stop
    # has to be reachable through the queue rather than from a timer thread.
    actions_q.put(_action())

    ready = wait_for_recording_ready(
        get_observation=get_observation,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=None,
        shutdown_sentinel=_SHUTDOWN,
        heartbeat_interval=0.01,
        min_action_streak=50,
        min_obs_streak=50,
        status_interval_s=100.0,
    )
    assert ready is False


def test_wait_for_recording_ready_returns_false_on_shutdown_sentinel():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    actions_q.put(_SHUTDOWN)

    ready = wait_for_recording_ready(
        get_observation=lambda: None,
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=None,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )
    assert ready is False
    assert stop_event.is_set()


def test_wait_for_recording_ready_treats_dispatch_failure_as_not_ready():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    state = {"n": 0}

    def dispatch(action: OrcaJointPositions) -> None:
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("servo offline")

    for i in range(7):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    ready = wait_for_recording_ready(
        get_observation=lambda: np.zeros(3),
        actions_q=actions_q,
        stop_event=stop_event,
        dispatch_action=dispatch,
        shutdown_sentinel=_SHUTDOWN,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )
    assert ready is True
    assert state["n"] == 7


def test_drain_actions_queue_discards_stale_commands():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    for i in range(4):
        actions_q.put(_action([float(i), 0.0, 0.0]))

    drained = drain_actions_queue(
        actions_q,
        stop_event=stop_event,
        shutdown_sentinel=_SHUTDOWN,
    )

    assert drained == 4
    assert actions_q.empty()


def test_poll_latest_action_returns_newest_command():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    actions_q.put(_action([1.0, 0.0, 0.0]))
    actions_q.put(_action([2.0, 0.0, 0.0]))

    latest = poll_latest_action(
        actions_q,
        last_action=None,
        stop_event=stop_event,
        shutdown_sentinel=_SHUTDOWN,
    )

    assert latest is not None
    assert latest.as_array(_JOINT_IDS)[0] == 2.0
    assert actions_q.empty()


def test_teleop_consumer_loop_mirrors_and_respects_dispatch_gate():
    actions_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    dispatch_enabled = threading.Event()
    mirror = TeleopActionMirror()
    dispatched: list[OrcaJointPositions] = []

    thread = threading.Thread(
        target=teleop_consumer_loop,
        args=(actions_q,),
        kwargs={
            "mirror": mirror,
            "stop_event": stop_event,
            "shutdown_sentinel": _SHUTDOWN,
            "dispatch_action": dispatched.append,
            "dispatch_enabled": dispatch_enabled,
            "heartbeat_interval": 0.01,
        },
        daemon=True,
    )
    thread.start()

    actions_q.put(_action([1.0, 0.0, 0.0]))
    assert wait_until(lambda: mirror.snapshot() is not None)
    assert dispatched == []

    dispatch_enabled.set()
    actions_q.put(_action([2.0, 0.0, 0.0]))
    assert wait_until(lambda: len(dispatched) == 1)
    assert dispatched[0].as_array(_JOINT_IDS)[0] == 2.0

    stop_event.set()
    thread.join(timeout=1.0)


def test_wait_for_teleop_mirror_ready_waits_for_mirror_updates():
    """One mirror update per probe, so the streak builds iff each poll re-reads
    ``update_count``. Feeding from a timed thread instead made the streak race
    the poll interval and could never converge."""
    stop_event = threading.Event()
    mirror = TeleopActionMirror()
    probes = {"n": 0}

    def get_observation():
        probes["n"] += 1
        mirror.update(_action([float(probes["n"]), 0.0, 0.0]))
        return {"ok": True}

    ready = wait_for_teleop_mirror_ready(
        get_observation=get_observation,
        mirror=mirror,
        stop_event=stop_event,
        heartbeat_interval=0.0,
        min_action_streak=5,
        min_obs_streak=5,
        status_interval_s=100.0,
    )
    assert ready is True
    assert probes["n"] == 6  # the first poll has no increase to observe yet
