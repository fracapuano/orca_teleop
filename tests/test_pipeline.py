"""Tests for orca_teleop.pipeline.

These tests cover the threading/queue plumbing only — they never touch real
hardware. Happy paths use `orca_core.test_mock.MockOrcaHand` (installed via
the `patch_mock_hand` fixture in conftest.py); failure modes install a small
custom stub class instead.
"""

from __future__ import annotations

import inspect
import queue
import threading
import time

import pytest
from orca_core import OrcaJointPositions
from orca_core.test_mock import MockOrcaHand

from orca_teleop.pipeline import (
    _SHUTDOWN,
    TeleopQueues,
    ingress_worker,
    retargeter_worker,
    robot_worker,
    run,
)

# ---------- helpers ----------------------------------------------------------


def _make_queues(maxsize: int = 8) -> TeleopQueues:
    return TeleopQueues(
        landmarks_q=queue.Queue(maxsize=maxsize),
        actions_q=queue.Queue(maxsize=maxsize),
    )


def _midpoint_action() -> OrcaJointPositions:
    """Build an OrcaJointPositions at every joint's ROM midpoint (degrees)."""
    roms = MockOrcaHand().config.joint_roms_dict
    return OrcaJointPositions({j: 0.5 * (lo + hi) for j, (lo, hi) in roms.items()})


def _start(target, *args, name: str | None = None) -> threading.Thread:
    t = threading.Thread(target=target, args=args, name=name, daemon=True)
    t.start()
    return t


# ---------- 1. public surface ------------------------------------------------


def test_public_exports():
    from orca_teleop import (  # noqa: F401
        TeleopQueues as _PQ,
    )


def test_pipeline_queues_dataclass():
    q = _make_queues()
    assert isinstance(q.landmarks_q, queue.Queue)
    assert isinstance(q.actions_q, queue.Queue)


def test_run_signature_stable():
    sig = inspect.signature(run)
    assert "model_path" in sig.parameters


# ---------- 2. _SHUTDOWN propagation ----------------------------------------


def test_ingress_emits_shutdown_on_stop():
    q = _make_queues()
    stop = threading.Event()
    t = _start(ingress_worker, q, stop, name="ingress")
    time.sleep(0.05)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    items = []
    while True:
        try:
            items.append(q.landmarks_q.get_nowait())
        except queue.Empty:
            break
    assert items[-1] is _SHUTDOWN


def test_retargeter_forwards_shutdown_downstream(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())
    q = _make_queues()
    stop = threading.Event()
    q.landmarks_q.put(_SHUTDOWN)
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert q.actions_q.get_nowait() is _SHUTDOWN


def test_retargeter_emits_shutdown_on_stop_event(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())
    q = _make_queues()
    stop = threading.Event()
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    time.sleep(0.05)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    items = []
    while True:
        try:
            items.append(q.actions_q.get_nowait())
        except queue.Empty:
            break
    assert items[-1] is _SHUTDOWN


def test_robot_exits_on_shutdown_sentinel(patch_mock_hand):
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    q.actions_q.put(_SHUTDOWN)
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(patch_mock_hand) == 1  # the hand was constructed


# ---------- 3. robot_worker happy path --------------------------------------


def test_robot_sets_ready_after_init(patch_mock_hand):
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    hand = patch_mock_hand[0]
    assert hand.is_connected()
    stop.set()
    t.join(timeout=2.0)


def test_robot_consumes_orca_joint_positions(patch_mock_hand):
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    action = _midpoint_action()
    q.actions_q.put(action)
    q.actions_q.put(action)
    q.actions_q.put(_SHUTDOWN)
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_robot_accepts_in_rom_positions(patch_mock_hand):
    """An OrcaJointPositions built from the mock's own ROM midpoints must
    flow through without raising. Locks the units the retargeter must emit."""
    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    q.actions_q.put(_midpoint_action())
    q.actions_q.put(_SHUTDOWN)
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)


# ---------- 4. robot_worker failure modes -----------------------------------


class _FailingConnectHand:
    def __init__(self, model_path=None):
        self.init_called = False
        self.disconnected = False

    def connect(self):
        return False, "boom"

    def init_joints(self):
        self.init_called = True

    def set_joint_positions(self, action):
        pass

    def disable_torque(self):
        pass

    def disconnect(self):
        self.disconnected = True


class _ExplodingHand:
    def __init__(self, model_path=None):
        self.disabled = False
        self.disconnected = False

    def connect(self):
        return True, "ok"

    def init_joints(self):
        pass

    def set_joint_positions(self, action):
        raise RuntimeError("kaboom")

    def disable_torque(self):
        self.disabled = True

    def disconnect(self):
        self.disconnected = True


def test_robot_connect_failure_leaves_ready_clear(monkeypatch):
    instances: list[_FailingConnectHand] = []

    def factory(model_path=None):
        h = _FailingConnectHand()
        instances.append(h)
        return h

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)

    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert not ready.is_set()
    assert not instances[0].init_called


def test_robot_finally_cleans_up_on_exception(monkeypatch):
    instances: list[_ExplodingHand] = []

    def factory(model_path=None):
        h = _ExplodingHand()
        instances.append(h)
        return h

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)

    q = _make_queues()
    stop = threading.Event()
    ready = threading.Event()
    q.actions_q.put(_midpoint_action())
    t = _start(robot_worker, q, stop, ready, None, name="robot")
    assert ready.wait(2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert instances[0].disabled and instances[0].disconnected


# ---------- 5. run() orchestration ------------------------------------------


def test_run_raises_on_connect_failure(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", _FailingConnectHand)
    with pytest.raises(RuntimeError, match="failed to connect"):
        run("ignored")


def test_run_does_not_start_producers_if_robot_fails(monkeypatch):
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", _FailingConnectHand)
    before = {t.name for t in threading.enumerate()}
    with pytest.raises(RuntimeError):
        run("ignored")
    time.sleep(0.05)
    after_names = {t.name for t in threading.enumerate() if t.is_alive()}
    leaked = after_names - before
    assert "retargeter" not in leaked
    assert "ingress" not in leaked


def test_run_starts_producers_then_stops_cleanly(monkeypatch, patch_mock_hand):
    """Interrupt the main-thread action loop after a few iterations and verify
    both producer threads shut down cleanly."""
    real_get = queue.Queue.get
    calls = {"n": 0}
    main_ident = threading.main_thread().ident

    def fake_get(self, *args, **kwargs):
        if threading.get_ident() == main_ident:
            calls["n"] += 1
            if calls["n"] >= 3:
                raise KeyboardInterrupt
        return real_get(self, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "get", fake_get)

    run(None)
    time.sleep(0.05)
    alive = {t.name for t in threading.enumerate() if t.is_alive()}
    assert "retargeter" not in alive
    assert "ingress" not in alive


# ---------- 6. backpressure --------------------------------------------------


def test_landmarks_queue_is_bounded():
    q = _make_queues(maxsize=2)
    q.landmarks_q.put_nowait("a")
    q.landmarks_q.put_nowait("b")
    with pytest.raises(queue.Full):
        q.landmarks_q.put("c", timeout=0.05)


# ---------- 7. active worker smoke tests -------------------------------------

import numpy as np


def test_ingress_emits_landmark_arrays_then_shutdown():
    """ingress_worker produces (21, 3) float32 arrays and terminates with _SHUTDOWN."""
    q = _make_queues()
    stop = threading.Event()
    t = _start(ingress_worker, q, stop, name="ingress")
    time.sleep(0.15)  # allow a few 30 Hz frames
    stop.set()
    t.join(timeout=2.0)
    items = []
    while True:
        try:
            items.append(q.landmarks_q.get_nowait())
        except queue.Empty:
            break
    assert items[-1] is _SHUTDOWN
    landmarks = [x for x in items if x is not _SHUTDOWN]
    assert len(landmarks) > 0
    for lm in landmarks:
        assert isinstance(lm, np.ndarray)
        assert lm.shape == (21, 3)
        assert lm.dtype == np.float32


def test_retargeter_forwards_joint_positions(monkeypatch):
    """retargeter_worker converts landmarks to OrcaJointPositions and enqueues them."""
    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", lambda mp=None: MockOrcaHand())
    q = _make_queues()
    stop = threading.Event()
    # Feed a few fake landmark arrays, then stop
    for _ in range(3):
        q.landmarks_q.put(np.zeros((21, 3), dtype=np.float32))
    t = _start(retargeter_worker, q, stop, None, name="retargeter")
    time.sleep(0.1)
    stop.set()
    t.join(timeout=2.0)
    items = []
    while True:
        try:
            items.append(q.actions_q.get_nowait())
        except queue.Empty:
            break
    actions = [x for x in items if x is not _SHUTDOWN]
    assert len(actions) > 0
    for action in actions:
        assert isinstance(action, OrcaJointPositions)
