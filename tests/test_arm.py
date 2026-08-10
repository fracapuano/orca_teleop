"""Tests for the arm worker: the checks an ArmSink implementer never writes."""

import threading
import time

import numpy as np
import pytest

from orca_teleop.arm import ArmSink, arm_worker
from orca_teleop.ingress.frames import LatestFrame, Pose, TeleopFrame

IDENTITY = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))


def _frame(**overrides) -> TeleopFrame:
    defaults = dict(
        timestamp_ns=1,
        recv_monotonic_ns=time.monotonic_ns(),
        stream_id=1,
        pose_epoch=0,
        handedness="right",
        tracking_valid=True,
        wrist=IDENTITY,
        head=None,
        wrist_angle_degrees=0.0,
    )
    return TeleopFrame(**{**defaults, **overrides})


class RecordingArmSink(ArmSink):
    def __init__(self, dispatch_raises: bool = False):
        self.dispatched: list[TeleopFrame] = []
        self.holds: list[str] = []
        self.references: list[tuple[int, int]] = []
        self.closed = False
        self._dispatch_raises = dispatch_raises

    def connect(self) -> None:
        pass

    def dispatch(self, frame: TeleopFrame) -> None:
        self.dispatched.append(frame)
        if self._dispatch_raises:
            raise RuntimeError("arm link exploded")

    def on_hold(self, reason: str) -> None:
        self.holds.append(reason)

    def on_reference_change(self, stream_id: int, pose_epoch: int) -> None:
        self.references.append((stream_id, pose_epoch))

    def close(self) -> None:
        self.closed = True


def _run_worker(frames: LatestFrame, sink: ArmSink, **kwargs) -> threading.Thread:
    stop_event = kwargs.pop("stop_event", threading.Event())
    thread = threading.Thread(
        target=arm_worker,
        kwargs=dict(frames=frames, sink=sink, stop_event=stop_event, **kwargs),
        daemon=True,
    )
    thread.start()
    return thread


def test_arm_worker_exits_on_stop_event():
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop)
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert sink.closed


def test_arm_worker_exits_on_ingress_close():
    """The ingress going away must not strand the arm thread."""
    frames, sink = LatestFrame(), RecordingArmSink()
    thread = _run_worker(frames, sink)
    time.sleep(0.05)
    frames.close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert sink.closed


def test_fresh_frames_are_dispatched():
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop)
    for index in range(3):
        frames.publish(_frame(timestamp_ns=index))
        time.sleep(0.03)
    stop.set()
    thread.join(timeout=2.0)
    assert [f.timestamp_ns for f in sink.dispatched] == [0, 1, 2]


def test_on_hold_is_edge_triggered():
    """One hold per transition, not one per poll."""
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop, poll_timeout_s=0.01)
    time.sleep(0.15)  # many polls, no frames
    stop.set()
    thread.join(timeout=2.0)
    assert sink.holds == ["no_frames"]


def test_hold_re_arms_after_frames_resume():
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop, poll_timeout_s=0.01)
    time.sleep(0.08)
    frames.publish(_frame())
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=2.0)
    assert sink.holds.count("no_frames") >= 1
    assert len(sink.dispatched) == 1


def test_stale_frame_holds_instead_of_dispatching():
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop, stale_after_s=0.01)
    frames.publish(_frame(recv_monotonic_ns=time.monotonic_ns() - int(1e9)))
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=2.0)
    assert sink.dispatched == []
    assert "stale" in sink.holds


def test_tracking_invalid_frame_holds():
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop)
    frames.publish(_frame(tracking_valid=False))
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=2.0)
    assert sink.dispatched == []
    assert "tracking_invalid" in sink.holds


def test_reference_change_fires_on_stream_id_and_pose_epoch():
    """Headset off/on (epoch) and a new publisher (stream) both re-clutch."""
    frames, sink, stop = LatestFrame(), RecordingArmSink(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop)
    frames.publish(_frame(stream_id=1, pose_epoch=7))
    time.sleep(0.05)
    frames.publish(_frame(stream_id=1, pose_epoch=7))  # same reference: no re-fire
    time.sleep(0.05)
    frames.publish(_frame(stream_id=1, pose_epoch=8))  # XR session re-pinned
    time.sleep(0.05)
    frames.release(1)
    frames.publish(_frame(stream_id=2, pose_epoch=8))  # different publisher
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=2.0)
    assert sink.references == [(1, 7), (1, 8), (2, 8)]


def test_dispatch_exception_is_contained():
    """A bad command must not kill the thread and strand the arm."""
    frames, sink, stop = LatestFrame(), RecordingArmSink(dispatch_raises=True), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop)
    frames.publish(_frame(timestamp_ns=1))
    time.sleep(0.05)
    frames.publish(_frame(timestamp_ns=2))
    time.sleep(0.05)
    assert thread.is_alive()
    stop.set()
    thread.join(timeout=2.0)
    assert len(sink.dispatched) == 2
    assert sink.holds  # entered a hold rather than pretending success
    assert sink.closed


def test_close_is_called_even_when_the_sink_misbehaves():
    class ExplodingOnHold(RecordingArmSink):
        def on_hold(self, reason: str) -> None:
            raise RuntimeError("hold failed")

    frames, sink, stop = LatestFrame(), ExplodingOnHold(), threading.Event()
    thread = _run_worker(frames, sink, stop_event=stop, poll_timeout_s=0.01)
    time.sleep(0.08)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert sink.closed


def test_arm_sink_requires_every_method():
    """ABC, not Protocol: a missing method fails at construction."""

    class Incomplete(ArmSink):
        def connect(self) -> None: ...
        def dispatch(self, frame) -> None: ...

    with pytest.raises(TypeError):
        Incomplete()
