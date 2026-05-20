"""State-machine tests for ``_drain_queue`` in ``scripts/teleop_arm_quest.py``.

``_drain_queue`` runs the per-side ``awaiting_anchor`` → ``tracking`` ⇄
``clutched`` state machine and is the highest-risk piece of the Quest path
(its behavior is what the operator actually feels). These tests drive it
directly with synthetic ``HandLandmarks`` against a stub IK (only
``forward_kinematics_full`` is exercised) and a stub retargeter (always
returns ``None``), so we exercise transitions without spinning up pinocchio
or pink.

The ``sys.path`` stunt below is intentional: ``_drain_queue`` ships in the
scripts/ directory and we want to test the version that actually runs. When
the state machine moves into the package as part of the planned refactor,
this can drop back to a regular ``from orca_teleop...`` import.
"""

from __future__ import annotations

import collections
import queue
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from teleop_arm_quest import (  # noqa: E402
    SIDES,
    MetaQuestDrainStats,
    _drain_latest_metaquest_frames,
    _drain_queue,
)

from orca_teleop.ingress.server import HandLandmarks, WristPose  # noqa: E402


class StubIK:
    """Returns a fixed transform from ``forward_kinematics_full``.

    The fixed translation lets a re-anchor test assert "T_home was reset to
    FK(q_prev, side)" without having to load the real URDF.
    """

    FK_TRANSLATION = np.array([0.42, 0.43, 0.44])

    def forward_kinematics_full(self, q: np.ndarray, side: str) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self.FK_TRANSLATION
        return T


class StubRetargeter:
    """No-op retargeter — returns ``None`` so the hand-targets branch is a no-op."""

    def retarget(self, target_pose):  # noqa: ARG002
        return None


def _wrist_pose(position, rotation=None) -> WristPose:
    return WristPose(
        position=np.asarray(position, dtype=np.float32),
        rotation=np.asarray(rotation if rotation is not None else np.eye(3), dtype=np.float32),
    )


def _landmark(position, *, side: str = "right", ts_ns: int = 0, rotation=None) -> HandLandmarks:
    return HandLandmarks(
        keypoints=np.zeros((21, 3), dtype=np.float32),
        handedness=side,  # type: ignore[arg-type]
        timestamp_ns=ts_ns,
        wrist_pose=_wrist_pose(position, rotation),
    )


def test_latest_metaquest_drain_reports_inactive_and_missing_wrist() -> None:
    q: queue.Queue = queue.Queue()
    q.put(_landmark([0.0, 0.0, 0.0], side="left", ts_ns=1))
    q.put(
        HandLandmarks(
            keypoints=np.zeros((21, 3), dtype=np.float32),
            handedness="right",
            timestamp_ns=2,
            wrist_pose=None,
        )
    )
    q.put(_landmark([0.1, 0.2, 0.3], side="right", ts_ns=3))
    stats = MetaQuestDrainStats()

    frames = _drain_latest_metaquest_frames(q, ("right",), stats=stats)

    assert len(frames) == 1
    assert frames[0].handedness == "right"
    assert stats.total_items == 3
    assert stats.by_side == {"left": 1, "right": 2}
    assert stats.usable_by_side == {"right": 1}
    assert stats.inactive_side == 1
    assert stats.missing_wrist_pose == 1


def _initial_state(*, still_window: int = 5) -> dict:
    """Fresh state dicts in the shape ``_drain_queue`` mutates."""
    return {
        "pose_window": {s: collections.deque(maxlen=still_window) for s in SIDES},
        "span_buffer": {s: collections.deque(maxlen=2_000) for s in SIDES},
        "last_refit_t": {s: 0.0 for s in SIDES},
        "clutch_start_t": {s: None for s in SIDES},
        "T_first": {},
        "T_home": {s: pin.SE3(np.eye(3), np.zeros(3)) for s in SIDES},
        "scale": {},
        "targets": {},
        "hand_targets": {},
        "new_target_acquire_ns": {},
    }


def _call(
    state: dict,
    q: queue.Queue,
    ik: StubIK,
    retargeters: dict[str, StubRetargeter],
    *,
    still_window: int = 5,
    clutch_grace_s: float = 0.5,
    manual_scale: float | None = 0.5,
) -> None:
    """Run ``_drain_queue`` with test defaults. Caller contract: clear
    ``new_target_acquire_ns`` first (matches the main loop)."""
    state["new_target_acquire_ns"].clear()
    _drain_queue(
        q,
        state["pose_window"],
        state["span_buffer"],
        state["last_refit_t"],
        state["clutch_start_t"],
        state["T_first"],
        state["T_home"],
        state["scale"],
        state["targets"],
        state["hand_targets"],
        state["new_target_acquire_ns"],
        ik,
        retargeters,
        np.zeros(50, dtype=np.float64),
        manual_scale=manual_scale,
        workspace_delta_limits_m={
            # Wide enough that the clip never bites for the synthetic test inputs.
            "left": ((-1.0, -1.0, -1.0), (+1.0, +1.0, +1.0)),
            "right": ((-1.0, -1.0, -1.0), (+1.0, +1.0, +1.0)),
        },
        auto_fit_margin=0.7,
        # min_span_samples high + period high → auto-fit never runs in tests.
        min_span_samples=10_000_000,
        span_refit_period_s=1e9,
        span_change_threshold=0.1,
        still_threshold_m=0.01,
        still_window_samples=still_window,
        clutch_grace_s=clutch_grace_s,
    )


@pytest.fixture
def stubs() -> tuple[StubIK, dict[str, StubRetargeter]]:
    return StubIK(), {s: StubRetargeter() for s in SIDES}


def test_anchors_on_stillness_window(stubs):
    """5 still poses fill the window → side anchors, target seeded at T_home."""
    ik, retargeters = stubs
    state = _initial_state(still_window=5)
    q: queue.Queue = queue.Queue()

    for _ in range(5):
        q.put(_landmark([0.0, 0.0, 0.0], ts_ns=11))
        _call(state, q, ik, retargeters)

    assert "right" in state["T_first"]
    assert "right" in state["targets"]
    np.testing.assert_allclose(
        state["targets"]["right"].translation,
        state["T_home"]["right"].translation,
    )
    # The seed-target write site records the source frame's ts so the main
    # loop can compute end-to-end lag at sink.update().
    assert state["new_target_acquire_ns"]["right"] == 11


def test_anchor_skipped_when_window_not_filled(stubs):
    """4 still poses (window=5) should NOT anchor — window not yet full."""
    ik, retargeters = stubs
    state = _initial_state(still_window=5)
    q: queue.Queue = queue.Queue()

    for _ in range(4):
        q.put(_landmark([0.0, 0.0, 0.0]))
        _call(state, q, ik, retargeters)

    assert "right" not in state["T_first"]
    assert "right" not in state["targets"]


def test_tracking_advances_target_on_motion(stubs):
    """After anchor, a moving wrist pose updates the IK target and stamps lag."""
    ik, retargeters = stubs
    state = _initial_state(still_window=5)
    q: queue.Queue = queue.Queue()

    for _ in range(5):
        q.put(_landmark([0.0, 0.0, 0.0]))
        _call(state, q, ik, retargeters)

    initial_target = state["targets"]["right"].translation.copy()

    q.put(_landmark([0.10, 0.0, 0.0], ts_ns=12345))
    _call(state, q, ik, retargeters)

    assert not np.allclose(state["targets"]["right"].translation, initial_target)
    assert state["new_target_acquire_ns"]["right"] == 12345


def test_per_axis_scale_seeded_on_anchor(stubs):
    """With manual_scale=0.5 the seeded scale is a (3,) array of 0.5."""
    ik, retargeters = stubs
    state = _initial_state(still_window=5)
    q: queue.Queue = queue.Queue()

    for _ in range(5):
        q.put(_landmark([0.0, 0.0, 0.0]))
        _call(state, q, ik, retargeters, manual_scale=0.5)

    np.testing.assert_array_equal(state["scale"]["right"], np.full(3, 0.5))


def test_clutch_engages_after_stillness_returns(stubs):
    """Anchored → moved off → goes still again → clutch latches."""
    ik, retargeters = stubs
    state = _initial_state(still_window=3)
    q: queue.Queue = queue.Queue()

    # Anchor.
    for _ in range(3):
        q.put(_landmark([0.0, 0.0, 0.0]))
        _call(state, q, ik, retargeters, still_window=3)

    assert "right" in state["T_first"]

    # Move off so the window stops being still.
    q.put(_landmark([0.05, 0.0, 0.0]))
    _call(state, q, ik, retargeters, still_window=3)
    assert state["clutch_start_t"]["right"] is None

    # Three still poses at the new position → window registers still again.
    for _ in range(3):
        q.put(_landmark([0.05, 0.0, 0.0]))
        _call(state, q, ik, retargeters, still_window=3)

    assert state["clutch_start_t"]["right"] is not None


def test_reanchor_on_motion_after_clutch_grace(stubs):
    """In a clutched state with grace already elapsed, the first motion sample
    resets T_first to the operator's current pose and T_home to FK(q_prev)."""
    ik, retargeters = stubs
    state = _initial_state(still_window=5)
    q: queue.Queue = queue.Queue()

    for _ in range(5):
        q.put(_landmark([0.0, 0.0, 0.0]))
        _call(state, q, ik, retargeters)

    initial_T_first_translation = state["T_first"]["right"].translation.copy()

    # Simulate "clutched 1e9 s ago". With clutch_grace_s=0.0 below, the
    # elapsed check passes immediately on the first moving sample.
    state["clutch_start_t"]["right"] = 0.0

    operator_pose_at_resume = np.array([0.10, 0.0, 0.0], dtype=np.float32)
    q.put(_landmark(operator_pose_at_resume, ts_ns=99))
    _call(state, q, ik, retargeters, clutch_grace_s=0.0)

    # clutch exited
    assert state["clutch_start_t"]["right"] is None
    # T_first re-anchored to operator's current (FLU-converted) pose, so it
    # must have moved from where the initial anchor put it.
    assert not np.allclose(state["T_first"]["right"].translation, initial_T_first_translation)
    # T_home re-anchored to FK(q_prev, side) — the stub's fixed translation.
    np.testing.assert_allclose(state["T_home"]["right"].translation, StubIK.FK_TRANSLATION)


def test_clutch_grace_window_freezes_robot(stubs):
    """Inside the grace window, motion samples are ignored — no target update."""
    ik, retargeters = stubs
    state = _initial_state(still_window=5)
    q: queue.Queue = queue.Queue()

    for _ in range(5):
        q.put(_landmark([0.0, 0.0, 0.0]))
        _call(state, q, ik, retargeters)

    target_at_anchor = state["targets"]["right"].translation.copy()

    # Set clutch start to NOW so elapsed ≪ grace. The most reliable cross-
    # platform way is to use time.monotonic() at the moment of seeding.
    import time as _time

    state["clutch_start_t"]["right"] = _time.monotonic()

    q.put(_landmark([0.10, 0.0, 0.0], ts_ns=7))
    _call(state, q, ik, retargeters, clutch_grace_s=60.0)

    # Target unchanged, clutch still engaged, no fresh latency stamp this tick.
    np.testing.assert_allclose(state["targets"]["right"].translation, target_at_anchor)
    assert state["clutch_start_t"]["right"] is not None
    assert "right" not in state["new_target_acquire_ns"]
