"""Shared fixtures for orca_teleop tests."""

from __future__ import annotations

import threading

import pytest
from orca_core.test_mock import MockOrcaHand


@pytest.fixture
def patch_mock_hand(monkeypatch):
    """Make `robot_worker` build a `MockOrcaHand` instead of a real one.

    Returns the list of instances created during the test (usually one).
    """
    created: list[MockOrcaHand] = []

    def factory(model_path=None):
        hand = MockOrcaHand()
        created.append(hand)
        return hand

    monkeypatch.setattr("orca_teleop.pipeline.OrcaHand", factory)
    return created


@pytest.fixture(autouse=True)
def _no_thread_leaks():
    """Fail any test that leaks a non-daemon thread it spawned."""
    before = {t.ident for t in threading.enumerate()}
    yield
    leaked = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
    # Give stragglers a brief moment to wind down before declaring a leak.
    for t in leaked:
        t.join(timeout=1.0)
    still_alive = [t.name for t in leaked if t.is_alive()]
    assert not still_alive, f"leaked threads: {still_alive}"
