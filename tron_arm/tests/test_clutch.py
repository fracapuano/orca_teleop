"""Clutch. The safety property: failure must read as RELEASED."""

from __future__ import annotations

import io
import os
import pty
import termios
import threading
import time

import pytest

from tron_arm.clutch import HOLD_TIMEOUT_S, KeyboardClutch, ScriptedClutch, raw_terminal


def stable(attrs: list) -> list:
    """termios attrs minus the kernel's transient PENDIN bit."""
    out = list(attrs)
    out[3] &= ~termios.PENDIN
    return out


class TestRawTerminal:
    def test_none_without_a_tty(self):
        with raw_terminal(io.StringIO()) as fd:
            assert fd is None

    def test_restores_and_keeps_isig(self):
        primary, secondary = pty.openpty()
        try:
            stream = os.fdopen(secondary, "rb", buffering=0)
            before = termios.tcgetattr(secondary)
            with raw_terminal(stream) as fd:
                assert fd == secondary
                during = termios.tcgetattr(secondary)
                assert not during[3] & termios.ECHO
                assert not during[3] & termios.ICANON
                assert during[3] & termios.ISIG, "cbreak must keep Ctrl-C working"
            assert stable(termios.tcgetattr(secondary)) == stable(before)
        finally:
            os.close(primary)

    def test_restores_when_the_body_raises(self):
        primary, secondary = pty.openpty()
        try:
            stream = os.fdopen(secondary, "rb", buffering=0)
            before = termios.tcgetattr(secondary)
            with pytest.raises(RuntimeError):
                with raw_terminal(stream):
                    raise RuntimeError("boom")
            assert stable(termios.tcgetattr(secondary)) == stable(before)
        finally:
            os.close(primary)


class TestKeyboardClutch:
    def test_no_tty_means_unavailable_and_released(self):
        clutch = KeyboardClutch(stream=io.StringIO())
        clutch.start()
        try:
            assert not clutch.available and not clutch.engaged
        finally:
            clutch.stop()

    def test_engages_on_press_and_releases_after_the_timeout(self):
        """A single press holds for the grace period covering the OS's
        delay-until-repeat, then releases."""
        primary, secondary = pty.openpty()
        try:
            now = [100.0]
            clutch = KeyboardClutch(" ", hold_timeout_s=0.25, clock=lambda: now[0],
                                    stream=os.fdopen(secondary, "rb", buffering=0))
            clutch.start()
            assert clutch.available
            os.write(primary, b" ")
            deadline = time.monotonic() + 2.0
            while not clutch.engaged and time.monotonic() < deadline:
                time.sleep(0.005)
            assert clutch.engaged
            now[0] += 0.4
            assert clutch.engaged, "dropped during the OS delay-until-repeat"
            now[0] += 0.8
            assert not clutch.engaged, "past the grace period -> released"
            clutch.stop()
        finally:
            os.close(primary)

    def test_other_keys_do_not_engage(self):
        primary, secondary = pty.openpty()
        try:
            clutch = KeyboardClutch(" ", stream=os.fdopen(secondary, "rb", buffering=0))
            clutch.start()
            os.write(primary, b"xyz")
            time.sleep(0.3)
            assert not clutch.engaged
            clutch.stop()
        finally:
            os.close(primary)

    def test_on_key_sees_everything_so_hotkeys_can_share_the_reader(self):
        primary, secondary = pty.openpty()
        seen: list[str] = []
        try:
            clutch = KeyboardClutch(" ", stream=os.fdopen(secondary, "rb", buffering=0),
                                    on_key=seen.append)
            clutch.start()
            os.write(primary, b"o q ")
            deadline = time.monotonic() + 2.0
            while len(seen) < 4 and time.monotonic() < deadline:
                time.sleep(0.005)
            clutch.stop()
        finally:
            os.close(primary)
        assert {"o", "q", " "} <= set(seen)

    def test_a_pedal_is_just_a_different_key_and_timeout(self):
        primary, secondary = pty.openpty()
        try:
            clutch = KeyboardClutch("p", hold_timeout_s=0.15,
                                    stream=os.fdopen(secondary, "rb", buffering=0))
            clutch.start()
            os.write(primary, b"p")
            deadline = time.monotonic() + 2.0
            while not clutch.engaged and time.monotonic() < deadline:
                time.sleep(0.005)
            assert clutch.engaged
            clutch.stop()
        finally:
            os.close(primary)

    def test_stop_is_idempotent_and_joins(self):
        primary, secondary = pty.openpty()
        try:
            clutch = KeyboardClutch(stream=os.fdopen(secondary, "rb", buffering=0))
            clutch.start()
            clutch.stop()
            clutch.stop()
            assert not clutch.available
            assert [t.name for t in threading.enumerate()].count("clutch") == 0
        finally:
            os.close(primary)

    def test_rejects_a_non_positive_timeout(self):
        with pytest.raises(ValueError, match="hold_timeout_s"):
            KeyboardClutch(" ", hold_timeout_s=0.0)

    def test_default_timeout_exceeds_a_typical_repeat_interval(self):
        assert HOLD_TIMEOUT_S >= 0.1


class TestScriptedClutch:
    def test_starts_released_and_is_settable(self):
        clutch = ScriptedClutch()
        assert not clutch.engaged
        clutch.engaged = True
        assert clutch.engaged


class TestKeyRepeatFlicker:
    """Found on hardware: macOS waits ~0.5 s before the first auto-repeat, which
    exceeded the 0.25 s hold timeout, so every press dropped the clutch once and
    re-latched. The session log showed latches in pairs 0.5 s apart."""

    def _clutch(self, now):
        return KeyboardClutch(" ", hold_timeout_s=0.25, clock=lambda: now[0],
                              stream=io.StringIO())

    def test_survives_the_os_delay_until_repeat(self):
        now = [100.0]
        clutch = self._clutch(now)
        clutch._last_seen = now[0]                    # first keydown
        for elapsed in (0.1, 0.3, 0.45):              # inside macOS's ~0.5 s delay
            now[0] = 100.0 + elapsed
            assert clutch.engaged, f"clutch dropped {elapsed}s after the press"

    def test_tightens_once_repeats_are_flowing(self):
        now = [100.0]
        clutch = self._clutch(now)
        clutch._last_seen = now[0]
        for i in range(1, 6):                          # 40 ms auto-repeats
            now[0] = 100.0 + 0.5 + i * 0.04
            clutch._repeat_interval = (0.04 if clutch._repeat_interval is None
                                       else min(clutch._repeat_interval, 0.04))
            clutch._last_seen = now[0]
        # Release is now detected in ~3 repeat intervals, not ~1 s.
        assert clutch.timeout_s <= 0.25
        now[0] += 0.30
        assert not clutch.engaged, "release was not detected promptly"

    def test_release_is_never_slower_than_the_initial_grace(self):
        now = [100.0]
        clutch = self._clutch(now)
        clutch._last_seen = now[0]
        now[0] += 1.5
        assert not clutch.engaged
