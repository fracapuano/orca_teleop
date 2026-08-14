"""Hold-to-engage clutch.

Safety property: anything that goes wrong must read as RELEASED, never as
engaged. No tty, a dead reader, an exception -- all mean released.

Hold-to-engage, not toggle: a toggle survives the operator letting go.
"""

from __future__ import annotations

import contextlib
import os
import selectors
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Iterator

__all__ = ["KeyboardClutch", "ScriptedClutch", "raw_terminal"]

#: A tty sends key *repeats* while a key is held, never a key-up, so "held" is
#: inferred from the gap since the last repeat.
#:
#: The catch: an OS waits a "delay until repeat" before the FIRST repeat --
#: ~0.5 s on macOS by default -- but then repeats every ~30-50 ms. One fixed
#: timeout cannot serve both: short enough for a prompt release (good) means the
#: clutch drops during that initial delay (bad, and observed on hardware as a
#: flickering clutch that re-latched twice per press).
#:
#: So the timeout adapts: generous until repeats are flowing, then tightened to a
#: multiple of the observed interval. Release stays prompt without the flicker.
HOLD_TIMEOUT_S = 0.25
#: Grace period covering the OS delay-until-repeat, used until we have measured
#: the real interval. macOS ranges to ~1 s; this covers the common settings.
INITIAL_HOLD_TIMEOUT_S = 0.9
#: Once repeats are flowing, hold for this many observed intervals.
REPEAT_TIMEOUT_FACTOR = 3.0


@contextlib.contextmanager
def raw_terminal(stream: Any = None) -> Iterator[int | None]:
    """cbreak mode, restored unconditionally. Yields None when there is no tty.

    cbreak rather than raw: keeps ISIG so Ctrl-C still works.
    """
    stream = stream or sys.stdin
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        yield None
        return
    if not os.isatty(fd):
        yield None
        return
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        # TCSADRAIN so queued output (e.g. a traceback) is flushed, not dropped.
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class KeyboardClutch:
    """Engaged only while ``key`` is held. A USB foot pedal presenting as a
    keyboard works unchanged -- pass its keycode and a shorter timeout.

    ``on_key`` sees every character, so hotkeys can share this one reader.
    """

    def __init__(
        self,
        key: str = " ",
        *,
        hold_timeout_s: float = HOLD_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        stream: Any = None,
        on_key: Callable[[str], None] | None = None,
    ) -> None:
        if hold_timeout_s <= 0.0:
            raise ValueError(f"hold_timeout_s must be > 0, got {hold_timeout_s!r}")
        self.key = key.lower()
        self._timeout = float(hold_timeout_s)
        self._clock = clock
        self._stream = stream or sys.stdin
        self._on_key = on_key
        self._last_seen: float | None = None
        self._repeat_interval: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._restore: contextlib.ExitStack | None = None
        self.available = False

    @property
    def timeout_s(self) -> float:
        """Current release timeout: generous until repeats have been measured."""
        if self._repeat_interval is None:
            return max(self._timeout, INITIAL_HOLD_TIMEOUT_S)
        return max(self._timeout, REPEAT_TIMEOUT_FACTOR * self._repeat_interval)

    @property
    def engaged(self) -> bool:
        last = self._last_seen
        return last is not None and (self._clock() - last) <= self.timeout_s

    def start(self) -> None:
        if self._thread is not None:
            return
        self._restore = contextlib.ExitStack()
        fd = self._restore.enter_context(raw_terminal(self._stream))
        if fd is None:
            # No tty: stay released rather than pretending. Callers check
            # `available` and warn.
            self._restore.close()
            self._restore = None
            return
        self.available = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._read, args=(fd,), name="clutch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        if self._restore is not None:
            self._restore.close()
            self._restore = None
        self.available = False
        self._last_seen = None
        self._repeat_interval = None

    def __enter__(self) -> "KeyboardClutch":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _read(self, fd: int) -> None:
        selector = selectors.DefaultSelector()
        selector.register(fd, selectors.EVENT_READ)
        try:
            while not self._stop.is_set():
                # Timeout, not a blocking read: stop() must be able to wake us.
                if not selector.select(timeout=0.05):
                    continue
                try:
                    data = os.read(fd, 16)
                except OSError:
                    return
                if not data:
                    return
                for char in data.decode("utf-8", "ignore"):
                    if char.lower() == self.key:
                        now = self._clock()
                        if self._last_seen is not None:
                            gap = now - self._last_seen
                            # Only learn from gaps that look like auto-repeat.
                            # A long gap is a fresh press, not a repeat, and
                            # must not stretch the release timeout.
                            if gap <= INITIAL_HOLD_TIMEOUT_S:
                                self._repeat_interval = (
                                    gap if self._repeat_interval is None
                                    else min(self._repeat_interval, gap))
                            else:
                                self._repeat_interval = None  # new press, re-learn
                        self._last_seen = now
                    if self._on_key is not None:
                        self._on_key(char)
        finally:
            selector.close()


class ScriptedClutch:
    """Test double: set ``engaged`` directly."""

    def __init__(self, engaged: bool = False) -> None:
        self.engaged = bool(engaged)
        self.available = True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __enter__(self) -> "ScriptedClutch":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass
