import time


class RateTicker:
    """Sleep to maintain a fixed loop rate.

    Usage::

        RENDER_FPS = 30
        limiter = RateLimiter(dt=1 / RENDER_FPS)
        while running:  # loops at RENDER_FPS
            # ... do work here
            limiter.tick()
    """

    def __init__(self, dt: float) -> None:
        self._dt = dt
        self._next_tick = time.perf_counter()

    def tick(self) -> None:
        """Sleep until the next scheduled tick, then advance the schedule.

        If the deadline was missed the schedule resets to now, preventing
        a spiral of back-to-back zero-sleep ticks.
        """
        self._next_tick += self._dt
        now = time.perf_counter()
        sleep_s = self._next_tick - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            self._next_tick = now
