"""Precise waits for the real-time loops (sim_server ticks, the video fake camera, the render clock).

macOS adds up to ~50 % timer leeway to a sleep (measured on the M2 Max: an 18 ms time.sleep returns 6-9 ms late,
a 2 ms one 1 ms late), which made 50 Hz ticks start ~8 ms late. Sleeping in shrinking chunks (60 % of what is left,
so even full leeway cannot pass the deadline) and spinning the last half millisecond lands within ~0.01 ms on both
macOS and Linux for ~5 % of one core at 50 Hz.
"""
from __future__ import annotations

import time


def sleep_until(deadline: float, clock=time.perf_counter, spin: float = 0.0005, stop=None) -> bool:
    """Wait until clock() >= deadline. stop: optional threading.Event; returns False if it was set meanwhile."""
    while True:
        rem = deadline - clock()
        if rem <= spin:
            break
        d = (rem - spin) * 0.6
        if stop is not None:
            if stop.wait(d):
                return False
        else:
            time.sleep(d)
    while clock() < deadline:
        pass
    return stop is None or not stop.is_set()
