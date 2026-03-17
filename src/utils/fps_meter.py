"""Rolling FPS counter для demo_live.py."""

from __future__ import annotations

import time
from collections import deque


class FPSMeter:
    """Измеряет FPS как rolling mean по последним N кадрам.

    Использование:
        meter = FPSMeter(window=30)
        while playing:
            t0 = time.perf_counter()
            process_frame()
            meter.update(time.perf_counter() - t0)
            print(f"FPS: {meter.fps:.1f}")
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: deque[float] = deque(maxlen=window)

    def update(self, dt_seconds: float) -> None:
        """Добавляет время обработки одного кадра."""
        if dt_seconds > 0:
            self._times.append(dt_seconds)

    def tick(self) -> None:
        """Автоматически замеряет время с предыдущего tick()."""
        now = time.perf_counter()
        if hasattr(self, "_last_tick"):
            self.update(now - self._last_tick)
        self._last_tick = now

    @property
    def fps(self) -> float:
        """Текущий FPS (rolling mean)."""
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    def reset(self) -> None:
        self._times.clear()
        if hasattr(self, "_last_tick"):
            del self._last_tick
