import time
from collections import deque


class FpsCounter:
    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)
    def tick(self):
        self._times.append(time.time())
    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / max(dt, 1e-9)
