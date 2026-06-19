"""Rolling background-speech loudness baseline (dBFS).

Estimates the loudness of speech-like background audio (a TV/radio talking to
itself) so the orchestrator can later tell it apart from a user deliberately
addressing the device — the premise being that a user is closer/louder than
ambient media. Fed only by transcriptions that did *not* follow a wakeword.

Holds a time-windowed set of recent background samples and reports a percentile,
floored at a quiet-room default so a truly silent room still sits at the floor
and a whispered command (above the floor) clears it. A noisy room raises the
reported value while samples keep arriving; once they stop, samples age out of
the window and it falls back to the floor — the dynamic, self-timing behaviour
without a background thread.

Nothing is acted on yet: this only produces the number logged alongside each
transcription for tuning.
"""

import time
from collections import deque
from typing import Optional

import numpy as np

# dBFS a "super quiet room" sits at. Background speech is absent in true
# silence, so defaulting here means a whispered command (above this floor)
# still clears the baseline. Tunable once real-room numbers are in.
DEFAULT_BASELINE_DBFS = -60.0
# How long a background sample influences the baseline. Samples older than this
# age out, so a room that goes quiet re-baselines down toward the floor.
WINDOW_SECONDS = 60.0
# Percentile of the windowed samples reported as the baseline. The median is a
# robust "typical background level" — a single loud burst doesn't dominate.
PERCENTILE = 50.0


class BackgroundNoiseBaseline:
    """Time-windowed percentile of recent background-speech loudness (dBFS)."""

    def __init__(
        self,
        default_dbfs: float = DEFAULT_BASELINE_DBFS,
        window_seconds: float = WINDOW_SECONDS,
        percentile: float = PERCENTILE,
    ):
        self.default_dbfs = default_dbfs
        self.window_seconds = window_seconds
        self.percentile = percentile
        # (monotonic_t, dbfs), oldest first.
        self._samples: deque = deque()

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def add(self, dbfs: float, now: Optional[float] = None) -> None:
        """Record one background-speech loudness sample."""
        now = time.monotonic() if now is None else now
        self._samples.append((now, dbfs))
        self._evict(now)

    def value(self, now: Optional[float] = None) -> float:
        """Current baseline: windowed percentile, floored at the quiet default."""
        now = time.monotonic() if now is None else now
        self._evict(now)
        if not self._samples:
            return self.default_dbfs
        pct = float(np.percentile([d for _, d in self._samples], self.percentile))
        return max(self.default_dbfs, pct)
