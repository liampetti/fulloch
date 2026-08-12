"""Wakeword gate contracts shared by optional streaming backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WakewordResult:
    matched: bool
    score: float
    detected_at: float = 0.0


class WakewordBackend(Protocol):
    def feed_pcm(self, satellite_id: str, pcm) -> WakewordResult: ...
    def reset(self, satellite_id: str) -> None: ...
    def self_test(self) -> None: ...


class ScoreGate:
    """Per-stream smoothing, debounce, and cooldown over raw KWS scores."""

    def __init__(self, threshold: float, smoothing_frames: int, cooldown_ms: int):
        self.threshold = float(threshold)
        self.smoothing_frames = max(1, int(smoothing_frames))
        self.cooldown_s = max(0, int(cooldown_ms)) / 1000.0
        self._scores: dict[str, list[float]] = {}
        self._cooldown_until: dict[str, float] = {}

    def feed(self, satellite_id: str, score: float) -> WakewordResult:
        now = time.monotonic()
        scores = self._scores.setdefault(satellite_id, [])
        scores.append(float(score))
        del scores[:-self.smoothing_frames]
        smoothed = sum(scores) / len(scores)
        matched = (
            len(scores) == self.smoothing_frames
            and smoothed >= self.threshold
            and now >= self._cooldown_until.get(satellite_id, 0.0)
        )
        if matched:
            self._cooldown_until[satellite_id] = now + self.cooldown_s
            scores.clear()
        return WakewordResult(matched, smoothed, now if matched else 0.0)

    def reset(self, satellite_id: str) -> None:
        self._scores.pop(satellite_id, None)
        self._cooldown_until.pop(satellite_id, None)
