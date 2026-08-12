"""Unit coverage for the dependency-light wakeword gate contract."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.wakeword import ScoreGate


def test_score_gate_requires_consecutive_smoothed_frames():
    gate = ScoreGate(threshold=0.5, smoothing_frames=3, cooldown_ms=0)
    assert not gate.feed("kitchen", 0.8).matched
    assert not gate.feed("kitchen", 0.8).matched
    result = gate.feed("kitchen", 0.8)
    assert result.matched
    assert result.score == pytest.approx(0.8)


def test_score_gate_keeps_satellites_isolated():
    gate = ScoreGate(threshold=0.5, smoothing_frames=2, cooldown_ms=0)
    assert not gate.feed("kitchen", 0.9).matched
    assert not gate.feed("office", 0.9).matched
    assert gate.feed("kitchen", 0.9).matched
    assert gate.feed("office", 0.9).matched


def test_score_gate_reset_discards_debounce_state():
    gate = ScoreGate(threshold=0.5, smoothing_frames=2, cooldown_ms=0)
    gate.feed("kitchen", 0.9)
    gate.reset("kitchen")
    assert not gate.feed("kitchen", 0.9).matched
