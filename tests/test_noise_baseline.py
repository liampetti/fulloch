"""Tests for the rolling background-speech loudness baseline.

The baseline reports a windowed percentile of recent background (non-wakeword)
loudness samples, floored at a quiet-room default so a silent room sits at the
floor and a whisper above it still clears. Samples age out of the window so a
room that goes quiet re-baselines down. Nothing acts on the value yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.noise_baseline import BackgroundNoiseBaseline  # noqa: E402


def test_empty_returns_quiet_floor():
    b = BackgroundNoiseBaseline(default_dbfs=-60.0)
    assert b.value(now=0.0) == -60.0


def test_floor_holds_when_samples_are_quieter_than_default():
    b = BackgroundNoiseBaseline(default_dbfs=-60.0)
    b.add(-72.0, now=0.0)
    # A whisper-quiet background can't drag the baseline below the floor, so a
    # genuine whisper (above the floor) still reads as louder.
    assert b.value(now=1.0) == -60.0


def test_reports_median_of_loud_background():
    b = BackgroundNoiseBaseline(default_dbfs=-60.0, percentile=50.0)
    for db in (-30.0, -20.0, -10.0):
        b.add(db, now=1.0)
    assert b.value(now=1.0) == -20.0  # median


def test_samples_age_out_of_window():
    b = BackgroundNoiseBaseline(default_dbfs=-60.0, window_seconds=60.0)
    b.add(-15.0, now=0.0)
    assert b.value(now=10.0) == -15.0
    # 61s later the sample has aged out — back to the quiet floor.
    assert b.value(now=61.0) == -60.0
