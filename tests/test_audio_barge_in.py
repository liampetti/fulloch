"""Tests for the configurable barge-in sensitivity (`barge_in_threshold_dbfs`).

The barge-in floor is the mic-silence threshold used only while the assistant
speaks: an interrupting voice must exceed it to be captured. It's kept in dBFS
(the unit transcription volume is logged in) and converted once to the linear
RMS the recorder compares against. A speakerphone that ducks its mic during
playback makes interrupts arrive quiet, so the floor needs lowering.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio import (  # noqa: E402
    BARGE_IN_THRESHOLD_DBFS,
    AudioCapture,
    dbfs_to_rms,
    rms_to_dbfs,
)


def test_dbfs_rms_round_trip():
    for dbfs in (-60.0, -48.0, -40.0, -20.0):
        assert math.isclose(rms_to_dbfs(dbfs_to_rms(dbfs)), dbfs, abs_tol=1e-9)


def test_default_is_minus_48_dbfs():
    assert BARGE_IN_THRESHOLD_DBFS == -48.0


def test_no_override_uses_default_dbfs():
    ac = AudioCapture(use_vad=False)
    assert ac.barge_in_threshold_dbfs == BARGE_IN_THRESHOLD_DBFS
    assert math.isclose(ac._barge_in_rms, dbfs_to_rms(BARGE_IN_THRESHOLD_DBFS), abs_tol=1e-9)


def test_dbfs_override_lowers_floor():
    # A lower dBFS floor = more sensitive = smaller linear RMS threshold.
    ac = AudioCapture(use_vad=False, barge_in_threshold_dbfs=-54.0)
    assert ac.barge_in_threshold_dbfs == -54.0
    assert math.isclose(ac._barge_in_rms, dbfs_to_rms(-54.0), abs_tol=1e-9)
    assert ac._barge_in_rms < dbfs_to_rms(BARGE_IN_THRESHOLD_DBFS)
