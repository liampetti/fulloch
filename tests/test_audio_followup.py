"""Tests for the follow-up window gating in `AudioCapture`.

A reply to the assistant during the wakeword-free follow-up window is often
one or two words and shorter than `MIN_UTTERANCE_MS`; the recorder must accept
it at the shorter `follow_up_min_utterance_samples` floor while the window is
open, and revert to the full floor once it closes/expires. These tests cover
the arm / clear / auto-expire state machine (`recorder_thread` itself opens a
real audio stream, so the enqueue decision is exercised at the unit level via
these helpers — construction does not touch sounddevice).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio import AudioCapture


def make_capture():
    # use_vad=False keeps the Silero model out of the test; __init__ does not
    # open the InputStream (that happens in recorder_thread).
    return AudioCapture(use_vad=False)


def test_follow_up_closed_by_default():
    ac = make_capture()
    assert ac._follow_up_open() is False
    assert ac.follow_up_active.is_set() is False


def test_arm_opens_window():
    ac = make_capture()
    ac.arm_follow_up(5.0)
    assert ac.follow_up_active.is_set() is True
    assert ac._follow_up_open() is True


def test_clear_closes_window():
    ac = make_capture()
    ac.arm_follow_up(5.0)
    ac.clear_follow_up()
    assert ac.follow_up_active.is_set() is False
    assert ac._follow_up_open() is False


def test_window_auto_expires_and_clears_event():
    ac = make_capture()
    ac.arm_follow_up(5.0)
    # Force the deadline into the past — the window must report closed and
    # clear its own event so the idle noise guard re-tightens.
    ac._follow_up_until = time.monotonic() - 1.0
    assert ac._follow_up_open() is False
    assert ac.follow_up_active.is_set() is False


def test_arm_includes_capture_slack():
    ac = make_capture()
    before = time.monotonic()
    ac.arm_follow_up(5.0)
    # Deadline is window + slack (silence_duration + 1.5s) past now.
    assert ac._follow_up_until > before + 5.0 + ac._follow_up_slack_s - 0.1


def test_follow_up_min_below_full_min():
    ac = make_capture()
    assert ac.follow_up_min_utterance_samples < ac.min_utterance_samples
