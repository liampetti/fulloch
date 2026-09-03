"""Tests for the follow-up window gating in `AudioCapture`.

A reply to the assistant during the wakeword-free follow-up window is often
one or two words and shorter than `MIN_UTTERANCE_MS`; the recorder must accept
it at the shorter `follow_up_min_utterance_samples` floor while the window is
open, and revert to the full floor once it closes/expires. The window itself
lives on `SatelliteSession.follow_up_deadline` (per-satellite, since A's
follow-up window must not affect B's) — these tests cover the arm / clear /
auto-expire state machine at the unit level via `AudioCapture`'s methods,
which take the session explicitly (`recorder_thread` itself opens a real
audio stream, so this is exercised below construction).
"""

import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio import AudioCapture  # noqa: E402
from core.satellite import SatelliteSession  # noqa: E402


def make_capture():
    # use_vad=False keeps the Silero model out of the test; __init__ does not
    # open the InputStream (that happens in recorder_thread).
    return AudioCapture(use_vad=False)


def make_session():
    return SatelliteSession(id="sat-a", chunk_q=queue.Queue())


def test_follow_up_closed_by_default():
    ac = make_capture()
    sess = make_session()
    assert ac._follow_up_open(sess) is False
    assert sess.follow_up_deadline == 0.0


def test_arm_opens_window():
    ac = make_capture()
    sess = make_session()
    ac.arm_follow_up(sess, 5.0)
    assert sess.follow_up_deadline > 0.0
    assert ac._follow_up_open(sess) is True


def test_clear_closes_window():
    ac = make_capture()
    sess = make_session()
    ac.arm_follow_up(sess, 5.0)
    ac.clear_follow_up(sess)
    assert sess.follow_up_deadline == 0.0
    assert ac._follow_up_open(sess) is False


def test_window_auto_expires_and_clears_event():
    ac = make_capture()
    sess = make_session()
    ac.arm_follow_up(sess, 5.0)
    # Force the deadline into the past — the window must report closed.
    sess.follow_up_deadline = time.monotonic() - 1.0
    assert ac._follow_up_open(sess) is False


def test_arm_includes_capture_slack():
    ac = make_capture()
    sess = make_session()
    before = time.monotonic()
    ac.arm_follow_up(sess, 5.0)
    # Deadline is window + slack (silence_duration + 1.5s) past now.
    assert sess.follow_up_deadline > before + 5.0 + ac._follow_up_slack_s - 0.1


def test_arm_can_start_after_scheduled_playback():
    ac = make_capture()
    sess = make_session()
    playback_end = time.monotonic() + 10.0

    ac.arm_follow_up(sess, 5.0, start_at=playback_end)

    assert sess.follow_up_deadline > playback_end + 5.0 + ac._follow_up_slack_s - 0.1
    assert ac._follow_up_open(sess) is True


def test_follow_up_is_per_satellite():
    ac = make_capture()
    a_sess = make_session()
    b_sess = SatelliteSession(id="sat-b", chunk_q=queue.Queue())
    ac.arm_follow_up(a_sess, 5.0)
    assert ac._follow_up_open(a_sess) is True
    assert ac._follow_up_open(b_sess) is False


def test_follow_up_min_below_full_min():
    ac = make_capture()
    assert ac.follow_up_min_utterance_samples < ac.min_utterance_samples


def test_early_wake_probe_suppresses_duplicate_soft_probe():
    class Endpointer:
        speech_started = True
        soft_endpointed = True
        endpointed = False
        speech_onset = 1.0
        voiced_rms = 0.1

        def process(self, _chunk):
            pass

        def reset(self):
            pass

    ac = make_capture()
    ac._use_vad_enabled = True
    ac._build_endpointer = lambda: Endpointer()
    ac.early_wake_probe_seconds = 0.0
    ac._enqueue = MagicMock()
    session = make_session()
    session.chunk_q.put(np.zeros(320, dtype=np.float32))
    session.chunk_q.put(None)

    recorder = threading.Thread(target=ac.satellite_recorder_thread, args=(session,))
    recorder.start()
    recorder.join(timeout=1)

    assert not recorder.is_alive()
    assert ac._enqueue.call_count == 1
    assert ac._enqueue.call_args.kwargs == {}
    assert ac._enqueue.call_args.args[-1] is True
