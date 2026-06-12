"""Tests for the TTS cancellation primitive."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tts_session import TtsSession, parse_barge_time  # noqa: E402


class TestTtsSession:
    def test_starts_idle_and_not_cancelled(self):
        session = TtsSession()
        assert session.active is False
        assert session.cancelled is False

    def test_stop_sets_cancelled(self):
        session = TtsSession()
        session.stop()
        assert session.cancelled is True

    def test_stop_is_idempotent(self):
        session = TtsSession()
        session.stop()
        session.stop()
        assert session.cancelled is True

    def test_stop_from_another_thread_is_visible(self):
        """speak_stream's consumer loop polls .cancelled — make sure cross-thread
        visibility actually works (it does, threading.Event is the whole point)."""
        session = TtsSession()
        observed = []

        def watcher():
            # Spin briefly then signal — mimic the transcriber thread firing
            # stop() while another thread is inside speak_stream.
            time.sleep(0.05)
            session.stop()

        def consumer():
            # Mimic speak_stream's polling loop.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if session.cancelled:
                    observed.append("cancelled")
                    return
                time.sleep(0.01)
            observed.append("timeout")

        t1 = threading.Thread(target=watcher)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert observed == ["cancelled"]

    def test_clear_via_stop_event_resets_state(self):
        """speak_stream clears the event at the top of each call so a prior
        stop() doesn't immediately abort the next playback."""
        session = TtsSession()
        session.stop()
        assert session.cancelled is True
        session.stop_event.clear()
        assert session.cancelled is False


class TestParseBargeTime:
    def test_zero_seconds(self):
        assert parse_barge_time("0s") == 0.0

    def test_n_seconds(self):
        assert parse_barge_time("5s") == 5.0
        assert parse_barge_time("2.5s") == 2.5

    def test_always_is_infinite(self):
        assert parse_barge_time("always") == float("inf")

    def test_case_and_whitespace_tolerant(self):
        assert parse_barge_time(" Always ") == float("inf")
        assert parse_barge_time("5S") == 5.0

    def test_none_defaults_to_zero(self):
        assert parse_barge_time(None) == 0.0

    def test_invalid_defaults_to_zero(self):
        assert parse_barge_time("five seconds") == 0.0
        assert parse_barge_time("") == 0.0

    def test_negative_clamped_to_zero(self):
        assert parse_barge_time("-3s") == 0.0

    def test_numeric_without_suffix(self):
        assert parse_barge_time("4") == 4.0
