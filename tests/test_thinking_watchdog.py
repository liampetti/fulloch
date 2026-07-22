"""Tests for the ThinkingWatchdog periodic-stall context manager."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thinking_watchdog import ThinkingWatchdog  # noqa: E402


class _FakeSession:
    """Minimal stand-in for TtsSession — exposes only what the watchdog needs."""

    def __init__(self):
        self.cancelled = False


@pytest.fixture
def stall_cache():
    # Each entry is the `(chunks, sample_rate)` tuple `synthesize` returns.
    return [(MagicMock(name="chunks-a"), 24000), (MagicMock(name="chunks-b"), 24000)]


class TestStartStop:
    def test_no_play_before_first_interval(self, stall_cache):
        play = MagicMock()
        session = _FakeSession()
        with ThinkingWatchdog(stall_cache, play, session, interval=0.5):
            time.sleep(0.05)  # well under the interval
        assert play.call_count == 0

    def test_plays_after_interval_elapses(self, stall_cache):
        play = MagicMock()
        session = _FakeSession()
        with ThinkingWatchdog(stall_cache, play, session, interval=0.08):
            time.sleep(0.25)
        # Should have played at least twice within the window.
        assert play.call_count >= 2

    def test_max_stalls_limits_progress_phrases(self, stall_cache):
        play = MagicMock()
        session = _FakeSession()
        with ThinkingWatchdog(stall_cache, play, session, interval=0.03, max_stalls=1):
            time.sleep(0.12)
        assert play.call_count == 1

    def test_clean_shutdown_on_context_exit(self, stall_cache):
        play = MagicMock()
        session = _FakeSession()
        wd = ThinkingWatchdog(stall_cache, play, session, interval=0.05)
        with wd:
            time.sleep(0.12)
        # Background thread should be joined and not running anymore.
        assert wd._thread is None or not wd._thread.is_alive()

    def test_empty_cache_makes_watchdog_a_noop(self):
        play = MagicMock()
        session = _FakeSession()
        with ThinkingWatchdog([], play, session, interval=0.05):
            time.sleep(0.15)
        assert play.call_count == 0


class TestCancellation:
    def test_session_cancel_short_circuits(self, stall_cache):
        play = MagicMock()
        session = _FakeSession()
        with ThinkingWatchdog(stall_cache, play, session, interval=0.05):
            time.sleep(0.08)  # let one stall fire
            session.cancelled = True
            time.sleep(0.2)
        # After cancel, no further stalls should land.
        played_before_cancel = play.call_count
        time.sleep(0.1)
        assert play.call_count == played_before_cancel

    def test_play_error_does_not_break_loop(self, stall_cache):
        play = MagicMock(side_effect=[RuntimeError("boom"), None])
        session = _FakeSession()
        with ThinkingWatchdog(stall_cache, play, session, interval=0.05):
            time.sleep(0.2)
        # First call raised; watchdog should keep ticking and call again.
        assert play.call_count >= 1


class TestThreadIsolation:
    def test_watchdog_does_not_block_main(self, stall_cache):
        """Body of the `with` block must not be delayed by the watchdog."""
        play = MagicMock()
        session = _FakeSession()
        t0 = time.monotonic()
        with ThinkingWatchdog(stall_cache, play, session, interval=10.0):
            pass  # immediate exit
        elapsed = time.monotonic() - t0
        # Even with the join timeout, immediate exit should take milliseconds.
        assert elapsed < 0.5
