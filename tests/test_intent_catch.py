"""
Tests for the regex intent catching module.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.intent_catch import (
    catchAll,
    extract_after_play,
    extract_stop,
    extract_skip,
    extract_resume,
    extract_timer,
    has_time_query,
    list_timers,
)


class TestExtractAfterPlay:
    """Tests for play command extraction."""

    def test_simple_play(self):
        result = extract_after_play("play some rock music")
        assert result == "some rock music"

    def test_play_artist(self):
        result = extract_after_play("play Taylor Swift")
        assert result == "Taylor Swift"

    def test_play_with_extra_words(self):
        result = extract_after_play("please play the Beatles")
        assert result is None  # Doesn't match pattern starting with play

    def test_no_play_command(self):
        result = extract_after_play("what's the weather")
        assert result is None

    def test_play_case_insensitive(self):
        result = extract_after_play("PLAY jazz music")
        assert result == "jazz music"


class TestExtractStop:
    """Tests for stop/pause command extraction."""

    def test_stop(self):
        assert extract_stop("stop") is True

    def test_pause(self):
        assert extract_stop("pause") is True

    def test_halt(self):
        assert extract_stop("halt") is True

    def test_stop_with_whitespace(self):
        assert extract_stop("  stop  ") is True

    def test_stop_case_insensitive(self):
        assert extract_stop("STOP") is True

    def test_not_stop(self):
        assert extract_stop("play music") is None


class TestExtractSkip:
    """Tests for skip command extraction."""

    def test_skip(self):
        assert extract_skip("skip") is True

    def test_skip_with_whitespace(self):
        assert extract_skip("  skip") is True

    def test_not_skip(self):
        assert extract_skip("play next") is None


class TestExtractResume:
    """Tests for resume command extraction."""

    def test_resume(self):
        assert extract_resume("resume") is True

    def test_resume_with_whitespace(self):
        assert extract_resume("  resume") is True

    def test_not_resume(self):
        assert extract_resume("continue playing") is None


class TestHasTimeQuery:
    """Tests for time query detection."""

    def test_what_time_is_it(self):
        assert has_time_query("what time is it") is True

    def test_whats_the_time(self):
        assert has_time_query("what's the time") is True

    def test_whats_the_time_no_apostrophe(self):
        assert has_time_query("whats the time") is True

    def test_not_time_query(self):
        assert has_time_query("set a timer") is None


class TestExtractTimer:
    """Tests for timer duration extraction."""

    def test_start_timer_minutes(self):
        result = extract_timer("start timer ten minutes")
        assert result == "ten minutes"

    def test_set_timer_for(self):
        result = extract_timer("set timer for 2 hours")
        assert result == "2 hours"

    def test_start_a_timer(self):
        result = extract_timer("start a timer thirty seconds please")
        assert result == "thirty seconds"

    def test_not_timer(self):
        result = extract_timer("what time is it")
        assert result is None


class TestListTimers:
    """Tests for list timers command."""

    def test_get_timers(self):
        assert list_timers("get timers") is True

    def test_get_timer(self):
        assert list_timers("get timer") is True

    def test_not_list_timers(self):
        assert list_timers("start timer") is None


class TestCatchAll:
    """Tests for the main catchAll function."""

    def test_catch_play(self):
        result = catchAll("play some jazz")
        assert result == {"intent": "play_song", "args": ["some jazz"]}

    def test_catch_stop(self):
        result = catchAll("stop")
        assert result == {"intent": "pause", "args": []}

    def test_catch_time(self):
        result = catchAll("what time is it")
        assert result == {"intent": "get_time", "args": []}

    def test_catch_skip(self):
        result = catchAll("skip")
        assert result == {"intent": "skip", "args": []}

    def test_catch_resume(self):
        result = catchAll("resume")
        assert result == {"intent": "resume", "args": []}

    def test_catch_timer(self):
        result = catchAll("start timer ten minutes")
        assert result == {"intent": "start_countdown", "args": ["ten minutes"]}

    def test_catch_list_timers(self):
        result = catchAll("get timers")
        assert result == {"intent": "list_timers", "args": []}

    def test_no_match_returns_original(self):
        original = "tell me a joke"
        result = catchAll(original)
        assert result == original

    def test_complex_unmatched_query(self):
        original = "what's the weather forecast for tomorrow"
        result = catchAll(original)
        assert result == original
