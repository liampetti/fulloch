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
    extract_deep_think,
    extract_summarize_thinking,
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

    def test_play_with_leading_words(self):
        # extract_after_play uses re.search, so leading words are tolerated.
        result = extract_after_play("please play the Beatles")
        assert result == "the Beatles"

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
        assert result == {"actions": [{"intent": "play_song", "args": ["some jazz"]}]}

    def test_catch_stop(self):
        result = catchAll("stop")
        assert result == {"actions": [{"intent": "pause", "args": []}]}

    def test_catch_time(self):
        result = catchAll("what time is it")
        assert result == {"actions": [{"intent": "get_time", "args": []}]}

    def test_catch_skip(self):
        result = catchAll("skip")
        assert result == {"actions": [{"intent": "skip", "args": []}]}

    def test_catch_resume(self):
        result = catchAll("resume")
        assert result == {"actions": [{"intent": "resume", "args": []}]}

    def test_catch_timer(self):
        result = catchAll("start timer ten minutes")
        assert result == {"actions": [{"intent": "start_countdown", "args": ["ten minutes"]}]}

    def test_catch_list_timers(self):
        result = catchAll("get timers")
        assert result == {"actions": [{"intent": "list_timers", "args": []}]}

    def test_no_match_returns_original(self):
        original = "tell me a joke"
        result = catchAll(original)
        assert result == original

    def test_complex_unmatched_query(self):
        original = "what's the weather forecast for tomorrow"
        result = catchAll(original)
        assert result == original

    def test_catch_deep_think(self):
        result = catchAll("think about whether I should switch banks")
        assert result == {"actions": [{"intent": "deep_think", "args": ["whether I should switch banks"]}]}


class TestExtractDeepThink:
    """Regex-fast-path for thinking-mode queries.

    Conservative: must be an explicit ask for thought, not a casual 'I think'.
    """

    @pytest.mark.parametrize("utterance,topic", [
        ("think about whether to switch banks", "whether to switch banks"),
        ("Think About my career options.", "my career options"),
        ("please think over the new schedule", "the new schedule"),
        ("consider carefully about my retirement plan", "my retirement plan"),
        ("reflect on the meeting", "the meeting"),
        ("ponder over the proposal", "the proposal"),
        ("let me think about my next steps", "my next steps"),
        ("let's think through this problem", "this problem"),
        ("what do you think about electric cars", "electric cars"),
        ("can you think about my workout routine", "my workout routine"),
    ])
    def test_matches(self, utterance, topic):
        assert extract_deep_think(utterance) == topic

    @pytest.mark.parametrize("utterance", [
        "i think it's raining",
        "i don't think so",
        "think fast",
        "consider it done",
        "think",
        "let me reflect for a second",  # no preposition + topic
        "what's the weather",
    ])
    def test_non_matches(self, utterance):
        assert extract_deep_think(utterance) is None


class TestExtractSummarizeThinking:
    """Regex catch for 'what do you have so far'-style follow-ups during
    a thinking turn that just got interrupted."""

    @pytest.mark.parametrize("utterance", [
        "summarise",
        "summarize your thoughts",
        "summarise what you've got",
        "summarise what you have so far",
        "what have you got so far",
        "what do you have so far",
        "what are you thinking so far",
        "what have you been thinking",
        "so what have you got so far",
        "give me your thoughts",
        "tell me what you have got",
        "tell me your thoughts",
        "stop thinking",
        "give up thinking",
        "that's enough",
    ])
    def test_matches(self, utterance):
        assert extract_summarize_thinking(utterance) is True

    @pytest.mark.parametrize("utterance", [
        "we discussed this so far",  # 'so far' alone shouldn't match
        "tell me a joke",
        "think about the weather",   # belongs to deep_think
        "what time is it",
        "play some music",
        "stop",                       # belongs to existing stop intent
    ])
    def test_non_matches(self, utterance):
        assert extract_summarize_thinking(utterance) is None

    def test_catch_all_routes_to_summarize_thinking(self):
        result = catchAll("summarise your thoughts")
        assert result == {"actions": [{"intent": "summarize_thinking", "args": []}]}

    def test_catch_all_summarize_takes_priority_over_deep_think(self):
        # "summarise what you've been thinking" mentions 'thinking' but
        # is asking for a summary, not a fresh deep_think.
        result = catchAll("summarise what you've been thinking")
        assert result == {"actions": [{"intent": "summarize_thinking", "args": []}]}
