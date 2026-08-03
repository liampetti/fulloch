"""
Tests for the regex intent catching module.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.intent_catch import (
    catchAll,
    extract_after_play,
    extract_area_light_state,
    extract_color,
    extract_cover,
    extract_deep_think,
    extract_dim_brighten,
    extract_light_brightness,
    extract_lock,
    extract_note_delete,
    extract_resume,
    extract_satellite_message,
    extract_skip,
    extract_stop,
    extract_summarize_thinking,
    extract_timer,
    extract_toggle,
    extract_turn_onoff,
    extract_volume_down,
    extract_volume_down_target,
    extract_volume_up,
    extract_volume_up_target,
    has_default_weather_forecast,
    has_time_query,
    is_contextual_web_search_request,
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
        # A polite prefix before the leading verb is still tolerated.
        result = extract_after_play("please play the Beatles")
        assert result == "the Beatles"

    def test_play_with_request_prefix(self):
        assert extract_after_play("can you play some jazz") == "some jazz"
        assert extract_after_play("i want you to play Taylor Swift") == "Taylor Swift"

    def test_no_play_command(self):
        result = extract_after_play("what's the weather")
        assert result is None

    def test_play_case_insensitive(self):
        result = extract_after_play("PLAY jazz music")
        assert result == "jazz music"

    def test_midsentence_play_not_caught(self):
        # Regression: "play" buried mid-sentence must NOT route to play_song.
        # Real failure (2026-06-18): a web-search request matched the play
        # fast-path because "play" appeared deep in the utterance.
        assert (
            extract_after_play(
                "search the web for the best new ps5 games that i can play "
                "with my seven-year-old daughter. she liked playing astro bot before"
            )
            is None
        )
        # And the user's correction attempt, which also contained "play":
        assert (
            extract_after_play(
                "i don't want you to play me a song. i want you to search about ps5 games"
            )
            is None
        )

    def test_midsentence_play_routes_to_slm(self):
        # End-to-end through catchAll: the over-matching turn falls through to
        # the SLM (returned unchanged) instead of emitting a play_song action.
        msg = "search the web for the best new ps5 games that i can play with my daughter"
        assert catchAll(msg) == msg


class TestAreaLightState:
    def test_extracts_area_and_requested_state(self):
        assert extract_area_light_state("what lights are on upstairs") == ("upstairs", "on")
        assert extract_area_light_state("Which lights are currently off in the office?") == ("the office", "off")

    def test_routes_status_question_to_area_state_tool(self):
        assert catchAll("what lights are on upstairs") == {
            "actions": [
                {
                    "intent": "get_entities_in_area_state",
                    "args": ["upstairs", "light", "on"],
                }
            ]
        }

    def test_inventory_question_still_reaches_agent(self):
        assert catchAll("what lights are upstairs") == "what lights are upstairs"


class TestContextualWebSearch:
    def test_matches_only_topicless_search_requests(self):
        assert is_contextual_web_search_request("can you search the internet")
        assert is_contextual_web_search_request("look it up")
        assert not is_contextual_web_search_request("search the internet for contaminated food")


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

    def test_midsentence_timer_not_caught(self):
        # Anchored like play: a passing mention of setting a timer must not fire.
        assert extract_timer("remind me later that i need to set a timer for the eggs") is None


class TestListTimers:
    """Tests for list timers command."""

    def test_get_timers(self):
        assert list_timers("get timers") is True

    def test_get_timer(self):
        assert list_timers("get timer") is True

    def test_not_list_timers(self):
        assert list_timers("start timer") is None

    def test_midsentence_get_timer_not_caught(self):
        # Anchored: "get the timer" mid-sentence must not hijack the turn.
        assert list_timers("can you help me get the timer working") is None


class TestLightBrightness:
    """Brightness fast-path → ha_set_brightness, skipping the SLM."""

    @pytest.mark.parametrize(
        "utterance,entity,pct",
        [
            ("set downstairs office lights to a hundred percent", "downstairs office lights", 100),
            ("set the kitchen lights to 50 percent", "kitchen lights", 50),
            ("dim the bedroom lamp to twenty percent", "bedroom lamp", 20),
            ("change the hallway light to seventy five percent", "hallway light", 75),
            ("turn the office lights to 30%", "office lights", 30),
            ("set the lounge to half percent", "lounge", 50),
        ],
    )
    def test_matches(self, utterance, entity, pct):
        assert extract_light_brightness(utterance) == (entity, pct)

    @pytest.mark.parametrize(
        "utterance",
        [
            "set the kitchen lights to bright",  # no parseable number
            "turn the office lights to maximum",  # no "percent"
            "set the volume to 50 percent",  # volume → SLM, not a light
            "set the lights and the fan to 50 percent",  # compound
            "set it to 50 percent",  # vague entity
            "what time is it",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_light_brightness(utterance) is None

    def test_catch_all_routes_to_brightness(self):
        result = catchAll("set downstairs office lights to a hundred percent")
        assert result == {
            "actions": [{"intent": "ha_set_brightness", "args": ["downstairs office lights", 100]}]
        }


class TestDimBrighten:
    """Bare dim/brighten fast-path → ha_set_brightness at fixed levels."""

    @pytest.mark.parametrize(
        "utterance,entity,pct",
        [
            ("can you dim the lights in the downstairs office", "downstairs office", 30),
            ("dim the lights", "lights", 30),
            ("dim the downstairs office lights", "downstairs office lights", 30),
            ("brighten the kitchen", "kitchen", 100),
            ("brighten the bedroom lamp", "bedroom lamp", 100),
            ("please dim the lounge", "lounge", 30),
        ],
    )
    def test_matches(self, utterance, entity, pct):
        assert extract_dim_brighten(utterance) == (entity, pct)

    @pytest.mark.parametrize(
        "utterance",
        [
            "dimitri called",  # \b stops mid-word match
            "dim it",  # vague entity
            "dim the lights and lock the door",  # compound
            "what time is it",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_dim_brighten(utterance) is None

    def test_percent_form_takes_priority(self):
        # "dim X to N percent" is handled by the brightness rule, not the
        # fixed-level dim rule — catchAll lists it first.
        assert catchAll("dim the kitchen lights to 20 percent") == {
            "actions": [{"intent": "ha_set_brightness", "args": ["kitchen lights", 20]}]
        }

    def test_catch_all_routes_to_dim(self):
        assert catchAll("can you dim the lights in the downstairs office") == {
            "actions": [{"intent": "ha_set_brightness", "args": ["downstairs office", 30]}]
        }


class TestTurnOnOff:
    """turn on/off fast-path → turn_on / turn_off."""

    @pytest.mark.parametrize(
        "utterance,state,entity",
        [
            ("turn on the kitchen lights", "on", "kitchen lights"),
            ("turn off the bedroom lamp", "off", "bedroom lamp"),
            ("turn the office lights on", "on", "office lights"),
            ("turn the hallway light off", "off", "hallway light"),
            ("please turn on the porch light", "on", "porch light"),
        ],
    )
    def test_matches(self, utterance, state, entity):
        assert extract_turn_onoff(utterance) == (state, entity)

    @pytest.mark.parametrize(
        "utterance",
        [
            "turn it off",  # vague entity
            "turn those lights back off",  # leading anaphor → resolve in SLM
            "turn these lamps off",  # leading anaphor
            "turn that light off",  # leading anaphor
            "turn on the lamp and the fan",  # compound
            "what time is it",
            "play some music",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_turn_onoff(utterance) is None

    @pytest.mark.parametrize(
        "utterance,state,entity",
        [
            # A re-do adverb ("back"/"again") must not corrupt an explicit entity.
            ("turn the dining room lights back off", "off", "dining room lights"),
            ("turn the porch light back on", "on", "porch light"),
            ("turn the kitchen lights off again", "off", "kitchen lights"),
        ],
    )
    def test_redo_adverb_stripped(self, utterance, state, entity):
        assert extract_turn_onoff(utterance) == (state, entity)

    def test_catch_all_routes_to_turn_on(self):
        assert catchAll("turn on the kitchen lights") == {
            "actions": [{"intent": "turn_on", "args": ["kitchen lights"]}]
        }

    def test_catch_all_routes_to_turn_off(self):
        assert catchAll("turn the bedroom lamp off") == {
            "actions": [{"intent": "turn_off", "args": ["bedroom lamp"]}]
        }


class TestToggle:
    """toggle fast-path → toggle."""

    def test_match(self):
        assert extract_toggle("toggle the office lights") == "office lights"

    def test_vague_entity(self):
        assert extract_toggle("toggle it") is None

    def test_catch_all_routes_to_toggle(self):
        assert catchAll("toggle the office lights") == {
            "actions": [{"intent": "toggle", "args": ["office lights"]}]
        }


class TestLock:
    """lock/unlock fast-path → ha_lock / ha_unlock."""

    @pytest.mark.parametrize(
        "utterance,action,entity",
        [
            ("lock the front door", "lock", "front door"),
            ("unlock the back door", "unlock", "back door"),
            ("please lock the garage door", "lock", "garage door"),
        ],
    )
    def test_matches(self, utterance, action, entity):
        assert extract_lock(utterance) == (action, entity)

    @pytest.mark.parametrize(
        "utterance",
        [
            "lock it",  # vague entity
            "lock the door and turn off the lights",  # compound
            "what time is it",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_lock(utterance) is None

    def test_catch_all_routes_to_lock(self):
        assert catchAll("lock the front door") == {
            "actions": [{"intent": "ha_lock", "args": ["front door"]}]
        }

    def test_catch_all_routes_to_unlock(self):
        assert catchAll("unlock the back door") == {
            "actions": [{"intent": "ha_unlock", "args": ["back door"]}]
        }


class TestCover:
    """open/close/shut fast-path → ha_open_cover / ha_close_cover."""

    @pytest.mark.parametrize(
        "utterance,action,entity",
        [
            ("open the blinds", "open", "blinds"),
            ("close the garage door", "close", "garage door"),
            ("shut the bedroom curtains", "close", "bedroom curtains"),
            ("open the living room shades", "open", "living room shades"),
        ],
    )
    def test_matches(self, utterance, action, entity):
        assert extract_cover(utterance) == (action, entity)

    @pytest.mark.parametrize(
        "utterance",
        [
            "open spotify",  # no cover keyword → SLM
            "close the app",  # no cover keyword
            "open the blinds and the curtains",  # compound
            "open it",  # vague
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_cover(utterance) is None

    def test_catch_all_routes_to_open(self):
        assert catchAll("open the blinds") == {
            "actions": [{"intent": "ha_open_cover", "args": ["blinds"]}]
        }

    def test_catch_all_routes_to_close(self):
        assert catchAll("close the garage door") == {
            "actions": [{"intent": "ha_close_cover", "args": ["garage door"]}]
        }


class TestColor:
    """set/make/turn colour fast-path → ha_set_color."""

    @pytest.mark.parametrize(
        "utterance,entity,color",
        [
            ("set the lights to red", "lights", "red"),
            ("make the office lights blue", "office lights", "blue"),
            ("turn the lights green", "lights", "green"),
            ("change the lamp to warm white", "lamp", "warm white"),
            ("set the bedroom lights cool white", "bedroom lights", "cool white"),
        ],
    )
    def test_matches(self, utterance, entity, color):
        assert extract_color(utterance) == (entity, color)

    @pytest.mark.parametrize(
        "utterance",
        [
            "set the lights to bright",  # not a known colour
            "set a timer for five minutes",  # no colour
            "make the lights red and blue",  # compound
            "what time is it",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_color(utterance) is None

    def test_catch_all_routes_to_color(self):
        assert catchAll("set the lights to red") == {
            "actions": [{"intent": "ha_set_color", "args": ["lights", "red"]}]
        }

    def test_warm_white_not_split_into_white(self):
        # The multi-word colour must win over the bare "white" alternation.
        assert catchAll("set the lights to warm white") == {
            "actions": [{"intent": "ha_set_color", "args": ["lights", "warm white"]}]
        }


class TestVolume:
    """volume up/down fast-path → ha_volume_up / ha_volume_down."""

    @pytest.mark.parametrize(
        "utterance",
        [
            "louder",
            "volume up",
            "turn up the volume",
            "turn the volume up",
            "turn it up",
            "turn up the sound",
        ],
    )
    def test_up_matches(self, utterance):
        assert extract_volume_up(utterance) is True

    @pytest.mark.parametrize(
        "utterance",
        [
            "quieter",
            "softer",
            "volume down",
            "turn down the volume",
            "turn the volume down",
            "turn it down",
            "turn down the sound",
        ],
    )
    def test_down_matches(self, utterance):
        assert extract_volume_down(utterance) is True

    @pytest.mark.parametrize(
        "utterance",
        [
            "turn on the lights",
            "turn it off",
            "play some music",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_volume_up(utterance) is None
        assert extract_volume_down(utterance) is None

    def test_catch_all_routes_to_volume_up(self):
        assert catchAll("louder") == {"actions": [{"intent": "ha_volume_up", "args": []}]}

    def test_catch_all_routes_to_volume_down(self):
        assert catchAll("turn the volume down") == {
            "actions": [{"intent": "ha_volume_down", "args": []}]
        }

    def test_targeted_volume_routes_to_the_named_room_or_player(self):
        assert extract_volume_down_target("turn the volume down in the living room") == "living room"
        assert extract_volume_up_target("turn up the volume on the Sonos speaker") == "Sonos speaker"
        assert catchAll("turn the volume down in the living room") == {
            "actions": [{"intent": "ha_volume_down", "args": ["living room"]}]
        }

    def test_pause_with_filler_and_music_object_uses_the_fast_path(self):
        assert catchAll("just pause the music then") == {
            "actions": [{"intent": "pause", "args": []}]
        }


class TestNoFalsePositives:
    """Conversational / ambiguous phrasings that mention trigger words but
    should NOT be hijacked by a fast-path — they must reach the SLM unchanged.

    A fast-path firing here would skip the agent entirely, so these guard
    against the regex over-reaching (anchoring, compound/vague/particle guards,
    colour + cover keyword gating, percent requirement).
    """

    @pytest.mark.parametrize(
        "utterance",
        [
            # Questions / statements, not commands (anchored verbs save us).
            "what color should I make the lights",
            "why is the light red",
            "how bright are the lights",
            "is the front door locked",
            "is it warmer in the office",
            "the garage door is open",
            "I set the lights to red yesterday",
            "i think the volume is too loud",
            "i dont think the lights work",
            "tell me about the color blue",
            # Trigger word buried mid-sentence (anchored to start).
            "remind me to lock the door",
            "should I turn the lights off later",
            "set a reminder for the dentist",
            # Phrasal-verb idioms, not device commands.
            "lock in your answer",
            "lock down the schedule",
            "open up about your feelings",
            # Verb belongs to a different domain than the fast-path.
            "close the meeting",
            "open a new note",
            "set the scene to movie time",
            # Brightness rule needs a parseable number AND "percent".
            "set the lights to bright",
            "turn the lights to maximum",
            # Colour rule needs a known colour word.
            "set the lights to teal",
            # Cover rule needs a cover keyword.
            "open the application",
            # \b stops mid-word verb matches.
            "dimitri called me",
            "locksmith is coming today",
        ],
    )
    def test_does_not_fire(self, utterance):
        # catchAll returns the original string (not an actions dict) on no match.
        assert catchAll(utterance) == utterance


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
        assert result == {"actions": [{"intent": "get_current_time", "args": []}]}

    @pytest.mark.parametrize(
        "utterance",
        [
            "what's the weather",
            "weather forecast",
            "forecast?",
            "what will the weather be like",
            "tell me the weather forecast",
        ],
    )
    def test_catch_default_weather_forecast(self, utterance):
        assert has_default_weather_forecast(utterance) is True
        assert catchAll(utterance) == {"actions": [{"intent": "get_weather_forecast", "args": []}]}

    @pytest.mark.parametrize(
        "utterance",
        [
            "what's the weather forecast for tomorrow",
            "what's the weather in London",
            "will it rain this weekend",
            "forecast for the kitchen",
        ],
    )
    def test_default_weather_forecast_rejects_arguments(self, utterance):
        assert has_default_weather_forecast(utterance) is None
        assert catchAll(utterance) == utterance

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
        assert result == {"actions": [{"intent": "get_timer_status", "args": []}]}

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
        assert result == {
            "actions": [{"intent": "deep_think", "args": ["whether I should switch banks"]}]
        }


class TestExtractNoteDelete:
    """Deletion/editing of notes is unsupported — these requests must be caught
    and refused deterministically, never routed to the SLM (which would
    confabulate doing it)."""

    @pytest.mark.parametrize(
        "utterance",
        [
            "delete the note",
            "remove that note",
            "remove the note you wrote in today's note regarding formula one",
            "can you delete the F1 entry from today's note",
            "erase my shopping note",
            "clear the daily note",
            "get rid of the boiler note",
            "forget that fact",
            "delete that fact you saved",
            "edit my shopping note",
            "change the F1 note",
            "update the journal entry",
            "i want you to remove the note about the trip",
        ],
    )
    def test_matches(self, utterance):
        assert extract_note_delete(utterance) is True

    @pytest.mark.parametrize(
        "utterance",
        [
            "add a note to delete the old config tomorrow",  # 'delete' is content
            "turn off the lights",
            "remove the milk from the shopping list",  # HA todo, not a note
            "delete the alarm",
            "write a note about the F1 standings",
            "read my notes from today",
            "what's in my shopping note",
            "forget it",  # dismissal, no note
            "cancel the timer",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_note_delete(utterance) is None

    def test_catch_all_refuses_with_reply(self):
        result = catchAll("delete the note you wrote about formula one")
        assert isinstance(result, dict)
        assert "reply" in result and "actions" not in result
        assert "dashboard" in result["reply"].lower()

    def test_catch_all_allows_editor_requests_when_edit_mode_is_enabled(self, monkeypatch):
        monkeypatch.setattr("utils.intent_catch._obsidian_edit_enabled", lambda: True)
        assert catchAll("delete the active note") == "delete the active note"


class TestExtractSatelliteMessage:
    def test_extracts_tell_with_target_and_message(self):
        assert extract_satellite_message("Tell downstairs that dinner is ready") == (
            "downstairs",
            "dinner is ready",
        )

    def test_catch_all_routes_satellite_message(self):
        assert catchAll("announce to the kitchen saying the guests are here") == {
            "actions": [
                {
                    "intent": "send_satellite_message",
                    "args": ["kitchen", "the guests are here"],
                }
            ]
        }


class TestExtractDeepThink:
    """Regex-fast-path for thinking-mode queries.

    Conservative: must be an explicit ask for thought, not a casual 'I think'.
    """

    @pytest.mark.parametrize(
        "utterance,topic",
        [
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
        ],
    )
    def test_matches(self, utterance, topic):
        assert extract_deep_think(utterance) == topic

    @pytest.mark.parametrize(
        "utterance",
        [
            "i think it's raining",
            "i don't think so",
            "think fast",
            "consider it done",
            "think",
            "let me reflect for a second",  # no preposition + topic
            "what's the weather",
        ],
    )
    def test_non_matches(self, utterance):
        assert extract_deep_think(utterance) is None


class TestExtractSummarizeThinking:
    """Regex catch for 'what do you have so far'-style follow-ups during
    a thinking turn that just got interrupted."""

    @pytest.mark.parametrize(
        "utterance",
        [
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
        ],
    )
    def test_matches(self, utterance):
        assert extract_summarize_thinking(utterance) is True

    @pytest.mark.parametrize(
        "utterance",
        [
            "we discussed this so far",  # 'so far' alone shouldn't match
            "tell me a joke",
            "think about the weather",  # belongs to deep_think
            "what time is it",
            "play some music",
            "stop",  # belongs to existing stop intent
        ],
    )
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
