"""Tests for `utils/completeness.py` — the early-endpoint / speculation gate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.completeness import (
    SPECULATION_UNSAFE_INTENTS,
    is_complete,
    should_commit_provisional,
)

# --- is_complete ----------------------------------------------------------

def test_closed_clause_is_complete():
    assert is_complete("turn off the kitchen lights") is True
    assert is_complete("what is the capital of France") is True


def test_trailing_conjunction_is_incomplete():
    assert is_complete("play some jazz and") is False
    assert is_complete("turn on the lamp then") is False


def test_trailing_preposition_or_article_is_incomplete():
    assert is_complete("set a reminder for") is False
    assert is_complete("tell me about the") is False
    assert is_complete("i want to") is False


def test_trailing_particle_is_incomplete():
    # "turn off" (forward, no entity yet) — the speaker is about to name a target.
    assert is_complete("turn off") is False
    assert is_complete("turn the volume up") is False


def test_filler_tail_is_incomplete():
    assert is_complete("what time is it um") is False


def test_trailing_punctuation_ignored():
    assert is_complete("dim the lights.") is True
    assert is_complete("and ...") is False


def test_empty_is_incomplete():
    assert is_complete("") is False
    assert is_complete("   ") is False
    assert is_complete("?!") is False


def test_single_closed_word_is_complete():
    assert is_complete("stop") is True


# --- should_commit_provisional --------------------------------------------

def _actions(*intents):
    return {"actions": [{"intent": i, "args": []} for i in intents]}


def test_safe_regex_intent_commits_immediately():
    # Reversible / read-only intents commit on the regex match alone.
    assert should_commit_provisional("turn off the lights", _actions("turn_off")) is True
    assert should_commit_provisional("what time is it", _actions("get_time")) is True
    assert should_commit_provisional("louder", _actions("ha_volume_up")) is True


def test_unsafe_regex_intent_waits():
    assert should_commit_provisional("unlock the front door", _actions("ha_unlock")) is False
    assert should_commit_provisional("play the beatles", _actions("play_song")) is False
    assert should_commit_provisional("set a timer for ten minutes",
                                     _actions("start_countdown")) is False


def test_mixed_actions_with_any_unsafe_waits():
    assert should_commit_provisional("x", _actions("turn_on", "play_song")) is False


def test_freeform_commits_only_when_complete():
    # No regex match → catchAll returns the raw string.
    assert should_commit_provisional("what is the capital of France",
                                     "what is the capital of France") is True
    assert should_commit_provisional("what is the capital of",
                                     "what is the capital of") is False


def test_reply_dict_commits_when_complete():
    # A {"reply": ...} (e.g. note-delete refusal) carries no actions — harmless.
    assert should_commit_provisional("delete my note", {"reply": "I can't."}) is True


def test_unsafe_set_membership():
    assert "ha_lock" in SPECULATION_UNSAFE_INTENTS
    assert "turn_on" not in SPECULATION_UNSAFE_INTENTS


# --- integration with the real catchAll (the transcriber's actual call) ----

def test_commit_decision_with_real_catchall():
    from utils.intent_catch import catchAll

    def decide(prompt):
        return should_commit_provisional(prompt, catchAll(prompt))

    # Safe smart-home / read-only commands commit early.
    assert decide("turn off the kitchen lights") is True
    assert decide("what time is it") is True
    assert decide("dim the office to 20 percent") is True

    # Unsafe commands wait for the hard endpoint.
    assert decide("unlock the front door") is False
    assert decide("play some jazz") is False
    assert decide("set a timer for ten minutes") is False

    # Free-form questions commit only when the clause reads finished.
    assert decide("who was the first person on the moon") is True
    assert decide("tell me about the") is False
