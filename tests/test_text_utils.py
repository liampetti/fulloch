"""Tests for `core.text_utils.split_sentences`.

Used by `_warm_and_announce` to split the startup greeting into per-sentence
TTS jobs so the CUDA-graph cache picks up multiple prefill shapes during
warmup (smooths first-real-turn latency).
"""

from core.text_utils import split_sentences


def test_empty_string_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence_no_punctuation():
    assert split_sentences("hello there") == ["hello there"]


def test_single_sentence_with_period():
    assert split_sentences("Online and ready.") == ["Online and ready."]


def test_two_sentences_split_and_preserve_punctuation():
    result = split_sentences("Online and ready. Octopuses have three hearts.")
    assert result == ["Online and ready.", "Octopuses have three hearts."]


def test_mixed_terminators():
    result = split_sentences("Are you there? Yes! I'm here.")
    assert result == ["Are you there?", "Yes!", "I'm here."]


def test_extra_whitespace_between_sentences():
    result = split_sentences("First sentence.   Second one.")
    assert result == ["First sentence.", "Second one."]


def test_trailing_whitespace_stripped():
    assert split_sentences("  Hi.  ") == ["Hi."]
