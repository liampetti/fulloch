"""Tests for `core.text_utils.split_sentences`.

Used by `_warm_and_announce` to split the startup greeting into per-sentence
TTS jobs so the CUDA-graph cache picks up multiple prefill shapes during
warmup (smooths first-real-turn latency).
"""

from core.text_utils import clean_for_tts, split_sentences


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


# --- clean_for_tts: reasoning-block stripping ------------------------------


def test_strips_qwen_think_block():
    out = clean_for_tts("<think>weighing options</think>The answer is yes.")
    assert out == "The answer is yes."


def test_strips_gemma_thought_channel():
    # Gemma 4 wraps reasoning in a `<|channel>thought ... <channel|>` block.
    raw = "<|channel>thought\nlet me reason about this\n<channel|>The answer is yes."
    assert clean_for_tts(raw) == "The answer is yes."


def test_strips_gemma_empty_ghost_channel():
    # The template emits an empty thought channel even with thinking off.
    raw = "<|channel>thought\n<channel|>It's sunny today."
    assert clean_for_tts(raw) == "It's sunny today."


def test_strips_stray_gemma_channel_markers():
    out = clean_for_tts("Hello <channel|> there.")
    assert "channel" not in out and "Hello" in out and "there." in out


def test_keeps_plain_text_untouched():
    assert clean_for_tts("The answer is forty two.") == "The answer is forty two."
