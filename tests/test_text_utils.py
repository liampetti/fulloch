"""Tests for `core.text_utils.split_sentences`.

Used by `_warm_and_announce` to split the startup greeting into per-sentence
TTS jobs so the CUDA-graph cache picks up multiple prefill shapes during
warmup (smooths first-real-turn latency).
"""

import pytest

from core.text_utils import clean_for_tts, split_clauses, split_sentences, spoken_for_tts


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


def test_normalises_news_semicolons_and_acronyms_without_sentence_fragments():
    cleaned = clean_for_tts("WA news; NRL and AFL in NY.")

    assert cleaned == "W A news. N R L and A F L in N Y."
    assert split_sentences(cleaned) == ["W A news.", "N R L and A F L in N Y."]
    assert list(split_clauses(cleaned)) == ["W A news.", "N R L and A F L in N Y."]


def test_preserves_commas_but_turns_semicolons_into_sentence_breaks():
    assert clean_for_tts("One, two; three") == "One, two. three"


def test_spoken_for_tts_expands_decimals_money_percentages_and_units():
    assert spoken_for_tts("It is 18.2°C, weighs 2.5 kg, and costs $12.50 at 75%.") == (
        "It is eighteen point two degrees Celsius, weighs two point five kilograms, "
        "and costs twelve dollars and fifty cents at seventy-five percent."
    )


def test_spoken_for_tts_preserves_decimal_precision_and_common_identifiers():
    assert spoken_for_tts("The result is 0.0033. Visit 192.168.1.1 on 2026-08-04 at 10:30.") == (
        "The result is zero point zero zero three three. Visit 192.168.1.1 on 2026-08-04 at ten thirty."
    )


def test_spoken_for_tts_localizes_numbers_without_english_unit_names():
    assert spoken_for_tts("Il fait 18,2 °C et 25 %.", "fr-FR") == (
        "Il fait dix-huit virgule deux °C et vingt-cinq %."
    )
    assert spoken_for_tts("Es sind 21 kg.", "de-DE") == "Es sind einundzwanzig kg."
    assert spoken_for_tts("Visit 192.168.1.1 at 10:30 on 2026-08-04.", "fr") == (
        "Visit 192.168.1.1 at 10:30 on 2026-08-04."
    )
    assert spoken_for_tts("Value 25.", "zz") == "Value 25."


@pytest.mark.parametrize("language,text,expected", [
    ("en", "The time is 2:30 PM.", "The time is two thirty PM."),
    ("en-GB", "It's 09:05.", "It's nine oh five."),
    ("en", "12:00 AM", "twelve AM"),
    ("de-DE", "Es ist 22:18 Uhr.", "Es ist zweiundzwanzig Uhr achtzehn."),
    ("de", "Es ist 09:05 Uhr.", "Es ist neun Uhr fünf."),
    ("de", "Es ist 09:00 Uhr.", "Es ist neun Uhr."),
    ("fr-FR", "Il est 14:30.", "Il est quatorze heures trente."),
    ("fr", "Il est 01:05.", "Il est une heure cinq."),
    ("fr", "Il est 21:00.", "Il est vingt et une heures."),
    ("fr", "00:00", "zéro heures"),
])
def test_spoken_clock_times(language, text, expected):
    assert spoken_for_tts(text, language) == expected


@pytest.mark.parametrize("text", [
    "The score is 2:30.", "Mix at a ratio of 1:20.", "https://example.com/09:05",
    "2026-11-05T09:05:00", "09:05:10", "25:00", "12:60", "23:00 PM",
])
def test_clock_normalization_leaves_ambiguous_or_invalid_values(text):
    assert spoken_for_tts(text, "en") == text


def test_clock_normalization_does_not_inject_english_for_unsupported_language():
    assert spoken_for_tts("22:18", "zz") == "22:18"
    assert spoken_for_tts("22:18", "es") == "22:18"


def test_number_words_regional_tags_ordinals_and_unsupported_language():
    from utils.number_words import number_words

    assert number_words(21, "fr-BE") == "vingt et un"
    assert number_words(3, "fr", ordinal=True) == "troisième"
    assert number_words(21, "zz") == "21"
