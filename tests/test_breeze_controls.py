import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.breeze_controls import (  # noqa: E402
    apply_delivery,
    delivery_for_request,
    extract_delivery_request,
    sanitize_for_breeze,
    strip_breeze_tags,
)


def test_breeze_controls_keep_documented_english_audio_tags_only():
    text = "(sighs) Hello. (clears throat) Ready? (invented sound)"

    assert sanitize_for_breeze(text) == "(sighs) Hello. (clears throat) Ready?"


def test_breeze_controls_cap_audio_tags_per_reply():
    text = " ".join("(laughs)" for _ in range(10))

    assert sanitize_for_breeze(text, max_tags=2) == "(laughs) (laughs)"


def test_breeze_display_text_has_no_audio_tags():
    assert strip_breeze_tags("(whispers) Keep this quiet.") == "Keep this quiet."


def test_breeze_explicit_delivery_uses_documented_tags_and_continues():
    whisper = delivery_for_request("whisper today's notes")

    assert whisper == "(whispers)"
    assert apply_delivery("Here are your notes.", whisper) == "(whispers) Here are your notes."
    assert delivery_for_request("continue", whisper) == whisper
    assert delivery_for_request("what time is it?", whisper) == ""


def test_breeze_delivery_extractor_strips_only_unambiguous_whispers():
    assert extract_delivery_request("Please whisper today's notes") == (
        "today's notes",
        "(whispers)",
    )
    assert extract_delivery_request("quietly tell me the weather") == (
        "tell me the weather",
        "(whispers)",
    )
    assert extract_delivery_request("Read me today's notes slowly") == (
        "Read me today's notes slowly",
        "",
    )
