import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.higgs_controls import (  # noqa: E402
    apply_delivery,
    delivery_for_request,
    extract_delivery_request,
    sanitize_for_higgs,
    split_leading_higgs_controls,
    strip_higgs_controls,
)


def test_higgs_controls_allow_known_tags_and_strip_unknown_tags():
    text = "<|emotion:affection|>Hello <|prosody:pause|><|bad:tag|>there"
    assert sanitize_for_higgs(text) == "<|emotion:affection|>Hello <|prosody:pause|>there"


def test_display_text_has_no_higgs_controls():
    assert strip_higgs_controls("<|style:whispering|> Hello") == "Hello"


def test_leading_controls_are_separated_from_speech_text():
    assert split_leading_higgs_controls("<|style:whispering|><|prosody:speed_slow|> Hello") == (
        "<|style:whispering|><|prosody:speed_slow|>",
        "Hello",
    )


def test_explicit_delivery_survives_a_continuation_only():
    whisper = delivery_for_request("whisper today's notes")
    assert apply_delivery("Here are your notes.", whisper).startswith("<|style:whispering|>")
    assert delivery_for_request("continue", whisper) == whisper
    assert delivery_for_request("what time is it?", whisper) == ""


def test_quietly_always_maps_to_the_whisper_delivery():
    assert delivery_for_request("what is the weather quietly") == "<|style:whispering|>"


def test_delivery_extractor_removes_explicit_readback_modifiers():
    assert extract_delivery_request("Read me today's notes quickly") == (
        "Read me today's notes",
        "<|prosody:speed_fast|>",
    )
    assert extract_delivery_request("quietly tell me the weather") == (
        "tell me the weather",
        "<|style:whispering|>",
    )
    assert extract_delivery_request("Please whisper today's notes") == (
        "today's notes",
        "<|style:whispering|>",
    )
    assert extract_delivery_request("Could you slowly read me todays notes?") == (
        "Could you read me todays notes?",
        "<|prosody:speed_slow|>",
    )


def test_delivery_extractor_keeps_ambiguous_content_and_continuations():
    assert extract_delivery_request("What happened quickly after the alarm?") == (
        "What happened quickly after the alarm?",
        "",
    )
    assert extract_delivery_request("Turn off the kitchen lights quickly") == (
        "Turn off the kitchen lights quickly",
        "",
    )
    assert extract_delivery_request("continue", "<|style:whispering|>") == (
        "continue",
        "<|style:whispering|>",
    )
