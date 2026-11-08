"""Narrow live-fact guards guide the agent without dispatching a fast path."""

import pytest

from core.live_fact_guard import live_fact_requirement


@pytest.mark.parametrize("prompt,tools,clock_intent", [
    ("what time is it in french", ("get_current_time",), "HassGetCurrentTime"),
    ("what day is it? in german", ("get_current_time",), "HassGetCurrentDate"),
    ("What was the news in Germany today? Tell me in German", ("external_information",), None),
    ("Was sind die neuesten Nachrichten heute? Auf Französisch", ("external_information",), None),
    ("Quelles sont les actualités aujourd'hui? En allemand", ("external_information",), None),
    ("What's the weather today in French?", ("get_weather_forecast", "external_information"), None),
    ("Wie ist das Wetter heute auf Englisch?", ("get_weather_forecast", "external_information"), None),
    ("what is 2 plus 2 in French?", ("calculate",), None),
    ("calculate 8 * 4", ("calculate",), None),
    ("what is 8 - 4?", ("calculate",), None),
])
def test_clear_live_questions_require_relevant_tool(prompt, tools, clock_intent):
    requirement = live_fact_requirement(prompt)
    assert requirement.tools == tools
    assert requirement.clock_intent == clock_intent


@pytest.mark.parametrize("prompt", [
    "tell me a joke about the news", "what does the word news mean?",
    "what time is it in London", "what time is the game on today?",
    "translate the words two plus two into French", "play French music",
])
def test_guard_does_not_expand_ambiguous_or_nonlive_requests(prompt):
    assert live_fact_requirement(prompt) is None


def test_multilingual_translated_clock_uses_existing_matcher(monkeypatch):
    from pathlib import Path

    from tools import ha_client
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    matcher = HassilFastPath(["de"], str(Path(__file__).parent / "fixtures/hassil"))
    requirement = live_fact_requirement("wie spät ist es auf französisch", matcher)
    assert requirement.tools == ("get_current_time",)
    assert requirement.clock_intent == "HassGetCurrentTime"
