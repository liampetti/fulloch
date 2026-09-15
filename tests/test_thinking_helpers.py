"""Thinking prompt contracts and bounded observation helpers."""

import pytest

from core import thinking_worker as worker
from core.text_utils import clean_for_tts
from tools.thinking_playbooks import matching_playbooks
from tools.tool_registry import Param, ToolSchema
from utils.prompts import (
    FOREGROUND_HISTORY_MESSAGES,
    assemble_foreground_history,
    get_thinking_report_answer_prompt,
    get_thinking_report_prompt,
    get_thinking_worker_prompt,
)


def test_clean_for_tts_strips_think_blocks():
    raw = (
        "<think>OK so they're asking about cars. Let me weigh the options...</think>"
        "Electric cars are probably worth it if you have home charging."
    )
    cleaned = clean_for_tts(raw)
    assert "<think>" not in cleaned and "weigh the options" not in cleaned
    assert "Electric cars" in cleaned


def test_foreground_history_is_copied_and_bounded():
    history = [
        {"role": "user", "content": "x" * 1000} for _ in range(FOREGROUND_HISTORY_MESSAGES + 1)
    ]
    compact = assemble_foreground_history(history)
    assert len(compact) == FOREGROUND_HISTORY_MESSAGES
    assert compact[0] is not history[1]
    assert len(compact[-1]["content"]) < 1000


def test_thinking_worker_prompt_excludes_foreground_personality_and_includes_inputs():
    prompt = get_thinking_worker_prompt(
        "Compare options",
        [{"role": "user", "content": "context"}],
        notes="saved note",
        job_state="one lead remains",
        capabilities="search_notes (read)",
    )
    assert "Compare options" in prompt
    assert "saved note" in prompt and "one lead remains" in prompt
    assert "search_notes (read)" in prompt
    assert "personality" not in prompt.lower()


def test_thinking_worker_prompt_uses_generic_progress_guardrails():
    prompt = get_thinking_worker_prompt("Compare options", [], capabilities="search_notes (read)")
    for instruction in (
        "highest information gain",
        "materially different",
        "runtime rejects duplicates",
        "sufficient findings collected",
        "useful preliminary report",
        "single source",
        "materially different available option",
        "plan",
        "Reactive question:",
        "Do not write the final report yourself",
        "Today is ",
    ):
        assert instruction in prompt
    assert "book" in prompt.lower()


def test_thinking_worker_prompt_has_no_domain_specific_playbooks():
    prompt = get_thinking_worker_prompt("Compare options", [], capabilities="search_notes(query)")
    for word in ("flight", "hotel", "iata"):
        assert word not in prompt.lower()
    assert "highest information gain" in prompt
    assert "untrusted data" in prompt and "<tool_observations>" in prompt


def test_thinking_worker_prompt_includes_matching_tool_owned_playbook():
    import tools.travel  # noqa: F401

    playbooks = matching_playbooks(
        "I am travelling from Paris to Rome, then Madrid around Easter next year.",
        {"plan_travel", "search_flights", "assess_itinerary"},
    )
    prompt = get_thinking_worker_prompt(
        "Check this itinerary",
        [],
        capabilities="plan_travel(request)",
        capability_playbooks="\n\n".join(playbook.render() for playbook in playbooks),
    )
    assert [playbook.name for playbook in playbooks] == ["travel planning"]
    for instruction in (
        "travel planning",
        "start with plan_travel",
        "timezone or calendar arithmetic alone",
        "failed itinerary assessment rejects only that candidate",
        "materially different date, route, or flight search",
        "exactly origin, destination, and a future ISO departure date",
        "Artifact reference",
        "never serialize schedule JSON",
    ):
        assert instruction in prompt
    assert playbooks[0].fallback_capability == "plan_travel"


def test_thinking_playbook_requires_an_enabled_capability():
    import tools.travel  # noqa: F401

    assert matching_playbooks("Find flights to Tokyo", {"calculate"}) == []


@pytest.mark.parametrize(
    "prompt",
    [
        "I want to go somewhere for lunch.",
        "I want to go somewhere to see the sunset.",
        "I want to go somewhere and fly overnight.",
        "I want to catch an early plane to somewhere so I arrive in time for dinner.",
    ],
)
def test_travel_playbook_matches_movement_not_meal_words(prompt):
    import tools.travel  # noqa: F401

    capabilities = {"plan_travel", "search_flights", "assess_itinerary"}
    assert [playbook.name for playbook in matching_playbooks(prompt, capabilities)] == [
        "travel planning"
    ]
    assert matching_playbooks("What should I have for dinner?", capabilities) == []


def test_thinking_transcript_compaction_preserves_each_tool_observation():
    transcript = ""
    for index in range(8):
        transcript = worker._append_thinking_observation(
            transcript, f"tool:search_{index}", f"result-{index} " + "x" * 2_900
        )
    assert len(transcript) <= worker.DEEP_THINK_TRANSCRIPT_MAX_CHARS
    for index in range(8):
        assert f"[tool:search_{index}]" in transcript and f"result-{index}" in transcript


def test_thinking_capability_description_includes_callable_signature():
    description = worker._describe_thinking_capability(
        "search_notes",
        ToolSchema(
            "search_notes", "Search saved notes.", [Param("query", True), Param("limit", False, 5)]
        ),
    )
    assert description == "- search_notes(query, limit=5): Search saved notes."


def test_thinking_report_prompt_requires_a_direct_evidence_scoped_summary():
    prompt = get_thinking_report_prompt("Compare heating options", "Retrieved prices")
    for instruction in (
        "## Summary",
        "exactly two or three short sentences",
        "central question",
        "beyond the retrieved evidence",
    ):
        assert instruction in prompt


def test_report_answer_prompt_is_grounded_to_one_completed_report():
    prompt = get_thinking_report_answer_prompt(
        "The installed price is $4,000.", {"artifacts": {"artifact-001": {"price": 4000}}}
    )
    for instruction in (
        "only the completed report",
        '"The report does not answer that."',
        "The installed price is $4,000.",
        "artifact-001",
    ):
        assert instruction in prompt
