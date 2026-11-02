"""Explicit task/source scope extraction remains deterministic and topic-neutral."""

import pytest

from core.thinking_scope import (
    apply_subject_candidates,
    named_entities,
    research_context,
    source_compatibility,
)
from core.thinking_worker import _planning_conversation


def test_planning_history_preserves_bounded_user_and_assistant_context():
    history = [{"role": "user", "content": "How did xyzq shares perform?"},
               {"role": "assistant", "content": "Example Widgets Inc · XYZQ · TEST", "ts": 123},
               {"role": "tool", "content": "not a user or assistant turn"},
               {"role": "user", "content": "Explain the possible risks."}]
    turns = _planning_conversation(history, "Give me a detailed analysis of xyzq shares.")
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user"]
    assert turns[1]["ts"] == 123
    assert "Example Widgets" in turns[1]["content"]
    assert len(_planning_conversation([{"role": "user", "content": str(i)} for i in range(30)], "task")) == 12


def test_planning_history_bounds_text_and_handles_missing_history():
    assert _planning_conversation(None, "task") == []
    turns = _planning_conversation([{"role": "assistant", "content": "x" * 5000}] * 10, "task")
    assert sum(len(turn["content"]) for turn in turns) == 4000


def test_stock_research_is_current_but_explicit_history_stays_historical():
    anchor = "2030-06-15T12:00:00+00:00"
    assert research_context("Give me a detailed analysis of xyzq shares", as_of=anchor)["temporal_intent"] == "current"
    assert research_context("Describe the weather last year", as_of=anchor)["temporal_intent"] == "historical"
    assert research_context("explain the proof of Pythagoras' theorem", as_of=anchor)["temporal_intent"] == "timeless"


def test_scope_retains_requested_dates_and_reports_conflicting_label_without_filtering():
    context = research_context("Research the upcoming Harbour 1000 race on 2026-10-08")

    compatibility, _reason, subject = source_compatibility(
        context, title="Harbour 12 Hour entry list", url="https://example.com/harbour-12-hour"
    )

    assert context["requested_dates"] == ["2026-10-08"]
    assert compatibility == "compatible"
    assert "assess individual claims" in _reason
    assert subject == "harbour 1000"


def test_scope_normalises_explicit_day_month_year_dates():
    context = research_context("Research an event on 8 October 2026")

    assert context["requested_dates"] == ["2026-10-08"]


def test_scope_inherits_explicit_entity_and_preserves_task_constraints():
    context = research_context(
        "Compare the upcoming race's schedule, ticket price, and limitations using official sources.",
        [{"role": "user", "content": "Research the Harbour 1000 for me."}],
    )

    assert context["temporal_intent"] == "future"
    assert "Harbour 1000" in context["entities"]
    assert "Harbour 1000" in context["inherited_entities"]
    assert {"schedule", "price", "limitations"} <= set(context["comparison_dimensions"])
    assert context["source_preferences"] == ["official_primary"]
    assert context["unresolved_references"] == []


def test_scope_marks_unresolvable_follow_up_reference_unknown():
    context = research_context("Tell me about the upcoming race.")

    assert context["unresolved_references"] == ["the upcoming race"]
    assert context["entities"] == []


def test_fresh_research_topic_does_not_inherit_previous_jobs_scope():
    task = "Research the latest local AI trends for running capable models at home"
    context = research_context(task, [
        {"role": "user", "content": "Research the Harbour 1000 main race on 2026-10-08 using official sources"},
        {"role": "user", "content": task},
    ])
    assert "Harbour 1000" not in context["entities"]
    assert context["inherited_entities"] == []
    assert context["requested_dates"] == []
    assert context["requested_category"] == ""
    assert context["source_preferences"] == []


def test_named_entities_keeps_explicit_acronyms_and_quoted_names():
    assert named_entities('Compare NVIDIA with "Private Photo Backup".') == [
        "Private Photo Backup",
        "NVIDIA",
    ]


def test_relative_event_edition_reports_different_calendar_without_filtering_evidence():
    context = research_context(
        "Research the imminent Harbour 1000 main race.",
        as_of="2026-10-11T00:00:00+11:00",
    )

    assert context["edition_year"] == 2026
    assert source_compatibility(
        context, title="Harbour 1000 2027 calendar", url="https://example.test/2027-calendar",
    )[0] == "compatible"


def test_explicit_category_mismatch_remains_available_for_claim_level_assessment():
    context = research_context("Research the imminent main race category.")

    assert context["requested_category"] == "main race"
    compatibility, reason, _subject = source_compatibility(
        context, title="GR Cup support field", url="https://example.test/gr-cup",
    )
    assert compatibility == "compatible"
    assert reason


def test_subject_identity_is_invalidated_if_support_becomes_incompatible():
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    card = {
        "id": "card-1", "kind": "atomic_claim", "compatibility": "compatible",
        "evidence_status": "single_source", "statement": "Harbour Grand Prix is on October 25, 2026.",
        "provenance": {"source_id": "source-1"},
    }
    proposal = {"name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": ["card-1"]}
    assert apply_subject_candidates(context, [proposal], [card])["state"] == "resolved"
    card["compatibility"] = "incompatible"
    assert apply_subject_candidates(context, [], [card])["state"] == "unresolved"


def test_subject_identity_drops_only_invalid_support_when_other_support_remains():
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    card = {
        "id": "card-1", "kind": "atomic_claim", "compatibility": "compatible",
        "evidence_status": "single_source", "statement": "Harbour Grand Prix is on October 25, 2026.",
        "provenance": {"source_id": "source-1"},
    }
    other = {**card, "id": "card-2", "provenance": {"source_id": "source-2"}}
    proposal = {"name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": ["card-1", "card-2"]}
    assert apply_subject_candidates(context, [proposal], [card, other])["card_ids"] == ["card-1", "card-2"]
    card["compatibility"] = "incompatible"
    result = apply_subject_candidates(context, [], [card, other])
    assert result["state"] == "resolved"
    assert result["card_ids"] == ["card-2"]
    assert set(result["candidates"][0]["source_bindings"]) == {"card-2"}


@pytest.mark.parametrize("change", ["conflict", "modified", "removed"])
def test_subject_identity_does_not_reuse_changed_or_removed_original_support(change):
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    card = {
        "id": "card-1", "kind": "atomic_claim", "compatibility": "compatible",
        "evidence_status": "single_source", "statement": "Harbour Grand Prix is on October 25, 2026.",
        "provenance": {"source_id": "source-1"},
    }
    proposal = {"name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": ["card-1"]}
    assert apply_subject_candidates(context, [proposal], [card])["state"] == "resolved"
    if change == "conflict":
        card["evidence_status"] = "conflicted"
    elif change == "modified":
        card["statement"] = "Harbour Grand Prix is on October 26, 2026."
    remaining = [] if change == "removed" else [card]
    assert apply_subject_candidates(context, [], remaining)["state"] == "unresolved"


def test_subject_identity_does_not_infer_a_year_or_join_separate_assertions():
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    card = {
        "id": "card-1", "kind": "atomic_claim", "compatibility": "compatible",
        "evidence_status": "single_source",
        "statement": "The 2026 calendar is available.\nHarbour Grand Prix is on October 25.",
        "provenance": {"source_id": "source-1"},
    }
    proposal = {"name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": ["card-1"]}
    result = apply_subject_candidates(context, [proposal], [card])
    assert result["state"] == "unresolved"
    assert result["rejected"][-1]["reason"] == "identity_date_binding_unproven"


def test_subject_identity_rejects_elapsed_event_even_with_exact_source():
    context = research_context("Research the next Harbour race", as_of="2026-10-29T00:00:00Z")
    card = {
        "id": "card-1", "kind": "atomic_claim", "compatibility": "compatible",
        "evidence_status": "single_source", "statement": "Harbour Grand Prix is on October 25, 2026.",
        "provenance": {"source_id": "source-1"},
    }
    proposal = {"name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": ["card-1"]}
    result = apply_subject_candidates(context, [proposal], [card])
    assert result["state"] == "unresolved"
    assert result["rejected"][-1]["reason"] == "elapsed_subject"


def _synthetic_schedule():
    from core.thinking_source_context import source_block_card, source_document_blocks

    text = ("2026 Harbour event schedule\nDate | Event\n"
            "Oct. 25-Oct. 26 | Harbour Grand Prix\nNov. 06-Nov. 08 | Island Grand Prix")
    source = {"text": text, "title": "Schedule", "url": "https://example.test/schedule"}
    registry = {"source-1": source}
    card_id, block = next((identifier, value) for identifier, value in source_document_blocks(registry).items()
                          if "|" in value["text"])
    return registry, source_block_card(card_id, block, source)


def test_identity_can_use_synthetic_calendar_heading_for_event_window():
    registry, card = _synthetic_schedule()
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    proposal = {"name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": [card["id"]]}
    result = apply_subject_candidates(
        context, [proposal], [card], source_registry=registry,
    )
    assert result["state"] == "resolved"
    assert result["card_ids"] == [card["id"]]
    assert result["date_scope"] == "event_window_boundary"
    assert "not necessarily" not in result["detail"]  # The validator does not claim a race time.
    assert "race or start date still requires checking" in result["detail"]
@pytest.mark.parametrize(("name", "date"), [
    ("Island Grand Prix", "2026-10-25"),
    ("Harbour Grand Prix", "2026-11-06"),
    ("Harbour Grand Prix", "2025-10-25"),
    ("Harbour City Grand Prix", "2026-10-25"),
])
def test_synthetic_table_does_not_join_unrelated_rows_or_invent_names(name, date):
    registry, card = _synthetic_schedule()
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    proposal = {"name": name, "date": date, "card_ids": [card["id"]]}
    result = apply_subject_candidates(
        context, [proposal], [card], source_registry=registry,
    )
    assert result["state"] == "unresolved"


@pytest.mark.parametrize(("heading", "date_cell", "candidate", "expected"), [
    ("2026 Harbour event schedule", "Oct. 25", "2026-10-25", "resolved"),
    ("2026 Harbour event schedule", "Oct. 25-26", "2026-10-26", "unresolved"),
    ("2026 Harbour event schedule", "Dec. 31-Jan. 2", "2026-12-31", "unresolved"),
    ("2026 Harbour historical events", "Oct. 25", "2026-10-25", "unresolved"),
    ("2025 and 2026 Harbour schedule", "Oct. 25", "2026-10-25", "unresolved"),
    ("Harbour event schedule", "Oct. 25", "2026-10-25", "unresolved"),
])
def test_year_qualified_schedule_context_is_conservative(heading, date_cell, candidate, expected):
    from core.thinking_source_context import source_block_card, source_document_blocks

    text = f"{heading}\nDate | Event\n{date_cell} | Harbour Grand Prix"
    source = {"text": text, "title": "Schedule", "url": "https://example.test/schedule"}
    registry = {"source-1": source}
    block_id, block = next((identifier, value) for identifier, value in source_document_blocks(registry).items()
                           if "|" in value["text"])
    card = source_block_card(block_id, block, source)
    context = research_context("Research the next Harbour race", as_of="2026-10-01T00:00:00Z")
    proposal = {"name": "Harbour Grand Prix", "date": candidate, "card_ids": [block_id]}
    result = apply_subject_candidates(context, [proposal], [card], source_registry=registry)
    assert result["state"] == expected
