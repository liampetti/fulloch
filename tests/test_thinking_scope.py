"""Explicit task/source scope extraction remains deterministic and topic-neutral."""

from core.thinking_scope import named_entities, research_context, source_compatibility


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
        [{"role": "user", "content": "Research the Bathurst 1000 for me."}],
    )

    assert context["temporal_intent"] == "future"
    assert "Bathurst 1000" in context["entities"]
    assert "Bathurst 1000" in context["inherited_entities"]
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
        {"role": "user", "content": "Research the Bathurst 1000 main race on 2026-10-08 using official sources"},
        {"role": "user", "content": task},
    ])
    assert "Bathurst 1000" not in context["entities"]
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
        "Research the imminent Bathurst 1000 main race.",
        as_of="2026-10-11T00:00:00+11:00",
    )

    assert context["edition_year"] == 2026
    assert source_compatibility(
        context, title="Bathurst 1000 2027 calendar", url="https://example.test/2027-calendar",
    )[0] == "compatible"


def test_explicit_category_mismatch_remains_available_for_claim_level_assessment():
    context = research_context("Research the imminent main race category.")

    assert context["requested_category"] == "main race"
    compatibility, reason, _subject = source_compatibility(
        context, title="GR Cup support field", url="https://example.test/gr-cup",
    )
    assert compatibility == "compatible"
    assert reason
