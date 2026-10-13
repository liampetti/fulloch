"""Reader-facing regression tests for the failures found in the Bathurst report."""

import json
import logging
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ThinkingBudget
from core.thinking_coverage import chapter_requirements, initial_requirements, validate_chapter_plan
from core.thinking_evidence import followup_evidence, report_evidence, restore_report_cards
from core.thinking_fact_cards import (
    build_fact_cards,
    report_card_ledger,
    serialize_synthesis_packet,
)
from core.thinking_fallback import evidence_fallback
from core.thinking_scope import research_context
from core.thinking_worker import (
    ThinkingWorker,
    _chapter_evidence_packet,
    _chapter_output_error,
    _clean_report_block,
    _parse_report_audit,
    _structured_chapter_output,
    _temporal_report_audit,
)
from tools.capabilities import ToolCapability
from tools.tool_registry import Param, ThinkingResult, ToolSchema

pytestmark = pytest.mark.usefixtures("synthetic_source_reviewer")


def supported_audit(prompt, cards):
    """Explicit model-auditor fixture for the supported paraphrase scenarios."""
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    assessments = []
    draft = prompt.split("Draft report:\n", 1)[1]
    for location, passage in re.findall(r"\[(P\d+)\]\s*(.*?)(?=\[P\d+\]|\Z)", draft, re.S):
        ids = {value.strip() for match in re.findall(r"<!--\s*cards:\s*(.*?)-->", passage)
               for value in match.split(",")}
        claim = re.sub(r"<!--.*?-->", "", passage, flags=re.S).strip()
        support = [{"card_id": card["id"], "span": span} for card in cards if card["id"] in ids
                   for span in [card.get("statement"), *strings(card.get("fields", {}))] if span]
        if support:
            assessments.append({"location": location, "claim": claim, "status": "supported", "support": support})
    return json.dumps({"verdict": "pass", "issues": [], "assessments": assessments})


def web_job():
    task = "Give me a deep analysis on the Bathurst 1000 race coming up"
    job = BackgroundJob("12345678", JobSnapshot(task))
    job.research_context = research_context(task, as_of="2026-10-01T00:00:00Z")
    sources = [{
        "url": "https://example.com/bathurst-1000", "title": "Bathurst 1000 schedule",
        "source_text": "The Bathurst 1000 starts at 11:30 AEDT on Sunday 11 October 2026. The race covers 161 laps.",
        "published_at": "2026-09-29", "eligible": True,
        # Simulate an old result so the persistence boundary is tested too.
        "source_html": "<script>junk</script>" * 10_000,
        "excluded_blocks": [{"text": "discarded navigation" * 10_000}],
    }, {
        "url": "https://example.com/bathurst-12-hour", "title": "Bathurst 12 Hour schedule",
        "evidence": "The Bathurst 12 Hour includes a different set of drivers and cars.", "eligible": True,
    }]
    for _ in range(2):
        job.record_outcome("external_information", "evidence", "2 sources retrieved.",
                           {"sources": sources}, (), {"type": "web_research", "sources": sources})
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, task, job.research_context,
                                     source_registry=job.source_registry)
    return job


def chapter_response(body, digest, aliases):
    """Model the structured chapter contract with packet-local source aliases."""
    def row(text):
        cited = [value.strip() for group in re.findall(
            r"<!--\s*cards:\s*([^>]+?)\s*-->", text,
        ) for value in group.split(",") if value.strip()]
        visible = re.sub(r"\s*<!--\s*cards:[^>]+-->", "", text).strip()
        return {"text": visible, "cards": cited or aliases[:1]}

    passages = [row(paragraph) for paragraph in re.split(r"\n\s*\n", body.strip()) if paragraph.strip()]
    digest_row = row(digest)
    return json.dumps({"passages": passages, "digest": digest_row["text"],
                       "digest_cards": digest_row["cards"]})


def finance_quote_cards():
    task = "Check the latest NVIDIA quote"
    job = BackgroundJob("quote-audit", JobSnapshot(task))
    retrieved_at = "2026-10-10T08:26:23Z"
    quote = {
        "name": "NVIDIA Corp", "symbol": "NVDA", "exchange": "NASDAQ",
        "currency": "USD", "price": "USD229.28", "numeric_price": 229.28,
        "quoted_at": "Oct 09 2026, 08:30:00 PM UTC-04:00",
        "cache_mode": "provider cache allowed (up to one hour)",
        "source_url": "https://finance.example/NVDA",
        "points": [{"time": "2026-10-09T19:00:00Z", "value": 237.18}],
    }
    job.record_outcome(
        "get_finance_quote", "evidence", "One current NVIDIA quote.", {"quote": quote}, (),
        {"type": "finance_quote", "quote": quote}, retrieved_at=retrieved_at,
    )
    cards = build_fact_cards(
        job.evidence, job.artifacts, task, job.research_context,
        source_registry=job.source_registry,
    )
    return job, cards


def test_saved_evidence_is_small_deduplicated_source_material_not_a_checkpoint():
    job = web_job()
    job.model_budget = {"responses": ["model-call statistics"]}
    job.report_draft = "BAD UNSUPPORTED DRAFT"
    job.report_audit = {"issues": ["audit diagnostics"]}
    before = job.checkpoint()

    evidence = report_evidence(job)
    encoded = json.dumps(evidence)

    assert len(encoded) < 10_000
    assert len(evidence["fact_cards"]) == 3
    assert len(evidence["source_registry"]) == 2
    assert "https://example.com/bathurst-1000" in encoded
    for junk in ("source_html", "excluded_blocks", "discarded navigation", "BAD UNSUPPORTED",
                 "model-call statistics", "audit diagnostics"):
        assert junk not in encoded
    assert job.checkpoint() == before
    restored = restore_report_cards(evidence)
    assert len(report_card_ledger(restored)) == 3
    assert {card["statement"] for card in restored} == {
        card["statement"] for card in report_card_ledger(job.fact_cards)
    }
    assert {card["source_url"] for card in restored} == {
        "https://example.com/bathurst-1000", "https://example.com/bathurst-12-hour",
    }


def test_followup_evidence_uses_compact_saved_facts_and_real_urls():
    evidence = report_evidence(web_job())
    evidence["model_budget"] = {"debug": "DO NOT INCLUDE EXECUTION TRACES"}
    packet = followup_evidence(evidence, "What race start time was found?")
    encoded = json.dumps(packet)
    assert "11:30 AEDT" in encoded and "https://example.com/bathurst-1000" in encoded
    assert "DO NOT INCLUDE EXECUTION TRACES" not in encoded
    assert "source_html" not in encoded


def test_followup_packet_omits_oversized_records_whole_not_price_prefixes():
    job = web_job()
    card = dict(job.fact_cards[0], kind="structured_record", statement="",
                fields={"price": 4_000, "terms": "Restrictions apply. " * 1_000})
    job.fact_cards = [card]
    packet = followup_evidence(report_evidence(job), "What price and restrictions were found?", max_chars=1_000)
    assert packet["claims"] == []


def test_research_packet_keeps_quote_and_provenance_without_full_chart_history():
    points = [{"time": f"2026-10-{day:02d}T16:00:00Z", "value": float(day)}
              for day in range(1, 61)]
    card = {
        "id": "quote-card",
        "kind": "structured_record",
        "statement": "",
        "fields": {
            "name": "Example Corp", "symbol": "EXM", "exchange": "NASDAQ",
            "currency": "USD", "price": "USD229.28", "quoted_at": "2026-10-09T20:00:00Z",
            "cache_mode": "provider cache allowed (up to one hour)",
            "window": "5D", "points": points,
        },
        "evidence_status": "single_source",
        "claim_identity": {"subject": "Example Corp", "claim_type": "market_snapshot"},
        "temporal_role": "provider_snapshot",
        "limitations": "Single provider snapshot.",
        "label": "Market record",
        "source_url": "https://example.com/quote",
        "provenance": {"source_id": "https://example.com/quote"},
    }

    packet = json.loads(_chapter_evidence_packet([card], max_chars=2_400))

    assert len(packet) == 1
    record = packet[0]["record"]
    assert record["price"] == "USD229.28"
    assert record["currency"] == "USD"
    assert record["exchange"] == "NASDAQ"
    assert record["quoted_at"] == "2026-10-09T20:00:00Z"
    assert record["cache_mode"] == "provider cache allowed (up to one hour)"
    assert "points" not in record
    assert record["history_summary"] == {
        "window": "5D", "point_count": 60,
        "first": points[0], "last": points[-1],
        "low": points[0], "high": points[-1],
    }
    assert packet[0]["source"]["source_id"] == "https://example.com/quote"
    assert card["fields"]["points"] == points


def test_extractive_fallback_summary_contains_findings_not_retrieval_diagnostics():
    job = web_job()
    requirements = [{
        "identifier": "timing", "question": "When and where is the Bathurst 1000?",
        "dimension": "Event timing", "state": "open", "evidence_terms": ["starts", "laps"],
        "detail": "Relevant material is available, but direct answer coverage is not verified.",
    }, {
        "identifier": "participants", "question": "Who will participate?",
        "dimension": "Driver pairings", "state": "open", "evidence_terms": ["drivers"],
        "detail": "Missing facts: No card offers enough evidence for src-internal-id.",
    }]
    report = evidence_fallback(job.snapshot.task, job.fact_cards, requirements=requirements)
    summary = report.split("## Summary\n\n")[1].split("\n\n## ")[0]

    assert "11:30 AEDT" in summary and "161 laps" in summary
    assert "Driver pairings" not in report
    assert "[Bathurst 1000 schedule](https://example.com/bathurst-1000)" in report
    assert "[1]" in summary
    assert "Partial answer" not in summary
    assert "What remains unassessed" not in report and "What remains uncertain" not in report
    for junk in ("not verified", "No card", "src-internal-id", "attachment", "evidence.json", "Source time", "single source"):
        assert junk not in report
    assert "12 Hour" not in report


def test_quote_fallback_is_readable_cited_partial_and_omits_chart_fields():
    job, cards = finance_quote_cards()
    quote_card = cards[0]
    quote_card["fields"]["points"] = [
        {"time": f"2026-10-09T{hour:02d}:00:00Z", "value": 237.18 + hour}
        for hour in range(24)
    ]
    requirements = [{
        "identifier": "quote", "question": "What is the latest quote?", "dimension": "Current quote",
        "chapter_id": "chapter-quote", "state": "resolved", "evidence_ids": [quote_card["id"]],
    }, {
        "identifier": "results-period", "question": "What reporting period was covered?",
        "dimension": "Recent results", "chapter_id": "chapter-results", "state": "open",
    }, {
        "identifier": "results-revenue", "question": "What revenue did NVIDIA report?",
        "dimension": "Reported financials", "chapter_id": "chapter-results", "state": "open",
    }, {
        "identifier": "news", "question": "What major recent news matters?",
        "dimension": "Major recent news", "chapter_id": "chapter-news", "state": "open",
    }]
    state = {}

    report = evidence_fallback(
        job.snapshot.task, cards, requirements=requirements, state=state, max_chars=3_000,
    )
    summary = report.split("## Summary\n\n", 1)[1].split("\n\n## ", 1)[0]
    assert "NVIDIA Corp (NVDA:NASDAQ)" in summary
    assert "USD 229.28" in summary and "NASDAQ" in report
    assert "on Oct 09 2026, 08:30:00 PM UTC-04:00" in summary
    assert "retrieved Oct 10 2026, 8:26:23 AM UTC" in summary
    assert "cache qualification: provider cache allowed (up to one hour)" in report
    assert "[1]" in summary
    assert "https://finance.example/NVDA" in report
    assert "points" not in report and "history_summary" not in report and "237.18" not in report
    assert "What remains uncertain" not in report
    assert "Recent results" not in report and "Major recent news" not in report
    assert "What revenue did NVIDIA report?" not in report
    assert len(report) <= 3_000
    assert "What remains unassessed" not in report
    assert "Partial answer" not in report
    assert state["partial_report"] is True and state["display_card_ids"] == [quote_card["id"]]


def test_fallback_summary_covers_distinct_topics_instead_of_only_the_first_chapter():
    job = web_job()
    # Synthetic legacy cards have no new finding binding to the original claim.
    legacy_card = {key: value for key, value in job.fact_cards[0].items() if key != "finding"}
    job.fact_cards.append(dict(legacy_card, id="pairing-card",
                               statement="Alex and Casey are paired in car 15."))
    requirements = [{"identifier": "timing", "question": "Event timing", "evidence_terms": ["starts", "laps"]},
                    {"identifier": "entries", "question": "Driver pairings", "evidence_terms": ["paired"]}]
    report = evidence_fallback(job.snapshot.task, job.fact_cards, requirements=requirements)
    summary = report.split("## Summary\n\n")[1].split("\n\n## ")[0]
    assert "11:30 AEDT" in summary and "Alex and Casey" in summary


def test_evidence_fallback_is_bounded_and_hides_ambiguous_table_metadata():
    job = web_job()
    legacy_card = {key: value for key, value in job.fact_cards[0].items() if key != "finding"}
    claims = [dict(legacy_card, id=f"fallback-{index}",
                   statement=f"A distinct verified Bathurst finding {index} is supported.")
              for index in range(12)]
    ambiguous = dict(
        legacy_card, id="fallback-ambiguous", kind="structured_record", statement="",
        fields={
            "headers": ["Event Information / Results / Race 1"] * 2,
            "cells": ["Round 11", "Round 11"],
            "relationship_status": "ambiguous columns; no relationships inferred",
            "table_caption": "2026 Bathurst 1000",
        },
    )

    state = {}
    report = evidence_fallback(job.snapshot.task, [*claims, ambiguous], state=state)

    assert "What remains unassessed" not in report
    assert "not a completed assessment or prediction" not in report
    assert "relationship_status" not in report
    assert "ambiguous columns" not in report
    assert "headers:" not in report
    assert len(re.findall(r"A distinct verified Bathurst finding \d+", report)) <= 8
    assert "## References" in report

    counts = state["evidence_counts"]
    assert counts["unsynthesized"] == len(state["selected_card_ids"])
    assert counts["display_omitted"] == counts["unsynthesized"] - len(state["display_card_ids"])


def test_specialized_plan_does_not_append_generic_duplicate_chapters():
    task = "Give me a deep analysis on the Bathurst 1000 race coming up"
    context = research_context(task, as_of="2026-10-01T00:00:00Z")
    questions = [
        "What is the timing and scope of the Bathurst 1000?",
        "Who are the principal participants, teams, drivers and their relationships?",
        "What recent results, form and changes inform the Bathurst 1000?",
        "Which evidence-backed factors distinguish likely outcomes for the Bathurst 1000?",
        "What uncertainty, risk or limitation constrains the Bathurst 1000 analysis?",
    ]
    value = {"chapters": [{"title": question[:90], "priority": "high", "questions": [question],
                           "tools": ["external_information"], "queries": [question]} for question in questions]}

    plan = validate_chapter_plan(value, task, initial_requirements(task, context), ["external_information"])

    assert plan is not None and len(plan) == 5
    all_questions = [question for chapter in plan for question in chapter["questions"]]
    assert [question["question"] for question in all_questions if question.get("source_requirement_id") != "subject-identity"] == questions
    assert sum(question.get("source_requirement_id") == "subject-identity" for question in all_questions) == 1
    requirements = chapter_requirements(plan, task, context)
    assert all(item["freshness"] == "fresh" for item in requirements)


def test_explicit_user_obligations_are_not_removed_by_generic_matching():
    task = "Compare photo backup options. Cover privacy and licensing."
    context = research_context(task)
    value = {"chapters": [{"title": "Mobile support", "priority": "high",
                           "questions": ["What mobile backup support is available?"],
                           "tools": ["lookup"], "queries": ["photo mobile backup"]}]}
    requirements = initial_requirements(task, context)

    plan = validate_chapter_plan(value, task, requirements, ["lookup"])

    questions = {question["question"] for chapter in plan for question in chapter["questions"]}
    assert all(item["question"] in questions for item in requirements)


def test_coverage_packet_preserves_relevance_order_not_url_hash_order():
    cards = [{"id": identifier, "statement": "The source supports backup.", "fields": {}}
             for identifier in ("z-relevant", "a-incidental")]
    packet = json.loads(_chapter_evidence_packet(cards))
    assert [row["id"] for row in packet] == ["z-relevant", "a-incidental"]


def test_model_table_view_preserves_values_but_not_duplicate_dom_diagnostics():
    job = web_job()
    card = dict(job.fact_cards[0], kind="structured_record", statement="15 | Alex / Casey")
    card.pop("finding")  # This synthetic table is a legacy record, not the original assertion.
    card["fields"] = {"table_index": 0, "row_index": 2, "column_group": 0,
                      "columns": {"Car": "15", "Drivers": "Alex / Casey", "Team": None},
                      "headers": ["Car", "Drivers", "Team"], "cells": ["15", "Alex / Casey", None],
                      "source_cells": [{"text": "Alex / Casey", "declared_rowspan": "1"}]}

    packet = serialize_synthesis_packet([card], job.source_registry)

    assert packet["claims"][0]["record"] == {"columns": {"Car": "15", "Drivers": "Alex / Casey", "Team": None}}
    assert "source_cells" in card["fields"]  # retained original is not mutated
    job.fact_cards = [card]
    restored = restore_report_cards(report_evidence(job))[0]
    assert restored["fields"] == packet["claims"][0]["record"]
    assert restored["provenance"]["table"] == 0
    assert restored["provenance"]["row"] == 2
    assert restored["provenance"]["group"] == 0


def test_compact_evidence_preserves_provider_scope_and_offer_constraints():
    job = BackgroundJob("provider", JobSnapshot("Compare current return flights"))
    scope = "Only returned provider offers are comparable; baggage fees are not included."
    job.record_outcome("flights", "evidence", scope, {}, (), {
        "type": "flight_search", "route": {"origin": "HAM", "destination": "SYD"},
        "departure_date": "2027-01-17", "return_date": "2027-03-03", "currency": "EUR",
        "offers": [{"price": 1_200, "airlines": ["Example"], "departure": {"id": "HAM"},
                    "arrival": {"id": "SYD"}, "baggage": None}],
    })
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, source_registry=job.source_registry)
    original = job.fact_cards[0]
    restored = restore_report_cards(report_evidence(job))[0]
    assert restored["fields"] == original["fields"]
    assert restored["scope"] == scope
    for key in ("route", "departure_date", "return_date", "currency"):
        assert restored["context"][key] == original["context"][key]
    assert restored["claim_eligibility"] == {
        key: value for key, value in original["claim_eligibility"].items() if value != []
    }
    report = evidence_fallback(job.snapshot.task, [restored])
    assert "currency: EUR" in report and "departure_date: 2027-01-17" in report


def test_future_schedule_is_not_rejected_as_past_because_it_says_to_be_held():
    job = web_job()
    card = dict(job.fact_cards[0], statement="The race is scheduled to be held on 2026-10-11.",
                temporal_role="document", context={"published_at": "2020-01-01"},
                claim_identity={"period": "2020-01-01"})
    paragraph = "The upcoming race is scheduled to be held on 2026-10-11. <!-- cards: " + card["id"] + " -->"
    assert _temporal_report_audit(paragraph, [card], job.research_context) == []


def test_a_completed_prior_race_still_cannot_be_framed_as_upcoming():
    job = web_job()
    card = dict(job.fact_cards[0], statement="The race was held on 2025-10-12.")
    paragraph = "The upcoming race is scheduled to take place on 2025-10-12. <!-- cards: " + card["id"] + " -->"
    assert _temporal_report_audit(paragraph, [card], job.research_context)


@pytest.mark.parametrize("quote_text", ["USD229.28", "USD 229.28"])
def test_money_and_quote_timestamps_are_field_aware_and_format_normalized(quote_text):
    job, cards = finance_quote_cards()
    card_id = cards[0]["id"]
    report = (
        f"The NVIDIA quote was {quote_text}, quoted at Oct 09 2026, 08:30:00 PM UTC-04:00; "
        f"retrieved at 2026-10-10T08:26:23Z. <!-- cards: {card_id} -->"
    )
    worker = ThinkingWorker(
        object(), None, MagicMock(), lambda *_args, **kwargs: supported_audit(kwargs["system_prompt"], cards),
        lambda: {}, None, lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, cards, [])

    clean, bindings, issues, _raw = worker._audit_candidate(
        job, report, cards, ledger, ThinkingBudget(worker.model, job.model_budget),
        lambda: False, allow_revision=False,
    )

    assert clean and bindings
    assert not issues
    assert json.loads(job.synthesis_state["passage_reviews"][0]["audit"])["assessments"]


@pytest.mark.parametrize(("reported", "issue_kind"), [
    ("The quote was EUR 229.28.", "monetary_currency"),
    ("The quote was USD 229.29.", "monetary_value"),
    ("The latest quote was USD 237.18.", "monetary_value"),
    ("The quote was USD 229.28, quoted at 2026-10-09T21:00:00Z.", "timestamp_role"),
    ("The quote was USD 229.28, retrieved at 2026-10-10T08:27:23Z.", "timestamp_role"),
    ("The quote was USD 229.28, quoted at 2026-10-10T08:26:23Z.", "timestamp_role"),
    ("The quote was USD 229.28, retrieved at 2026-10-09T20:00:00Z.", "timestamp_role"),
])
def test_field_aware_audit_rejects_changed_money_or_timestamp_even_when_model_passes(reported, issue_kind):
    job, cards = finance_quote_cards()
    card_id = cards[0]["id"]
    report = f"{reported} <!-- cards: {card_id} -->"
    worker = ThinkingWorker(
        object(), None, MagicMock(), lambda *_args, **_kwargs: '{"verdict":"pass","issues":[]}',
        lambda: {}, None, lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, cards, [])

    _clean, bindings, issues, _raw = worker._audit_candidate(
        job, report, cards, ledger, ThinkingBudget(worker.model, job.model_budget),
        lambda: False, allow_revision=False,
    )

    assert not bindings
    assert any(issue_kind in issue for issue in issues)
    assert job.synthesis_state["passage_reviews"][0]["audit"] == '{"verdict":"pass","issues":[]}'


def test_live_audit_correction_requires_exact_draft_and_cited_source_anchors():
    job = web_job()
    card = job.fact_cards[0]
    worker = ThinkingWorker(
        object(), None, MagicMock(), lambda *_args, **_kwargs: '{"verdict":"pass","issues":[]}',
        lambda: {}, None, lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, [card], [])
    draft_quote = "The Bathurst 1000 starts at 11:00 AEDT on Sunday 11 October 2026."
    source_quote = card["statement"]
    report = f"{draft_quote} <!-- cards: {card['id']} -->"
    anchored = {
        "verdict": "revise", "issues": [{
            "location": "P1", "draft_quote": draft_quote,
            "source_id": card["provenance"]["source_id"], "source_quote": source_quote,
            "type": "literal_mismatch", "reason": "The start time differs from the source.",
            "fix": "Use the cited 11:30 AEDT start time.",
        }],
    }

    parsed = _parse_report_audit(json.dumps(anchored), report, [card], ledger)

    assert len(parsed) == 1 and "P1" in parsed[0]
    assert source_quote in parsed[0] and "Use the cited 11:30" in parsed[0]
    for invalid in (
        {**anchored, "issues": [{**anchored["issues"][0], "source_quote": "invented source text"}]},
        {**anchored, "issues": [{**anchored["issues"][0], "source_id": "unrelated-source"}]},
        {**anchored, "issues": [{**anchored["issues"][0], "draft_quote": "not in the report"}]},
    ):
        assert _parse_report_audit(json.dumps(invalid), report, [card], ledger) == []


def test_unsupported_cross_section_conclusion_is_rejected_without_mutating_accepted_sections():
    job = web_job()
    card = job.fact_cards[0]
    accepted_body = "The source schedules the race for Sunday."
    job.chapter_plan = [{"id": "chapter-01", "body": accepted_body, "analysis_state": "complete"}]
    draft = f"Alpha is a safer choice than Beta. <!-- cards: {card['id']} -->"
    issue = {
        "location": "P1", "draft_quote": draft,
        "source_id": card["provenance"]["source_id"],
        "source_quote": card["statement"], "type": "unsupported_comparison",
        "reason": "The cited source makes no comparison with Beta.",
        "fix": "Remove the unsupported comparison.",
    }
    worker = ThinkingWorker(
        object(), None, MagicMock(),
        lambda *_args, **_kwargs: json.dumps({"verdict": "revise", "issues": [issue]}),
        lambda: {}, None, lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, [card], [])
    clean, bindings, issues, _raw = worker._audit_candidate(
        job, draft, [card], ledger, ThinkingBudget(worker.model, job.model_budget),
        lambda: False, allow_revision=False,
    )

    assert "Alpha is a safer choice than Beta" not in clean
    assert not bindings
    assert issues
    assert job.chapter_plan[0]["body"] == accepted_body


def test_bogus_audit_correction_cannot_rewrite_an_exact_retained_finding():
    job = web_job()
    card = job.fact_cards[0]
    source_quote = card["statement"]
    response = json.dumps({"verdict": "revise", "issues": [{
        "location": "P1", "draft_quote": source_quote,
        "source_id": card["provenance"]["source_id"], "source_quote": "invented source quotation",
        "type": "literal_mismatch", "reason": "The source says a different time.",
        "fix": "Change the start time to 10:00 AEDT.",
    }]})
    calls = []
    worker = ThinkingWorker(
        object(), None, MagicMock(),
        lambda _model, **kwargs: calls.append(kwargs["user_prompt"]) or response,
        lambda: {}, None, lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, [card], [])

    clean, bindings, issues, _raw = worker._audit_candidate(
        job, f"{source_quote} <!-- cards: {card['id']} -->", [card], ledger,
        ThinkingBudget(worker.model, job.model_budget), lambda: False,
    )

    assert source_quote in clean
    assert "10:00 AEDT" not in clean
    assert bindings
    assert calls == ["Audit the draft report now."]
    assert not issues


def test_anchored_audit_repair_is_rechecked_against_the_original_source():
    job = web_job()
    card = job.fact_cards[0]
    source_quote = card["statement"]
    draft_quote = "The Bathurst 1000 starts at 11:00 AEDT on Sunday 11 October 2026."
    issue = {
        "location": "P1", "draft_quote": draft_quote,
        "source_id": card["provenance"]["source_id"], "source_quote": source_quote,
        "type": "literal_mismatch", "reason": "The start time differs from the source.",
        "fix": "Use the cited 11:30 AEDT start time.",
    }
    audit_prompts = []

    def generate(_model, **kwargs):
        stage = kwargs["user_prompt"]
        if stage == "Audit the draft report now.":
            audit_prompts.append(kwargs["system_prompt"])
            return json.dumps({"verdict": "revise", "issues": [issue]}) if len(audit_prompts) == 1 else (
                '{"verdict":"pass","issues":[]}'
            )
        if stage == "Revise this section only.":
            return f"{source_quote} <!-- cards: {card['id']} -->"
        raise AssertionError(stage)

    worker = ThinkingWorker(
        object(), None, MagicMock(), generate, lambda: {}, None,
        lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, [card], [])

    clean, bindings, issues, _raw = worker._audit_candidate(
        job, f"{draft_quote} <!-- cards: {card['id']} -->", [card], ledger,
        ThinkingBudget(worker.model, job.model_budget), lambda: False,
    )

    assert source_quote in clean and draft_quote not in clean
    assert bindings and not issues
    assert len(audit_prompts) == 2
    assert source_quote in audit_prompts[1]
    assert source_quote in audit_prompts[1].split("Draft report:", 1)[1]


def test_failed_citation_binding_is_checkpointed_without_logging_marker_text(caplog):
    job = web_job()
    card = job.fact_cards[0]
    forged_id = "forged-source-card"
    report = f"{card['statement']} <!-- cards: {forged_id} -->"
    worker = ThinkingWorker(
        object(), None, MagicMock(), lambda *_args, **_kwargs: '{"verdict":"pass","issues":[]}',
        lambda: {}, None, lambda *_args: "", threading.Lock(), 2,
    )
    _prompt, ledger = worker._packet_prompt(job, [card], [])

    with caplog.at_level(logging.INFO):
        _clean, bindings, issues, _raw = worker._audit_candidate(
            job, report, [card], ledger, ThinkingBudget(worker.model, job.model_budget),
            lambda: False, allow_revision=False,
        )

    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    assert not bindings and issues
    assert any(
        event["category"] == "citation_failure" and event["reason"] == "unknown_card_id"
        for event in restored.stage_log
    )
    assert forged_id not in caplog.text
    assert card["statement"] not in caplog.text


def test_live_chapter_workflow_accepts_fenced_outputs_and_produces_a_cited_summary():
    claim = "The provider supports local mobile backup, but requires a Wi-Fi connection."
    finding = "Local mobile backup is supported, but only with a Wi-Fi connection."
    source = {"url": "https://docs.example/backup", "title": "Mobile backup guide",
              "evidence": claim, "eligible": True}
    schema = ToolSchema("lookup", "Read provider documentation.", [Param("query", True, None)], thinking_outcome=True)
    capability = ToolCapability("lookup", MagicMock(return_value=ThinkingResult(
        claim, evidence={"sources": [source]}, scope="Provider documentation.",
        artifact={"type": "web_research", "sources": [source]},
    )), "native", 1, lambda result: result, "read")
    stages = []

    def generate(_model, **kwargs):
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        stages.append(stage)
        if stage == "Return the chapter plan object.":
            value = {"chapters": [{"title": "Mobile backup support", "priority": "high",
                "questions": ["Explain mobile backup support."], "tools": ["lookup"],
                "queries": ["provider mobile backup support"]}]}
        elif stage == "Choose the next read-only action.":
            value = {"tool": "lookup", "args": ["provider mobile backup support"]}
        elif stage == "Check direct answer coverage.":
            question = re.search(r'"question_id":"([^"]+)"', prompt)[1]
            card = re.search(r'"id":"(src-[^"]+)"', prompt)[1]
            value = {"answers": [{"question_id": question, "answered": True,
                                  "card_ids": [card], "missing_facts": ""}]}
        elif stage == "Extract the chapter's compact facts.":
            value = {"cards": [{"alias": "C1", "fact": claim}], "disagreements": [], "missing_information": []}
        elif stage == "Write the chapter body and digest.":
            value = {"passages": [{"text": finding, "cards": ["C1"]}],
                     "digest": finding, "digest_cards": ["C1"]}
        elif stage == "Assemble the opening summary and closing conclusions.":
            identifier = re.search(r'"supporting_card_ids":\["([^"]+)"', prompt)[1]
            value = {"summary": finding + f" <!-- cards: {identifier} -->", "conclusions": ""}
        elif stage == "Audit the draft report now.":
            value = json.loads(supported_audit(prompt, job.fact_cards))
        else:
            raise AssertionError(f"Unexpected stage/retry: {stage}")
        return "```json\n" + json.dumps(value) + "\n```"

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {"lookup": capability},
                            SimpleNamespace(_schemas={"lookup": schema}, canonical_name=lambda name: name),
                            lambda *_: "", threading.Lock(), 2)
    job = BackgroundJob("chapter-test", JobSnapshot("Explain mobile backup support."))

    report, _ = worker.run(job, lambda: False)

    assert finding in report and "[1]" in report
    assert "## Mobile backup support\n\n" + finding in report
    assert report.count(finding) == 2  # Summary overlap must not erase the chapter.
    assert job.report_bindings[0]["source_url"] == source["url"]
    assert job.chapter_plan[0]["analysis_state"] == "complete"
    assert not job.synthesis_state.get("limited_evidence_fallback")
    assert job.model_budget["stages"]["chapter_analysis"]["calls"] == 1
    assert "Repair the chapter output." not in stages
    assert not job.synthesis_state["passage_reviews"][0]["repairs"]


def test_inherited_quote_flows_through_additional_research_into_multiple_cited_chapters():
    fixture_path = Path(__file__).parent / "fixtures/thinking/inherited_quote_research_responses.json"
    responses = json.loads(fixture_path.read_text(encoding="utf-8"))
    task = "Brief NVIDIA's latest quote, recent results, and recent news."
    retrieved_at = "2026-10-10T08:26:23Z"
    quote = {
        "name": "NVIDIA Corp", "symbol": "NVDA", "exchange": "NASDAQ",
        "currency": "USD", "price": "USD229.28", "numeric_price": 229.28,
        "quoted_at": "Oct 09 2026, 08:30:00 PM UTC-04:00",
        "cache_mode": "provider cache allowed (up to one hour)",
        "source_url": "https://finance.example/NVDA",
        "points": [{"time": "2026-10-09T19:00:00Z", "value": 237.18}],
    }
    handoff = {
        "original_request": task,
        "research_task": task,
        "observations": [{
            "tool": "get_finance_quote", "args": [], "text": "One current NVIDIA quote.",
            "status": "evidence", "scope": "One current NVIDIA quote snapshot.",
            "evidence": {"quote": quote},
            "artifact": {"type": "finance_quote", "quote": quote},
            "retrieved_at": retrieved_at,
        }],
    }
    source_records = {
        "results": {
            "url": "https://investors.example/nvidia/q2-results",
            "title": "NVIDIA reports fiscal 2027 second-quarter results",
            "source_text": (
                "NVIDIA reported revenue of USD 46.7 billion for its fiscal 2027 second quarter, "
                "ended July 27, 2026."
            ),
            "published_at": "2026-08-19", "eligible": True,
        },
        "news": {
            "url": "https://news.example/nvidia/ai-partnership",
            "title": "NVIDIA announces AI infrastructure partnership",
            "source_text": (
                "NVIDIA reported a new AI infrastructure partnership with two cloud computing companies "
                "on October 8, 2026, expanding deployment capacity for its accelerator platform."
            ),
            "published_at": "2026-10-08", "eligible": True,
        },
    }
    external_calls = []

    def retrieve(args, _kwargs):
        query = args[0]
        external_calls.append(query)
        key = "results" if "quarterly" in query.casefold() else "news"
        source = source_records[key]
        return ThinkingResult(
            "One relevant source retrieved.", status="evidence", scope=f"One NVIDIA {key} source.",
            evidence={"sources": [source]},
            artifact={"type": "web_research", "sources": [source]},
        )

    external_schema = ToolSchema("external_information", "Web retrieval", [
        Param("query", False, "get me the latest news stories"),
        Param("profile", False, "conversation"), Param("category", False, "auto"),
        Param("time_range", False, ""), Param("page", False, 1),
    ], thinking_outcome=True)
    quote_schema = ToolSchema("get_finance_quote", "Finance quote", [], thinking_outcome=True)
    external_capability = ToolCapability(
        "external_information", retrieve, "native", 1, str, "read",
    )
    quote_invoke = MagicMock()
    quote_capability = ToolCapability("get_finance_quote", quote_invoke, "native", 1, str, "read")
    chapter_action_calls = {}
    observed_stages = []

    def generate(_model, **kwargs):
        stage = kwargs["user_prompt"]
        prompt = kwargs["system_prompt"]
        observed_stages.append(stage)
        if stage == "Return the chapter plan object.":
            return responses["chapter_plan"]
        if stage == "Check direct answer coverage.":
            return responses["coverage"]
        if stage == "Choose the next read-only action.":
            chapter = json.loads(re.search(r"Chapter: (\{[^\n]+\})", prompt)[1])["title"]
            index = chapter_action_calls.get(chapter, 0)
            chapter_action_calls[chapter] = index + 1
            fixture_actions = responses["actions"][chapter]
            return fixture_actions[min(index, len(fixture_actions) - 1)]
        if stage == "Write the chapter body and digest.":
            chapter = re.search(r"\nChapter: ([^\n]+)", prompt)[1]
            return responses["chapter_analysis"][chapter]
        if stage == "Audit the draft report now.":
            return supported_audit(prompt, job.fact_cards)
        if stage in {"Assemble the opening summary and closing conclusions.",
                     "Repair final assembly output."}:
            return responses["final_assembly"]
        raise AssertionError(f"Unexpected fixture stage: {stage}")

    worker = ThinkingWorker(
        object(), None, MagicMock(), generate,
        lambda: {
            "external_information": external_capability,
            "get_finance_quote": quote_capability,
        },
        SimpleNamespace(
            _schemas={"external_information": external_schema, "get_finance_quote": quote_schema},
            canonical_name=lambda name: name,
        ),
        lambda *_: "", threading.Lock(), 2,
    )
    job = BackgroundJob("inherited-quote-e2e", JobSnapshot(task, handoff=handoff))
    job.research_context = research_context(task, as_of="2026-10-10T12:00:00Z")

    report, _findings = worker.run(job, lambda: False)

    assert quote_invoke.call_count == 0
    assert len(external_calls) == 2
    assert any("quarterly" in query.casefold() for query in external_calls)
    assert any("news" in query.casefold() for query in external_calls)
    assert {chapter["title"] for chapter in job.chapter_plan} == {
        "Latest NVIDIA quote", "Recent NVIDIA results", "Major recent NVIDIA news",
    }
    assert "USD 229.28" in report
    assert "Oct 09 2026, 08:30:00 PM UTC-04:00" in report
    assert "cache data up to one hour old" in report
    assert "USD 46.7 billion" in report
    assert "AI infrastructure partnership" in report
    titles = (
        "Latest NVIDIA quote", "Recent NVIDIA results", "Major recent NVIDIA news",
    )
    assert all(f"## {title}" in report for title in titles)
    for title in titles:
        chapter_text = report.split(f"## {title}\n\n", 1)[1].split("\n\n## ", 1)[0]
        assert re.search(r"\[\d+(?:,\s*\d+)*\]", chapter_text)
    assert {binding["source_url"] for binding in job.report_bindings} == {
        "https://finance.example/NVDA",
        "https://investors.example/nvidia/q2-results",
        "https://news.example/nvidia/ai-partnership",
    }
    assert {
        "Return the chapter plan object.", "Choose the next read-only action.",
        "Write the chapter body and digest.", "Audit the draft report now.",
    } <= set(observed_stages)


def analytical_chapter_job():
    job = BackgroundJob("multi-chapter", JobSnapshot("Compare local mobile backup, privacy and licensing"))
    job.research_context = research_context(job.snapshot.task, as_of="2026-10-01T00:00:00Z")
    topics = {
        "Mobile support": [
            "The application supports automatic mobile uploads over Wi-Fi. It requires local server authentication. "
            "Uploads pause during disconnection and resume when the network returns.",
            "Background uploads remain subject to the phone's battery restrictions. "
            "The vendor does not guarantee immediate background transfers while the application is closed.",
        ],
        "Privacy boundaries": [
            "The server stores encrypted photographs locally. The household retains the encryption keys. "
            "The vendor does not receive those keys or access the stored photographs.",
            "Encryption protects stored photographs, but does not replace a separate recovery copy. "
            "A lost encryption key prevents access to the stored photographs.",
        ],
        "Licensing limits": [
            "The application is licensed for household use without a subscription. "
            "Commercial deployments require a separate license. The household license excludes business use.",
            "The commercial license includes email support. The household license does not include that service. "
            "The vendor publishes no fixed response-time commitment for either license.",
        ],
    }
    plan = {"chapters": []}
    for index, (title, paragraphs) in enumerate(topics.items()):
        url = f"https://docs.example/topic-{index}"
        text = title + "\n" + "\n".join(paragraphs)
        source = {"url": url, "title": title, "source_text": text, "eligible": True}
        job.record_outcome("lookup", "evidence", "Provider documentation", {"sources": [source]}, (),
                           {"type": "web_research", "sources": [source]})
        plan["chapters"].append({"title": title, "priority": "high", "questions": [title + " and qualifications?"],
                                 "tools": ["lookup"], "queries": [title]})
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, source_registry=job.source_registry)
    job.chapter_plan = validate_chapter_plan(plan, job.snapshot.task, [], ["lookup"])
    job.evidence_requirements = chapter_requirements(job.chapter_plan, job.snapshot.task, job.research_context)
    for index, (chapter, requirement) in enumerate(zip(job.chapter_plan, job.evidence_requirements, strict=True)):
        ids = [card["id"] for card in job.fact_cards if card["source_url"].endswith(f"topic-{index}")]
        chapter["card_ids"] = ids
        requirement.update(state="resolved", evidence_ids=ids)
    return job, topics


@pytest.mark.parametrize("fail_final", [False, True])
def test_multiple_substantive_chapters_survive_summary_and_digest_overlap(fail_final):
    job, topics = analytical_chapter_job()
    writing_calls = []

    def generate(_model, **kwargs):
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        if stage == "Write the chapter body and digest.":
            title = re.search(r"\nChapter: ([^\n]+)", prompt)[1]
            view = json.loads(prompt.split("Source claims, records and contextual paragraphs: ", 1)[1])
            assert view["passages"] and topics[title][0] in json.dumps(view["passages"])
            aliases = [row["alias"] for row in view["cards"]]
            marker = " <!-- cards: " + ", ".join(aliases) + " -->"
            writing_calls.append(title)
            body = "\n\n".join(paragraph + marker for paragraph in topics[title])
            return chapter_response(body, topics[title][0] + marker, aliases)
        if stage in {"Assemble the opening summary and closing conclusions.", "Repair final assembly output."}:
            if fail_final:
                return ""
            rows = json.loads(prompt.split("Chapter digests and their supporting source-card references: ", 1)[1])
            return json.dumps({"summary": "The cards establish the following findings. " + rows[0]["digest"],
                               "conclusions": rows[1]["digest"]})
        if stage == "Audit the draft report now.":
            return supported_audit(prompt, job.fact_cards)
        raise AssertionError(stage)

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 2)
    report, bindings = worker._synthesise_chapters(job, job.fact_cards,
                                                ThinkingBudget(worker.model, job.model_budget), lambda: False)
    assert writing_calls == list(topics)
    assert report.startswith("## Summary\n\n")
    for title, paragraphs in topics.items():
        section = report.split(f"## {title}\n\n", 1)[1].split("\n\n## ", 1)[0]
        assert all(paragraph in section for paragraph in paragraphs)
        assert len(section) > 300
    assert len({binding["source_url"] for binding in bindings}) == 3
    assert not job.synthesis_state["partial_report"]
    assert not re.search(r"\bcards?\b|src-[a-f0-9]|<!--", report)
    assert "card_extraction" not in job.model_budget["stages"]


def test_internal_source_terms_are_removed_without_dropping_factual_paragraphs_or_markers():
    text = ("The cards establish the race date. The race runs on Sunday 11 October. "
            "<!-- cards: src-0123456789abcdef-c-0-20 -->")
    cleaned = _clean_report_block(text)
    assert "The race runs on Sunday 11 October." in cleaned
    assert "the sources establish" in cleaned.casefold()
    assert "<!-- cards: src-0123456789abcdef-c-0-20 -->" in cleaned
    assert "The cards establish" not in cleaned
    assert _clean_report_block("The graphics cards require adequate cooling. The cards use two fans.") == (
        "The graphics cards require adequate cooling. The cards use two fans.")


def test_digest_word_limit_repair_targets_the_digest_not_the_chapter_body():
    body = "A substantive chapter paragraph with preserved qualifications."
    error = _chapter_output_error({"body": body, "digest": "finding " * 121})
    assert "digest exceeds 120" in error and "not the analytical body" in error
    assert _chapter_output_error({"body": body, "digest": "A finding. <!-- cards: C1, C2 -->"}) == ""
    assert "complete JSON" in _chapter_output_error(None)


def test_structured_chapter_output_attaches_only_explicit_packet_aliases():
    aliases = {"C1": "src-one", "C2": "src-two"}
    value = {
        "passages": [
            {"text": "The provider supports background uploads.", "cards": ["C1"]},
            {"text": "Transfers may pause while disconnected.", "cards": ["C1", "C2"]},
        ],
        "digest": "Uploads resume after connectivity returns.", "digest_cards": ["C2"],
    }

    parsed, error = _structured_chapter_output(value, aliases)

    assert not error
    assert parsed == {
        "body": (
            "The provider supports background uploads. <!-- cards: C1 -->\n\n"
            "Transfers may pause while disconnected. <!-- cards: C1, C2 -->"
        ),
        "digest": "Uploads resume after connectivity returns. <!-- cards: C2 -->",
    }


@pytest.mark.parametrize("value", [
    {"body": "Legacy body. <!-- cards: C1 -->", "digest": "Legacy digest. <!-- cards: C1 -->"},
    {"passages": [{"text": "Unsupported citation.", "cards": ["invented"]}],
     "digest": "A digest.", "digest_cards": ["C1"]},
    {"passages": [{"text": "First paragraph.\n\nSecond paragraph.", "cards": ["C1"]}],
     "digest": "A digest.", "digest_cards": ["C1"]},
    {"passages": [{"text": "Text <!-- cards: C1 -->", "cards": ["C1"]}],
     "digest": "A digest.", "digest_cards": ["C1"]},
    {"passages": [{"text": "No support array.", "cards": []}],
     "digest": "A digest.", "digest_cards": ["C1"]},
])
def test_structured_chapter_output_rejects_model_authored_or_unknown_citations(value):
    parsed, error = _structured_chapter_output(value, {"C1": "src-one"})

    assert parsed is None
    assert error


def test_failed_digest_repair_preserves_the_independently_audited_chapter_body():
    job, topics = analytical_chapter_job()
    mobile_facts = topics["Mobile support"]

    def generate(_model, **kwargs):
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        if stage in {"Write the chapter body and digest.", "Repair the chapter output."}:
            title = re.search(r"\nChapter: ([^\n]+)", prompt)[1]
            view = json.loads(prompt.split("Source claims, records and contextual paragraphs: ", 1)[1])
            marker = " <!-- cards: " + ", ".join(row["alias"] for row in view["cards"]) + " -->"
            body = "\n\n".join(fact + marker for fact in topics[title])
            digest = ("finding " * 121 + marker
                      if title == "Mobile support" else topics[title][0] + marker)
            return chapter_response(body, digest, [row["alias"] for row in view["cards"]])
        if stage == "Assemble the opening summary and closing conclusions.":
            rows = json.loads(prompt.split("Chapter digests and their supporting source-card references: ", 1)[1])
            assert len(rows) == 2  # The body survives; only its invalid digest is omitted.
            return json.dumps({"summary": rows[0]["digest"], "conclusions": ""})
        if stage == "Audit the draft report now.":
            return supported_audit(prompt, job.fact_cards)
        raise AssertionError(stage)

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 2)
    report, bindings = worker._synthesise_chapters(
        job, job.fact_cards, ThinkingBudget(worker.model, job.model_budget), lambda: False,
    )

    mobile = next(chapter for chapter in job.chapter_plan if chapter["title"] == "Mobile support")
    assert mobile["analysis_state"] in {"complete", "partial"}
    assert mobile["digest"] == ""
    assert all(fact in report for fact in mobile_facts)
    assert mobile["id"] in job.synthesis_state["digest_omitted_chapters"]
    assert bindings


def test_failed_chapter_keeps_qualified_source_facts_instead_of_disappearing():
    job, topics = analytical_chapter_job()

    def generate(_model, **kwargs):
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        if stage in {"Write the chapter body and digest.", "Repair the chapter output."}:
            title = re.search(r"\nChapter: ([^\n]+)", prompt)[1]
            if title == "Licensing limits":
                return '{"body":"truncated'
            view = json.loads(prompt.split("Source claims, records and contextual paragraphs: ", 1)[1])
            marker = " <!-- cards: " + ", ".join(row["alias"] for row in view["cards"]) + " -->"
            return chapter_response("\n\n".join(text + marker for text in topics[title]),
                                    topics[title][0] + marker,
                                    [row["alias"] for row in view["cards"]])
        if stage == "Assemble the opening summary and closing conclusions.":
            rows = json.loads(prompt.split("Chapter digests and their supporting source-card references: ", 1)[1])
            assert len(rows) == 2  # Extracts do not masquerade as audited digests.
            return json.dumps({"summary": rows[0]["digest"], "conclusions": ""})
        if stage == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        raise AssertionError(stage)

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 2)
    report, bindings = worker._synthesise_chapters(job, job.fact_cards,
                                                ThinkingBudget(worker.model, job.model_budget), lambda: False)
    section = report.split("## Licensing limits\n\n", 1)[1]
    assert "**Unverified source facts:**" not in section
    assert "Commercial deployments require a separate license." in section
    assert "The household license excludes business use." in section
    # The partial-answer commentary is allowed only in the trailing section,
    # never in the summary the reader sees first.
    summary_text = report.split("## Summary\n\n", 1)[1].split("\n\n## ", 1)[0]
    assert "**Partial answer:**" not in summary_text
    assert "**Partial answer:**" not in report
    assert job.chapter_plan[-1]["analysis_state"] == "evidence_only"
    assert job.synthesis_state["partial_report"]
    assert job.synthesis_state["extractive_only_card_ids"]
    assert set(job.synthesis_state["extractive_only_card_ids"]) <= set(job.synthesis_state["unsynthesized_card_ids"])
    assert len({binding["source_url"] for binding in bindings}) == 3


def test_low_priority_failed_chapter_is_still_declared_partial():
    job, topics = analytical_chapter_job()
    job.evidence_requirements[-1]["importance"] = "low"

    def generate(_model, **kwargs):
        stage, prompt = kwargs["user_prompt"], kwargs["system_prompt"]
        if stage == "Write the chapter body and digest.":
            title = re.search(r"\nChapter: ([^\n]+)", prompt)[1]
            if title == "Licensing limits":
                return "incomplete output"
            view = json.loads(prompt.split("Source claims, records and contextual paragraphs: ", 1)[1])
            marker = " <!-- cards: " + ", ".join(row["alias"] for row in view["cards"]) + " -->"
            return chapter_response("\n\n".join(text + marker for text in topics[title]),
                                    topics[title][0] + marker,
                                    [row["alias"] for row in view["cards"]])
        if stage == "Assemble the opening summary and closing conclusions.":
            rows = json.loads(prompt.split("Chapter digests and their supporting source-card references: ", 1)[1])
            return json.dumps({"summary": rows[0]["digest"], "conclusions": ""})
        if stage == "Audit the draft report now.":
            return '{"verdict":"pass","issues":[]}'
        if stage == "Repair the chapter output.":
            return "still incomplete"
        raise AssertionError(stage)

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 2)
    report, _ = worker._synthesise_chapters(job, job.fact_cards,
                                         ThinkingBudget(worker.model, job.model_budget), lambda: False)
    summary_text = report.split("## Summary\n\n", 1)[1].split("\n\n## ", 1)[0]
    assert "**Partial answer:**" not in summary_text
    assert "**Partial answer:**" not in report
    assert job.synthesis_state["partial_report"]


def test_when_every_chapter_fails_fallback_is_short_evidence_only_and_cited():
    job, _topics = analytical_chapter_job()

    def generate(_model, **kwargs):
        if kwargs["user_prompt"] in {"Write the chapter body and digest.", "Repair the chapter output."}:
            return "not valid chapter JSON"
        raise AssertionError(f"Unexpected stage after all chapters fail: {kwargs['user_prompt']}")

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 2)
    report, bindings = worker._synthesise_chapters(
        job, job.fact_cards, ThinkingBudget(worker.model, job.model_budget), lambda: False,
    )

    assert "What remains unassessed" not in report
    assert "not a completed assessment or prediction" not in report
    assert 0 < len(bindings) <= 8
    assert len({binding["source_url"] for binding in bindings}) == 3
    assert job.synthesis_state["limited_evidence_fallback"]
    assert len(job.synthesis_state["display_card_ids"]) <= 8
    assert all(chapter["analysis_state"] == "missing" for chapter in job.chapter_plan)
    assert "<!-- cards:" not in report and "src-" not in report


def test_many_short_indexed_claims_do_not_spend_model_calls_rephrasing_each_sentence():
    job = web_job()
    cards = [dict(job.fact_cards[0], id=f"short-{index}") for index in range(122)]
    chapter = {"id": "chapter-01", "title": "Timing"}

    def generate(*_args, **_kwargs):
        raise AssertionError("Short exact facts are already compact.")

    worker = ThinkingWorker(object(), None, MagicMock(), generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 2)
    facts, _, _ = worker._extract_chapter_cards(job, chapter, cards,
                                               ThinkingBudget(worker.model, job.model_budget), lambda: False)
    assert len(facts) == 122 and set(facts.values()) == {cards[0]["statement"]}
    assert job.model_budget["calls"] == 0
    assert chapter["card_extraction_state"] == "exact"
