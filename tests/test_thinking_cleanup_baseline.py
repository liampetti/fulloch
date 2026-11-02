"""Synthetic evidence remains replayable and exercises report safeguards."""

import json
from pathlib import Path

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_fallback import _fallback_fact_body, chapter_excerpts
from core.thinking_replay import _acceptance_metrics, load_replay, replay, restore_replay_chapters
from core.thinking_worker import (
    _clean_report_block,
    _dedup_chapter_passages,
    _deterministic_report_audit,
    _extract_report_bindings,
)
from utils.prompts import get_thinking_user_prompt

FIXTURE = Path(__file__).parent / "deep-think/synthetic-finance.evidence.json"


def test_synthetic_replay_retains_evidence_and_open_requirements():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    job = load_replay(FIXTURE)
    restore_replay_chapters(job)
    assert job.chapter_plan and job.fact_cards
    assert data["fixture_metadata"]["synthetic"] is True
    assert job.fact_cards[0]["fields"]["name"] == "Example Corporation"
    assert any(item["state"] == "open" for item in data["coverage_requirements"])


def test_fallback_does_not_render_historical_table_without_its_year():
    historical = {"kind": "source_excerpt", "statement": "Date | Event\nOct 10 | Harbour\nOct 11 | Island"}
    assert _fallback_fact_body(historical) == ""


def test_fallback_keeps_table_that_contains_its_temporal_context():
    card = {"kind": "source_excerpt", "statement": (
        "Date | Result\n2024-01-01 | 5\n2025-01-01 | 6"
    )}
    assert "2024-01-01" in _fallback_fact_body(card)


@pytest.mark.parametrize("date", [
    "Oct. 23-25",
    "Nov. 06-08",
])
def test_fallback_does_not_present_yearless_calendar_row_as_dated_fact(date):
    card = {"kind": "structured_record", "fields": {"columns": {"Date": date, "Event": "Harbour"}}}
    assert _fallback_fact_body(card) == ""


@pytest.mark.parametrize("columns", [
    {"Date": "Oct. 23-25, 2026", "Event": "Harbour Grand Prix"},
    {"Date": "23/10", "Year": "2026", "Event": "Harbour Grand Prix"},
])
def test_fallback_keeps_calendar_row_with_its_own_explicit_year(columns):
    card = {"kind": "structured_record", "fields": {"columns": columns}}
    assert "Harbour Grand Prix" in _fallback_fact_body(card)


def test_fallback_does_not_infer_calendar_year_from_retrieval_metadata():
    card = {"kind": "structured_record", "retrieved_at": "2026-10-29T00:00:00Z",
            "fields": {"columns": {"Date": "23/10", "Event": "Harbour Grand Prix"}}}
    assert _fallback_fact_body(card) == ""


def test_source_table_renders_in_report_viewer_with_its_citation():
    from bs4 import BeautifulSoup

    from server.routes_chat import _render_markdown_document

    card = {"id": "table-1", "kind": "atomic_claim", "statement": "Team | Driver\nExample | Avery",
            "source_url": "https://example.test/table", "source_ids": ["source-1"],
            "evidence_status": "single_source", "compatibility": "compatible", "label": "Example table",
            "source_class": "secondary_reporting", "freshness": "unknown", "scope": "Synthetic data", "fields": {}}
    excerpt = chapter_excerpts([card])
    body = _dedup_chapter_passages(set(), _clean_report_block(excerpt))
    report, bindings = _extract_report_bindings(body, [card])
    document = BeautifulSoup(_render_markdown_document(report), "html.parser")
    table = document.find("table")
    assert table is not None
    assert "Team [1]" in table.get_text(" ", strip=True)
    assert "Avery" in table.get_text(" ", strip=True)
    assert [binding["card_id"] for binding in bindings] == [card["id"]]


def test_irregular_source_table_is_not_fabricated():
    from core.thinking_fallback import _markdown_source_table

    assert _markdown_source_table("Team | Driver\nExample Team | Avery | Extra") == ""


def test_source_table_preserves_empty_cells_and_escapes_html():
    from core.thinking_fallback import _markdown_source_table

    table = _markdown_source_table("Name | Note | Status\nExample | <script>bad</script> |")
    assert "| Example | &lt;script&gt;bad&lt;/script&gt; |  |" in table


def test_partial_packet_cannot_establish_venue_missing_from_all_sources():
    # Audits may see only a subset of a synthetic evidence ledger.
    cited = []
    statement = "The exact circuit name is not provided in the supplied sources."
    assert any("partial source packet" in issue for issue in _deterministic_report_audit(statement, cited))
    reversed_statement = "The supplied sources do not provide the specific circuit name."
    assert any("partial source packet" in issue for issue in _deterministic_report_audit(reversed_statement, cited))
    scoped = "The cited source does not specify the circuit name."
    assert not any("partial source packet" in issue for issue in _deterministic_report_audit(scoped, cited))
    positive = "The circuit name is provided in the supplied sources."
    assert not any("partial source packet" in issue for issue in _deterministic_report_audit(positive, cited))


@pytest.mark.parametrize("statement", [
    "The exact date, circuit, and location are not provided in the available evidence.",
    "The exact circuit name and session times are not specified in the sources.",
    "The available evidence identifies the race, but it does not provide the circuit name.",
    "The provided sources do not verify the exact date.",
    "The retained evidence does not include the quote time.",
    "The exact date and circuit are not provided in the supplied cards.",
    "The evidence cards do not identify the event date.",
])
def test_broad_absence_claims_are_not_proven_by_one_partial_packet(statement):
    assert any("partial source packet" in issue for issue in _deterministic_report_audit(statement, []))


@pytest.mark.parametrize("statement", [
    "The cited source does not specify the circuit name.",
    "The evidence says the provider does not offer mobile backup.",
    "The evidence supports the date. The circuit is not named in the source.",
    "The available evidence includes the race date.",
    "The graphics cards do not include a power cable.",
    "These playing cards do not contain any jokers.",
])
def test_broad_absence_guard_preserves_source_scoped_and_positive_claims(statement):
    assert not any("partial source packet" in issue for issue in _deterministic_report_audit(statement, []))


@pytest.mark.parametrize("statement", [
    "The specific circuit name and precise location details are not provided in these sources.",
    "The cited source does not specify the circuit name.",
    "The source does not list a winner for the race.",
])
def test_absence_from_whole_source_needs_more_than_an_excerpt(statement):
    assert any("cannot establish absence from an entire source" in issue
               for issue in _deterministic_report_audit(statement, []))


def test_explicit_source_absence_assertion_is_not_rejected_as_an_omission():
    statement = "The source does not list a winner for the race."
    job = load_replay(FIXTURE)
    card = dict(job.fact_cards[0], statement=statement)
    assert not any("cannot establish absence from an entire source" in issue
                   for issue in _deterministic_report_audit(statement, [card]))
    assert any("cannot establish absence from an entire source" in issue
               for issue in _deterministic_report_audit(
                   statement + " <!-- cards: another-card -->", [card],
               ))


def test_non_f1_replay_preserves_open_question(tmp_path):
    fixture = tmp_path / "backup.evidence.json"
    fixture.write_text(json.dumps({
        "schema_version": 2,
        "task": "Explain backup limitations",
        "fact_cards": [],
        "coverage_requirements": [{
            "identifier": "backup-q", "question": "Which uploads are unsupported?",
            "importance": "high", "state": "open", "evidence_ids": [],
        }],
    }), encoding="utf-8")
    job = load_replay(fixture)
    restore_replay_chapters(job)
    assert job.evidence_requirements[0]["state"] == "open"
    assert job.evidence_requirements[0]["evidence_ids"] == []
    assert job.chapter_plan[0]["questions"][0]["question"] == "Which uploads are unsupported?"


def test_synthetic_non_finance_evidence_is_replayable_without_retrieval():
    path = Path(__file__).parent / "deep-think/synthetic-travel.evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    job = load_replay(path)
    restore_replay_chapters(job)
    assert data["fixture_metadata"]["synthetic"] is True
    assert len(job.fact_cards) > 0 and len(job.source_registry) > 0
    assert any(req["state"] == "open" for req in job.evidence_requirements)


def test_production_replay_reassesses_saved_coverage_without_rebuilding_cards():
    class RecordingModel:
        def __init__(self):
            self.prompts = []

        def generate(self, **kwargs):
            self.prompts.append(kwargs.get("user_prompt"))
            return "{}"

    model = RecordingModel()
    saved_ids = {card["id"] for card in load_replay(FIXTURE).fact_cards}
    result = replay(FIXTURE, model=model, production=True)
    assert result["error"] == ""
    assert get_thinking_user_prompt("chapter_coverage") in model.prompts
    assert saved_ids <= {card["id"] for card in result["checkpoint"]["fact_cards"]}
    assert result["checkpoint"]["capability_calls"] == {}
    requirements = result["checkpoint"]["evidence_requirements"]
    assert requirements


def test_production_replay_resolves_identity_only_from_saved_source(tmp_path):
    statement = "Harbour Grand Prix is on October 25, 2026."
    fixture = tmp_path / "event.evidence.json"
    fixture.write_text(json.dumps({
        "schema_version": 2,
        "task": "Research the next Harbour race",
        "research_context": {"as_of": "2026-10-01T00:00:00Z"},
        "source_registry": {"source-1": {
            "title": "Event notice", "url": "https://example.test/event", "text": statement,
        }},
        "fact_cards": [{
            "id": "card-1", "source": "source-1", "kind": "atomic_claim", "claim": statement,
            "status": "single_source", "provenance": {"source_id": "source-1"},
        }],
        "coverage_requirements": [{
            "identifier": "chapter-01-q01", "chapter_id": "chapter-01",
            "question": "Resolve the exact upcoming subject, date and version from original source evidence.",
            "importance": "high", "state": "open", "is_subject_identity": True, "evidence_ids": [],
        }],
    }), encoding="utf-8")

    class IdentityModel:
        def generate(self, **kwargs):
            if kwargs.get("user_prompt") == get_thinking_user_prompt("chapter_coverage"):
                return json.dumps({"answers": [], "findings": [], "subject_candidates": [{
                    "name": "Harbour Grand Prix", "date": "2026-10-25", "card_ids": ["C1"],
                }]})
            return "{}"

    result = replay(fixture, model=IdentityModel(), production=True)
    assert result["error"] == ""
    checkpoint = result["checkpoint"]
    assert checkpoint["research_context"]["subject_resolution"]["state"] == "resolved"
    assert checkpoint["research_context"]["subject_resolution"]["card_ids"] == ["card-1"]
    assert checkpoint["capability_calls"] == {}


def test_replay_metrics_distinguish_excerpts_from_delivered_answers():
    job = BackgroundJob("backup", JobSnapshot("Explain backup limitations"))
    job.evidence_requirements = [{
        "identifier": "backup-q", "importance": "high", "state": "resolved",
        "evidence_ids": ["backup-card"], "delivered_synthesis": "not_started",
    }]
    cards = [{"id": "backup-card"}]
    job.fact_cards = cards
    job.report_bindings = [{"card_id": "backup-card"}]
    job.synthesis_state["extractive_only_card_ids"] = ["backup-card"]

    metrics = _acceptance_metrics(job, cards, "A cited extract.", 1)
    assert metrics["delivery_status"] == "evidence_only"
    assert metrics["evidence_only_fallback"] is True
    assert metrics["supported_answer_coverage"]["fully_delivered"] == 0
    assert job.evidence_requirements[0]["delivered_synthesis"] == "not_started"

    job.synthesis_state["extractive_only_card_ids"] = []
    metrics = _acceptance_metrics(job, cards, "A cited answer.", 1)
    assert metrics["delivery_status"] == "complete"
    assert metrics["supported_answer_coverage"]["fully_delivered"] == 1
