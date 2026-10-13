"""All originals reach model review; only real source bindings can reach writing."""

import json
import threading
from types import SimpleNamespace

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ThinkingBudget
from core.thinking_evidence import render_card_references
from core.thinking_fact_cards import eligible_cards_for_requirement, select_requirement_cards
from core.thinking_replay import restore_replay_chapters
from core.thinking_source_context import source_block_card, source_document_blocks
from core.thinking_worker import ThinkingWorker


def worker():
    return ThinkingWorker(
        None, None, SimpleNamespace(checkpoint=lambda *_: None, update_stage=lambda *_: None),
        lambda *_args, **_kwargs: "", lambda: {}, SimpleNamespace(_schemas={}), lambda *_: "",
        threading.Lock(), 1,
    )


def test_whole_timetable_and_undated_background_are_not_assertion_filtered():
    sources = {
        "schedule": {"text": "FRIDAY 23 OCTOBER | | Local time\nFORMULA 1 | Practice | 12:30 - 13:30",
                     "title": "Official United States GP", "url": "https://example.org/timetable"},
        "track": {"text": "The uphill first corner offers overtaking opportunities.",
                  "title": "COTA", "url": "https://example.org/cota"},
    }
    blocks = source_document_blocks(sources)
    assert len(blocks) == 2
    cards = [source_block_card(identifier, block, sources[block["source"]]) for identifier, block in blocks.items()]
    requirement = {"freshness": "fresh", "subject_resolution_required": True,
                   "subject_resolution": {"state": "resolved", "name": "United States Grand Prix (Austin)"}}
    assert eligible_cards_for_requirement(cards, requirement) == cards
    for card in cards:
        p = card["provenance"]
        assert sources[p["source_id"]]["text"][p["start"]:p["end"]] == card["statement"]
    assert source_document_blocks(sources) == blocks
    sources["track"]["text"] += " Conditions may vary."
    assert set(source_document_blocks(sources)) != set(blocks)


def test_model_review_selects_originals_and_keeps_summaries_out_of_evidence(monkeypatch):
    job = BackgroundJob("review", JobSnapshot("Analyse the next race"))
    job.source_registry = {
        "schedule": {"text": "FRIDAY | Local time\nPractice | 12:30", "title": "Official timetable",
                     "url": "https://example.org/schedule"},
        "bot": {"text": "Making sure you're not a bot!", "url": "https://example.org/bot"},
    }
    job.evidence_requirements = [{"identifier": "timing", "question": "When is practice?"}]
    calls = []

    def generate(*_args, **kwargs):
        calls.append(kwargs)
        assert "Making sure you're not a bot!" in kwargs["system_prompt"]
        pages = json.loads(kwargs["system_prompt"].split("Original pages: ", 1)[1])
        schedule_alias = next(iter(next(page for page in pages if page["source"] == "schedule")["blocks"]))
        return json.dumps({"pages": [
            {"source": "schedule", "summary": "Navigation summary is not source evidence.",
             "findings": [{"summary": "Practice timing", "questions": ["timing"], "blocks": [schedule_alias]}]},
            {"source": "bot", "summary": "Bot challenge; no usable facts.", "findings": []},
        ]})

    monkeypatch.setattr("core.thinking_worker._budgeted_generate", generate)
    service = worker()
    budget = ThinkingBudget(None, job.model_budget)
    service._review_source_documents(job, budget, lambda: False)
    assert job.synthesis_state["source_review_progress"]["complete"]
    assert len(job.fact_cards) == 1
    assert job.fact_cards[0]["statement"] == job.source_registry["schedule"]["text"]
    assert "Navigation summary" not in str(job.fact_cards)
    selected, _ = select_requirement_cards(job.fact_cards, job.evidence_requirements)
    assert selected == job.fact_cards
    service._review_source_documents(job, budget, lambda: False)
    assert len(calls) == 1  # Reviewed original versions survive incremental writes.


def test_failed_review_does_not_mark_unseen_sources_irrelevant(monkeypatch):
    job = BackgroundJob("bad-review", JobSnapshot("Analyse the next race"))
    job.source_registry = {"source": {"text": "Original retained prose.", "url": "https://example.org/page"}}
    job.evidence_requirements = [{"identifier": "answer", "question": "What happened?"}]
    monkeypatch.setattr("core.thinking_worker._budgeted_generate", lambda *_args, **_kwargs: '{"pages":[]}')
    worker()._review_source_documents(job, ThinkingBudget(None, job.model_budget), lambda: False)
    assert not job.evidence_requirements[0]["source_review_complete"]
    assert job.synthesis_state["source_review_progress"]["reviewed_blocks"] == 0


def test_every_retained_source_version_reaches_review_inventory():
    sources = {"same-url": {"text": "A corrected current claim.", "versions": {
        "v1": {"text": "The earlier claim."}, "v2": {"text": "A corrected current claim."},
    }}}
    blocks = source_document_blocks(sources)
    assert {row["text"] for row in blocks.values()} == {"The earlier claim.", "A corrected current claim."}
    assert any(row["source_version"] == "v1" for row in blocks.values())


def test_measured_output_releases_unused_reservation_but_resume_keeps_unfinished_charge():
    state = {}
    budget = ThinkingBudget(None, state)
    reserved, started = budget.begin("audit", system_prompt="Audit a supplied claim.")
    budget.finish("audit", started, "A checked response.", diagnostics={"completion_tokens": 128})
    assert state["reserved_tokens"] == reserved
    assert state["charged_tokens"] == 128
    next_reserved, _ = budget.begin("audit", system_prompt="Another supplied claim.")
    resumed = ThinkingBudget(None, state)
    assert resumed.state["charged_tokens"] == 128 + next_reserved
    assert resumed.state["calls"] == 2


def test_offline_bibliography_requires_real_bound_cards_and_does_not_initialize_notes():
    cards = [{"id": "real", "compatibility": "compatible", "label": "Original schedule",
              "source_url": "https://example.org/schedule"}]
    bindings = [{"card_id": "invented", "source_url": "https://example.org/unseen"},
                {"card_id": "real", "source_url": "https://example.org/schedule"},
                {"card_id": "real", "source_url": "https://example.org/schedule"}]
    references = render_card_references(cards, bindings)
    assert references == "## References\n\n1. [Original schedule](https://example.org/schedule)"


def test_review_navigation_does_not_replace_an_accepted_answer_source_identity():
    sources = {"source": {"text": "The race is on October 25, 2026.", "url": "https://example.org/page"}}
    identifier, block = next(iter(source_document_blocks(sources).items()))
    card = source_block_card(identifier, block, sources["source"])
    requirement = {"identifier": "answer", "question": "When?", "state": "resolved",
                   "evidence_ids": [identifier], "source_review_complete": True, "model_selected_card_ids": []}
    selected, _ = select_requirement_cards([card], [requirement])
    assert selected == [card]


def test_production_replay_preserves_saved_scope_and_groups_without_tools():
    job = BackgroundJob("replay", JobSnapshot("Analyse the next race"))
    job.research_context = {"as_of": "2026-10-13T02:07:22Z", "entities": ["Formula 1"],
                            "subject_resolution": {"state": "resolved", "name": "Austin", "date": "2026-10-25"}}
    job.evidence_requirements = [
        {"identifier": "chapter-01-q01", "question": "When is the race?"},
        {"identifier": "chapter-02-q01", "question": "Who leads?"},
    ]
    restore_replay_chapters(job)
    assert len(job.chapter_plan) == 2
    assert all(chapter["tools"] == [] for chapter in job.chapter_plan)
    assert job.evidence_requirements[1]["subject_resolution"]["name"] == "Austin"
    assert job.evidence_requirements[1]["as_of"] == "2026-10-13T02:07:22Z"
