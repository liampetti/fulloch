"""One batch assessment assigns original partial facts without closing gaps."""

import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from core.background_jobs import BackgroundJob
from core.thinking_budget import ThinkingBudget
from core.thinking_fact_cards import assign_batch_findings, link_chapter_cards
from core.thinking_worker import ThinkingWorker
from scripts.compare_thinking_sources import source_only_job

FIXTURES = Path(__file__).parent / "fixtures/thinking"


def backup_job():
    return source_only_job(json.loads((FIXTURES / "source-only-backup.json").read_text()))


def test_batch_assignment_retains_original_partial_finding_and_gap_without_promoting_analysis():
    job = backup_job()
    chapter, requirement, card = job.chapter_plan[0], job.evidence_requirements[0], job.fact_cards[0]
    original = deepcopy(card["finding"])
    response = {"answers": [{"question_id": requirement["identifier"], "answered": False,
                             "card_ids": [], "missing_facts": "Independent privacy verification."}],
                "findings": [{"question_id": requirement["identifier"], "card_ids": [card["id"]],
                              "meaning": "Invented encryption guarantee must never be evidence."}]}
    assign_batch_findings(chapter, job.evidence_requirements, job.fact_cards, response)
    link_chapter_cards(job.fact_cards, job.chapter_plan)
    assert card["finding"]["meaning"] == original["meaning"]
    assert card["finding"]["support"] == original["support"]
    assert card["finding"]["context_spans"] == original["context_spans"]
    assert card["finding"]["question_ids"] == [requirement["identifier"]]
    assert card["finding"]["section_ids"] == [chapter["id"]]
    assert requirement["state"] == "open" and not requirement.get("accepted_answer")
    assert card["finding"]["assessment"] == "assigned"
    assert not card["finding"].get("verified")
    assert chapter["section_card"]["state"] == "unassessed"


def test_batch_assignments_validate_ids_without_overriding_model_relevance():
    job = backup_job()
    chapter, requirement = job.chapter_plan[0], job.evidence_requirements[0]
    assign_batch_findings(chapter, job.evidence_requirements, job.fact_cards, {"findings": [
        {"question_id": "other-thread", "card_ids": [job.fact_cards[0]["id"]]},
        {"question_id": requirement["identifier"], "card_ids": ["fabricated-source"]},
    ]})
    assert not chapter["questions"][0].get("finding_bindings")
    f1 = source_only_job(json.loads((FIXTURES / "f1-recovery.json").read_text()))
    tutorial = next(card for card in f1.fact_cards if "f1dataR" in card["label"])
    assign_batch_findings(f1.chapter_plan[0], f1.evidence_requirements, f1.fact_cards, {"findings": [
        {"question_id": f1.evidence_requirements[0]["identifier"], "card_ids": [tutorial["id"]]},
    ]})
    assert tutorial["id"] in f1.chapter_plan[0]["questions"][0]["finding_bindings"]
    assert f1.evidence_requirements[0]["state"] != "resolved"


def test_batch_source_version_bindings_survive_restart_and_invalidate_selectively():
    job = backup_job()
    chapter, requirement = job.chapter_plan[0], job.evidence_requirements[0]
    assign_batch_findings(chapter, job.evidence_requirements, job.fact_cards, {"findings": [
        {"question_id": requirement["identifier"], "card_ids": [card["id"] for card in job.fact_cards]},
    ]})
    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    question = restored.chapter_plan[0]["questions"][0]
    assert question["finding_bindings"] == chapter["questions"][0]["finding_bindings"]
    changed = restored.fact_cards[0]
    changed["provenance"]["source_version"] = "another-version"
    link_chapter_cards(restored.fact_cards, restored.chapter_plan)
    assert changed["id"] not in question["finding_bindings"]
    assert len(question["finding_bindings"]) == len(job.fact_cards) - 1
    assert restored.evidence_requirements[0]["state"] == "open"


def test_previously_assessed_batch_does_not_rephrase_long_originals_in_an_extra_model_gate():
    job = backup_job()
    card = job.fact_cards[0]
    # Use a complete legacy assertion to isolate admission, not fabricate v4 provenance.
    card.pop("finding")
    card["version"] = 3
    card["statement"] = "The vendor does not guarantee uploads when " + "the declared network condition applies " * 20 + "."
    chapter = job.chapter_plan[0]
    chapter["coverage_checked"] = True
    calls = []
    service = ThinkingWorker(object(), None, SimpleNamespace(), lambda *args, **kwargs: calls.append(kwargs),
                             lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    budget = ThinkingBudget(service.model, job.model_budget)
    facts, disagreements, gaps = service._extract_chapter_cards(job, chapter, [card], budget, lambda: False)
    assert facts[card["id"]] == card["statement"]
    assert not calls and budget.state["calls"] == 0
    assert chapter["card_extraction_state"] == "exact"
    assert not disagreements and not gaps


def test_actual_run_assigns_partial_source_facts_and_gaps_in_one_batch_call_before_pause():
    job = backup_job()
    calls = []

    def generate(_model, **kwargs):
        calls.append(kwargs["user_prompt"])
        assert kwargs["user_prompt"] == "Check direct answer coverage."
        return json.dumps({"answers": [{"question_id": job.evidence_requirements[0]["identifier"],
                                        "answered": False, "card_ids": [], "missing_facts": "A privacy condition is unresolved."}],
                           "findings": [{"question_id": job.evidence_requirements[0]["identifier"],
                                         "card_ids": [card["id"] for card in job.fact_cards[:3]]}]})

    service = ThinkingWorker(object(), None, SimpleNamespace(update_stage=lambda *_: None, checkpoint=lambda *_: None),
                             generate, lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    report, _ = service.run(job, lambda: bool(job.chapter_plan[0].get("coverage_checked")))
    assert not report
    assert calls == ["Check direct answer coverage."]
    assert job.evidence_requirements[0]["state"] == "open"
    assert "privacy condition" in job.evidence_requirements[0]["detail"]
    assert job.chapter_plan[0]["questions"][0]["finding_bindings"]
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert resumed.chapter_plan[0]["questions"][0]["finding_bindings"] == job.chapter_plan[0]["questions"][0]["finding_bindings"]
    assert resumed.model_budget["calls"] == 1


def test_batch_local_aliases_decode_to_originals_before_eligibility_and_checkpointing():
    job = backup_job()

    def generate(_model, **kwargs):
        assert kwargs["user_prompt"] == "Check direct answer coverage."
        packet, _ = json.JSONDecoder().raw_decode(kwargs["system_prompt"].split("Eligible evidence cards:", 1)[1].strip())
        assert packet and all(row["id"].startswith("C") for row in packet)
        return json.dumps({"answers": [{"question_id": job.evidence_requirements[0]["identifier"],
                                        "answered": False, "card_ids": [], "missing_facts": "Privacy is unresolved."}],
                           "findings": [{"question_id": job.evidence_requirements[0]["identifier"],
                                         "card_ids": [packet[0]["id"], "C999"]}], "subject_candidates": []})

    service = ThinkingWorker(object(), None, SimpleNamespace(update_stage=lambda *_: None, checkpoint=lambda *_: None),
                             generate, lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    service.run(job, lambda: bool(job.chapter_plan[0].get("coverage_checked")))
    bindings = job.chapter_plan[0]["questions"][0]["finding_bindings"]
    assert len(bindings) == 1 and set(bindings) <= {card["id"] for card in job.fact_cards}
    restored = BackgroundJob.from_checkpoint(job.checkpoint())
    assert restored.chapter_plan[0]["questions"][0]["finding_bindings"] == bindings
    assert restored.evidence_requirements[0]["state"] == "open"
