"""Evidence traces distinguish recorded navigation from actual answer delivery."""

import json
from pathlib import Path

from core.background_jobs import BackgroundJob
from core.thinking_evidence import report_evidence
from core.thinking_replay import load_replay, replay
from core.thinking_trace import card_evidence_trace, question_evidence_trace

FIXTURE = Path(__file__).parent / "deep-think/synthetic-finance.evidence.json"


def test_question_trace_retains_distinct_navigation_and_delivery_states():
    job = load_replay(FIXTURE)
    job.fact_cards.extend([dict(job.fact_cards[0], id="synthetic-card-2"),
                           dict(job.fact_cards[0], id="synthetic-card-3")])
    first, second, third = [card["id"] for card in job.fact_cards[:3]]
    job.evidence_requirements = [{
        "identifier": "q1", "chapter_id": "chapter", "state": "open", "evidence_ids": [],
        "source_review_complete": True, "model_selected_card_ids": [second],
    }]
    job.chapter_plan = [{"id": "chapter", "analyzed_card_ids": [first, third],
                         "refined_requirements": {"q1": {
                             "seen_card_ids": [third], "stop_reason": "question_round_limit",
                             "limits_at_stop": ["question_round_limit", "chapter_call_limit"],
                             "unseen_eligible_card_count": len(job.fact_cards) - 1,
                         }}}]
    job.synthesis_state = {"selected_card_ids": [first], "selection_reasons": {first: ["q1: keyword"]}}
    job.report_bindings = [{"card_id": first}]

    trace = question_evidence_trace(job)["q1"]
    assert trace["selected_for_question_ids"] == [first]
    assert trace["source_review_selected_ids"] == [second]
    assert trace["refinement_shown_ids"] == [third]
    assert trace["associated_ids_in_chapter_analysis"] == sorted([first, third])
    assert trace["associated_ids_cited_in_report"] == [first]
    assert trace["binding_based_delivery"] == "missing"  # A citation alone is not an answer.
    assert trace["refinement_stop_reason"] == "question_round_limit"
    assert trace["refinement_limits_at_stop"] == ["question_round_limit", "chapter_call_limit"]
    assert trace["other_eligible_count"] == trace["eligible_inventory_count"] - 1
    assert "delivered_synthesis" not in job.evidence_requirements[0]  # Measurement is read-only.
    assert json.loads(json.dumps(trace)) == trace


def test_question_trace_keeps_unrecorded_stages_unknown():
    job = load_replay(FIXTURE)
    trace = question_evidence_trace(job)[job.evidence_requirements[0]["identifier"]]
    assert trace["selection_recorded"] is False
    assert trace["selected_for_question_ids"] is None
    assert trace["other_eligible_count"] is None
    assert trace["source_review_selected_ids"] is None
    assert trace["refinement_shown_ids"] is None
    assert trace["associated_ids_in_chapter_analysis"] is None


def test_evidence_export_includes_scheduling_reason():
    job = load_replay(FIXTURE)
    job.action_selections = [{"tool": "lookup", "args": ["synthetic query"],
                              "requirement_id": "q1", "chapter_id": "c1",
                              "scheduling_reason": "breadth_first"}]
    actions = report_evidence(job)["run_metadata"]["selected_actions"]
    assert actions[0]["scheduling_reason"] == "breadth_first"
    assert actions[0]["requirement_id"] == "q1"


def test_card_trace_records_selection_packets_audit_and_extractive_delivery():
    job = load_replay(FIXTURE)
    first = job.fact_cards[0]["id"]
    second = "fictional-card-2"
    job.fact_cards.append(dict(job.fact_cards[0], id=second))
    job.evidence_requirements = [{"identifier": "q1", "model_selected_card_ids": [first]}]
    job.chapter_plan = [{"id": "c1", "analysis_groups": {"batch": {
        "card_ids": [first], "state": "failed", "reason": "no_verified_chapter_body"}}}]
    job.work_queue = [{"kind": "source_sample", "state": "blocked", "references": [first],
                       "reason": "Indivisible source sample: verification_context_does_not_fit"}]
    job.synthesis_state = {
        "selected_card_ids": [first], "selection_decisions": {
            first: {"q1": "selected"}, second: {"q1": "per_question_limit"}},
        "chapter_selection_decisions": {"c1": {first: {"q1": "selected"}}},
        "extractive_only_card_ids": [first],
        "passage_reviews": [{"passage_outcomes": [{"card_ids": [first], "outcome": "rejected",
                                                   "reason_codes": ["support"]}]}],
    }
    job.report_bindings = [{"card_id": first}]
    trace = card_evidence_trace(job)
    assert trace["cards"][first]["eligibility"] == "eligible"
    assert trace["cards"][first]["source_review_questions"] == ["q1"]
    assert trace["cards"][first]["chapter_selection"] == {"c1": {"q1": "selected"}}
    assert trace["cards"][first]["chapter_packets"][0]["reason"] == "no_verified_chapter_body"
    assert trace["cards"][first]["work_units"][0]["reason"] == "verification_context_does_not_fit"
    assert trace["cards"][first]["audit_outcomes"][0]["reason_codes"] == ["support"]
    assert trace["cards"][first]["delivery"] == "extractive"
    assert trace["cards"][second]["selection_by_requirement"] == {"q1": "per_question_limit"}
    assert trace["cards"][second]["delivery"] == "not_cited"
    assert not trace["truncated"]
    job.synthesis_state["card_evidence_trace"] = trace
    assert report_evidence(job)["run_metadata"]["card_evidence_trace"] == trace


def test_card_trace_keeps_unrecorded_stages_unknown_and_bounds_output():
    job = load_replay(FIXTURE)
    job.fact_cards.append(dict(job.fact_cards[0], id="another-card"))
    result = card_evidence_trace(job, max_cards=1)
    assert result["truncated"] and len(result["cards"]) == 1
    entry = next(iter(result["cards"].values()))
    assert entry["selection_by_requirement"] is None and entry["delivery"] is None
    assert not entry["packet_trace_recorded"] and not entry["audit_trace_recorded"]


def test_replay_persists_question_trace_in_checkpoint():
    fixture = FIXTURE.with_name("synthetic-finance.evidence.json")
    result = replay(fixture, production=True)
    assert not result["error"]
    restored = BackgroundJob.from_checkpoint(result["checkpoint"])
    trace = restored.synthesis_state["question_evidence_trace"]
    assert set(trace) == {item["identifier"] for item in restored.evidence_requirements}
    assert all("binding_based_delivery" in row for row in trace.values())
