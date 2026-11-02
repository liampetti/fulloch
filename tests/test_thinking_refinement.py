"""Small-subset evidence refinement lets a compact model answer open questions
by reading a few related cards at a time and rephrasing from what they reveal,
instead of hunting one answer across the whole chapter in a single batch."""

import json
import threading
from types import SimpleNamespace

import pytest

from core import thinking_coverage as coverage
from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ThinkingBudget
from core.thinking_fact_cards import report_card_ledger
from core.thinking_worker import (
    ThinkingWorker,
    _refine_open_chapter_coverage,
    _refinement_card_binding,
    _sync_all_chapter_questions,
    _sync_chapter_questions,
)

REQUIREMENT = {
    "identifier": "backup-q",
    "question": "Which uploads are unsupported on the backup service?",
    "chapter_id": "backup",
    "importance": "high",
    "state": "open",
    "evidence_terms": ["unsupported uploads", "local network"],
}

# Three statements with distinct vocabularies so a rephrased question genuinely
# re-ranks the subset: the answer card shares the gap words, the near-miss card
# shares only the rephrased words, and the off-topic card shares neither.
ANSWER_STATEMENT = "The backup service does not support uploads when the local network is unavailable."
NEAR_MISS_STATEMENT = "Encrypted file sync requires the desktop client version 4 or later."
OFF_TOPIC_STATEMENT = "The service stores backup history for thirty days by default."


def _valid_card(card_id: str, statement: str, source: str) -> dict:
    """A structurally valid retained card (passes validate_fact_card)."""
    return {
        "id": card_id,
        "kind": "source_excerpt",
        "label": "backup",
        "source_class": "secondary_reporting",
        "freshness": "stable",
        "scope": "backup",
        "compatibility": "compatible",
        "fields": {},
        "source_ids": [source],
        "evidence_status": "single_source",
        "source_url": "https://fixture.invalid/" + source,
        "statement": statement,
    }


def _job(*, open_question: bool = True):
    job = BackgroundJob("refine", JobSnapshot("Explain backup limitations"))
    statements = [
        ANSWER_STATEMENT, NEAR_MISS_STATEMENT, OFF_TOPIC_STATEMENT,
    ]
    for index, statement in enumerate(statements):
        source = {"url": f"https://fixture.invalid/{index}", "title": f"Service {index}",
                  "source_text": statement, "eligible": True}
        job.record_outcome("lookup", "evidence", "Bounded documentation",
                           {"sources": [source]}, (),
                           {"type": "web_research", "sources": [source]})
    from core.thinking_fact_cards import build_fact_cards
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts,
                                      source_registry=job.source_registry)
    # The build order is stable; pin the mapping the assertions rely on.
    by_statement = {card["statement"]: card["id"] for card in job.fact_cards}
    job._fixture_ids = {
        "answer": by_statement[ANSWER_STATEMENT],
        "near_miss": by_statement[NEAR_MISS_STATEMENT],
        "off_topic": by_statement[OFF_TOPIC_STATEMENT],
    }
    ids = [card["id"] for card in job.fact_cards]
    job.evidence_requirements = [{
        "identifier": "backup-q",
        "question": "Which uploads are unsupported on the backup service?",
        "chapter_id": "backup",
        "importance": "high",
        "state": "open" if open_question else "resolved",
        "evidence_terms": ["unsupported uploads", "local network"],
        "evidence_ids": [] if open_question else ids,
        "answer_coverage": "open" if open_question else "evidence_scoped",
    }]
    job.chapter_plan = [{
        "id": "backup", "title": "Backup limitations", "priority": "high",
        "questions": [{"requirement_id": "backup-q", "state": "open",
                       "supporting_card_ids": []}],
        "requirement_ids": ["backup-q"],
    }]
    return job


def _worker(job, generate):
    return ThinkingWorker(
        SimpleNamespace(context_tokens=16000), None,
        SimpleNamespace(update_stage=lambda *_: None),
        lambda _, **kw: generate(**kw), lambda: {}, None,
        lambda *_: "", threading.Lock(), 1,
    )


def _chapter():
    return {
        "id": "backup", "title": "Backup limitations", "priority": "high",
        "questions": [{"requirement_id": "backup-q", "state": "open",
                       "supporting_card_ids": []}],
        "requirement_ids": ["backup-q"],
    }


def _refine(job, generate, chapter=None):
    chapter = chapter or _chapter()
    worker = _worker(job, generate)
    budget = ThinkingBudget(SimpleNamespace(context_tokens=16000), {})
    _refine_open_chapter_coverage(
        worker, job, chapter, budget, lambda: False,
        requirements_by_chapter={chapter["id"]: [
            item for item in job.evidence_requirements if item.get("chapter_id") == chapter["id"]
        ]},
    )
    return chapter


def _packet_alias(prompt: str, statement: str) -> str | None:
    """Return the C-alias the refinement packet assigned to a card's claim."""
    import re
    packet = prompt.split("Evidence cards:", 1)[1]
    for candidate in re.findall(r'"(C\d+)"', packet):
        window = packet.split(f'"{candidate}"', 1)[1][:400]
        if statement[:40] in window:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Subsample mechanics (deterministic, no model)
# ---------------------------------------------------------------------------

def test_subsample_first_round_keeps_related_cards_bounded():
    cards = [_valid_card(f"card-{i}",
                         f"Backup statement {i} mentions the local network upload limit.",
                         f"src-{i}")
             for i in range(40)]
    subset = coverage.refinement_subsample(cards, dict(REQUIREMENT))
    assert subset
    assert len(subset) <= coverage.MAX_REFINEMENT_CARDS
    ids = [card["id"] for card in subset]
    assert len(ids) == len(set(ids))
    # The most on-point card ranks first on the original gap vocabulary.
    assert ids[0] == "card-0"


def test_subsample_rephrase_steers_to_different_neighbourhood():
    job = _job()
    ledger = report_card_ledger(job.fact_cards)
    requirement = dict(REQUIREMENT)
    original = coverage.refinement_subsample(ledger, requirement)
    # The rephrase uses the near-miss card's own vocabulary, so that card must
    # climb above the off-topic card and the subset ordering shifts.
    requirement["rephrased_question"] = "What version rule applies to encrypted sync?"
    shifted = coverage.refinement_subsample(ledger, requirement)
    original_ids = [card["id"] for card in original]
    shifted_ids = [card["id"] for card in shifted]
    assert job._fixture_ids["near_miss"] in shifted_ids
    assert shifted_ids != original_ids
    # The off-topic card (no shared vocabulary) sinks below the near-miss card.
    assert shifted_ids.index(job._fixture_ids["near_miss"]) < \
        shifted_ids.index(job._fixture_ids["off_topic"])


def test_subsample_pushes_seen_cards_to_tail_but_keeps_them():
    job = _job()
    ledger = report_card_ledger(job.fact_cards)
    requirement = dict(REQUIREMENT)
    requirement["rephrased_question"] = "What version rule applies to encrypted sync?"
    first = coverage.refinement_subsample(ledger, requirement)
    second = coverage.refinement_subsample(
        ledger, requirement, seen_ids={card["id"] for card in first})
    first_ids = {card["id"] for card in first}
    second_ids = [card["id"] for card in second]
    # Seen cards remain eligible (a rephrase may legitimately need one) but sink
    # below any unseen card; the least-relevant seen card is last.
    assert set(second_ids) == first_ids
    assert second_ids[-1] == job._fixture_ids["off_topic"]
    # The answer card (most relevant, seen) stays ahead of the seen off-topic card.
    assert second_ids.index(job._fixture_ids["answer"]) < \
        second_ids.index(job._fixture_ids["off_topic"])


def test_subsample_rotates_beyond_first_window_without_rephrasing():
    cards = [_valid_card(f"card-{i:02}", f"Backup upload limit {i}.", f"src-{i}")
             for i in range(7)]
    first = coverage.refinement_subsample(cards, dict(REQUIREMENT), max_cards=3)
    second = coverage.refinement_subsample(
        cards, dict(REQUIREMENT), seen_ids={card["id"] for card in first}, max_cards=3,
    )
    assert len(first) == len(second) == 3
    assert {card["id"] for card in first}.isdisjoint(card["id"] for card in second)


def test_refinement_can_inspect_cards_omitted_by_complete_source_review():
    reviewed = _valid_card("reviewed", "Backup upload policy is described here.", "one")
    omitted = _valid_card("omitted", ANSWER_STATEMENT, "two")
    requirement = {**REQUIREMENT, "source_review_complete": True,
                   "model_selected_card_ids": ["reviewed"]}
    subset = coverage.refinement_subsample(
        [reviewed, omitted], requirement, seen_ids={"reviewed"}, max_cards=1,
    )
    assert [card["id"] for card in subset] == ["omitted"]
    assert requirement["source_review_complete"] is True


def test_refinement_records_unseen_inventory_without_marking_it_missing():
    job = _job()
    chapter = _refine(job, lambda **_: json.dumps({"answered": False, "card_ids": []}))
    progress = chapter["refined_requirements"]["backup-q"]
    assert progress["eligible_card_count"] == len(report_card_ledger(job.fact_cards))
    assert progress["unseen_eligible_card_count"] == (
        progress["eligible_card_count"] - len(progress["seen_card_ids"])
    )
    assert job.evidence_requirements[0]["state"] == "open"


def test_refinement_records_when_chapter_budget_prevents_a_question():
    job = _job()
    chapter = _chapter()
    chapter["refinement_calls"] = coverage.MAX_REFINEMENT_CALLS_PER_CHAPTER
    _refine(job, lambda **_: pytest.fail("model should not be called"), chapter)
    progress = chapter["refined_requirements"]["backup-q"]
    assert progress["stop_reason"] == "chapter_call_limit"
    assert progress["unseen_eligible_card_count"] == len(report_card_ledger(job.fact_cards))
    assert job.evidence_requirements[0]["state"] == "open"


def test_refinement_records_both_question_and_chapter_limits():
    job = _job()
    chapter = _chapter()
    chapter["refinement_calls"] = coverage.MAX_REFINEMENT_CALLS_PER_CHAPTER
    chapter["refined_requirements"] = {"backup-q": {
        "rounds_done": coverage.MAX_REFINEMENT_ROUNDS,
        "seen_card_ids": [job._fixture_ids["answer"]],
    }}
    _refine(job, lambda **_: pytest.fail("model should not be called"), chapter)
    progress = chapter["refined_requirements"]["backup-q"]
    assert progress["stop_reason"] == "chapter_call_limit"
    assert progress["limits_at_stop"] == ["question_round_limit", "chapter_call_limit"]
    assert progress["seen_card_ids"] == [job._fixture_ids["answer"]]
    assert job.evidence_requirements[0]["state"] == "open"


def test_refinement_sends_different_cards_in_later_model_calls():
    job = _job()
    job.fact_cards = [_valid_card(f"card-{i:02}", f"Backup record number {i:02}.", f"src-{i:02}")
                      for i in range(50)]
    prompts = []

    def generate(**kwargs):
        prompts.append(kwargs["system_prompt"])
        return json.dumps({"answered": False, "card_ids": []})

    chapter = _refine(job, generate)
    assert len(prompts) == 3
    first = {i for i in range(50) if _packet_alias(prompts[0], f"Backup record number {i:02}.")}
    second = {i for i in range(50) if _packet_alias(prompts[1], f"Backup record number {i:02}.")}
    assert first and second and first.isdisjoint(second)
    progress = chapter["refined_requirements"]["backup-q"]
    assert progress["unseen_eligible_card_count"] == 50 - len(progress["seen_card_ids"])


def test_refinement_carries_original_partial_card_into_next_window(monkeypatch):
    monkeypatch.setattr(coverage, "MAX_REFINEMENT_CARDS", 2)
    job = _job()
    job.fact_cards = [_valid_card(f"card-{i}", f"Backup evidence part {i}.", f"src-{i}")
                      for i in range(5)]
    packets = []

    def generate(**kwargs):
        packet = json.loads(kwargs["system_prompt"].split("Evidence cards:", 1)[1])
        packets.append(packet)
        if len(packets) == 1:
            return json.dumps({"answered": False, "partial_card_ids": [packet[0]["id"], "invented"]})
        assert packet[0]["claim"] == packets[0][0]["claim"]
        assert packet[1]["claim"] not in {row["claim"] for row in packets[0]}
        return json.dumps({"answered": True, "answer": "The two evidence parts answer the question.",
                           "card_ids": [packet[0]["id"], packet[1]["id"]]})

    chapter = _refine(job, generate)
    assert len(packets) == 2
    assert job.evidence_requirements[0]["state"] == "resolved"
    assert len(job.evidence_requirements[0]["evidence_ids"]) == 2
    assert chapter["refined_requirements"]["backup-q"]["stop_reason"] == "answered"


def test_changed_partial_card_is_not_carried_on_resume(monkeypatch):
    monkeypatch.setattr(coverage, "MAX_REFINEMENT_CARDS", 1)
    job = _job()
    original = _valid_card("old", "Original backup record.", "one")
    job.fact_cards = [_valid_card("old", "Changed backup record.", "one"),
                      _valid_card("new", "New backup record.", "two")]
    chapter = _chapter()
    chapter["refined_requirements"] = {"backup-q": {
        "rounds_done": 1, "seen_card_ids": ["old"],
        "partial_bindings": {"old": _refinement_card_binding(original)},
    }}
    prompts = []

    def generate(**kwargs):
        prompts.append(kwargs["system_prompt"])
        return json.dumps({"answered": False, "card_ids": []})

    _refine(job, generate, chapter)
    packet = json.loads(prompts[0].split("Evidence cards:", 1)[1])
    assert [row["claim"] for row in packet] == ["New backup record."]


def test_refinement_does_not_accept_unknown_or_unshown_support():
    job = _job()

    def generate(**kwargs):
        alias = _packet_alias(kwargs["system_prompt"], ANSWER_STATEMENT)
        return json.dumps({"answered": True, "answer": ANSWER_STATEMENT,
                           "card_ids": [alias, "invented"]})

    _refine(job, generate)
    assert job.evidence_requirements[0]["state"] == "open"
    assert job.evidence_requirements[0]["evidence_ids"] == []


def test_refinement_does_not_hide_conflict_peers_outside_current_window(monkeypatch):
    monkeypatch.setattr(coverage, "MAX_REFINEMENT_CARDS", 1)
    job = _job()
    first = _valid_card("card-a", "Backup is available.", "one")
    second = _valid_card("card-b", "Backup is unavailable.", "two")
    for card in (first, second):
        card.update(evidence_status="conflicted", claim_identity={"subject": "backup"})
    job.fact_cards = [first, second]

    def generate(**kwargs):
        packet = json.loads(kwargs["system_prompt"].split("Evidence cards:", 1)[1])
        return json.dumps({"answered": True, "answer": packet[0]["claim"],
                           "card_ids": [packet[0]["id"]]})

    _refine(job, generate)
    assert job.evidence_requirements[0]["state"] == "open"
    assert job.evidence_requirements[0]["evidence_ids"] == []


# ---------------------------------------------------------------------------
# Worker loop (fake backend, no real model)
# ---------------------------------------------------------------------------

def test_chapter_question_state_is_derived_from_requirements():
    chapter = {"questions": [
        {"requirement_id": "one", "state": "open"},
        {"requirement_id": "two", "state": "answered", "supporting_card_ids": ["old"]},
        {"requirement_id": "other", "state": "unchanged"},
    ]}
    evidence = ["card-1"]
    requirements = [
        {"identifier": "one", "state": "resolved", "evidence_ids": evidence},
        {"identifier": "two", "state": "open", "detail": "Missing facts: source"},
    ]
    _sync_chapter_questions(chapter, requirements)
    assert chapter["questions"][0]["state"] == "answered"
    assert chapter["questions"][0]["supporting_card_ids"] == evidence
    assert chapter["questions"][0]["supporting_card_ids"] is not evidence
    assert chapter["questions"][1]["state"] == "open"
    assert chapter["questions"][1]["supporting_card_ids"] == []
    assert chapter["questions"][1]["missing_facts"] == "Missing facts: source"
    assert chapter["questions"][2]["state"] == "unchanged"


def test_cross_chapter_prerequisite_invalidation_updates_question_view():
    job = _job(open_question=False)
    requirement = job.evidence_requirements[0]
    requirement["dependencies"] = ["identity-q"]
    job.evidence_requirements.append({
        "identifier": "identity-q", "question": "Identify the service",
        "chapter_id": "other", "state": "open", "evidence_ids": [],
    })
    job.chapter_plan[0]["questions"][0].update(
        state="answered", supporting_card_ids=list(requirement["evidence_ids"]),
    )
    coverage.refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
    _sync_all_chapter_questions(job)
    assert requirement["state"] == "open"
    question = job.chapter_plan[0]["questions"][0]
    assert question["state"] == "open"
    assert question["supporting_card_ids"] == []


def test_refinement_resolves_question_from_small_subset_with_evidence_ids():
    job = _job()
    target_id = job._fixture_ids["answer"]

    def generate(**kwargs):
        alias = _packet_alias(kwargs["system_prompt"], ANSWER_STATEMENT)
        assert alias is not None, "answer card not present in refinement packet"
        return json.dumps({
            "answered": True,
            "answer": "Uploads over the local network are unsupported when offline.",
            "card_ids": [alias],
            "rephrased_question": "",
            "missing_facts": "",
            "close_facts": [],
        })

    chapter = _refine(job, generate)
    requirement = job.evidence_requirements[0]
    assert requirement["state"] == "resolved"
    assert requirement["evidence_ids"] == [target_id]
    assert requirement["answer_coverage"] == "evidence_scoped"
    assert requirement["accepted_answer"]["text"]
    assert chapter["questions"][0]["state"] == "answered"
    assert chapter["questions"][0]["supporting_card_ids"] == [target_id]
    assert chapter["refined_requirements"]["backup-q"]["rounds_done"] == 1
    assert chapter["refined_requirements"]["backup-q"]["stop_reason"] == "answered"
    assert chapter["refinement_calls"] == 1


def test_refinement_preserves_answer_when_cross_chapter_premise_is_resolved():
    job = _job()
    target_id = job._fixture_ids["answer"]
    job.evidence_requirements[0]["dependencies"] = ["identity-q"]
    job.evidence_requirements.append({
        "identifier": "identity-q", "chapter_id": "other", "question": "Identify the service",
        "is_subject_identity": True, "state": "resolved", "evidence_ids": [target_id],
        "answer_coverage": "evidence_scoped",
    })

    def generate(**kwargs):
        alias = _packet_alias(kwargs["system_prompt"], ANSWER_STATEMENT)
        return json.dumps({"answered": True, "answer": ANSWER_STATEMENT,
                           "card_ids": [alias], "close_facts": []})

    _refine(job, generate)
    assert job.evidence_requirements[0]["state"] == "resolved"
    assert job.evidence_requirements[0]["evidence_ids"] == [target_id]


def test_refinement_does_not_bypass_open_cross_chapter_premise():
    job = _job()
    job.evidence_requirements[0]["dependencies"] = ["identity-q"]
    job.evidence_requirements.append({
        "identifier": "identity-q", "chapter_id": "other", "question": "Identify the service",
        "is_subject_identity": True, "state": "open", "evidence_ids": [],
    })

    def generate(**kwargs):
        alias = _packet_alias(kwargs["system_prompt"], ANSWER_STATEMENT)
        return json.dumps({"answered": True, "answer": ANSWER_STATEMENT,
                           "card_ids": [alias], "close_facts": []})

    _refine(job, generate)
    assert job.evidence_requirements[0]["state"] == "open"
    assert job.evidence_requirements[0]["evidence_ids"] == []
    assert job.evidence_requirements[0]["candidate_answer"]["unresolved_dependencies"] == ["identity-q"]


def test_refinement_does_not_accept_a_missing_prerequisite():
    job = _job()
    job.evidence_requirements[0]["dependencies"] = ["missing-q"]

    def generate(**kwargs):
        alias = _packet_alias(kwargs["system_prompt"], ANSWER_STATEMENT)
        return json.dumps({"answered": True, "answer": ANSWER_STATEMENT,
                           "card_ids": [alias], "close_facts": []})

    _refine(job, generate)
    assert job.evidence_requirements[0]["state"] == "open"
    assert job.evidence_requirements[0]["candidate_answer"]["unresolved_dependencies"] == ["missing-q"]


def test_refinement_stays_open_and_bounded_when_no_card_answers():
    job = _job()
    calls = {"n": 0}

    def generate(**kwargs):
        calls["n"] += 1
        return json.dumps({
            "answered": False,
            "card_ids": [],
            "rephrased_question": "What version rule applies to encrypted sync?",
            "missing_facts": "no card states the upload rule",
            "close_facts": ["encrypted sync exists"],
        })

    chapter = _refine(job, generate)
    requirement = job.evidence_requirements[0]
    assert requirement["state"] == "open"
    assert requirement["evidence_ids"] == []
    assert calls["n"] <= coverage.MAX_REFINEMENT_ROUNDS
    assert chapter["refinement_calls"] == calls["n"]
    assert requirement["rephrased_question"]
    assert "no card states the upload rule" in requirement["detail"]
    assert chapter["questions"][0]["state"] == "open"
    progress = chapter["refined_requirements"]["backup-q"]
    assert progress["rounds_done"] == calls["n"]
    assert progress["seen_card_ids"]
    assert progress["stop_reason"] == "question_round_limit"
    assert "encrypted sync exists" in requirement["close_facts"]


def test_refinement_resolves_on_later_round_after_rephrase():
    job = _job()
    target_id = job._fixture_ids["near_miss"]
    calls = {"n": 0}

    def generate(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({
                "answered": False, "card_ids": [],
                "rephrased_question": "What version rule applies to encrypted sync?",
                "missing_facts": "nothing about the version rule yet",
                "close_facts": ["uploads are mentioned"],
            })
        alias = _packet_alias(kwargs["system_prompt"], NEAR_MISS_STATEMENT)
        assert alias is not None, "near-miss card not present in round-2 packet"
        return json.dumps({
            "answered": True,
            "answer": "Encrypted sync requires desktop client version 4 or later.",
            "card_ids": [alias],
            "rephrased_question": "", "missing_facts": "", "close_facts": [],
        })

    _refine(job, generate)
    requirement = job.evidence_requirements[0]
    assert requirement["state"] == "resolved"
    assert requirement["evidence_ids"] == [target_id]
    assert calls["n"] == 2
    assert "uploads are mentioned" in requirement["close_facts"]


def test_refinement_persists_close_facts_across_rounds():
    job = _job()
    calls = {"n": 0}

    def generate(**kwargs):
        calls["n"] += 1
        return json.dumps({
            "answered": False, "card_ids": [],
            "rephrased_question": f"Round {calls['n']} rephrase",
            "missing_facts": "still nothing",
            "close_facts": [f"fact from round {calls['n']}"],
        })

    chapter = _refine(job, generate)
    requirement = job.evidence_requirements[0]
    assert "fact from round 1" in requirement["close_facts"]
    assert "fact from round 2" in requirement["close_facts"]
    progress = chapter["refined_requirements"]["backup-q"]
    assert "fact from round 1" in progress["close_facts"]
    assert "fact from round 2" in progress["close_facts"]


def test_refinement_resumes_with_remaining_rounds_only():
    job = _job()
    chapter = _chapter()
    calls = []

    def interrupted(**kwargs):
        calls.append(kwargs["system_prompt"])
        if len(calls) == 2:
            raise ValueError("interrupted")
        return json.dumps({"answered": False, "card_ids": [],
                           "rephrased_question": "version rule", "close_facts": []})

    _refine(job, interrupted, chapter)
    assert chapter["refined_requirements"]["backup-q"]["rounds_done"] == 1
    assert chapter["refined_requirements"]["backup-q"]["stop_reason"] == "model_error"
    resumed = []

    def continue_rounds(**kwargs):
        resumed.append(kwargs["system_prompt"])
        return json.dumps({"answered": False, "card_ids": [],
                           "rephrased_question": "version rule", "close_facts": []})

    _refine(job, continue_rounds, chapter)
    assert len(resumed) == coverage.MAX_REFINEMENT_ROUNDS - 1
    assert all("follow-up pass" in prompt for prompt in resumed)
    assert chapter["refined_requirements"]["backup-q"]["rounds_done"] == coverage.MAX_REFINEMENT_ROUNDS
    assert chapter["refined_requirements"]["backup-q"]["stop_reason"] == "question_round_limit"
    _refine(job, continue_rounds, chapter)
    assert len(resumed) == coverage.MAX_REFINEMENT_ROUNDS - 1


def test_cached_batch_coverage_passes_chapter_mapping_to_refinement():
    class MatchingFingerprint:
        def __eq__(self, other):
            return isinstance(other, str)

    class ReachedCachedRefinement(Exception):
        pass

    job = _job()
    chapter = job.chapter_plan[0]
    chapter.update(tools=[], queries=[], batch_assessment_fingerprint=MatchingFingerprint())
    worker = _worker(job, lambda **_kwargs: "")
    worker.registry = SimpleNamespace(_schemas={})

    def inspect_refinement(_job, _chapter, _budget, _cancelled, *, requirements_by_chapter):
        assert requirements_by_chapter == {chapter["id"]: job.evidence_requirements}
        raise ReachedCachedRefinement

    worker._refine_open_chapter_coverage = inspect_refinement
    with pytest.raises(ReachedCachedRefinement):
        worker.run(job, lambda: False)


def test_refinement_is_noop_when_nothing_open_or_no_eligible_cards():
    # Already-resolved question: no calls at all.
    job = _job(open_question=False)
    calls = {"n": 0}
    chapter = _refine(job, lambda **kw: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 0
    assert "refined_requirements" not in chapter

    # Open question but empty ledger: no calls.
    job = _job()
    job.fact_cards = []
    calls = {"n": 0}
    _refine(job, lambda **kw: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 0


def test_refinement_budget_exhaustion_stops_without_error():
    job = _job()
    calls = {"n": 0}

    def generate(**kwargs):
        calls["n"] += 1
        return json.dumps({"answered": False, "card_ids": [],
                           "rephrased_question": "x", "missing_facts": "",
                           "close_facts": []})

    chapter = _chapter()
    worker = _worker(job, generate)
    budget = ThinkingBudget(SimpleNamespace(context_tokens=16000), {})
    # Pre-spend the whole chapter_coverage call allowance.
    budget.state["stages"]["chapter_coverage"] = {"calls": 18}
    _refine_open_chapter_coverage(
        worker, job, chapter, budget, lambda: False,
        requirements_by_chapter={chapter["id"]: job.evidence_requirements},
    )
    assert calls["n"] == 0
    assert job.evidence_requirements[0]["state"] == "open"


# ---------------------------------------------------------------------------
# Delivery promotion (Item B regression)
# ---------------------------------------------------------------------------

def test_loop_resolved_requirement_promotes_delivery_off_evidence_only():
    """A requirement the refinement loop resolves (evidence_ids set) must move
    classify_delivered_synthesis off evidence_only once its card is cited in
    the synthesis — no gate change, the existing evidence_ids path does it."""
    job = _job()
    target_id = job._fixture_ids["answer"]

    def generate(**kwargs):
        alias = _packet_alias(kwargs["system_prompt"], ANSWER_STATEMENT)
        assert alias is not None, "answer card not present in refinement packet"
        return json.dumps({
            "answered": True,
            "answer": "Uploads over the local network are unsupported when offline.",
            "card_ids": [alias],
            "rephrased_question": "",
            "missing_facts": "",
            "close_facts": [],
        })

    _refine(job, generate)
    requirement = job.evidence_requirements[0]
    assert requirement["state"] == "resolved"
    assert requirement["evidence_ids"] == [target_id]

    # Without the loop the requirement stays open and delivery is evidence_only.
    before = dict(requirement, state="open", evidence_ids=[])
    baseline = coverage.classify_delivered_synthesis(
        [before], [{"card_id": target_id}],
    )
    assert baseline["status"] == "evidence_only"

    # After the loop resolves it and the synthesis cites the card, delivery
    # is no longer evidence-only.
    after = coverage.classify_delivered_synthesis(
        [requirement], [{"card_id": target_id}],
    )
    assert requirement["delivered_synthesis"] == "evidence_scoped"
    assert after["status"] == "complete"
