import json

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_coverage import (
    apply_decomposition,
    apply_reasoned_coverage,
    apply_supported_coverage,
    chapter_requirements,
    fallback_chapter_plan,
    initial_requirements,
    refresh_coverage,
    validate_chapter_plan,
)
from core.thinking_fact_cards import build_fact_cards
from core.thinking_scope import research_context


def test_temporal_scope_and_coverage_share_coming_up_and_frozen_relative_dates():
    task = "Give me a deep analysis on the Bathurst 1000 race coming up"
    scope = research_context(task, as_of="2026-09-30T05:00:00Z")
    assert scope["temporal_intent"] == "future"
    requirements = initial_requirements(task, scope)
    assert any(row["identifier"] == "current-context" for row in requirements)
    assert all(row["temporal_scope"] == "future" for row in requirements)
    travel = research_context("Flights on 17th January next year, returning 3rd of March next year",
                              as_of=scope["as_of"])
    assert travel["requested_dates"] == ["2027-01-17", "2027-03-03"]
    ambiguous = research_context("Flights on 17 January", as_of=scope["as_of"])
    assert not ambiguous["requested_dates"] and ambiguous["ambiguous_dates"]


def test_bathurst_decomposition_is_task_derived_validated_and_preserves_constraints():
    task = "Give me a deep analysis on the Bathurst 1000 race coming up; do not predict a winner."
    scope = research_context(task, as_of="2026-09-30T05:00:00Z")
    dimensions = {"timing": ["schedule", "date"], "entrants/co-drivers": ["driver", "co-driver", "entrant"],
                  "recent form": ["form", "finished", "results"], "race factors": ["weather", "strategy"],
                  "uncertainty": ["uncertainty", "risk"]}
    plan = json.dumps({"requirements": [{"question": f"What is known about {name} for Bathurst 1000?",
        "dimension": name, "importance": "high", "temporal_scope": "future",
        "completion_criteria": f"Evidence directly establishes {name}; missing details stay visible.",
        "evidence_terms": terms} for name, terms in dimensions.items()]})
    requirements = apply_decomposition(initial_requirements(task, scope), plan, task, scope)
    assert {row["dimension"] for row in requirements if row.get("decomposed")} == set(dimensions)
    assert all(row["constraints"] == task for row in requirements)
    assert apply_decomposition(requirements, '{"requirements":[{"question":"invented"}]}', task, scope) is None

    evidence = [{"status": "evidence", "evidence": {"type": "web_research", "sources": [{
        "title": "Bathurst entrants recent form race factors schedule", "url": "https://hotel.example/event",
        "evidence": "Our accommodation offers rooms near Bathurst 1000.", "eligible": True,
    }]}}]
    cards = build_fact_cards(evidence, {}, task, scope)
    refresh_coverage(requirements, cards)
    assert requirements[0]["state"] == "open"
    assert all(row["state"] == "open" for row in requirements if row.get("decomposed"))
    assert all(row["delivered_synthesis"] == "not_started" for row in requirements)


def test_older_checkpoint_does_not_claim_verified_or_delivered_coverage():
    old = BackgroundJob("old", JobSnapshot("Research subject")).checkpoint()
    old.pop("checkpoint_version")
    old["evidence_requirements"] = [{"identifier": "answer", "state": "resolved"}]
    resumed = BackgroundJob.from_checkpoint(old)
    assert resumed.checkpoint_version == 6
    assert resumed.evidence_requirements[0]["state"] == "open"
    assert resumed.evidence_requirements[0]["answer_coverage"] == "not_verified"
    assert resumed.chapter_plan == []
    assert resumed.planning_complete is False


def test_requirement_decomposition_models_dependencies_and_rejects_cycles():
    task = "Analyze a sample topic"
    scope = research_context(task)
    rows = [
        {"question": "What is the initial condition?", "dimension": "initial", "importance": "high",
         "temporal_scope": "task-dependent", "completion_criteria": "Direct evidence",
         "evidence_terms": ["initial"], "required": True, "dependencies": []},
        {"question": "What follows from it?", "dimension": "result", "importance": "medium",
         "temporal_scope": "task-dependent", "completion_criteria": "Evidence plus premise",
         "evidence_terms": ["result"], "required": False, "dependencies": ["question-initial"]},
    ]
    requirements = apply_decomposition(initial_requirements(task, scope),
                                       json.dumps({"requirements": rows}), task, scope)
    assert requirements is not None
    assert requirements[-1]["required"] is False
    assert requirements[-1]["dependencies"] == ["question-initial"]
    assert requirements[-1]["completion_condition"] == "Evidence plus premise"

    rows[0]["dependencies"] = ["question-result"]
    assert apply_decomposition(initial_requirements(task, scope),
                               json.dumps({"requirements": rows}), task, scope) is None

    duplicate_rows = [dict(rows[0], dimension="battery runtime", dependencies=[]),
                      dict(rows[1], dimension="runtime battery", dependencies=[])]
    assert apply_decomposition(initial_requirements(task, scope),
                               json.dumps({"requirements": duplicate_rows}), task, scope) is None


def test_supported_answer_unit_persists_claim_support_premises_and_revision():
    cards = build_fact_cards([{"status": "evidence", "scope": "Vendor documentation",
        "evidence": {"type": "web_research", "sources": [{
            "title": "Backup guide", "url": "https://docs.example/backup",
            "evidence": "The service supports mobile backup.",
        }]}}], {}, "Does the service support mobile backup?")
    reqs = initial_requirements("Does the service support mobile backup?")
    apply_supported_coverage(reqs, cards, {"answers": [{
        "question_id": "answer", "answered": True, "card_ids": [cards[0]["id"]],
        "answer": "Yes, mobile backup is supported.", "claim": "Mobile backup is supported.",
        "premises": ["The vendor says it supports mobile backup."],
        "unresolved_fields": ["Backup retention duration"],
    }]})
    answer = reqs[0]["accepted_answer"]
    assert answer["supporting_card_ids"] == [cards[0]["id"]]
    assert answer["premises"] == ["The vendor says it supports mobile backup."]
    assert answer["unresolved_fields"] == ["Backup retention duration"]
    assert answer["revision"] == 1


def test_conceptual_answer_uses_explicit_premises_without_web_cards_but_current_claim_does_not():
    requirements = [
        {"identifier": "logic", "freshness": "timeless", "state": "open",
         "reasoning_allowed": True, "explicit_premises": ["Premise A", "Premise B"]},
        {"identifier": "current", "freshness": "fresh", "state": "open"},
    ]
    apply_reasoned_coverage(requirements, {"answers": [
        {"question_id": "logic", "answered": True, "answer_kind": "reasoned_conversation",
         "answer": "The conclusion follows.", "premises": ["Premise A", "Premise B"]},
        {"question_id": "current", "answered": True, "answer_kind": "reasoned_conversation",
         "answer": "The current price is $10.", "premises": ["A prior price was $10"]},
    ]})
    assert requirements[0]["accepted_answer"]["answer_kind"] == "reasoned_conversation"
    assert requirements[0]["accepted_answer"]["supporting_card_ids"] == []
    assert requirements[1]["state"] == "open"


def test_reasoned_answer_is_rejected_without_explicit_premises_when_contract_declares_them():
    """A contract that declares explicit premises binds the answer to them."""
    requirements = [
        {"identifier": "logic", "freshness": "timeless", "state": "open",
         "reasoning_allowed": True, "explicit_premises": ["The user said A implies B"]},
        {"identifier": "logic2", "freshness": "timeless", "state": "open",
         "reasoning_allowed": True, "explicit_premises": []},
        {"identifier": "forbidden", "freshness": "timeless", "state": "open",
         "reasoning_allowed": False},
    ]
    apply_reasoned_coverage(requirements, {"answers": [
        {"question_id": "logic", "answered": True, "answer_kind": "reasoned_conversation",
         "answer": "B follows.", "premises": ["An invented premise"]},
        {"question_id": "logic2", "answered": True, "answer_kind": "reasoned_conversation",
         "answer": "Nothing rests on declared premises.", "premises": ["A user premise"]},
        {"question_id": "forbidden", "answered": True, "answer_kind": "reasoned_conversation",
         "answer": "A conclusion.", "premises": ["A declared premise"]},
    ]})
    assert all(requirement["state"] == "open" for requirement in requirements)
    # A premise drawn from the declared list is still accepted.
    apply_reasoned_coverage(requirements, {"answers": [
        {"question_id": "logic", "answered": True, "answer_kind": "reasoned_conversation",
         "answer": "B follows.", "premises": ["The user said A implies B"]},
    ]})
    assert requirements[0]["state"] == "resolved"
    assert requirements[0]["accepted_answer"]["premises"] == ["The user said A implies B"]
    assert requirements[1]["state"] == "open"
    assert requirements[2]["state"] == "open"


def _chapter_plan_requirements():
    task = "Analyze a sample topic"
    scope = research_context(task)
    rows = [
        {"question": "What is the initial condition?", "dimension": "initial", "importance": "high",
         "temporal_scope": "task-dependent", "completion_criteria": "Direct evidence",
         "evidence_terms": ["initial"], "required": True, "dependencies": []},
        {"question": "What follows from it?", "dimension": "result", "importance": "medium",
         "temporal_scope": "task-dependent", "completion_criteria": "Evidence plus premise",
         "evidence_terms": ["result"], "required": False, "dependencies": ["question-initial"]},
    ]
    return task, scope, apply_decomposition(
        initial_requirements(task, scope), json.dumps({"requirements": rows}), task, scope)


def test_chapter_plan_translates_dependencies_and_preserves_claim_bindings():
    task, scope, requirements = _chapter_plan_requirements()
    plan = validate_chapter_plan(
        {"chapters": [{
            "title": "Initial condition and its result",
            "priority": "high",
            "questions": [
                {"question": "What is the initial condition?", "priority": "high"},
                {"question": "What follows from it?", "priority": "medium"},
            ],
            "tools": ["web"], "queries": ["What is the initial condition?"],
        }]},
        task, requirements, ["web"],
    )
    assert plan is not None
    by_question = {question["question"]: question for chapter in plan
                   for question in chapter["questions"]}
    first = by_question["What is the initial condition?"]
    second = by_question["What follows from it?"]
    # Dependencies travel with the question across the source-ID to requirement-ID
    # translation performed by the validator.
    assert second["dependencies"] == [first["requirement_id"]]
    assert second["source_requirement_id"] == "question-result"
    assert second["required"] is False
    assert second["completion_condition"] == "Evidence plus premise"
    # Question dependencies must not be confused with card-ID claim dependencies.
    built = chapter_requirements(plan, task, scope)
    built_by_id = {item["identifier"]: item for item in built}
    assert built_by_id[second["requirement_id"]]["dependencies"] == [first["requirement_id"]]
    assert built_by_id[second["requirement_id"]]["claim_dependencies"] == []
    assert built_by_id[second["requirement_id"]]["dependencies"] == [first["requirement_id"]]
    assert built_by_id[second["requirement_id"]]["required"] is False


def test_chapter_plan_rejects_dependency_cycles_and_dangling_references():
    task, scope, requirements = _chapter_plan_requirements()
    cycle_rows = [
        {"question": "What is the initial condition?", "dimension": "initial", "importance": "high",
         "temporal_scope": "task-dependent", "completion_criteria": "Direct evidence",
         "evidence_terms": ["initial"], "required": True, "dependencies": ["question-result"]},
        {"question": "What follows from it?", "dimension": "result", "importance": "medium",
         "temporal_scope": "task-dependent", "completion_criteria": "Evidence plus premise",
         "evidence_terms": ["result"], "required": False, "dependencies": ["question-initial"]},
    ]
    cycled = apply_decomposition(initial_requirements(task, scope),
                                 json.dumps({"requirements": cycle_rows}), task, scope)
    assert cycled is None
    # The decomposed requirement list is the only source of dependency truth; a
    # plan that cannot be built from a valid decomposition never reaches
    # dispatch with a cycle.

    dangling_rows = [
        {"question": "What is the initial condition?", "dimension": "initial", "importance": "high",
         "temporal_scope": "task-dependent", "completion_criteria": "Direct evidence",
         "evidence_terms": ["initial"], "required": True, "dependencies": []},
        {"question": "What follows from it?", "dimension": "result", "importance": "medium",
         "temporal_scope": "task-dependent", "completion_criteria": "Evidence plus premise",
         "evidence_terms": ["result"], "required": False, "dependencies": ["question-missing"]},
    ]
    assert apply_decomposition(initial_requirements(task, scope),
                               json.dumps({"requirements": dangling_rows}), task, scope) is None

    # A manually constructed chapter list that still carries a dangling or cyclic
    # dependency is rejected by the validator even though it bypasses
    # apply_decomposition.
    def chapter_with(dependencies):
        return {"chapters": [{
            "title": "Initial condition and its result",
            "priority": "high",
            "questions": [
                {"question": "What is the initial condition?", "priority": "high"},
                {"question": "What follows from it?", "priority": "medium"},
            ],
            "tools": ["web"], "queries": ["What is the initial condition?"],
        }]}
    clean = validate_chapter_plan(chapter_with(None), task, requirements, ["web"])
    assert clean is not None
    clean_chapter = clean[0]
    for question in clean_chapter["questions"]:
        if question["question"] == "What follows from it?":
            question["dependencies"] = ["chapter-00-q09"]  # dangling
        elif question["question"] == "What is the initial condition?":
            question["dependencies"] = ["chapter-00-q02"]  # cycle
    assert validate_chapter_plan(clean, task, requirements, ["web"]) is None
    clean_chapter["questions"][0]["dependencies"] = ["chapter-00-q01"]  # self-cycle
    assert validate_chapter_plan(clean, task, requirements, ["web"]) is None


def test_fallback_chapter_plan_preserves_dependency_metadata():
    task, scope, requirements = _chapter_plan_requirements()
    plan = fallback_chapter_plan(task, requirements, ["web"])
    assert plan is not None
    questions = {question["question"]: question for chapter in plan
                 for question in chapter["questions"]}
    first = questions["What is the initial condition?"]
    second = questions["What follows from it?"]
    assert second["source_requirement_id"] == "question-result"
    assert second["required"] is False
    assert second["completion_condition"] == "Evidence plus premise"
    # The fallback keeps source-obligation IDs; chapter_requirements resolves
    # them into this plan's requirement IDs, separately from card IDs.
    built = chapter_requirements(plan, task, scope)
    built_by_id = {item["identifier"]: item for item in built}
    assert built_by_id[second["requirement_id"]]["dependencies"] == [first["requirement_id"]]
    assert built_by_id[second["requirement_id"]]["claim_dependencies"] == []
