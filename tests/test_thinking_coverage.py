"""Evidence-coverage planning stays deterministic around the model worker."""

from core import thinking_coverage as coverage


def test_fallback_search_seeds_are_facts_not_repeated_report_assignments():
    task = "give me a deep analysis on the next Formula 1 race coming up"
    requirements = coverage.initial_requirements(task)
    plan = coverage.fallback_chapter_plan(task, requirements, ["external_information"])
    queries = [query for chapter in plan for query in chapter["queries"]]
    assert queries and all(len(query) <= 240 for query in queries)
    assert all("give me" not in query and "deep analysis" not in query and " — " not in query for query in queries)
    identity = next(item for item in requirements if item["identifier"] == "subject-identity")
    query = coverage.searchable_gap_query(task, identity)
    assert "Formula 1" in query and "official date schedule" in query
    assert "Resolve the exact" not in query and "source evidence" not in query


def test_gap_search_preserves_pin_and_specific_needed_fact_without_rewriting_task():
    task = "Compare the upcoming release with version 4.2, excluding cloud-only deployments"
    gap = {"question": "Which local runtime limitations remain?", "subject_resolution": {
        "state": "resolved", "name": "Example Runtime", "date": "2026-11-12", "version": "4.3",
    }}
    query = coverage.searchable_gap_query(task, gap)
    assert "Example Runtime 2026-11-12 4.3" in query
    assert "local runtime limitations" in query
    assert task == "Compare the upcoming release with version 4.2, excluding cloud-only deployments"


def test_initial_requirements_capture_current_research_tradeoffs():
    requirements = coverage.initial_requirements("Research the latest options and compare them")

    assert [item["identifier"] for item in requirements] == [
        "answer",
        "current-context",
        "tradeoffs",
    ]
    assert all(item["state"] == "open" for item in requirements)


def test_local_model_research_requires_identity_freshness_runtime_and_trend_evidence():
    task = "Find latest info on trending local models with 9B parameters; keep sources recent"
    requirements = coverage.initial_requirements(task)
    by_id = {item["identifier"]: item for item in requirements}

    assert {
        "model-identity", "model-freshness", "model-runtime-fit", "model-trend-evidence",
    } <= set(by_id)
    assert "9B parameters" in by_id["model-identity"]["constraints"]
    assert "active" in by_id["model-identity"]["question"]
    assert "primary model records" in by_id["model-freshness"]["question"]
    assert "quantization" in by_id["model-runtime-fit"]["question"]
    assert "dated usage" in by_id["model-trend-evidence"]["question"]


def test_valid_but_incomplete_generated_plan_retains_model_answer_obligations():
    task = "Find latest info on trending local models with 9B parameters; keep sources recent"
    requirements = coverage.initial_requirements(task)
    plan = coverage.validate_chapter_plan({"chapters": [{
        "title": "Local model overview", "priority": "high",
        "questions": ["Which models are relevant?"],
        "tools": ["external_information"], "queries": ["latest local 9B model releases"],
    }]}, task, requirements, ["external_information"])

    assert plan is not None
    represented = {
        question["question"] for chapter in plan for question in chapter["questions"]
    }
    required = {
        item["question"] for item in requirements
        if item["identifier"] in {
            "model-identity", "model-freshness", "model-runtime-fit", "model-trend-evidence",
        }
    }
    assert required <= represented
    assert any("9B parameters" in query for chapter in plan for query in chapter["queries"])

    fallback = coverage.fallback_chapter_plan(task, requirements, ["external_information"])
    fallback_questions = {
        question["question"] for chapter in fallback for question in chapter["questions"]
    }
    assert required <= fallback_questions


def test_upcoming_claim_requires_current_context_and_ineligible_evidence_does_not_resolve():
    requirements = coverage.initial_requirements("Deep analysis of the upcoming race")

    assert [item["identifier"] for item in requirements][:3] == ["answer", "current-context", "tradeoffs"]
    assert any(item["identifier"] == "subject-identity" for item in requirements)
    assert {"scope-timing", "actors-options", "recent-evidence", "decisive-factors"} <= {
        item["identifier"] for item in requirements
    }
    coverage.resolve_requirement(requirements, requirements[0], status="evidence", scope="Wrong event.", eligible=False)
    assert requirements[0]["state"] == "open"


def test_requirement_resolution_and_bounded_exhaustion():
    requirements = coverage.initial_requirements("Compare local models")
    selected = coverage.open_requirement(requirements)

    coverage.resolve_requirement(
        requirements, selected, status="evidence", scope="One provider result."
    )

    assert requirements[0]["state"] == "open"
    assert requirements[0]["evidence_availability"] == "available"
    assert coverage.coverage_complete(requirements) is False
    coverage.mark_remaining_exhausted(requirements, "No distinct action remains.")
    assert coverage.coverage_complete(requirements) is False
    assert requirements[0]["state"] == "exhausted"
    assert requirements[1]["state"] == "exhausted"


def test_delivery_gate_requires_every_original_obligation_to_have_cited_synthesis():
    requirements = [
        {"identifier": "model-identity", "state": "resolved", "evidence_ids": ["card-a"]},
        {"identifier": "model-trend-evidence", "state": "open", "evidence_ids": []},
    ]
    assessment = coverage.classify_delivered_synthesis(
        requirements, [{"card_id": "card-a"}, {"card_id": "generic-card"}],
    )

    assert requirements[0]["delivered_synthesis"] == "evidence_scoped"
    assert requirements[1]["delivered_synthesis"] == "missing"
    assert assessment == {
        "status": "partial",
        "requirements": {"evidence_scoped": 1, "partial": 0, "extracted_only": 0, "missing": 1},
    }


def test_generic_citations_cannot_close_a_topicless_task_and_extracts_are_evidence_only():
    requirements = [{"identifier": "answer", "state": "open", "evidence_ids": []}]

    generic = coverage.classify_delivered_synthesis(
        requirements, [{"card_id": "generic-one"}, {"card_id": "generic-two"}],
    )
    assert requirements[0]["delivered_synthesis"] == "missing"
    assert generic["status"] == "evidence_only"

    requirements[0]["evidence_ids"] = ["extract-1"]
    evidence_only = coverage.classify_delivered_synthesis(
        requirements, [{"card_id": "extract-1"}], ["extract-1"],
    )
    assert requirements[0]["delivered_synthesis"] == "extracted_only"
    assert evidence_only["status"] == "evidence_only"


def test_delivery_gate_reports_complete_only_when_all_obligations_are_fully_delivered():
    requirements = [
        {"identifier": "identity", "state": "resolved", "evidence_ids": ["a"]},
        {"identifier": "freshness", "state": "resolved", "evidence_ids": ["b", "c"]},
    ]
    assessment = coverage.classify_delivered_synthesis(
        requirements, [{"card_id": "a"}, {"card_id": "b"}, {"card_id": "c"}],
    )

    assert assessment["status"] == "complete"
    assert all(item["delivered_synthesis"] == "evidence_scoped" for item in requirements)


def test_lexical_duplicate_rejects_reworded_action():
    selections = [{"tool": "external_information", "args": ["Bathurst 1000 entry list"]}]

    assert coverage.is_near_duplicate_action(
        "external_information", ["Who is on the Bathurst 1000 entry list?"], selections
    )


def test_duplicate_allows_changed_date_or_source():
    selections = [{"tool": "external_information", "args": ["Bathurst results 2026-10-08"]}]

    assert not coverage.is_near_duplicate_action(
        "external_information", ["Bathurst results 2026-10-09"], selections
    )


def test_lexical_guard_rejects_reordered_words():
    selections = [{"tool": "external_information", "args": ["Bathurst 1000 entry list driver"]}]

    assert coverage.is_near_duplicate_action(
        "external_information", ["Bathurst 1000 driver entry list"], selections
    )


def test_duplicate_guard_allows_changed_profile_or_period_but_not_only_a_changed_gap():
    selections = [
        {
            "tool": "external_information",
            "args": ["Bathurst 1000 schedule", "research", "news", "week"],
            "requirement_id": "answer",
        },
        {
            "tool": "search_flights",
            "args": [{"origin": "SYD", "destination": "TYO", "departure_date": "2027-05-12"}],
            "requirement_id": "answer",
        },
    ]

    assert not coverage.is_near_duplicate_action(
        "external_information", ["Bathurst 1000 schedule", "conversation", "news", "week"], selections
    )
    assert not coverage.is_near_duplicate_action(
        "external_information", ["Bathurst 1000 schedule Q2 2027", "research", "news", "week"], selections
    )
    assert not coverage.is_near_duplicate_action(
        "search_flights",
        [{"origin": "SYD", "destination": "TYO", "departure_date": "2027-05-13"}],
        selections,
    )
    assert coverage.is_near_duplicate_action(
        "external_information",
        ["Bathurst 1000 schedule", "research", "news", "week"],
        selections,
        {"identifier": "tradeoffs"},
    )


def test_general_retrieval_is_directed_to_the_named_gap_without_rewriting_structured_tools():
    requirement = {
        "identifier": "entrants",
        "question": "Which entrants and driver pairings are confirmed?",
        "evidence_terms": ["entrants", "drivers", "pairings"],
    }

    assert coverage.direct_retrieval_to_gap(
        "external_information", ["Bathurst 1000", "research"], requirement
    )[0] == "Bathurst 1000 entrants drivers pairings"
    assert coverage.direct_retrieval_to_gap(
        "external_information", ["Bathurst 1000 entrants", "research"], requirement
    )[0] == "Bathurst 1000 entrants"
    structured = [{"route": {"origin": "SYD", "destination": "TYO"}}]
    assert coverage.direct_retrieval_to_gap("search_flights", structured, requirement) == structured


def test_third_capability_call_requires_measured_gain():
    stalled = [
        {"tool": "external_information", "new_eligible_cards": 0, "coverage_gain": 0},
        {"tool": "external_information", "new_eligible_cards": 0, "coverage_gain": 0},
    ]
    productive = [*stalled[:1], {
        "tool": "external_information", "new_eligible_cards": 2, "coverage_gain": 1,
    }]

    assert coverage.effective_capability_call_limit("external_information", stalled) == 2
    assert coverage.effective_capability_call_limit("external_information", productive) == 3
    assert coverage.effective_capability_call_limit("external_information", [{}]) == 3


def test_irrelevant_card_volume_cannot_extend_retrieval_and_distinct_verification_can():
    selections = [{"tool": "lookup", "new_eligible_cards": 50, "useful_finding_gain": 0, "coverage_gain": 0}] * 2
    assert coverage.effective_capability_call_limit("lookup", selections) == 2
    assert not coverage.is_near_duplicate_query("lookup", ["Example release 2026-11-01 site:official.example"], [
        {"tool": "lookup", "args": ["Example release 2026-10-01 site:other.example"]},
    ])


def test_broad_analysis_uncertainty_cannot_be_completed_by_an_unrelated_alternative():
    requirement = next(
        item for item in coverage.initial_requirements("Give me a deep analysis of the upcoming race")
        if item["identifier"] == "tradeoffs"
    )

    assert requirement["dimension"] == "uncertainty"
    assert "alternative" not in coverage.requirement_terms(requirement)
