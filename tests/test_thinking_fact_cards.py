"""Canonical fact-card derivation from bounded typed evidence."""

from core.thinking_fact_cards import (
    _key_facts,
    build_fact_cards,
    card_is_fresh_for_current_claim,
    eligible_cards_for_requirement,
    reconcile_fact_cards,
    report_card_ledger,
    report_limitations,
    select_requirement_cards,
    validate_fact_card,
)
from core.thinking_fallback import _fallback_fact_body


def test_navigation_title_and_non_answer_comment_cannot_supply_coverage():
    source_text = (
        "Just realised that the best of us do this. "
        "'Do the needful' actually means 'do what is necessary'."
    )
    evidence = [{
        "status": "evidence",
        "evidence": {"type": "web_research", "sources": [{
            "url": "https://example.test/comment", "title": "Just realised that the best of us do this",
            "source_text": source_text, "evidence": source_text, "eligible": True,
        }]},
    }]

    cards = build_fact_cards(evidence, {}, "Research what the timing display's green C means")

    assert cards == []


def _card_for_freshness(*, temporal_role, context=None, fields=None, retrieved_at=None):
    return {
        "id": f"card-{temporal_role}",
        "kind": "atomic_claim",
        "label": "Test card",
        "fields": fields or {},
        "source_ids": ["artifact-001"],
        "source_class": "official_primary",
        "retrieved_at": retrieved_at,
        "freshness": "unknown",
        "evidence_status": "single_source",
        "scope": "Test scope.",
        "compatibility": "compatible",
        "temporal_role": temporal_role,
        "context": context or {},
    }


def test_key_facts_extract_bounded_date_and_value_sentences():
    text = (
        "The 2026 Bathurst 1000 is scheduled for 8–11 October 2026 at Mount Panorama. "
        "Drivers confirmed their entries on 2026-09-20. The race covers 161 laps. "
        "It is one of the most famous endurance races in motorsport. "
        "Ticket prices start at AUD 85 for general admission."
    )
    facts = _key_facts(text)
    assert len(facts) <= 5
    assert any("2026-09-20" in fact for fact in facts)
    assert any("161 laps" in fact for fact in facts)
    assert any("AUD 85" in fact for fact in facts)
    # Context-only sentences (no date, no value) are not extracted.
    assert not any("most famous" in fact for fact in facts)


def test_key_facts_ignores_navigation_boilerplate():
    text = (
        "Home Menu News About Contact. The event took place on 2026-10-08 with 43 cars on the grid."
    )
    facts = _key_facts(text)
    assert len(facts) == 1
    assert "43 cars" in facts[0]


def test_web_card_key_facts_and_claim_key_reflect_facts_not_url():
    evidence = [
        {
            "status": "evidence",
            "scope": "Two sources.",
            "artifact_id": "artifact-001",
            "evidence": {},
        }
    ]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "web_research",
                "sources": [
                    {
                        "host": "a.example",
                        "url": "https://a.example/1000",
                        "evidence": "The 2026 Bathurst 1000 starts on 2026-10-08. Entry closed on 2026-09-25 with 43 cars confirmed.",
                        "eligible": True,
                    },
                    {
                        "host": "b.example",
                        "url": "https://b.example/12-hour",
                        "evidence": "The 2025 Bathurst 12 Hour finished on 2025-10-05. The winning car completed 161 laps in the event.",
                        "eligible": True,
                    },
                ],
            }
        }
    }

    cards = build_fact_cards(evidence, artifacts)

    assert len(cards) == 4
    by_url = {}
    for card in cards:
        by_url.setdefault(card["source_url"], []).append(card)
    c1000 = by_url["https://a.example/1000"]
    c12 = by_url["https://b.example/12-hour"]
    assert any("2026-10-08" in card["statement"] for card in c1000)
    assert any("2025-10-05" in card["statement"] for card in c12)
    assert {card["claim_key"] for card in c1000}.isdisjoint(
        {card["claim_key"] for card in c12}
    )


def test_current_freshness_is_claim_type_aware_without_age_cutoff():
    provider = _card_for_freshness(
        temporal_role="provider_snapshot",
        context={"observed_at": "2026-09-28T12:00:00Z"},
    )
    undated_document = _card_for_freshness(temporal_role="document", retrieved_at="2026-09-28T12:00:00Z")
    future_schedule = _card_for_freshness(
        temporal_role="event_context",
        context={"observed_at": "2026-09-28T12:00:00Z", "event_dates": ["2026-10-08"]},
    )
    dated_paper = _card_for_freshness(
        temporal_role="publication", context={"published_at": "2019-01-01"}
    )

    assert card_is_fresh_for_current_claim(provider)
    assert not card_is_fresh_for_current_claim(undated_document)
    assert card_is_fresh_for_current_claim(future_schedule)
    assert card_is_fresh_for_current_claim(dated_paper)
    # The same undated stable document remains usable when the requirement is
    # not current, rather than being discarded globally for lacking a date.
    assert eligible_cards_for_requirement(
        [undated_document], {"freshness": "task-dependent"}
    ) == [undated_document]
    # Unknown freshness is supplied to the model, not a reason to hide a source.
    assert eligible_cards_for_requirement([undated_document], {"freshness": "fresh"}) == [undated_document]


def test_flight_cards_keep_complete_offers_together():
    offer = {
        "price": 2451,
        "outbound": {"departure": {"time": "2027-01-17 14:40"}},
        "inbound": {"departure": {"time": "2027-03-03 21:45"}},
    }
    evidence = [
        {"status": "evidence", "scope": "One offer.", "artifact_id": "artifact-001", "evidence": {}}
    ]
    artifacts = {"artifact-001": {"data": {"type": "flight_search", "offers": [offer]}}}

    cards = build_fact_cards(evidence, artifacts)

    assert len(cards) == 1
    assert cards[0]["kind"] == "structured_record"
    assert cards[0]["fields"] == offer
    assert cards[0]["source_class"] == "specialised_provider"


def test_partial_structured_card_stays_eligible_for_scoped_fact_not_full_option():
    evidence = [{"status": "evidence", "scope": "One offer.", "artifact_id": "artifact-001", "evidence": {}}]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "flight_search",
                "route": {"origin": "AAA", "destination": "BBB"},
                "offers": [{"price": 2451, "departure": "2027-01-17", "arrival": "2027-01-18"}],
            }
        }
    }

    cards = build_fact_cards(evidence, artifacts)

    assert report_card_ledger(cards) == cards
    assert cards[0]["claim_eligibility"] == {
        "scoped_claims": True,
        "complete_structured_option": False,
        "missing_option_fields": ["currency"],
    }
    assert "Missing fields for a complete structured option: currency." in report_limitations(evidence, cards)[0]["limitation"]


def test_card_validation_rejects_invalid_citeable_source_url():
    card = _card_for_freshness(temporal_role="document")
    card["source_url"] = "file:///private/source"

    assert validate_fact_card(card) == "card source URL must be an HTTP(S) URL"


def test_web_cards_preserve_source_provenance_and_excerpt():
    evidence = [{"status": "evidence", "scope": "Two sources.", "evidence": {"sources": []}}]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "web_research",
                "sources": [
                    {
                        "host": "example.com",
                        "url": "https://example.com/a",
                        "evidence": "A source-backed fact is supported by enough extracted text to make an eligible evidence card for this test.",
                    }
                ],
            }
        }
    }
    evidence[0]["artifact_id"] = "artifact-001"

    cards = build_fact_cards(evidence, artifacts)

    assert cards[0]["statement"].startswith("A source-backed fact")
    assert "url" not in cards[0]["fields"]
    assert cards[0]["source_host"] == "example.com"


def test_unknown_typed_evidence_has_a_scoped_generic_card():
    cards = build_fact_cards(
        [
            {
                "tool": "read_sensor",
                "status": "evidence",
                "scope": "One current state.",
                "evidence": {"state": "on"},
            }
        ],
        {},
    )

    assert cards[0]["kind"] == "generic_observation"
    assert "No specialised" in cards[0]["limitations"]


def test_invalid_card_is_rejected():
    assert validate_fact_card({"id": "missing"}) is not None


def test_reconciliation_does_not_confirm_matching_text_without_claim_identity():
    cards = [
        {
            "id": "one",
            "kind": "atomic_claim",
            "label": "One",
            "fields": {"value": 1},
            "source_ids": ["one"],
            "source_class": "secondary_reporting",
            "retrieved_at": None,
            "freshness": "unknown",
            "evidence_status": "single_source",
            "scope": "One.",
            "independence_key": "a",
            "claim_key": "same claim",
        },
        {
            "id": "two",
            "kind": "atomic_claim",
            "label": "Two",
            "fields": {"value": 1},
            "source_ids": ["two"],
            "source_class": "secondary_reporting",
            "retrieved_at": None,
            "freshness": "unknown",
            "evidence_status": "single_source",
            "scope": "One.",
            "independence_key": "b",
            "claim_key": "same claim",
        },
    ]

    reconcile_fact_cards(cards)

    assert [card["evidence_status"] for card in cards] == ["single_source", "single_source"]


def test_reconciliation_uses_subject_type_and_period_boundaries():
    prior_event = _card_for_freshness(
        temporal_role="event_context",
        context={"event_dates": ["2025-02-02"], "observed_at": "2026-09-28T00:00:00Z"},
        fields={"key_facts": ["The Bathurst 12 Hour finished on 2025-02-02."]},
    )
    upcoming_event = _card_for_freshness(
        temporal_role="event_context",
        context={"event_dates": ["2026-10-08"], "observed_at": "2026-09-28T00:00:00Z"},
        fields={"key_facts": ["The Bathurst 1000 starts on 2026-10-08."]},
    )
    for card, period, value, independence in (
        (prior_event, "2025-02-02", "finished", "prior"),
        (upcoming_event, "2026-10-08", "starts", "upcoming"),
    ):
        card.update(
            claim_identity={"subject": "Bathurst", "claim_type": "event_schedule", "period": period},
            fields={"value": value},
            independence_key=independence,
        )

    reconcile_fact_cards([prior_event, upcoming_event])

    assert prior_event["evidence_status"] == "single_source"
    assert upcoming_event["evidence_status"] == "single_source"


def test_reconciliation_confirms_or_conflicts_only_with_matching_identity():
    cards = []
    for identifier, value, independence in (
        ("one", "8 October", "publisher-a"),
        ("two", "8 October", "publisher-b"),
    ):
        card = _card_for_freshness(
            temporal_role="event_context",
            context={"event_dates": ["2026-10-08"], "observed_at": "2026-09-28T00:00:00Z"},
            fields={"value": value},
        )
        card.update(
            id=identifier,
            claim_identity={"subject": "Bathurst 1000", "claim_type": "event_schedule", "period": "2026-10-08"},
            independence_key=independence,
        )
        cards.append(card)

    reconcile_fact_cards(cards)

    assert [card["evidence_status"] for card in cards] == ["confirmed", "confirmed"]
    cards[1]["fields"] = {"value": "9 October"}
    cards[1]["evidence_status"] = "single_source"
    reconcile_fact_cards(cards)
    assert [card["evidence_status"] for card in cards] == ["conflicted", "conflicted"]


def test_explicit_disagreement_is_flagged_even_without_independent_publishers():
    cards = []
    for value in ("Three wildcards were announced.", "Two wildcard pairings were listed."):
        card = _card_for_freshness(temporal_role="event_context", context={"event_dates": ["2026"]})
        card.update(
            statement=value,
            claim_identity={"subject": "Bathurst 1000", "claim_type": "wildcard_count", "period": "2026"},
            independence_key="unknown-shared-origin",
        )
        cards.append(card)

    reconcile_fact_cards(cards)

    assert [card["evidence_status"] for card in cards] == ["conflicted", "conflicted"]


def test_structured_count_list_inconsistency_is_flagged():
    count = _card_for_freshness(temporal_role="event_context", fields={"count": 3})
    listed = _card_for_freshness(temporal_role="event_context", fields={"pairings": ["A/B", "C/D"]})
    for card in (count, listed):
        card["claim_identity"] = {"subject": "event", "claim_type": "wildcards", "period": "2026"}

    reconcile_fact_cards([count, listed])

    assert count["evidence_status"] == listed["evidence_status"] == "conflicted"


def test_incomplete_table_headers_are_inspectable_but_cannot_support_relationships():
    card = {
        "kind": "structured_record",
        "fields": {
            "headers": ["Driver", ""], "cells": ["Alex Example", "Example Racing"],
            "relationship_status": "incomplete headers; no relationships inferred",
        },
    }

    assert _fallback_fact_body(card) == ""


def test_ambiguous_table_relationships_are_not_admitted_as_direct_support():
    text = "Driver | Team\nAlex Example | Example Racing | extra cell"
    evidence = [{
        "status": "evidence", "artifact_id": "artifact-table", "evidence": {},
        "scope": "One table source.",
    }]
    artifacts = {"artifact-table": {"data": {"type": "web_research", "sources": [{
        "url": "https://example.test/entry-list", "title": "Entry list",
        "source_text": text, "evidence": text, "eligible": True,
    }]}}}

    cards = build_fact_cards(evidence, artifacts, "Research the main race driver and team pairings")

    assert len(cards) == 1
    assert cards[0]["fields"]["relationship_status"] == "ambiguous columns; no relationships inferred"
    assert cards[0]["compatibility"] == "ambiguous"
    assert "Table relationships are unresolved" in cards[0]["limitations"]
    assert report_card_ledger(cards) == []


def test_late_parameter_correction_conflicts_across_page_dates_and_reopens_coverage():
    from core.thinking_coverage import refresh_coverage

    sources = [
        {
            "url": "https://models.example.test/model-01", "title": "Model-01 card",
            "source_text": "Model-01 has 9 billion total parameters.",
            "published_at": "2026-09-01", "eligible": True,
        },
        {
            "url": "https://corrections.example.test/model-01", "title": "Model-01 correction",
            "source_text": "Model-01 has 10 billion total parameters.",
            "published_at": "2026-09-30", "eligible": True,
        },
    ]
    evidence = [{
        "status": "evidence", "artifact_id": "artifact-models", "evidence": {},
        "scope": "Model records.",
    }]
    artifacts = {"artifact-models": {"data": {"type": "web_research", "sources": sources}}}

    cards = build_fact_cards(evidence, artifacts, "Verify Model-01's total parameter count")

    assert len(cards) == 2
    assert {card["evidence_status"] for card in cards} == {"conflicted"}
    requirement = {
        "identifier": "model-parameter-count", "question": "What is Model-01's total parameter count?",
        "state": "resolved", "answer_coverage": "evidence_scoped",
        "evidence_ids": [cards[0]["id"]], "freshness": "task-dependent",
    }
    refresh_coverage([requirement], cards)

    assert requirement["state"] == "open"
    assert requirement["answer_coverage"] != "evidence_scoped"
    assert requirement["evidence_ids"] == []


def test_selection_limits_working_set_without_dropping_relevant_qualifications():
    cards = [{
        "id": f"record-{index}", "kind": "atomic_claim", "label": f"Record {index}",
        "statement": ("Battery runtime is 12 hours under laboratory conditions; "
                      "results vary by temperature." if index == 17 else f"Gardening note {index}."),
        "fields": {"record_id": index, "qualification": "temperature-dependent" if index == 17 else ""},
        "source_ids": [f"source-{index}"], "source_class": "official_primary",
        "freshness": "unknown", "evidence_status": "single_source", "scope": "Complete source record.",
        "compatibility": "compatible", "context": {}, "provenance": {"source_id": f"source-{index}"},
    } for index in range(40)]
    selected, _ = select_requirement_cards(cards, [{
        "identifier": "runtime", "question": "What battery runtime is supported?",
        "dimension": "battery runtime", "freshness": "task-dependent", "state": "open",
    }], max_cards=3)

    assert len(selected) <= 3
    assert any("12 hours" in card["statement"] for card in selected)
    assert "temperature" in next(card["statement"] for card in selected if "12 hours" in card["statement"])
    assert next(card["fields"]["qualification"] for card in selected if "12 hours" in card["statement"])


def test_unrelated_conflict_does_not_invalidate_a_supported_answer():
    from core.thinking_coverage import refresh_coverage

    supported = _card_for_freshness(temporal_role="document")
    supported.update(
        id="supported", statement="The device supports local processing.",
        claim_identity={"subject": "device", "claim_type": "document_claim", "property": "supports local processing", "period": ""},
    )
    unrelated = _card_for_freshness(temporal_role="document")
    unrelated.update(
        id="unrelated-conflict", evidence_status="conflicted",
        statement="Another product weighs 3 kg.",
        claim_identity={"subject": "other product", "claim_type": "document_claim", "property": "weighs <value> kg", "period": ""},
    )
    requirement = {
        "identifier": "device-processing", "question": "Does the device support local processing?",
        "state": "resolved", "answer_coverage": "evidence_scoped",
        "evidence_ids": ["supported"], "freshness": "task-dependent",
    }

    refresh_coverage([requirement], [supported, unrelated])

    assert requirement["state"] == "resolved"
    assert requirement["answer_coverage"] == "evidence_scoped"
    assert requirement["evidence_ids"] == ["supported"]


def test_conflict_in_a_declared_claim_dependency_reopens_answer():
    from core.thinking_coverage import refresh_coverage

    supported = _card_for_freshness(temporal_role="document")
    supported.update(
        id="conclusion", statement="The option fits the stated budget.",
        claim_identity={"subject": "option", "claim_type": "document_claim", "property": "fits budget", "period": ""},
    )
    premise = _card_for_freshness(temporal_role="document")
    premise.update(
        id="premise", statement="The price is USD 20.", evidence_status="conflicted",
        claim_identity={"subject": "option", "claim_type": "document_claim", "property": "price <value>", "period": ""},
    )
    requirement = {
        "identifier": "budget-fit", "question": "Does it fit the budget?",
        "state": "resolved", "answer_coverage": "evidence_scoped",
        "evidence_ids": ["conclusion"], "claim_dependencies": ["premise"],
        "freshness": "task-dependent",
    }

    refresh_coverage([requirement], [supported, premise])

    assert requirement["state"] == "open"
    assert requirement["answer_coverage"] != "evidence_scoped"
    assert requirement["evidence_ids"] == []


def test_unknown_claim_identity_does_not_create_a_synthetic_conflict():
    left = _card_for_freshness(temporal_role="document", fields={"value": "Alpha"})
    right = _card_for_freshness(temporal_role="document", fields={"value": "Beta"})
    left.update(id="left", claim_key="same broad topic")
    right.update(id="right", claim_key="same broad topic")

    reconcile_fact_cards([left, right])

    assert left["evidence_status"] == right["evidence_status"] == "single_source"


def test_derived_status_requires_complete_scalar_deterministic_lineage():
    valid = _card_for_freshness(
        temporal_role="observation",
        fields={"tool": "calculate", "inputs": {"expression": "2 + 2"}, "result": 4},
    )
    valid["source_class"] = "typed_tool_output"
    missing_input = _card_for_freshness(
        temporal_role="observation",
        fields={"tool": "calculate", "inputs": {"expression": ""}, "result": 4},
    )
    missing_input["source_class"] = "typed_tool_output"
    invalid_result = _card_for_freshness(
        temporal_role="observation",
        fields={"tool": "calculate", "inputs": {"expression": "2 + 2"}, "result": float("nan")},
    )
    invalid_result["source_class"] = "typed_tool_output"

    reconcile_fact_cards([valid, missing_input, invalid_result])

    assert valid["evidence_status"] == "derived"
    assert missing_input["evidence_status"] == "single_source"
    assert invalid_result["evidence_status"] == "single_source"


def test_unknown_secondary_origins_cannot_confirm_each_other():
    evidence = [{"status": "evidence", "scope": "Sources.", "artifact_id": "artifact-001", "evidence": {}}]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "web_research",
                "sources": [
                    {
                        "url": "https://publisher-a.example/report",
                        "title": "Harbour 1000 schedule",
                        "evidence": "The Harbour 1000 starts on 2026-10-08 with practice on Thursday, followed by qualifying and a full race weekend at the circuit.",
                        "mentioned_dates": ["2026-10-08"],
                        "eligible": True,
                    },
                    {
                        "url": "https://publisher-b.example/copy",
                        "title": "Harbour 1000 schedule copy",
                        "evidence": "The Harbour 1000 starts on 2026-10-08 with practice on Thursday, followed by qualifying and a full race weekend at the circuit.",
                        "mentioned_dates": ["2026-10-08"],
                        "eligible": True,
                    },
                ],
            }
        }
    }

    cards = build_fact_cards(evidence, artifacts, "Research the Harbour 1000")

    assert [card["independence_key"] for card in cards] == [
        "secondary:unverified-origin",
        "secondary:unverified-origin",
    ]
    assert [card["evidence_status"] for card in cards] == ["single_source", "single_source"]
    artifacts["artifact-001"]["data"]["sources"][1]["originating_source"] = "https://wire-b.example/story"
    cards = build_fact_cards(evidence, artifacts, "Research the Harbour 1000")
    assert cards[1]["independence_key"] == "origin:wire-b.example"


def test_web_card_rejects_a_generic_explicit_competing_event_label():
    evidence = [
        {
            "status": "evidence",
            "scope": "One source.",
            "artifact_id": "artifact-001",
            "evidence": {},
        }
    ]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "web_research",
                "sources": [
                    {
                        "url": "https://example.com/harbour-12-hour-grid/",
                        "host": "example.com",
                        "title": "Harbour 12 Hour starting grid",
                        "evidence": "The Harbour 12 Hour GT3 field has completed qualifying and will race on Sunday after a full weekend of practice sessions.",
                        "eligible": True,
                        "retrieved_at": "2026-09-28T00:00:00+00:00",
                    }
                ],
            }
        }
    }

    cards = build_fact_cards(evidence, artifacts, "Deep analysis of the upcoming Harbour 1000 race")

    assert cards[0]["compatibility"] == "compatible"
    assert report_card_ledger(cards)


def test_finance_cards_retain_provider_record_and_news_as_separate_cards():
    evidence = [
        {"status": "evidence", "scope": "One quote.", "artifact_id": "artifact-001", "evidence": {}}
    ]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "finance_quote",
                "retrieved_at": "2026-09-28T00:00:00+00:00",
                "quote": {
                    "symbol": "ABC",
                    "price": 10,
                    "source_url": "https://finance.example/abc",
                },
                "news": [
                    {
                        "title": "ABC results",
                        "url": "https://news.example/abc",
                        "published_at": "2026-09-27",
                    }
                ],
            }
        }
    }

    cards = build_fact_cards(evidence, artifacts)

    assert [card["kind"] for card in cards] == ["structured_record", "atomic_claim"]
    assert cards[1]["source_url"] == "https://news.example/abc"


def test_hotel_cards_preserve_search_constraints_as_context():
    evidence = [
        {"status": "evidence", "scope": "One hotel.", "artifact_id": "artifact-001", "evidence": {}}
    ]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "hotel_search",
                "location": "Tokyo",
                "check_in": "2027-05-12",
                "check_out": "2027-05-14",
                "currency": "JPY",
                "offers": [{"name": "Example Hotel", "price": 100}],
            }
        }
    }

    cards = build_fact_cards(evidence, artifacts)

    assert cards[0]["context"]["location"] == "Tokyo"
    assert cards[0]["context"]["check_in"] == "2027-05-12"


def test_normalizers_share_scope_metadata_without_fabricating_dates():
    evidence = [
        {"status": "evidence", "scope": "Records.", "artifact_id": "artifact-001", "evidence": {}},
        {
            "status": "evidence",
            "scope": "State.",
            "artifact_id": "artifact-002",
            "retrieved_at": "2026-09-28T00:00:00+00:00",
            "evidence": {},
        },
    ]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "paper_search",
                "retrieved_at": "2026-09-28T00:00:00+00:00",
                "papers": [
                    {
                        "title": "Efficient inference",
                        "source": "arXiv",
                        "url": "https://arxiv.org/abs/1234.5678v2",
                        "published_at": "2025-01-02",
                        "updated_at": "2025-02-03",
                        "version": "v2",
                    }
                ],
            }
        },
        "artifact-002": {
            "data": {"type": "tool_observation", "tool": "read_sensor", "state": "on"},
        },
    }

    cards = build_fact_cards(evidence, artifacts)

    paper, observation = cards
    assert paper["context"]["temporal_role"] == "publication"
    assert paper["context"]["version"] == "v2"
    assert paper["context"]["peer_review_scope"].startswith("Preprint")
    assert paper["context"]["published_at"] == "2025-01-02"
    assert observation["context"]["temporal_role"] == "observation"
    assert observation["context"]["observed_at"] == observation["retrieved_at"]
    assert "published_at" not in observation["context"]


def test_travel_and_finance_cards_keep_scope_with_the_record():
    evidence = [
        {"status": "evidence", "scope": "Travel.", "artifact_id": "artifact-001", "evidence": {}},
        {"status": "evidence", "scope": "Quote.", "artifact_id": "artifact-002", "evidence": {}},
    ]
    artifacts = {
        "artifact-001": {
            "data": {
                "type": "travel_plan",
                "request": "Sydney to Tokyo then Berlin",
                "leg_offers": [[{"price": 500}], [{"price": 700}]],
                "leg_contexts": [
                    {"route": {"origin": "SYD", "destination": "TYO"}, "departure_date": "2027-05-12", "currency": "AUD", "retrieved_at": "2026-09-28T00:00:00+00:00"},
                    {"route": {"origin": "TYO", "destination": "BER"}, "departure_date": "2027-05-20", "currency": "AUD", "retrieved_at": "2026-09-28T00:00:00+00:00"},
                ],
            }
        },
        "artifact-002": {
            "data": {
                "type": "finance_quote",
                "quote": {
                    "name": "Example Corp", "symbol": "EXM", "exchange": "NASDAQ", "currency": "USD",
                    "quoted_at": "2026-09-28T12:00:00Z", "cache_mode": "cache bypass requested",
                    "source_url": "https://finance.example/exm", "financials": {"reporting_period": "Q2 2026"},
                },
            }
        },
    }

    cards = build_fact_cards(evidence, artifacts)

    assert cards[0]["context"]["leg_index"] == 1
    assert cards[0]["context"]["route"] == {"origin": "SYD", "destination": "TYO"}
    assert cards[1]["context"]["leg_index"] == 2
    quote = cards[2]
    assert quote["context"]["exchange"] == "NASDAQ"
    assert quote["context"]["currency"] == "USD"
    assert quote["context"]["reporting_period"] == "Q2 2026"
    assert quote["context"]["observed_at"] == "2026-09-28T12:00:00Z"


def test_per_question_working_set_is_bounded_before_the_global_cap():
    """A single broad requirement cannot crowd the whole inventory in."""
    cards = [{
        "id": f"record-{index}", "kind": "atomic_claim", "label": f"Record {index}",
        "statement": f"Note about runtime {index} for the battery.",
        "fields": {"record_id": index},
        "source_ids": [f"source-{index}"], "source_class": "official_primary",
        "freshness": "unknown", "evidence_status": "single_source",
        "scope": "Complete source record.", "compatibility": "compatible",
        "context": {}, "provenance": {"source_id": f"source-{index}"},
    } for index in range(20)]
    requirements = [{
        "identifier": "runtime", "question": "What battery runtime is supported?",
        "dimension": "battery runtime", "freshness": "task-dependent", "state": "open",
    }]
    selected, reasons = select_requirement_cards(
        cards, requirements, per_requirement_max_cards=5, max_cards=128)

    assert len(selected) <= 5
    # The per-question cap is enforced first: it is never silently widened by
    # the larger global limit.
    assert len(selected) == 5
    assert any("per-question" in reason for reason in reasons.get("_omitted", []))


def test_evidence_timestamps_must_not_postdate_the_task_as_of():
    """Fetched-now is not evidence the observation is current at the as-of time."""
    as_of = "2026-09-30T00:00:00Z"
    postdated = _card_for_freshness(
        temporal_role="provider_snapshot",
        context={"observed_at": "2026-09-30T10:00:00Z"},
    )
    in_skew = _card_for_freshness(
        temporal_role="provider_snapshot",
        context={"observed_at": "2026-09-30T00:14:00Z"},
    )
    within_task = _card_for_freshness(
        temporal_role="provider_snapshot",
        context={"observed_at": "2026-09-30T00:00:00Z"},
    )

    assert not card_is_fresh_for_current_claim(postdated, as_of)
    assert card_is_fresh_for_current_claim(in_skew, as_of)
    assert card_is_fresh_for_current_claim(within_task, as_of)
    # Without an as-of instant, the evidence time is only checked for presence.
    assert card_is_fresh_for_current_claim(postdated)
    # An unparseable timestamp cannot establish freshness against an as-of time.
    unparseable = _card_for_freshness(
        temporal_role="provider_snapshot",
        context={"observed_at": "not-a-timestamp"},
    )
    assert not card_is_fresh_for_current_claim(unparseable, as_of)
