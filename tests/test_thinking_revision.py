"""Regression fixtures for compact cards, requirement packets and revision scope."""

import json
import re
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ThinkingBudget
from core.thinking_coverage import (
    adapt_section_plan,
    initial_requirements,
    planner_inventory,
    refresh_coverage,
    requirement_terms,
)
from core.thinking_fact_cards import (
    _text_claim_identity,
    build_fact_cards,
    reconcile_fact_cards,
    select_requirement_cards,
    serialize_synthesis_packet,
    validate_fact_card,
)
from core.thinking_scope import research_context
from core.thinking_worker import ThinkingWorker, _raw_card_packet, _recover_retained_evidence

EXAMPLE_TASKS = [
    "give me a deep analysis on the Bathurst 1000 race coming up",
    "Give me a deep, neutral briefing on NVIDIA: its latest quoted price, recent results, major recent news, and the uncertainties that matter. Do not give me investment advice.",
    "cheapest flights hamburg to sydney return from 17th january till 3rd of march next year",
    "Plan a trip from Sydney to Tokyo, then Tokyo to Berlin, leaving on 12 May 2027. Find the best retrieved flight options, explain the schedule trade-offs, and clearly state what has not been checked.",
    "Research the latest local AI trends for running capable models at home, and compare recent relevant arXiv papers on efficient inference, quantisation, and agent reliability. Separate paper-backed findings from current product news.",
    "Research and compare the current options for self-hosting a private household photo backup service. Cover privacy, local hardware requirements, mobile backup support, licensing, and important limitations using current source-backed evidence.",
    "Explain the current recommended way to back up a home NAS, and note which recommendations come from current vendor docs versus older but still-valid documentation.",
    "What is the current price of gold, and how has it changed over the past month?",
]


def _evidence(artifact_id="artifact-001"):
    return [{"status": "evidence", "scope": "One retained source record.",
             "artifact_id": artifact_id, "evidence": {}}]


def test_generic_tool_observations_have_artifact_scoped_ids_and_exact_content_paths():
    evidence = [*_evidence("artifact-001"), *_evidence("artifact-002")]
    for entry in evidence:
        entry["tool"] = "lookup"
    artifacts = {
        "artifact-001": {"data": {"content": "The service supports local backup."}},
        "artifact-002": {"data": {"content": "The service supports cloud backup."}},
    }
    registry, extraction = {}, {}

    cards = build_fact_cards(evidence, artifacts, source_registry=registry, extraction_state=extraction)
    packet = serialize_synthesis_packet(cards, registry)

    assert len({card["id"] for card in cards}) == 2
    assert len(registry) == len(extraction) == 2
    assert {card["provenance"]["record_path"] for card in cards} == {"/content"}
    assert len({claim["source"] for claim in packet["claims"]}) == 2


def test_shared_claim_prefix_does_not_drop_distinct_qualifications():
    prefix = "The service supports local backup " + "under documented conditions " * 15
    statements = [prefix + "but excludes phones.", prefix + "but excludes tablets."]
    artifacts = {"artifact-001": {"data": {"type": "web_research", "sources": [
        {"title": "Vendor guide", "url": "https://docs.example/backup", "evidence": " ".join(statements)}
    ]}}}

    cards = build_fact_cards(_evidence(), artifacts)

    assert len(cards) == 2
    assert {card["statement"] for card in cards} == set(statements)


def test_same_url_paper_versions_remain_distinct_records():
    evidence = [*_evidence("artifact-001"), *_evidence("artifact-002")]
    artifacts = {
        f"artifact-00{version}": {"data": {"type": "paper_detail", "paper": {
            "title": "Efficient inference", "source": "arxiv", "url": "https://arxiv.org/abs/1234",
            "version": version, "abstract": f"Version {version} has different limitations.",
        }}} for version in (1, 2)
    }

    cards = build_fact_cards(evidence, artifacts)

    assert {card["fields"]["version"] for card in cards} == {1, 2}


def test_duplicate_discovery_preserves_all_artifact_and_legacy_bindings():
    evidence = [*_evidence("artifact-001"), *_evidence("artifact-002")]
    artifacts = {identifier: {"data": {"type": "web_research", "sources": [{
        "title": "Vendor guide", "url": "https://docs.example/backup",
        "evidence": "The service supports local backup.",
    }]}} for identifier in ("artifact-001", "artifact-002")}

    cards = build_fact_cards(evidence, artifacts)

    assert len(cards) == 1
    assert cards[0]["source_ids"] == ["artifact-001", "artifact-002"]
    assert cards[0]["legacy_ids"] == ["artifact-001-web-1", "artifact-002-web-1"]


def test_packet_preserves_explicit_unknowns_empty_fields_and_list_positions():
    record = {"name": "Option", "currency": None, "price": 0,
              "legs": [None, {"departure": "", "arrival": None}, []], "restrictions": {}}
    artifacts = {"artifact-001": {"data": {"records": [record]}}}
    cards = build_fact_cards(_evidence(), artifacts)

    packet = serialize_synthesis_packet(cards, {})

    assert packet["claims"][0]["record"] == record
    assert cards[0]["fields"] == record


def test_card_validation_rejects_an_empty_source_binding():
    cards = build_fact_cards(_evidence(), {"artifact-001": {"data": {"records": [{"name": "Option"}]}}})
    cards[0]["source_ids"] = []

    assert validate_fact_card(cards[0]) == "card source_ids must be non-empty strings"


@pytest.mark.parametrize(("corrected", "status"), [(10, "conflicted"), (9, "confirmed")])
def test_explicit_dated_parameter_correction_preserves_original_but_compares_asserted_value(corrected, status):
    originals = ["Model-01 has 9 billion total parameters.",
                 f"Correction dated 2026-09-30: Model-01 has {corrected} billion total parameters, not 8 billion."]
    sources = [{"url": f"https://source-{index}.example/model", "title": "Model card", "source_text": text,
                "published_at": f"2026-09-{index + 1:02}", "source_class": "official_primary"} for index, text in enumerate(originals)]
    cards = build_fact_cards(_evidence(), {"artifact-001": {"data": {"type": "web_research", "sources": sources}}})
    reconcile_fact_cards(cards)
    assert {card["statement"] for card in cards} == set(originals)
    assert {card["evidence_status"] for card in cards} == {status}
    assert all(card["claim_identity"]["subject"] == "model-01" for card in cards)
    assert all(not card["claim_identity"]["period"] for card in cards)


def test_parameter_correction_does_not_erase_adapter_declared_version_period():
    identity = _text_claim_identity("Correction dated 2026-09-30: Model-01 has 10 billion total parameters, not 9 billion.",
                                    {"context": {"reporting_period": "version 2"}})
    assert identity["period"] == "version 2"


@pytest.mark.parametrize("field,record_path", [
    ("quote", "/quote"), ("exchange_rate", "/exchange_rate"),
    ("quotes", "/quotes/0"), ("watchlist", "/watchlist/0"), ("markets", "/markets/0"),
])
def test_finance_provenance_points_to_the_actual_retained_record(field, record_path):
    record = {"symbol": "ABC", "currency": "USD", "price": 10}
    data = {"type": "finance_record", field: [record] if field.endswith("s") or field == "watchlist" else record}
    artifacts = {"artifact-001": {"data": data}}

    cards = build_fact_cards(_evidence(), artifacts)

    assert cards[0]["provenance"]["record_path"] == record_path


def test_raw_rollback_packet_links_every_record_to_its_constraints_and_eligibility():
    artifacts = {"artifact-001": {"data": {
        "type": "flight_search", "route": {"origin": "HAM", "destination": "SYD"},
        "departure_date": "2027-01-17", "currency": "EUR",
        "offers": [{"price": 700, "departure": None, "arrival": None}],
    }}}
    cards = build_fact_cards(_evidence(), artifacts)

    packet = _raw_card_packet(cards, {})
    scopes = {scope["id"]: scope["scope"] for scope in packet["scopes"]}
    record = packet["records"][0]

    assert scopes[record["scope_id"]]["constraints"]["route"] == {"origin": "HAM", "destination": "SYD"}
    assert record["eligibility"]["missing_option_fields"] == ["schedule"]
    assert record["evidence"]["departure"] is None


def test_missing_claims_are_recovered_from_retained_evidence_with_a_durable_allowance():
    body = "Catalog entry. " * 16_000 + "The service supports automatic mobile backup."
    job = BackgroundJob("job", JobSnapshot("Compare mobile backup"))
    job.evidence = _evidence()
    job.artifacts = {"artifact-001": {"data": {"content": body}}}
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts,
                                     source_registry=job.source_registry, extraction_state=job.extraction_state)
    assert not job.fact_cards

    assert _recover_retained_evidence(job)

    assert any("supports automatic mobile backup" in card["statement"] for card in job.fact_cards)
    assert job.synthesis_state["extraction_recoveries"] == 1
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert not _recover_retained_evidence(resumed)
    assert resumed.synthesis_state["extraction_recoveries"] == 1


def test_retained_extraction_recovery_never_resets_its_budget_on_resume():
    job = BackgroundJob("job", JobSnapshot("Research"))
    job.evidence_requirements = initial_requirements(job.snapshot.task)
    job.extraction_state = {"source": {"text_chars": 1_000_000, "unprocessed_spans": [[0, 1_000_000]]}}
    job.synthesis_state["extraction_recoveries"] = 4
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())

    assert not _recover_retained_evidence(resumed)
    assert resumed.extraction_state == job.extraction_state


def test_all_eight_acceptance_tasks_freeze_constraints_and_keep_every_requirement_group():
    for task in EXAMPLE_TASKS:
        context = research_context(task)
        requirements = initial_requirements(task, context)
        groups = adapt_section_plan(requirements)

        assert requirements[0]["question"] == task
        assert all(item["constraints"] == task for item in requirements)
        assert {identifier for group in groups for identifier in group["requirement_ids"]} == {
            item["identifier"] for item in requirements
        }
    finance = initial_requirements(EXAMPLE_TASKS[1], research_context(EXAMPLE_TASKS[1]))[0]
    assert "Do not give me investment advice" in finance["constraints"]
    photo_requirements = initial_requirements(EXAMPLE_TASKS[5], research_context(EXAMPLE_TASKS[5]))
    assert "fact-and-important-limitations-using-current-source-backed-evidence" not in {
        item["identifier"] for item in photo_requirements
    }
    assert {"fact-privacy", "fact-hardware", "fact-mobile_support", "fact-licensing"} <= {
        item["identifier"] for item in photo_requirements
    }
    assert any(item["identifier"] == "tradeoffs" for item in photo_requirements)
    assert any(item["identifier"] == "fact-limitations" for item in initial_requirements("Explain limitations of this feature"))
    assert not any("invest" in item.casefold() for item in research_context("Explain how it works")
                   ["entities"])


def test_both_synthesis_paths_run_for_all_eight_frozen_acceptance_tasks():
    """Exercise real budget accounting for compact and hierarchical packets."""

    def make_cards(task, requirements, count):
        cards = []
        sources = {}
        assignable = [
            item for item in requirements
            if not (item["identifier"] == "answer" and item.get("broad"))
        ]
        for index in range(count):
            card_id = f"acceptance-{index}"
            url = f"https://acceptance-{index}.example/evidence"
            requirement = assignable[index % len(assignable)]
            if requirement["identifier"] == "answer":
                other_terms = set().union(*(
                    requirement_terms(item) for item in requirements
                    if item["identifier"] != "answer"
                )) if len(requirements) > 1 else set()
                unique_terms = requirement_terms(requirement) - other_terms
                claim = sorted(unique_terms)[0] if unique_terms else task
            elif requirement["identifier"] == "current-context":
                claim = "current latest quote schedule status news and price"
            elif requirement["identifier"] == "tradeoffs":
                claim = "material limitations alternatives trade-offs risks and uncertainty"
            else:
                claim = " ".join(requirement.get("evidence_terms") or []) or requirement.get("dimension") or requirement["question"]
            detail = ""
            cards.append({
                "version": 3, "id": card_id, "kind": "atomic_claim", "label": f"Primary record {index}",
                "fields": {}, "source_ids": [f"artifact-{index}"], "source_class": "official_primary",
                "retrieved_at": "2026-09-30T00:00:00Z", "freshness": "dated",
                "evidence_status": "single_source", "scope": "One retrieved primary record.",
                "statement": f"This source addresses {claim}.{detail}",
                "source_host": f"acceptance-{index}.example", "independence_key": f"primary-{index}",
                "claim_key": f"claim-{index}", "source_url": url, "compatibility": "compatible",
                "temporal_role": "document", "subject": f"Subject {index}",
                "context": {"source_identity": url, "published_at": "2026-09-01"},
                "claim_identity": {"subject": f"subject {index}", "claim_type": "document_claim", "period": "2026"},
                "claim_eligibility": {"scoped_claims": True, "complete_structured_option": True,
                                      "missing_option_fields": []},
                "provenance": {"source_id": url, "artifact_id": f"artifact-{index}",
                               "record_path": "/source_text", "start": 0, "end": 74},
            })
            sources[url] = {"id": url, "title": f"Primary record {index}", "url": url,
                            "class": "official_primary", "published_at": "2026-09-01"}
        return cards, sources

    def run_path(task, card_count, context_tokens):
        job = BackgroundJob("acceptance", JobSnapshot(task))
        job.evidence_requirements = initial_requirements(task, research_context(task))
        cards, sources = make_cards(task, job.evidence_requirements, card_count)
        job.source_registry = sources
        budget = ThinkingBudget(object(), job.model_budget, context_tokens=context_tokens)

        def packet_card_ids(prompt):
            match = re.search(r"Verified fact-card ledger:\n(.*?)\n\nMaterial evidence limitations:", prompt, re.S)
            if not match:
                return []
            return [claim["id"] for claim in json.loads(match.group(1)).get("claims", [])]

        def generate(_model, **kwargs):
            count = budget.count_messages(
                system_prompt=kwargs.get("system_prompt", ""), user_prompt=kwargs.get("user_prompt"),
                history=kwargs.get("history"), thinking_mode=kwargs.get("thinking_mode", False),
            )
            assert count + budget.headroom + kwargs["max_new_tokens"] <= context_tokens
            user_prompt = kwargs.get("user_prompt", "")
            if user_prompt == "Audit the draft report now.":
                return '{"verdict":"pass","issues":[]}'
            ids = packet_card_ids(kwargs.get("system_prompt", ""))
            if user_prompt == "Produce the final report now.":
                return "\n\n".join(card["statement"].split(".", 1)[0] + ". <!-- cards: " + card["id"] + " -->"
                                     for card in cards if card["id"] in ids)
            if user_prompt.startswith("Synthesize section:"):
                assert ids
                return "\n\n".join(card["statement"].split(".", 1)[0] + ". <!-- cards: " + card["id"] + " -->"
                                     for card in cards if card["id"] in ids)
            if user_prompt.startswith("Reconcile"):
                return ""
            raise AssertionError(user_prompt)

        worker = ThinkingWorker(
            model=object(), grammar=None, jobs=MagicMock(), generate=generate,
            capabilities=lambda: {}, registry=SimpleNamespace(_schemas={}),
            summarise_search=lambda *_args: "", model_lock=threading.Lock(), server_slots=1,
            context_tokens=context_tokens,
        )
        report, bindings = worker._synthesise(job, cards, budget, lambda: False, "")
        assert report
        assert {item["card_id"] for item in bindings} == {card["id"] for card in cards}
        return job.synthesis_state

    for task in EXAMPLE_TASKS:
        requirements = initial_requirements(task, research_context(task))
        compact_state = run_path(task, max(2, len(requirements)), 12_288)
        assert compact_state["selected_card_ids"]
        hierarchical_state = run_path(task, 18, 8_192)
        assert hierarchical_state["sections"]
        assert hierarchical_state["overflow_retries"] >= 1
        expected_sections = {
            item["identifier"] for item in requirements
            if not (item["identifier"] == "answer" and item.get("broad"))
            and item["identifier"] != "subject-identity"  # Undated synthetic prose cannot close identity.
        }
        assert expected_sections <= {
            requirement_id for section in hierarchical_state["sections"]
            for requirement_id in section["requirement_ids"]
        }


def test_budgeted_planner_inventory_distinguishes_extraction_from_missing_retrieval():
    cards = [
        {"id": "claim-1", "statement": "The service supports mobile backup.",
         "label": "Vendor docs", "fields": {}, "compatibility": "compatible",
         "evidence_status": "single_source", "claim_identity": {"subject": "service"}},
        {"id": "claim-2", "statement": "The service does not support remote backup.",
         "label": "Vendor docs", "fields": {}, "compatibility": "compatible",
         "evidence_status": "conflicted", "claim_identity": {"subject": "service"}},
    ]
    requirements = initial_requirements("Compare service mobile support")
    extraction = {"source-1": {"state": "partial", "unprocessed_spans": [[100, 200]]}}
    refresh_coverage(requirements, cards, extraction)
    inventory = planner_inventory(cards, requirements, extraction, max_chars=5_000)

    assert inventory["established"]
    assert inventory["contradictions"]
    assert inventory["extraction"][0]["state"] == "partial"
    assert any(item["state"] == "open" for item in inventory["missing_facts"])


def test_planner_inventory_budget_keeps_whole_claims_and_summarizes_extraction_diagnostics():
    cards = [
        {"id": f"claim-{index}", "statement": f"Complete qualitative claim {index}. " + ("qualified detail " * 18),
         "label": "Vendor docs", "fields": {}, "compatibility": "compatible",
         "evidence_status": "single_source", "claim_identity": {"subject": f"service-{index}"}}
        for index in range(45)
    ]
    requirements = initial_requirements("Compare service options")
    extraction = {"source": {
        "state": "partial", "processed_until": 8_000, "text_chars": 90_000, "claims": 41,
        "ambiguous_spans": [[index, index + 10] for index in range(1200)],
        "oversized_spans": [[index, index + 10] for index in range(300)],
        "unprocessed_spans": [[index, index + 10] for index in range(600)],
    }}

    inventory = planner_inventory(cards, requirements, extraction, max_chars=2_800)
    rendered = json.dumps(inventory, ensure_ascii=True, separators=(",", ":"))

    assert len(rendered.encode("utf-8")) <= 2_800
    assert inventory["established"]
    assert inventory["omitted_established_count"] > 0
    assert len(inventory["extraction"][0]["unprocessed_spans"]) <= 4
    assert "ambiguous_spans" not in inventory["extraction"][0]


def test_planner_inventory_bounds_conflict_and_extraction_ledgers_as_whole_records():
    cards = [
        {"id": f"conflict-{index}", "statement": f"Complete conflicting claim {index}.",
         "label": "Source", "fields": {}, "compatibility": "compatible",
         "evidence_status": "conflicted", "claim_identity": {"subject": f"service-{index}",
                                                                  "property": "different value"}}
        for index in range(35)
    ]
    requirements = initial_requirements("Compare service options")
    extraction = {
        f"source-{index}": {
            "state": "partial", "processed_until": 8_000, "text_chars": 20_000,
            "claims": 12, "ambiguous_spans": [[0, 20]] * 50,
            "oversized_spans": [[20, 50]] * 30, "unprocessed_spans": [[100, 200]] * 40,
        }
        for index in range(25)
    }

    inventory = planner_inventory(cards, requirements, extraction, max_chars=2_800)
    rendered = json.dumps(inventory, ensure_ascii=True, separators=(",", ":"))

    assert len(rendered.encode("utf-8")) <= 2_800
    assert inventory["omitted_contradiction_count"] + len(inventory["contradictions"]) == 35
    assert inventory["omitted_extraction_count"] + len(inventory["extraction"]) == 25
    assert inventory["omitted_established_count"] == 35 - len(inventory["established"])


def test_web_claim_cards_have_exact_spans_and_prompt_packet_removes_source_text_duplication():
    excerpt = (
        "Immich supports automatic iOS photo backup, but background uploads depend on OS policy. "
        "It does not provide end-to-end encryption for the server library."
    )
    evidence = _evidence()
    artifacts = {"artifact-001": {"data": {"type": "web_research", "sources": [
        {"title": "Immich vendor guide", "url": "https://docs.example/immich",
         "evidence": excerpt, "eligible": True, "source_class": "official_primary",
         "published_at": "2026-08-01", "originating_source": "docs.example"}
    ]}}}
    registry = {}
    extraction = {}
    cards = build_fact_cards(evidence, artifacts, "Compare Immich mobile backup and encryption",
                             source_registry=registry, extraction_state=extraction)
    packet = serialize_synthesis_packet(cards, registry)
    rendered = json.dumps(packet)

    assert len(cards) == 2
    assert all(card["provenance"]["start"] < card["provenance"]["end"] for card in cards)
    assert any("does not provide" in card["statement"] for card in cards)
    assert all("source_text" not in claim and "excerpt" not in claim
               for claim in packet["claims"])
    assert excerpt not in rendered
    assert "end-to-end encryption" in rendered
    assert all("text" not in value for value in registry.values())
    assert artifacts["artifact-001"]["data"]["sources"][0]["evidence"] == excerpt
    assert all(card["provenance"]["record_path"].endswith("/evidence") for card in cards)


def test_prompt_view_omits_selection_scores_and_duplicate_per_card_limitations():
    artifacts = {"artifact-001": {"data": {"records": [{"name": "Option", "limitation": "Cannot upload in the background."}]}}}
    cards = build_fact_cards(_evidence(), artifacts)
    card_id = cards[0]["id"]
    packet = serialize_synthesis_packet(cards, {}, reasons={card_id: ["answer: name; score=5", "answer: mobile; score=2"]})

    assert packet["claims"][0]["requirements"] == ["answer"]
    assert "score=" not in json.dumps(packet)
    job = BackgroundJob("job", JobSnapshot("Compare options"))
    job.evidence = _evidence()
    worker = ThinkingWorker(object(), None, MagicMock(), lambda *_args: "", lambda: {},
                            SimpleNamespace(_schemas={}), lambda *_args: "", threading.Lock(), 1)
    prompt, ledger = worker._packet_prompt(job, cards, [])
    assert "Cannot upload in the background." in ledger
    limitations = json.loads(re.search(r"Material evidence limitations:\n(.*?)\n\n", prompt, re.S).group(1))
    assert not limitations.get("evidence")


def test_distinct_claims_from_one_source_survive_but_repeated_discovery_deduplicates():
    text = "The service supports mobile backup. The service lacks selective sync."
    evidence = _evidence()
    artifacts = {"artifact-001": {"data": {"type": "web_research", "sources": [
        {"title": "Docs", "url": "https://docs.example/backup", "evidence": text,
         "eligible": True},
        {"title": "Docs copy", "url": "https://docs.example/backup", "evidence": text,
         "eligible": True},
    ]}}}

    cards = build_fact_cards(evidence, artifacts, "Compare mobile backup and sync")

    assert len(cards) == 2
    assert {card["statement"] for card in cards} == {
        "The service supports mobile backup.", "The service lacks selective sync."
    }


def test_material_value_conflict_reconciles_only_with_matching_event_year():
    entries = []
    artifacts = {}
    for index, date in enumerate(("2026-10-08", "2026-10-09", "2025-10-08"), 1):
        artifact_id = f"artifact-{index:03d}"
        entries.append({"status": "evidence", "scope": "Retrieved event page.",
                        "artifact_id": artifact_id, "evidence": {}})
        artifacts[artifact_id] = {"data": {"type": "web_research", "sources": [{
            "title": "Bathurst 1000 schedule", "url": f"https://publisher-{index}.example/event",
            "evidence": f"The Bathurst 1000 starts on {date} after practice and qualifying.",
            "mentioned_dates": [date], "eligible": True, "source_class": "official_primary",
        }]}}

    cards = build_fact_cards(entries, artifacts, "Research the upcoming Bathurst 1000")
    by_period = {}
    for card in cards:
        by_period.setdefault(card["claim_identity"]["period"], []).append(card)

    assert by_period["2026"]
    assert {card["evidence_status"] for card in by_period["2026"]} == {"conflicted"}
    assert {card["evidence_status"] for card in by_period["2025"]} == {"single_source"}


def test_large_offer_collections_and_all_fields_remain_intact_in_selected_packets():
    offers = [
        {"offer_id": f"offer-{index}", "price": index, "currency": "AUD",
         "departure": {"date": "2027-01-17", "time": f"{index % 24:02}:00"},
         "arrival": {"date": "2027-01-18", "time": "09:00"},
         "airlines": [f"Carrier {index}"], "stops": index % 3,
         "qualification": "subject to retrieved provider availability"}
        for index in range(30)
    ]
    cards = build_fact_cards(_evidence(), {"artifact-001": {"data": {
        "type": "flight_search", "provider": "Flights", "route": {"origin": "HAM", "destination": "SYD"},
        "currency": "AUD", "offers": offers,
    }}}, "Compare flights HAM to SYD")
    registry = {}
    build_fact_cards(_evidence(), {"artifact-001": {"data": {
        "type": "flight_search", "provider": "Flights", "route": {"origin": "HAM", "destination": "SYD"},
        "currency": "AUD", "offers": offers,
    }}}, "Compare flights HAM to SYD", source_registry=registry)
    packet = serialize_synthesis_packet(cards, registry)
    records = [item["record"] for item in packet["claims"]]

    assert len(cards) == len(offers) == len(records)
    assert {record["offer_id"] for record in records} == {offer["offer_id"] for offer in offers}
    assert all("qualification" in record and "currency" in record for record in records)
    assert all(card["provenance"]["record_path"].startswith("/offers/") for card in cards)


def test_selection_is_requirement_reasoned_and_exposes_unselected_cards():
    cards = [
        {"id": "mobile", "kind": "atomic_claim", "statement": "The app supports mobile photo backup.",
         "label": "Docs", "fields": {}, "source_ids": ["mobile-artifact"],
         "source_url": "https://docs.example/mobile", "source_class": "official_primary",
         "freshness": "dated", "scope": "Vendor docs.", "compatibility": "compatible",
         "evidence_status": "single_source", "claim_identity": {"subject": "photos"}},
        {"id": "license", "kind": "atomic_claim", "statement": "The license prohibits commercial hosting.",
         "label": "License", "fields": {}, "source_ids": ["license-artifact"],
         "source_url": "https://docs.example/license", "source_class": "official_primary",
         "freshness": "dated", "scope": "License document.", "compatibility": "compatible",
         "evidence_status": "single_source", "claim_identity": {"subject": "license"}},
    ]
    requirements = initial_requirements("Compare a photo server for mobile backup")
    selected, reasons = select_requirement_cards(cards, requirements)

    assert "mobile" in {card["id"] for card in selected}
    assert reasons["mobile"]
    assert all("license" not in reasons[card["id"]] for card in selected if card["id"] == "mobile")


def test_navigation_ranks_candidates_but_does_not_exclude_by_query_words():
    requirements = initial_requirements("Give me a deep analysis of the upcoming Harbour 1000 race")
    cards = []
    for identifier, statement in (
        ("entrants", "The confirmed entries include 26 cars with paired drivers."),
        ("motel", "Harbour accommodation offers an alternative place to stay for the race."),
    ):
        cards.append({
            "version": 3, "id": identifier, "kind": "atomic_claim", "label": "Source",
            "fields": {}, "source_ids": [identifier], "source_class": "secondary_reporting",
            "retrieved_at": "2026-09-30T00:00:00Z", "freshness": "dated",
            "evidence_status": "single_source", "scope": "One article.",
            "statement": statement, "source_url": f"https://example.com/{identifier}",
            "compatibility": "compatible", "claim_identity": {"subject": identifier},
        })

    selected, _reasons = select_requirement_cards(cards, requirements)

    assert {card["id"] for card in selected} == {"entrants", "motel"}
    assert selected[0]["id"] == "entrants"
    for requirement in requirements:
        requirement.update(source_review_complete=True, model_selected_card_ids=["entrants"])
    selected, _reasons = select_requirement_cards(cards, requirements)
    assert {card["id"] for card in selected} == {"entrants"}


def test_equal_relevance_prefers_fresher_and_authoritative_cards():
    """Among equally relevant cards, authoritative and fresher sources are
    ordered first so a stale roundup does not crowd out a primary source."""
    cards = [
        {"version": 3, "id": "stale-roundup", "kind": "atomic_claim",
         "label": "Roundup", "fields": {}, "source_ids": ["stale"],
         "source_class": "secondary_reporting",
         "retrieved_at": "2026-08-12T00:00:00Z", "freshness": "dated",
         "evidence_status": "single_source", "scope": "Roundup article.",
         "statement": "The provider documented quarterly revenue growth.",
         "source_url": "https://news.example/roundup",
         "compatibility": "compatible", "claim_identity": {"subject": "revenue"}},
        {"version": 3, "id": "fresh-primary", "kind": "atomic_claim",
         "label": "IR release", "fields": {}, "source_ids": ["fresh"],
         "source_class": "official_primary",
         "retrieved_at": "2026-10-10T00:00:00Z", "freshness": "dated",
         "evidence_status": "single_source", "scope": "Investor release.",
         "statement": "The provider reported quarterly revenue growth.",
         "source_url": "https://ir.example/release",
         "compatibility": "compatible", "claim_identity": {"subject": "revenue"}},
        {"version": 3, "id": "undated", "kind": "atomic_claim",
         "label": "Undated", "fields": {}, "source_ids": ["undated"],
         "source_class": "secondary_reporting",
         "freshness": "undated", "evidence_status": "single_source",
         "scope": "Undated page.",
         "statement": "The provider noted quarterly revenue growth.",
         "source_url": "https://blog.example/note",
         "compatibility": "compatible", "claim_identity": {"subject": "revenue"}},
    ]
    # A task-dependent requirement keeps every eligible card in play, so the
    # ordering among equals is what under test.
    requirements = [{
        "identifier": "chapter-01-q01", "question": "What quarterly revenue did the provider report?",
        "importance": "high", "freshness": "task-dependent", "state": "open",
        "evidence_terms": ["quarterly", "revenue", "growth", "provider"],
    }]

    selected, _reasons = select_requirement_cards(cards, requirements)

    ids = [card["id"] for card in selected]
    assert "fresh-primary" in ids
    assert "stale-roundup" in ids
    assert "undated" in ids
    # Fresh primary source must come before the stale roundup and undated card.
    assert ids.index("fresh-primary") < ids.index("stale-roundup")
    assert ids.index("fresh-primary") < ids.index("undated")


def test_claim_event_scope_overrides_incidental_historical_page_label():
    text = (
        "The Harbour 1000 includes 26 current entrant pairings.\n"
        "The Harbour 500 included 20 entrants in the historical race."
    )
    artifacts = {"artifact-001": {"data": {"type": "web_research", "sources": [{
        "title": "Harbour 500 archive", "url": "https://example.com/harbour-500-history",
        "evidence": text, "eligible": True,
    }]}}}

    cards = build_fact_cards(_evidence(), artifacts, "Research the upcoming Harbour 1000 race")

    by_statement = {card["statement"]: card for card in cards}
    assert by_statement["The Harbour 1000 includes 26 current entrant pairings."]["compatibility"] == "compatible"
    assert by_statement["The Harbour 500 included 20 entrants in the historical race."]["compatibility"] == "compatible"
    assert by_statement["The Harbour 500 included 20 entrants in the historical race."]["scope_conflict"]


def test_mixed_current_and_historical_tables_scope_each_complete_row_by_caption():
    artifacts = {"artifact-001": {"data": {"type": "web_research", "sources": [{
        "title": "Harbour race entry archive", "url": "https://example.com/harbour-entries",
        "evidence": "The page includes current and historical entrant tables for comparison.",
        "eligible": True,
        "tables": [
            {"caption": "Harbour 1000 current entrants", "records": [{
                "headers": ["Car", "Primary", "Co-driver"],
                "cells": ["15", "Alex Driver", "Casey Driver"],
                "columns": {"Car": "15", "Primary": "Alex Driver", "Co-driver": "Casey Driver"},
            }]},
            {"caption": "Harbour 500 historical entrants", "records": [{
                "headers": ["Car", "Primary", "Co-driver"],
                "cells": ["8", "Historic Driver", "Past Driver"],
                "columns": {"Car": "8", "Primary": "Historic Driver", "Co-driver": "Past Driver"},
            }]},
        ],
    }]}}}

    cards = [
        card for card in build_fact_cards(
            _evidence(), artifacts, "Research the upcoming Harbour 1000 race"
        )
        if card["kind"] == "structured_record"
    ]

    assert len(cards) == 2
    assert cards[0]["fields"]["columns"]["Co-driver"] == "Casey Driver"
    assert cards[0]["fields"]["table_caption"] == "Harbour 1000 current entrants"
    assert cards[0]["compatibility"] == "compatible"
    assert cards[1]["compatibility"] == "compatible"
    assert cards[1]["scope_conflict"]


def test_reconciliation_keeps_missing_temporal_scope_separate():
    cards = []
    for identifier, sentence, independence in (
        ("one", "The service supports automatic photo backup.", "publisher-a"),
        ("two", "The service does not support automatic photo backup.", "publisher-b"),
    ):
        evidence = _evidence(identifier)
        artifacts = {identifier: {"data": {"type": "web_research", "sources": [{
            "title": "Photo docs", "url": f"https://{independence}.example/guide",
            "evidence": sentence, "eligible": True,
        }]}}}
        cards.extend(build_fact_cards(evidence, artifacts, "Compare photo services"))

    reconcile_fact_cards(cards)

    assert [card["evidence_status"] for card in cards] == ["single_source", "single_source"]


def test_explicit_positive_and_negative_claims_conflict_within_the_same_report_period():
    cards = []
    for index, sentence in enumerate((
        "The service supports automatic mobile backup.",
        "The service does not support automatic mobile backup.",
    ), 1):
        artifact_id = f"period-{index}"
        evidence = _evidence(artifact_id)
        artifacts = {artifact_id: {"data": {"type": "web_research", "sources": [{
            "title": "Vendor backup guide", "url": f"https://vendor-{index}.example/backup",
            "evidence": sentence, "eligible": True, "source_class": "official_primary",
            "published_at": "2026-09-01",
        }]}}}
        cards.extend(build_fact_cards(evidence, artifacts, "Compare backup support"))

    reconcile_fact_cards(cards)
    assert len(cards) == 2
    assert {card["claim_identity"]["property"] for card in cards} == {"support automatic mobile backup."}
    assert {card["claim_identity"]["polarity"] for card in cards} == {"positive", "negative"}
    assert {card["evidence_status"] for card in cards} == {"conflicted"}
