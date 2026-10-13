"""Context-bearing records remain source-bound across assignment and restart."""

import json
from copy import deepcopy

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_evidence import report_evidence, restore_report_cards
from core.thinking_extraction import extract_claims
from core.thinking_fact_cards import (
    build_fact_cards,
    eligible_cards_for_requirement,
    link_chapter_cards,
    refresh_section_card,
    serialize_synthesis_packet,
    validate_fact_card,
)
from core.thinking_source_context import contextual_source_registry, source_passages


def finding_job(text, title="Example Grand Prix preview"):
    job = BackgroundJob("findings", JobSnapshot("Analyse Example Grand Prix"))
    source = {"url": "https://fixture.invalid/preview", "title": title,
              "source_text": text, "eligible": True, "published_at": "2026-10-12T00:00:00Z"}
    job.record_outcome("lookup", "evidence", "Preview", {"sources": [source]}, (),
                       {"type": "web_research", "sources": [source]})
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, source_registry=job.source_registry)
    return job


def test_finding_retains_negation_antecedent_comparison_basis_and_event_context_on_restart():
    text = (
        "Example Grand Prix — 2026-10-25\n"
        "Team Alpha uses the revised package. It does not improve straight-line speed "
        "relative to the old package when both use the same fuel load.\n"
        "The comparison applies only to dry practice runs; race performance remains unknown."
    )
    job = finding_job(text)
    card = next(card for card in job.fact_cards if "does not improve" in card["statement"])
    finding = card["finding"]
    assert finding["meaning"] == card["statement"]
    assert finding["support"] == [card["provenance"]]
    assert finding["assessment"] == "unassessed"
    spans = finding["context_spans"]
    assert all("Example Grand Prix" not in str(card.get("subject")) for card in job.fact_cards)
    assert text[spans[0][0]:spans[0][1]] == text
    resumed = BackgroundJob.from_checkpoint(json.loads(json.dumps(job.checkpoint())))
    assert resumed.fact_cards == job.fact_cards
    packet = serialize_synthesis_packet([card], resumed.source_registry)
    passages = source_passages(packet["claims"], resumed.source_registry)
    assert passages[0]["text"] == text
    assert passages[0]["source_version"] == card["provenance"]["source_version"]
    assert "same fuel load" in passages[0]["text"] and "only to dry" in passages[0]["text"]


def test_whole_table_relationships_and_original_headers_survive_finding_checkpoint():
    text = "Name | Team | Time (s)\nAlex | Alpha | 90.2\nSam | Beta | 91.1"
    claims, _ = extract_claims(text)
    assert len(claims) == 2
    assert all(text[s:e] == "Name | Team | Time (s)" for claim in claims
               for s, e in claim["context_spans"][:1])
    job = finding_job(text)
    resumed = BackgroundJob.from_checkpoint(json.loads(json.dumps(job.checkpoint())))
    records = [card for card in resumed.fact_cards if card["kind"] == "structured_record"]
    assert [card["fields"]["columns"] for card in records] == [
        {"Name": "Alex", "Team": "Alpha", "Time (s)": "90.2"},
        {"Name": "Sam", "Team": "Beta", "Time (s)": "91.1"},
    ]
    assert all(card["finding"]["record_ref"] == card["id"] for card in records)


def test_instruction_role_is_visible_for_model_relevance_assessment():
    job = finding_job("You can use historical race results to model driver performance.",
                      "Race analysis in Python tutorial")
    assert job.fact_cards and all(card["finding"]["role"] == "instruction" for card in job.fact_cards)
    assert eligible_cards_for_requirement(job.fact_cards, {"question": "What are the race results?"}) == job.fact_cards
    assert eligible_cards_for_requirement(job.fact_cards, {"question": "How to model historical results?"})


def test_section_card_does_not_promote_assignment_or_finished_writing_to_answer_coverage():
    job = finding_job("Team Alpha has scored 20 points, but only in dry races.")
    card = job.fact_cards[0]
    chapter = {"id": "form", "title": "Competitive form", "questions": [
        {"requirement_id": "form-q", "supporting_card_ids": [card["id"]],
         "state": "partial", "missing_facts": "Wet-weather performance is unknown."},
    ]}
    job.chapter_plan = [chapter]
    link_chapter_cards(job.fact_cards, job.chapter_plan)
    assert card["finding"]["question_ids"] == ["form-q"]
    assert card["finding"]["section_ids"] == ["form"]
    assert card["finding"]["assessment"] == "assigned"
    assert chapter["section_card"]["state"] == "unassessed"
    assert not chapter["section_card"]["body_ref"]
    binding = {"card_id": card["id"], "source_url": card["source_url"]}
    chapter.update(body="Accepted qualified analysis.", digest="A qualified finding.",
                   analysis_state="complete", analysis_groups={"group": {"state": "complete", "bindings": [binding]}})
    refresh_section_card(chapter, job.fact_cards)
    section = chapter["section_card"]
    assert section["state"] == "useful_partial" and section["gaps"]
    assert section["body_ref"] == "body" and section["support_bindings"] == [binding]
    resumed = BackgroundJob.from_checkpoint(json.loads(json.dumps(job.checkpoint())))
    assert resumed.chapter_plan[0]["section_card"] == section
    chapter.update(body="", digest="", analysis_state="stale")
    refresh_section_card(chapter, job.fact_cards)
    assert chapter["section_card"]["state"] == "unassessed"
    assert chapter["section_card"]["support_bindings"] == []


def test_saved_report_roundtrip_preserves_finding_context_and_instruction_role():
    job = finding_job("Analysis workflow\nYou can compare race results only with matched fuel loads.", "Analysis tutorial")
    evidence = report_evidence(job)
    restored = restore_report_cards(evidence)
    assert restored and all(validate_fact_card(card) is None for card in restored)
    assert restored[0]["finding"]["context_spans"] == job.fact_cards[0]["finding"]["context_spans"]
    assert restored[0]["finding"]["role"] == "instruction"
    assert source_passages(restored, evidence["source_registry"])[0]["text"].startswith("Analysis workflow")
    assert eligible_cards_for_requirement(restored, {"question": "Race strategy?"}) == restored


def test_finding_validation_rejects_shortened_qualifiers_and_changed_support():
    card = finding_job("Team Alpha does not gain speed when fuel loads are matched.").fact_cards[0]
    changed = deepcopy(card)
    changed["finding"]["meaning"] = "Team Alpha gains speed."
    assert validate_fact_card(changed) == "finding meaning must retain the complete source assertion"
    changed = deepcopy(card)
    changed["finding"]["support"][0]["source_version"] = "different-version"
    assert validate_fact_card(changed) == "finding support differs from its original binding"


def test_source_context_from_different_versions_is_not_merged_or_rebound():
    job = finding_job("Team Alpha has scored 20 points under dry conditions.")
    card = job.fact_cards[0]
    source_id = card["provenance"]["source_id"]
    source = job.source_registry[source_id]
    changed_text = "Team Alpha has scored 30 points under wet conditions."
    other = finding_job(changed_text).fact_cards[0]
    source["versions"][other["provenance"]["source_version"]] = {"text": changed_text}
    passages = source_passages([card, other], job.source_registry)
    assert len(passages) == 2
    assert {row["text"] for row in passages} == {card["statement"], changed_text}
    stale = deepcopy(card)
    stale["provenance"]["source_version"] = "missing-original-version"
    assert source_passages([stale], job.source_registry) == []


def test_unfamiliar_text_tool_keeps_context_and_adapter_owned_qualifications():
    job = BackgroundJob("unfamiliar", JobSnapshot("Compare equipment"))
    text = "Alpha hardware comparison\nThe new design has lower latency only under light load."
    job.record_outcome("novel_readonly", "evidence", "Comparison", {"text": text}, (),
                       {"text": text}, metadata={"qualifications": {"unit": "milliseconds", "scope": "light load"}})
    job.fact_cards = build_fact_cards(job.evidence, job.artifacts, source_registry=job.source_registry)
    card = job.fact_cards[0]
    assert card["finding"]["qualifications"] == {"unit": "milliseconds", "scope": "light load"}
    sources = contextual_source_registry(job.evidence, job.artifacts, job.fact_cards, job.source_registry)
    assert source_passages(job.fact_cards, sources)[0]["text"] == text
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert resumed.fact_cards == job.fact_cards


def test_legacy_card_assignment_does_not_fabricate_missing_source_context():
    job = finding_job("Team Alpha has scored 20 points.")
    card = job.fact_cards[0]
    card.pop("finding")
    chapter = {"id": "form", "questions": [{"requirement_id": "form-q", "state": "answered",
                                             "supporting_card_ids": [card["id"]]}]}
    link_chapter_cards([card], [chapter])
    assert card["finding"]["context_state"] == "unavailable"
    assert card["finding"]["context_spans"] == []
    assert chapter["section_card"]["state"] == "unassessed"
