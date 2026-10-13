"""Relative subjects are pinned to source-owned names and dates, not task labels."""

import json
from pathlib import Path

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_fact_cards import build_fact_cards, eligible_cards_for_requirement
from core.thinking_scope import apply_subject_candidates, needs_subject_resolution, research_context


def source_cards(task, texts):
    context = research_context(task, as_of="2026-10-12T04:19:14Z")
    sources = [{"title": title, "url": f"https://fixture.invalid/{index}", "source_text": text,
                "published_at": "2026-10-10T00:00:00Z", "retrieved_at": context["as_of"], "eligible": True}
               for index, (title, text) in enumerate(texts)]
    registry = {}
    cards = build_fact_cards(
        [{"status": "evidence", "artifact_id": "fixture", "evidence": {"sources": sources}}],
        {"fixture": {"data": {"type": "web_research", "sources": sources}}},
        task, context, source_registry=registry,
    )
    return context, cards


@pytest.mark.parametrize("task,name,version", [
    ("Analyse the next race", "Example Grand Prix", ""),
    ("Analyse the next Widget release", "Widget 2.1", "2.1"),
    ("Investigate the next train departure", "Train T42", ""),
])
def test_identity_binding_is_preserved_without_hiding_dependent_evidence(task, name, version):
    context, cards = source_cards(task, [
        (name, f"The next {name} is scheduled for 2026-10-25."),
        ("Methodology tutorial", "Data analysis is useful for modelling future results."),
    ])
    identity_card = next(card for card in cards if name in card["statement"])
    assert needs_subject_resolution(context)
    requirement = {"subject_resolution_required": True, "subject_resolution": {"state": "unresolved"}}
    assert eligible_cards_for_requirement(cards, requirement) == cards
    resolution = apply_subject_candidates(context, [{"name": name, "date": "2026-10-25", "version": version,
                                                    "card_ids": [identity_card["id"]]}], cards)
    assert resolution["state"] == "resolved" and resolution["version"] == version
    requirement["subject_resolution"] = resolution
    assert eligible_cards_for_requirement(cards, requirement) == cards
    job = BackgroundJob("scope-checkpoint", JobSnapshot(task), research_context=context)
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert resumed.research_context["subject_resolution"] == resolution


def test_elapsed_preview_tutorial_and_spliced_event_date_never_resolve_subject():
    context, cards = source_cards("Analyse the next race", [
        ("Old event preview", "Old Grand Prix is scheduled for 2026-10-11."),
        ("Analysis tutorial", "Data analysis is useful for modelling future results."),
        ("Next event schedule", "When is the New Grand Prix?\nThe Old Grand Prix is scheduled for 2026-10-25."),
    ])
    proposals = [
        {"name": "Old Grand Prix", "date": "2026-10-11", "card_ids": [cards[0]["id"]]},
        {"name": "New Grand Prix", "date": "2026-10-25", "card_ids": [card["id"] for card in cards]},
    ]
    resolution = apply_subject_candidates(context, proposals, cards)
    assert resolution["state"] == "unresolved"
    assert {row["reason"] for row in resolution["rejected"]} == {"elapsed_subject", "identity_date_binding_unproven"}
    assert all(card.get("subject") != "next race" for card in cards)


def test_competing_schedules_remain_ambiguous_and_version_changes_invalidate_support():
    context, cards = source_cards("Analyse the next Widget release", [
        ("Widget release", "Widget 2.1 is scheduled for 2026-10-25."),
        ("Widget release", "Widget 2.1 is scheduled for 2026-10-26."),
    ])
    proposals = [{"name": "Widget 2.1", "date": date, "version": "2.1", "card_ids": [card["id"]]}
                 for card, date in zip(cards, ("2026-10-25", "2026-10-26"), strict=True)]
    assert apply_subject_candidates(context, proposals, cards)["state"] == "ambiguous"
    cards[1]["provenance"]["source_version"] = "changed-original-version"
    assert apply_subject_candidates(context, [], cards)["state"] == "resolved"
    assert context["subject_resolution"]["date"] == "2026-10-25"
    cards[0]["evidence_status"] = "conflicted"
    assert apply_subject_candidates(context, [], cards)["state"] == "unresolved"


def test_f1_saved_sources_do_not_supply_a_guessed_calendar_answer():
    fixture = json.loads((Path(__file__).parent / "fixtures/thinking/f1-recovery.json").read_text())
    context, cards = source_cards(fixture["task"], [(s["title"], "\n\n".join(row["text"] for row in s["excerpts"]))
                                                 for s in fixture["sources"]])
    resolution = apply_subject_candidates(context, [
        {"name": "United States Grand Prix", "date": "2026-10-25", "card_ids": [card["id"] for card in cards]},
        {"name": "Singapore Grand Prix", "date": "2026-10-11", "card_ids": [card["id"] for card in cards]},
    ], cards)
    assert resolution["state"] == "unresolved"
    tutorials = [card for card in cards if "f1dataR" in card["label"]]
    assert tutorials and all("next formula" not in card["subject"] for card in tutorials)


def test_provider_field_identity_cannot_bind_arrival_as_departure():
    context = research_context("Investigate the next train departure", as_of="2026-10-12T00:00:00Z")
    card = {"id": "provider", "kind": "structured_record", "evidence_status": "single_source",
            "compatibility": "compatible", "fields": {"name": "Train T42", "arrival_date": "2026-10-25"},
            "provenance": {"record_id": "T42"}}
    proposal = {"name": "Train T42", "date": "2026-10-25", "card_ids": ["provider"]}
    assert apply_subject_candidates(context, [proposal], [card])["state"] == "unresolved"
    card["fields"]["departure_date"] = "2026-10-25"
    assert apply_subject_candidates(context, [proposal], [card])["state"] == "resolved"
