"""Source-quality witnesses from the saved failure; no web or model needed."""

import json
from datetime import datetime
from pathlib import Path

from core.thinking_fact_cards import build_fact_cards, select_requirement_cards
from core.thinking_scope import research_context

FIXTURES = Path(__file__).parent / "fixtures" / "thinking"


def load_fixture():
    return json.loads((FIXTURES / "f1-recovery.json").read_text())


def source_only_cards(fixture):
    sources = [{**source, "source_text": "\n\n".join(row["text"] for row in source["excerpts"]),
                "eligible": True} for source in fixture["sources"]]
    registry = {}
    context = research_context(fixture["task"], as_of=fixture["as_of"])
    cards = build_fact_cards(
        [{"status": "evidence", "artifact_id": "fixture", "evidence": {"sources": sources}}],
        {"fixture": {"data": {"type": "web_research", "sources": sources}}},
        fixture["task"], context, source_registry=registry,
    )
    return cards, registry


def test_reviewed_fixture_is_bounded_and_preserves_exact_source_excerpt_lengths():
    fixture = load_fixture()
    assert len(fixture["sources"]) == 4
    assert sum(len(row["text"]) for s in fixture["sources"] for row in s["excerpts"]) < 5000
    for source in fixture["sources"]:
        assert len(source["original_content_hash"]) == 64
        assert source["url"].startswith("https://")
        assert source["reviewed_role"]
        for excerpt in source["excerpts"]:
            assert len(excerpt["text"]) == excerpt["original_end"] - excerpt["original_start"]


def test_saved_report_reproduces_tutorial_promotion_and_elapsed_event_mixing():
    fixture = load_fixture()
    report = (FIXTURES / fixture["observed"]["report_fixture"]).read_text()
    by_id = {s["id"]: s for s in fixture["sources"]}
    for promotion in fixture["observed_promotions"]:
        original = "\n\n".join(row["text"] for row in by_id[promotion["source_id"]]["excerpts"])
        assert promotion["claim"] in original and promotion["claim"] in report
    singapore = by_id["src-6010e9635ba9f7c8"]
    assert "Sunday, October 11" in singapore["excerpts"][0]["text"]
    assert datetime.fromisoformat(fixture["as_of"]).date().isoformat() == "2026-10-12"
    assert fixture["observed"]["sections_synthesized"] == 0
    assert fixture["observed"]["questions_unanswered"] == 7
    assert "United States Grand Prix" not in report
    assert len(fixture["expected_deficits"]) == 5


def test_calendar_witness_preserves_ambiguity_not_an_expected_real_world_answer():
    fixture = load_fixture()
    schedule = next(s for s in fixture["sources"] if s["id"] == "src-67b0f600ea6fedc6")
    text = schedule["excerpts"][0]["text"]
    assert "Round 18: United States Grand Prix" in text
    assert "When is the United States Grand Prix?\nThe Singapore Grand Prix" in text
    assert "expected_event" not in fixture and "expected_race_date" not in fixture


def test_saved_originals_retain_methodology_role_for_model_review():
    fixture = load_fixture()
    cards, registry = source_only_cards(fixture)
    selected, reasons = select_requirement_cards(cards, [{
        "identifier": "fixture-timing", "question": "What is the relevant timing and scope of the requested subject?",
        "importance": "high", "state": "open",
    }], source_registry=registry)
    # Inventory eligibility is not an assertion that a tutorial answers an event
    # question. Its instructional role must remain available to model review.
    tutorials = [card for card in cards if "f1dataR" in card["label"]]
    assert tutorials and all(card["finding"]["role"] == "instruction" for card in tutorials)
    assert any(card in tutorials for card in selected)
    assert all(card["id"] in reasons for card in selected)
