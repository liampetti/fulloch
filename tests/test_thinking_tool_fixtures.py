"""Offline replay smoke tests for synthetic typed-tool evidence.

These fixtures test retained records, not live provider availability or quality.
"""

import json
from pathlib import Path

import pytest

from core.thinking_replay import load_replay, replay

FIXTURES = Path(__file__).parent / "deep-think"


def test_evidence_fixture_inventory_is_bounded():
    fixtures = list(FIXTURES.glob("synthetic-*.evidence.json"))
    assert len(fixtures) <= 5
    assert all(json.loads(path.read_text())["fixture_metadata"]["synthetic"] is True for path in fixtures)


@pytest.mark.parametrize(("name", "tool"), [
    ("synthetic-finance", "get_finance_quote"),
    ("synthetic-travel", "search_flights"),
    ("synthetic-ha-history", "get_entity_history"),
])
def test_synthetic_tool_fixture_replays_without_retrieval(name, tool):
    path = FIXTURES / f"{name}.evidence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fixture_metadata"] == {
        "synthetic": True,
        "purpose": "Offline tool-record regression; not live or current evidence.",
        "tool": tool,
    }
    result = replay(path, production=True)
    assert result["error"] == ""
    assert result["cards"] > 0
    assert result["production_chapters"] is True


def test_synthetic_tool_records_keep_their_important_fields():
    finance = load_replay(FIXTURES / "synthetic-finance.evidence.json").fact_cards[0]
    assert finance["fields"]["price"] == 42.5
    assert finance["fields"]["currency"] == "USD"
    assert finance["fields"]["quoted_at"] == "2026-10-29T09:59:00+00:00"

    travel = load_replay(FIXTURES / "synthetic-travel.evidence.json").fact_cards[0]
    assert travel["context"]["route"] == {"origin": "AAA", "destination": "BBB"}
    assert travel["fields"]["outbound"]["departure"]["time"] == "2027-01-17 14:40"
    assert travel["fields"]["inbound"]["departure"]["time"] == "2027-03-03 21:45"

    history = load_replay(FIXTURES / "synthetic-ha-history.evidence.json").fact_cards[0]
    assert history["context"]["tool"] == "get_entity_history"
    assert history["fields"]["unit"] == "°C"
    assert [point["value"] for point in history["fields"]["points"]] == [18.5, 20.0]
