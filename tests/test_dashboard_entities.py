"""Dashboard Entities tab endpoints (voice deny-list management).

`/entities` lists HA entities with their voice allow/deny state; `POST /entities`
toggles an entity. Both no-op cleanly when Home Assistant isn't configured.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from server.dashboard import create_app
from tools import ha_client as ha


def _stub_assistant():
    assistant = MagicMock()
    assistant.register_turn_listener = MagicMock()
    assistant.get_state.return_value = "idle"
    assistant.audio_capture.transcribing = True
    assistant.wakeword = "hey atticus"
    assistant._history = []
    return assistant


def test_entities_unavailable_without_ha(monkeypatch):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {})  # no home_assistant block
    client = TestClient(create_app(_stub_assistant()))
    r = client.get("/entities")
    assert r.status_code == 200
    assert r.json() == {"available": False, "entities": []}


def test_entities_list_and_toggle(monkeypatch, tmp_path):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})

    monkeypatch.setattr(ha, "_loaded", True)
    monkeypatch.setattr(ha, "_ENTITY_ALIASES", {"front door": "lock.front_door"})
    monkeypatch.setattr(ha, "_ENTITY_ALIASES_MULTI", {"front door": ["lock.front_door"]})
    monkeypatch.setattr(ha, "_DENIED_ENTITIES", frozenset())
    monkeypatch.setattr(ha, "_DENYLIST_PATH", str(tmp_path / "denylist.json"))

    client = TestClient(create_app(_stub_assistant()))

    r = client.get("/entities")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["entities"] == [
        {"entity_id": "lock.front_door", "name": "front door", "domain": "lock", "denied": False},
    ]

    r = client.post("/entities", json={"entity_id": "lock.front_door", "denied": True})
    assert r.status_code == 200
    assert ha.get_denylist() == {"lock.front_door"}
    assert ha._load_denylist() == frozenset({"lock.front_door"})
    assert r.json()["entities"][0]["denied"] is True


def test_entities_toggle_rejects_empty_id(monkeypatch):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})

    client = TestClient(create_app(_stub_assistant()))
    r = client.post("/entities", json={"entity_id": "  ", "denied": True})
    assert r.status_code == 400


def test_entities_toggle_404_without_ha(monkeypatch):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {})
    client = TestClient(create_app(_stub_assistant()))
    r = client.post("/entities", json={"entity_id": "lock.front_door", "denied": True})
    assert r.status_code == 404


def test_bulk_entity_updates_are_atomic_and_preserve_other_entries(monkeypatch, tmp_path):
    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})
    monkeypatch.setattr(ha, "_loaded", True)
    monkeypatch.setattr(ha, "_ENTITY_ALIASES_MULTI", {
        "lamp": ["light.lamp"], "door": ["lock.door"], "fan": ["fan.study"],
    })
    monkeypatch.setattr(ha, "_ENTITY_ALIASES", {})
    monkeypatch.setattr(ha, "_DENIED_ENTITIES", frozenset({"fan.study", "switch.retired"}))
    monkeypatch.setattr(ha, "_DENYLIST_PATH", str(tmp_path / "denylist.json"))
    client = TestClient(create_app(_stub_assistant()))

    result = client.post("/entities/bulk", json={"entity_ids": ["light.lamp", "lock.door"], "denied": True})
    assert result.status_code == 200
    assert ha.get_denylist() == {"light.lamp", "lock.door", "fan.study", "switch.retired"}
    assert ha._load_denylist() == frozenset(ha.get_denylist())
    assert all(entity["denied"] for entity in result.json()["entities"])

    result = client.post("/entities/bulk", json={"entity_ids": ["light.lamp", "lock.door"], "denied": False})
    assert result.status_code == 200
    assert ha.get_denylist() == {"fan.study", "switch.retired"}
    assert ha._load_denylist() == frozenset(ha.get_denylist())

    for ids in ([], [""], ["light.lamp", "light.unknown"]):
        result = client.post("/entities/bulk", json={"entity_ids": ids, "denied": True})
        assert result.status_code == 400
        assert ha.get_denylist() == {"fan.study", "switch.retired"}


def test_bulk_entity_update_without_ha(monkeypatch):
    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {})
    client = TestClient(create_app(_stub_assistant()))
    assert client.post("/entities/bulk", json={"entity_ids": ["light.lamp"], "denied": True}).status_code == 404


def test_bulk_persistence_failure_does_not_change_memory(monkeypatch):
    monkeypatch.setattr(ha, "_DENIED_ENTITIES", frozenset({"lock.door"}))
    monkeypatch.setattr(ha, "_persist_denylist", MagicMock(side_effect=OSError("disk failure")))
    with pytest.raises(OSError):
        ha.set_entities_denied(["light.lamp", "lock.door"], False)
    assert ha.get_denylist() == {"lock.door"}
