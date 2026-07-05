"""Dashboard Entities tab endpoints (voice deny-list management).

`/entities` lists HA entities with their voice allow/deny state; `POST /entities`
toggles an entity. Both no-op cleanly when Home Assistant isn't configured.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server.dashboard import create_app


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


def test_entities_list_and_toggle(monkeypatch):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})

    import tools.home_assistant as ha

    sample = [
        {"entity_id": "lock.front_door", "name": "Front Door", "domain": "lock", "denied": False},
    ]
    monkeypatch.setattr(ha, "list_entities", lambda: sample)
    calls = []
    monkeypatch.setattr(ha, "set_entity_denied", lambda eid, denied: calls.append((eid, denied)))

    client = TestClient(create_app(_stub_assistant()))

    r = client.get("/entities")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["entities"] == sample

    r = client.post("/entities", json={"entity_id": "lock.front_door", "denied": True})
    assert r.status_code == 200
    assert calls == [("lock.front_door", True)]


def test_entities_toggle_rejects_empty_id(monkeypatch):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {"home_assistant": {}})
    import tools.home_assistant as ha

    monkeypatch.setattr(ha, "list_entities", lambda: [])
    monkeypatch.setattr(ha, "set_entity_denied", lambda eid, denied: None)

    client = TestClient(create_app(_stub_assistant()))
    r = client.post("/entities", json={"entity_id": "  ", "denied": True})
    assert r.status_code == 400


def test_entities_toggle_404_without_ha(monkeypatch):

    import tools._config as cfg

    monkeypatch.setattr(cfg, "config", {})
    client = TestClient(create_app(_stub_assistant()))
    r = client.post("/entities", json={"entity_id": "lock.front_door", "denied": True})
    assert r.status_code == 404
