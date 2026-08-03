"""Focused tests for the restricted Home Assistant integration API."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server.integration_api import create_integration_app
from server.lifecycle import READY, AppContext, Lifecycle


def _context() -> AppContext:
    assistant = MagicMock()
    assistant.get_state.return_value = "idle"
    assistant.audio_capture.mic_globally_enabled = True
    return AppContext(lifecycle=Lifecycle(phase=READY), assistant=assistant)


def _client(monkeypatch):
    monkeypatch.setattr("server.integration_api.get_credential", lambda key: ["integration-token"])
    return TestClient(create_integration_app(_context()))


def test_integration_api_exposes_only_hacs_routes(monkeypatch):
    client = _client(monkeypatch)
    paths = {route.path for route in client.app.routes}
    assert paths == {"/status", "/speak", "/chat", "/mic", "/stream"}
    assert client.get("/static/index.js").status_code == 401


def test_integration_api_requires_its_own_bearer_token(monkeypatch):
    client = _client(monkeypatch)
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer satellite-token"}).status_code == 401

    response = client.get("/status", headers={"Authorization": "Bearer integration-token"})
    assert response.status_code == 200
    assert response.json()["state"] == "idle"


def test_integration_api_controls_mic_and_rejects_empty_speech(monkeypatch):
    client = _client(monkeypatch)
    headers = {"Authorization": "Bearer integration-token"}

    assert client.post("/mic", headers=headers, json={"enabled": False}).json() == {
        "ok": True,
        "mic_enabled": False,
    }
    assert client.post("/speak", headers=headers, json={"text": "  "}).status_code == 400
