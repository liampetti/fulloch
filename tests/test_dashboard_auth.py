"""Dashboard bearer-token auth (Tier 0.2).

Verifies that FULLOCH_DASHBOARD_TOKEN gates the sensitive routes while leaving
the unauthenticated shell (HTML page + logo) reachable, and that an unset token
disables auth entirely (the zero-config local-only path).
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server.dashboard import create_app


def _stub_assistant():
    """Minimal assistant surface create_app + the auth-exercised routes need."""
    assistant = MagicMock()
    assistant.register_turn_listener = MagicMock()
    assistant.get_state.return_value = "idle"
    assistant.audio_capture.transcribing = True
    assistant.wakeword = "hey atticus"
    assistant._history = []
    return assistant


def test_no_token_means_no_auth(monkeypatch):
    monkeypatch.delenv("FULLOCH_DASHBOARD_TOKEN", raising=False)
    client = TestClient(create_app(_stub_assistant()))
    # Sensitive route reachable without any credential.
    assert client.get("/status").status_code == 200
    assert client.get("/history").status_code == 200


def test_token_blocks_unauthenticated_requests(monkeypatch):
    monkeypatch.setenv("FULLOCH_DASHBOARD_TOKEN", "s3cret")
    client = TestClient(create_app(_stub_assistant()))

    # Shell stays open so the SPA can load and prompt for the token.
    assert client.get("/").status_code == 200
    assert client.get("/logo.png").status_code in (200, 404)  # file may be absent in CI

    # Sensitive routes are gated.
    assert client.get("/status").status_code == 401
    assert client.get("/history").status_code == 401
    assert client.get("/notes").status_code == 401


def test_token_accepted_via_header(monkeypatch):
    monkeypatch.setenv("FULLOCH_DASHBOARD_TOKEN", "s3cret")
    client = TestClient(create_app(_stub_assistant()))
    r = client.get("/status", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_token_accepted_via_query_param(monkeypatch):
    # EventSource can't set headers, so /stream-style auth uses ?token=.
    monkeypatch.setenv("FULLOCH_DASHBOARD_TOKEN", "s3cret")
    client = TestClient(create_app(_stub_assistant()))
    assert client.get("/status?token=s3cret").status_code == 200


def test_wrong_token_rejected(monkeypatch):
    monkeypatch.setenv("FULLOCH_DASHBOARD_TOKEN", "s3cret")
    client = TestClient(create_app(_stub_assistant()))
    r = client.get("/status", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert client.get("/status?token=nope").status_code == 401
