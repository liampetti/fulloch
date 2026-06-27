"""Parloch branding: when the LLM runs off-device (remote OpenAI endpoint) the
dashboard swaps the Fulloch character + favicon to Parloch and flags it in
/status, so the UI is honest that the setup is no longer fully local.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from server.dashboard import create_app

ROOT = Path(__file__).resolve().parent.parent


def _assistant():
    with patch("core.assistant.AudioCapture") as mac:
        mac.return_value = MagicMock()
        from core.assistant import Assistant

        a = Assistant(barge_in="wakeword", wakeword="hey atticus")
    a.audio_capture.transcribing = False  # real bool so /status serialises
    return a


def _client(a, monkeypatch):
    monkeypatch.delenv("FULLOCH_DASHBOARD_TOKEN", raising=False)  # /logo.png is exempt anyway
    return TestClient(create_app(a))


def test_logo_is_fulloch_for_local_llm(monkeypatch):
    a = _assistant()
    a.llm_backend = "llama"  # local llama.cpp — fully local
    r = _client(a, monkeypatch).get("/logo.png")
    assert r.status_code == 200
    assert r.content == (ROOT / "fulloch.png").read_bytes()


def test_logo_swaps_to_parloch_for_remote_llm(monkeypatch):
    a = _assistant()
    a.llm_backend = "openai"  # off-device — Parloch
    r = _client(a, monkeypatch).get("/logo.png")
    assert r.status_code == 200
    assert r.content == (ROOT / "parloch.png").read_bytes()
    # no-cache so a backend change is reflected on the next reload.
    assert r.headers.get("cache-control") == "no-cache"


def test_status_reports_remote_llm_flag(monkeypatch):
    a = _assistant()
    client = _client(a, monkeypatch)
    a.llm_backend = "openai"
    assert client.get("/status").json()["remote_llm"] is True
    a.llm_backend = "llama"
    assert client.get("/status").json()["remote_llm"] is False


def test_remote_query_param_overrides_running_state(monkeypatch):
    # The wizard/settings preview Parloch via ?remote=1 before any restart,
    # regardless of what's actually running.
    a = _assistant()
    client = _client(a, monkeypatch)
    fulloch = (ROOT / "fulloch.png").read_bytes()
    parloch = (ROOT / "parloch.png").read_bytes()

    a.llm_backend = "llama"  # running local...
    assert client.get("/logo.png?remote=1").content == parloch  # ...but preview Parloch

    a.llm_backend = "openai"  # running remote...
    assert client.get("/logo.png?remote=0").content == fulloch  # ...but preview Fulloch


def test_default_assistant_is_local(monkeypatch):
    # No models block resolves to the local Qwen/llama stack -> not Parloch.
    a = _assistant()
    assert a.llm_backend != "openai"
