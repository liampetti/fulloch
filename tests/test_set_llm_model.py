"""Live LLM model hot-swap: Assistant.set_llm_model swaps the remote model with
no restart, refuses a local backend, and the /llm/model endpoint persists it."""

import threading
import types
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from core.assistant import Assistant
from server.dashboard import create_app
from server.lifecycle import READY, AppContext, Lifecycle


def _fake_self(handle):
    """Minimal stand-in exercising set_llm_model without full construction."""
    return types.SimpleNamespace(slm_model=handle, _turn_lock=threading.Lock())


def test_swaps_remote_model_under_lock():
    handle = types.SimpleNamespace(
        _fulloch_remote=True,
        set_model=lambda m: setattr(handle, "model", m),
    )
    out = Assistant.set_llm_model(_fake_self(handle), "  new-model ")
    assert out == {"ok": True, "model": "new-model"}
    assert handle.model == "new-model"


def test_refuses_local_backend():
    handle = types.SimpleNamespace()  # no _fulloch_remote -> local llama/none
    out = Assistant.set_llm_model(_fake_self(handle), "x")
    assert out["ok"] is False and "not OpenAI" in out["error"]


def test_blank_model_rejected():
    handle = types.SimpleNamespace(_fulloch_remote=True, set_model=lambda m: None)
    assert Assistant.set_llm_model(_fake_self(handle), "  ")["ok"] is False


def test_endpoint_swaps_and_persists(tmp_path):
    cfg = tmp_path / "config.yml"
    cfg.write_text("models:\n  llm:\n    backend: openai\n    model: old\n")
    assistant = MagicMock()
    assistant.set_llm_model.return_value = {"ok": True, "model": "fresh"}
    context = AppContext(lifecycle=Lifecycle(phase=READY), assistant=assistant,
                         config_path=str(cfg))
    client = TestClient(create_app(context=context))

    r = client.post("/llm/model", json={"model": "fresh"})

    assert r.status_code == 200 and r.json()["ok"] is True
    assistant.set_llm_model.assert_called_once_with("fresh")
    # persisted so it survives a restart
    from server import config_store as cs
    assert cs.read_config(str(cfg))["models"]["llm"]["model"] == "fresh"
