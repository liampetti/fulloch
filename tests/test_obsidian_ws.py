"""WebSocket behaviour for the Obsidian plugin bridge.

Covers: token auth, `vault_metadata` adoption (sticky override), `context`
message → assistant.set_vault_context, `file_changed` reindex, and
disconnect cleanup.
"""
import json
import secrets
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from server.dashboard import create_app  # noqa: E402
from server.lifecycle import READY, AppContext, Lifecycle  # noqa: E402
from tools import notes, notes_root  # noqa: E402


def _stub_assistant():
    a = MagicMock()
    a.register_turn_listener = MagicMock()
    a.get_state.return_value = "idle"
    a.audio_capture.transcribing = True
    a.wakeword = "hey atticus"
    a._history = []
    a.set_vault_context = MagicMock()
    return a


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    notes_root.set_notes_root(None, persist=False)
    monkeypatch.setattr(notes_root, "_override", None)
    monkeypatch.setattr(notes_root, "_migrated", False)
    ctx = AppContext(lifecycle=Lifecycle(phase=READY))
    ctx.assistant = _stub_assistant()
    ctx.obsidian_token = secrets.token_hex(8)
    return ctx


def _client_for(ctx):
    return TestClient(create_app(context=ctx))


def _connect(client, token):
    return client.websocket_connect(f"/ws/obsidian?token={token}")


def test_tokenless_when_unset(ctx):
    """When obsidian_token is empty, no auth is required."""
    ctx.obsidian_token = ""
    client = _client_for(ctx)
    with client.websocket_connect("/ws/obsidian"):
        pass  # connect succeeded


def test_token_mismatch_rejects(ctx):
    ctx.obsidian_token = "right-token"
    client = _client_for(ctx)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/obsidian?token=wrong-token"):
            pass


def test_vault_metadata_records_editor_path_without_redirecting_notes(ctx, tmp_path):
    client = _client_for(ctx)
    vault = tmp_path / "some-vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    notes_root_before = notes_root.get_notes_root()
    with _connect(client, ctx.obsidian_token) as ws:
        ws.send_text(json.dumps({"type": "vault_metadata", "vault_path": str(vault), "files": []}))
        import time
        time.sleep(0.1)
    assert notes_root.get_notes_root() == notes_root_before
    assert ctx.obsidian_vault_state["connected"] is False  # cleared on disconnect
    assert ctx.obsidian_vault_state["vault_path"] == str(vault.resolve())


def test_vault_metadata_marks_differing_plugin_and_server_paths(ctx, tmp_path, monkeypatch):
    container_vault = tmp_path / "container-vault"
    container_vault.mkdir()
    (container_vault / ".obsidian").mkdir()
    monkeypatch.setattr(
        "tools.notes_root.translate_vault_path", lambda _path: container_vault
    )
    client = _client_for(ctx)

    with _connect(client, ctx.obsidian_token) as ws:
        ws.send_text(json.dumps({"type": "vault_metadata", "vault_path": "/host/vault", "files": []}))
        import time
        time.sleep(0.1)

    assert ctx.obsidian_vault_state["plugin_vault_path"] == "/host/vault"
    assert ctx.obsidian_vault_state["path_navigation_mismatch"] is True


def test_context_message_routes_to_assistant(ctx):
    client = _client_for(ctx)
    with _connect(client, ctx.obsidian_token) as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "context",
                    "file": {
                        "path": "foo.md",
                        "name": "foo",
                        "tags": [],
                        "links": [],
                        "backlinks": [],
                        "selection": "selected words",
                    },
                }
            )
        )
        import time
        time.sleep(0.1)
    ctx.assistant.set_vault_context.assert_called()
    context_call = next(
        call for call in ctx.assistant.set_vault_context.call_args_list
        if call.kwargs.get("current_file") is not None
    )
    assert context_call.kwargs["current_file"]["selection"] == "selected words"


def test_file_changed_reindexes(ctx, tmp_path, monkeypatch):
    notes_root.set_notes_root(tmp_path / "vault", persist=False)
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "hello.md").write_text("hello world", encoding="utf-8")

    fake_index = MagicMock()
    monkeypatch.setattr(notes, "_get_index", lambda: fake_index)

    client = _client_for(ctx)
    with _connect(client, ctx.obsidian_token) as ws:
        ws.send_text(json.dumps({"type": "file_changed", "path": "hello.md"}))
        import time
        time.sleep(0.2)
    fake_index.index_file.assert_called_once()


def test_disconnect_clears_state(ctx):
    client = _client_for(ctx)
    with _connect(client, ctx.obsidian_token) as ws:
        ws.send_text(json.dumps({"type": "vault_metadata", "vault_path": "/tmp/x", "files": []}))
        import time
        time.sleep(0.05)
    assert ctx.obsidian_vault_state["connected"] is False
    assert ctx.obsidian_cmd_q is None
