"""Dashboard API for the Obsidian integration."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from server.credentials_store import (  # noqa: E402
    get_credential,
    set_credential,
)
from server.dashboard import create_app  # noqa: E402
from server.lifecycle import READY, AppContext, Lifecycle  # noqa: E402
from tools import notes, notes_root  # noqa: E402


def _stub_assistant():
    a = MagicMock()
    a.register_turn_listener = MagicMock()
    a.get_state.return_value = "idle"
    a.audio_capture.transcribing = True
    a.wakeword = "hey atticus"
    return a


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("OBSIDIAN_TOKEN", "test-tok")
    set_credential("obsidian_token", "test-tok", path=str(tmp_path / "data" / "credentials.json"))
    ctx = AppContext(lifecycle=Lifecycle(phase=READY))
    ctx.assistant = _stub_assistant()
    notes_root.set_notes_root(None, persist=False)
    monkeypatch.setattr(notes_root, "_override", None)
    monkeypatch.setattr(notes_root, "_migrated", False)
    monkeypatch.setattr(notes, "NOTES_DIR_LEGACY", (tmp_path / "data" / "notes").resolve())
    return ctx


def _client(ctx):
    return TestClient(create_app(context=ctx))


def test_status_returns_state(ctx):
    client = _client(ctx)
    r = client.get("/api/obsidian/status")
    assert r.status_code == 200
    body = r.json()
    assert "connected" in body
    assert "vault_path" in body
    assert "indexing_progress" in body


def test_status_does_not_leak_token(ctx):
    client = _client(ctx)
    r = client.get("/api/obsidian/status")
    assert "token" not in r.json()


def test_regenerate_token_writes_new_value(ctx):
    client = _client(ctx)
    r = client.post("/api/obsidian/regenerate-token")
    assert r.status_code == 200
    new_token = r.json()["token"]
    assert new_token and new_token != "test-tok"
    on_disk = get_credential("obsidian_token", path=str(Path.cwd() / "data" / "credentials.json"))
    assert on_disk == new_token


def test_switch_vault_sets_override(ctx, tmp_path):
    new_vault = tmp_path / "switched-vault"
    new_vault.mkdir()
    (new_vault / ".obsidian").mkdir()
    client = _client(ctx)
    r = client.post("/api/obsidian/switch-vault", json={"path": str(new_vault)})
    assert r.status_code == 200
    assert notes_root.get_notes_root() == new_vault.resolve()


def test_switch_vault_rejects_non_vault(ctx, tmp_path):
    bad = tmp_path / "not-a-vault"
    bad.mkdir()
    client = _client(ctx)
    r = client.post("/api/obsidian/switch-vault", json={"path": str(bad)})
    assert r.status_code == 400
    assert "not_a_vault" in r.text or "vault" in r.text.lower()


def test_switch_vault_translates_host_path(ctx, tmp_path, monkeypatch):
    """Docker: the user pastes a host path; the server must remap to the
    in-container path before validating the .obsidian/ folder exists."""
    # Set up the in-container vault, plus a config that maps the host path.
    container_vault = tmp_path / "container" / "MyVault"
    container_vault.mkdir(parents=True)
    (container_vault / ".obsidian").mkdir()
    import yaml
    (tmp_path / "data" / "config.yml").write_text(yaml.dump({
        "obsidian": {"path_translation": {"/Users/jane": str(tmp_path / "container")}}
    }))
    from tools import notes_root
    # conftest.py points _CONFIG_PATH at data/config.example.yml (absolute,
    # fixed at import time) so this test's own tmp_path/data/config.yml is
    # only picked up if we redirect it here too.
    monkeypatch.setattr(notes_root, "_CONFIG_PATH", tmp_path / "data" / "config.yml")
    notes_root.reload_translation_map()
    client = _client(ctx)
    r = client.post("/api/obsidian/switch-vault", json={"path": "/Users/jane/MyVault"})
    assert r.status_code == 200
    assert notes_root.get_notes_root() == container_vault.resolve()


def test_migration_decision_copy(ctx, tmp_path):
    src = tmp_path / "data" / "notes"
    src.mkdir(parents=True)
    (src / "thing.md").write_text("hello", encoding="utf-8")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    notes_root.set_notes_root(vault, persist=False)

    client = _client(ctx)
    r = client.post("/api/obsidian/migration-decision", json={"action": "copy"})
    assert r.status_code == 200
    assert (vault / "Inbox" / "fulloch-import" / "thing.md").exists()


def test_migration_decision_skip_marks_dismissed(ctx, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    notes_root.set_notes_root(vault, persist=False)

    client = _client(ctx)
    r = client.post("/api/obsidian/migration-decision", json={"action": "skip"})
    assert r.status_code == 200
    assert notes_root.get_migrated() is True


def test_migration_decision_dismiss_never_asks_again(ctx, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    notes_root.set_notes_root(vault, persist=False)

    client = _client(ctx)
    r = client.post("/api/obsidian/migration-decision", json={"action": "dismiss"})
    assert r.status_code == 200
    assert notes_root.get_migrated() is True
