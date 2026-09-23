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
    monkeypatch.setitem(notes.config, "obsidian", {})
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
    assert body["path_navigation_mismatch"] is False
    assert body["allow_edit_delete"] is False


def test_edit_capability_persists_and_returns_current_state(ctx):
    client = _client(ctx)
    r = client.post("/api/obsidian/edit-capability", json={"enabled": True})
    assert r.status_code == 200
    assert r.json() == {"allow_edit_delete": True}
    assert client.get("/api/obsidian/status").json()["allow_edit_delete"] is True


def test_status_does_not_leak_token(ctx):
    client = _client(ctx)
    r = client.get("/api/obsidian/status")
    assert "token" not in r.json()


def test_documents_list_and_viewer_are_limited_to_the_notes_folder(ctx, tmp_path, monkeypatch):
    root = tmp_path / "documents"
    monkeypatch.setattr(notes_root, "_override", root)
    (root / "Projects").mkdir(parents=True)
    document = root / "Projects" / "garden-plan.md"
    document.write_text("# Garden plan\n\nPlant herbs in spring.", encoding="utf-8")
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "hidden.md").write_text("hidden", encoding="utf-8")

    client = _client(ctx)
    listing = client.get("/api/documents")

    assert listing.status_code == 200
    documents = listing.json()["documents"]
    assert {
        "path": "Projects/garden-plan.md",
        "title": "garden plan",
        "modified_at": document.stat().st_mtime,
    } in documents
    assert all(not item["path"].startswith(".obsidian/") for item in documents)
    viewer = client.get("/documents/Projects/garden-plan.md")
    assert viewer.status_code == 200
    assert viewer.headers["content-type"].startswith("text/html")
    assert "<h1>Garden plan</h1>" in viewer.text and "Plant herbs in spring." in viewer.text
    assert client.get("/documents/../credentials.md").status_code == 404


def test_report_link_renders_formatted_html(ctx, tmp_path, monkeypatch):
    root = tmp_path / "documents"
    monkeypatch.setattr(notes_root, "_override", root)
    report = root / "fulloch-reports" / "2026-08-27-12345678.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Report\n\nA useful finding.", encoding="utf-8")

    response = _client(ctx).get("/reports/fulloch-reports/2026-08-27-12345678")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Report" in response.text and "A useful finding." in response.text


@pytest.mark.parametrize("url", [
    "/documents/fulloch-reports/2026-08-27-12345678.md",
    "/reports/fulloch-reports/2026-08-27-12345678",
])
def test_document_edits_save_markdown_and_reindex(ctx, tmp_path, monkeypatch, url):
    root = tmp_path / "documents"
    monkeypatch.setattr(notes_root, "_override", root)
    after_write = MagicMock()
    monkeypatch.setattr(notes, "_after_write", after_write)
    path = root / "fulloch-reports/2026-08-27-12345678.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Original\n", encoding="utf-8")
    client = _client(ctx)
    response = client.put(url, json={"content": "# Edited\n\n**Updated**\n"})
    assert response.status_code == 200
    assert "<strong>Updated</strong>" in response.json()["html"]
    assert path.read_text(encoding="utf-8") == "# Edited\n\n**Updated**\n"
    after_write.assert_called_once_with(path)
    assert "# Edited" in client.get(url).text
    assert client.put("/documents/missing.md", json={"content": "new"}).status_code == 404
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    (root / "escape.md").symlink_to(outside)
    assert client.put("/documents/escape.md", json={"content": "changed"}).status_code == 404
    assert outside.read_text(encoding="utf-8") == "private"


def test_plugin_archive_download(ctx, tmp_path, monkeypatch):
    archive = tmp_path / "fulloch-obsidian-plugin.zip"
    archive.write_bytes(b"plugin")
    monkeypatch.setattr("server.routes_obsidian._OBSIDIAN_PLUGIN_ZIP", archive)

    r = _client(ctx).get("/api/obsidian/plugin.zip")

    assert r.status_code == 200
    assert r.content == b"plugin"
    assert r.headers["content-type"] == "application/zip"


def test_regenerate_token_writes_new_value(ctx):
    client = _client(ctx)
    r = client.post("/api/obsidian/regenerate-token")
    assert r.status_code == 200
    new_token = r.json()["token"]
    assert new_token and new_token != "test-tok"
    on_disk = get_credential("obsidian_token", path=str(Path.cwd() / "data" / "credentials.json"))
    assert on_disk == new_token


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
    r = client.post("/api/setup/obsidian-vault", json={"path": "/Users/jane/MyVault"})
    assert r.status_code == 200
    assert notes_root.get_notes_root() == container_vault.resolve()
