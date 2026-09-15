"""Persistence and transport failure contracts at the extracted boundaries."""

import queue

from tools import notes, notes_obsidian, notes_root, notes_storage


def test_markdown_frontmatter_survives_append_and_dashboard_save(tmp_path, monkeypatch):
    monkeypatch.setattr(notes_root, "get_notes_root", lambda: tmp_path)
    written = []
    monkeypatch.setattr(notes, "_after_write", written.append)
    path = tmp_path / "project.md"
    original = "---\ntags: [project]\naliases:\n  - My Project\n---\n\n# Project\n"
    notes_storage.write_markdown(path, original)

    notes.append_to_note("Project", "Next step")
    assert notes.read_note_file("project") == original + "\n- Next step\n"
    replacement = original + "\nUpdated body"
    assert notes.save_note_file("project", replacement)
    assert notes.read_note_file("project") == replacement + "\n"
    assert written == [path, path]


def test_failed_atomic_save_preserves_document_and_skips_notification(tmp_path, monkeypatch):
    monkeypatch.setattr(notes_root, "get_notes_root", lambda: tmp_path)
    path = tmp_path / "project.md"
    path.write_text("original\n", encoding="utf-8")
    written = []

    def fail_replace(source, target):
        raise OSError("replacement failed")

    monkeypatch.setattr(notes_storage.os, "replace", fail_replace)
    assert not notes_storage.save_note_file("project", "new", after_write=written.append)
    assert path.read_text(encoding="utf-8") == "original\n"
    assert written == []
    assert list(tmp_path.iterdir()) == [path]


def test_storage_resolves_shared_root_on_each_call(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "one.md").write_text("one", encoding="utf-8")
    (second / "two.md").write_text("two", encoding="utf-8")
    monkeypatch.setattr(notes_root, "get_notes_root", lambda: first)
    assert notes.list_note_files() == [{"name": "one", "title": "one"}]
    monkeypatch.setattr(notes_root, "get_notes_root", lambda: second)
    assert notes.list_note_files() == [{"name": "two", "title": "two"}]
    assert notes.read_note_file("one") is None


def test_obsidian_transport_backpressure_and_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(notes_obsidian, "_command_queue", None)
    commands = queue.Queue(maxsize=1)
    notes.set_obsidian_cmd_q(commands)
    assert notes_obsidian.send_command({"type": "delete_active"})
    assert not notes_obsidian.send_command({"type": "insert", "text": "queued"})
    notes_obsidian.open_file(tmp_path / "saved.md")
    assert commands.get_nowait() == {"type": "delete_active"}
    notes.set_obsidian_cmd_q(None)
    assert not notes_obsidian.send_command({"type": "delete_active"})
