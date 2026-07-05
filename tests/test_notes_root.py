"""Tests for the sticky notes-root override (tools/notes_root.py).

Each test gets a fresh tmp cwd so the override file (`./data/notes_root_override.json`)
isn't shared with the real install. `monkeypatch.chdir(tmp_path)` is used to redirect
the override file to a sandbox location; `monkeypatch.setattr` resets the module-level
override between tests.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import notes_root  # noqa: E402


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    """Each test runs in a fresh tmp cwd; reset module-level state."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(notes_root, "_override", None)
    monkeypatch.setattr(notes_root, "_migrated", False)
    # conftest.py points the real module at a fixed scratch file (via
    # FULLOCH_NOTES_ROOT_OVERRIDE_PATH) so pytest never touches the real
    # sticky override; redirect it into this test's sandboxed tmp_path too.
    monkeypatch.setattr(notes_root, "_OVERRIDE_PATH", tmp_path / "data" / "notes_root_override.json")


def _write_config(path: Path, notes_path: str = "./data/notes") -> None:
    cfg = {"notes": {"path": notes_path}}
    (path / "data" / "config.yml").write_text(
        __import__("yaml").dump(cfg), encoding="utf-8"
    )


def test_default_is_config_path(tmp_path):
    _write_config(tmp_path, "./data/notes")
    assert notes_root.get_notes_root() == (tmp_path / "data" / "notes").resolve()


def test_default_is_dotted_when_config_missing(tmp_path):
    assert notes_root.get_notes_root() == (tmp_path / "data" / "notes").resolve()


def test_set_notes_root_returns_true_on_change():
    new = Path("/tmp/vault-a").resolve()
    assert notes_root.set_notes_root(new) is True
    assert notes_root.get_notes_root() == new


def test_set_notes_root_returns_false_when_unchanged():
    new = Path("/tmp/vault-b").resolve()
    notes_root.set_notes_root(new)
    assert notes_root.set_notes_root(new) is False


def test_set_notes_root_persists_to_disk(tmp_path):
    new = (tmp_path / "vault").resolve()
    notes_root.set_notes_root(new)
    raw = (tmp_path / "data" / "notes_root_override.json").read_text()
    assert json.loads(raw)["path"] == str(new)


def test_init_loads_override_from_disk(tmp_path):
    new = (tmp_path / "vault").resolve()
    payload = json.dumps({"path": str(new), "migrated": True, "set_at": "2026-07-02T00:00:00Z"})
    (tmp_path / "data" / "notes_root_override.json").write_text(payload, encoding="utf-8")
    notes_root.init()
    assert notes_root.get_notes_root() == new
    assert notes_root.get_migrated() is True


def test_clear_overrides_back_to_default(tmp_path):
    _write_config(tmp_path, "./data/notes")
    notes_root.set_notes_root(Path("/tmp/somewhere-else").resolve())
    notes_root.set_notes_root(None)
    assert notes_root.get_notes_root() == (tmp_path / "data" / "notes").resolve()
    raw = json.loads((tmp_path / "data" / "notes_root_override.json").read_text())
    assert "path" not in raw


def test_index_listener_fires_on_change():
    calls: list[tuple[Path | None, Path | None]] = []
    notes_root.register_index_listener(lambda old, new: calls.append((old, new)))
    new_path = Path("/tmp/vault-c").resolve()
    notes_root.set_notes_root(new_path)
    assert calls == [(None, new_path)]
    notes_root.set_notes_root(None)
    assert calls == [(None, new_path), (new_path, None)]


def test_index_listener_not_fired_when_unchanged():
    calls: list = []
    new_path = Path("/tmp/vault-d").resolve()
    notes_root.set_notes_root(new_path)
    notes_root.register_index_listener(lambda *a: calls.append(a))
    notes_root.set_notes_root(new_path)  # no-op
    assert calls == []


def test_set_migrated_persists(tmp_path):
    new = (tmp_path / "vault").resolve()
    notes_root.set_notes_root(new)
    notes_root.set_migrated(True)
    raw = json.loads((tmp_path / "data" / "notes_root_override.json").read_text())
    assert raw["migrated"] is True
    assert notes_root.get_migrated() is True


# ---- Path translation (Docker host→container vault mapping) -----------------

def test_translate_vault_path_no_map_returns_input(monkeypatch, tmp_path):
    (tmp_path / "data" / "config.yml").write_text("general: {}\n")
    notes_root.reload_translation_map()
    p = notes_root.translate_vault_path("/Users/jane/Documents/MyVault")
    assert str(p) == "/Users/jane/Documents/MyVault"


def test_translate_vault_path_applies_longest_prefix(monkeypatch, tmp_path):
    cfg = {
        "obsidian": {
            "path_translation": {
                "/Users/jane": "/vault",
                "/Users/jane/Documents": "/docs",
            }
        }
    }
    (tmp_path / "data" / "config.yml").write_text(yaml.dump(cfg))
    notes_root.reload_translation_map()
    # /Users/jane/Documents/MyVault → /docs/MyVault (longest prefix wins)
    p = notes_root.translate_vault_path("/Users/jane/Documents/MyVault")
    assert str(p) == "/docs/MyVault"
    # /Users/jane/Pictures/X → /vault/Pictures/X (shorter prefix)
    p = notes_root.translate_vault_path("/Users/jane/Pictures/X")
    assert str(p) == "/vault/Pictures/X"


def test_translate_vault_path_exact_match(monkeypatch, tmp_path):
    cfg = {"obsidian": {"path_translation": {"/Users/jane": "/vault"}}}
    (tmp_path / "data" / "config.yml").write_text(yaml.dump(cfg))
    notes_root.reload_translation_map()
    p = notes_root.translate_vault_path("/Users/jane")
    assert str(p) == "/vault"


def test_translate_vault_path_no_match_passthrough(monkeypatch, tmp_path):
    cfg = {"obsidian": {"path_translation": {"/Users/jane": "/vault"}}}
    (tmp_path / "data" / "config.yml").write_text(yaml.dump(cfg))
    notes_root.reload_translation_map()
    # /home/alice/... doesn't match /Users/jane — passthrough unchanged.
    p = notes_root.translate_vault_path("/home/alice/Documents/Vault")
    assert str(p) == "/home/alice/Documents/Vault"


def test_translate_vault_path_ignores_non_string_values(monkeypatch, tmp_path):
    cfg = {"obsidian": {"path_translation": {"/Users/jane": "", "/Bob": None, "": "/x"}}}
    (tmp_path / "data" / "config.yml").write_text(yaml.dump(cfg))
    notes_root.reload_translation_map()
    # Only valid pairs kept — empty/None values dropped.
    p = notes_root.translate_vault_path("/Users/jane/Documents")
    assert str(p) == "/Users/jane/Documents"  # passthrough (empty value)
