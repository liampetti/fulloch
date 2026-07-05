"""Index invalidation when the notes root changes (vault path adoption).

Uses the same stub encoder as `test_notes_index.py` so the heavy model isn't
loaded. Verifies that the index's chunks + mtimes are cleared on root change
and that the next `scan()` rebuilds from the new root.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import notes_index, notes_root  # noqa: E402


class _StubEncoder:
    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(
        self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(__import__("hashlib").sha1(token.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


@pytest.fixture
def fresh_index(tmp_path, monkeypatch):
    """Build a NotesIndex and register it with notes_root."""
    # Hermetic: clear any listener / override leaked from a prior test.
    monkeypatch.setattr(notes_root, "_index_listener", None)
    monkeypatch.setattr(notes_root, "_override", None)
    monkeypatch.setattr(notes_root, "_migrated", False)

    base = tmp_path / "vault"
    base.mkdir()
    notes_root.set_notes_root(base, persist=False)
    idx = notes_index.NotesIndex(
        notes_root=base,
        index_path=tmp_path / "notes_index",
        spoken_filter=lambda s: s,
    )
    idx._model = _StubEncoder()
    idx._loaded = True
    return idx, base


def test_index_clears_on_root_change(fresh_index, tmp_path):
    idx, base = fresh_index
    (base / "a.md").write_text(
        "alpha paragraph contents here\n\nbeta paragraph contents here",
        encoding="utf-8",
    )
    idx.scan()
    assert len(idx._chunks) > 0

    new_base = tmp_path / "vault2"
    new_base.mkdir()
    notes_root.set_notes_root(new_base, persist=False)

    assert idx._chunks == []
    assert idx._mtimes == {}


def test_index_rebuilds_on_next_scan_after_change(fresh_index, tmp_path):
    idx, _ = fresh_index
    new_base = tmp_path / "vault2"
    new_base.mkdir()
    (new_base / "note.md").write_text(
        "hello world paragraph contents here",
        encoding="utf-8",
    )

    notes_root.set_notes_root(new_base, persist=False)
    idx.scan()
    assert len(idx._chunks) >= 1
    assert any(c.file == "note.md" for c in idx._chunks)


def test_listener_handles_cleared_override(fresh_index):
    """set_notes_root(None) updates _notes_root to the effective default."""
    idx, _ = fresh_index
    # Set a known override first.
    other = idx._notes_root.parent / "other"
    other.mkdir()
    notes_root.set_notes_root(other, persist=False)
    assert idx._notes_root == other
    # Now clear it.
    notes_root.set_notes_root(None, persist=False)
    # _notes_root should now equal the effective default (whatever that is).
    assert idx._notes_root == notes_root.get_notes_root()
