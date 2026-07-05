"""Tests for the semantic note index.

The actual embedding model (BAAI/bge-small-en-v1.5) is too heavy to load
inside a unit test, so a stub encoder is injected. It hashes input text to
a small deterministic vector so similarity behaves like a fast keyword
overlap proxy — sufficient to verify the index machinery (chunking, mtime
diffing, persistence, write hooks, top-k ordering) without exercising the
real model.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    notes,  # noqa: E402
    notes_index,  # noqa: E402
    notes_root,  # noqa: E402
)


class _StubEncoder:
    """Deterministic toy encoder: vectors derived from hashed unigrams.

    Captures roughly the keyword overlap intuition we'd want from BGE
    without loading 130MB of weights per test run.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(
        self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.sha1(token.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(notes_root, "_override", None)
    monkeypatch.setattr(notes_root, "_migrated", False)
    base = tmp_path / "notes"
    base.mkdir()
    notes_root.set_notes_root(base, persist=False)
    monkeypatch.setattr(notes, "DAILY_SUBDIR", None)
    # Wipe the lazy singleton between tests so each gets a fresh index.
    monkeypatch.setattr(notes, "_index", None)
    return base


@pytest.fixture
def stub_index(tmp_path, monkeypatch, notes_dir):
    """Build a NotesIndex backed by the stub encoder."""
    encoder = _StubEncoder()
    idx = notes_index.NotesIndex(
        notes_root=notes_dir,
        index_path=tmp_path / "notes_index",
        spoken_filter=notes._to_spoken,
    )
    # Skip the real model load + mark loaded so _ensure_loaded is a no-op.
    idx._model = encoder
    idx._loaded = True
    return idx


class TestChunking:
    def test_splits_on_blank_lines(self):
        content = "first paragraph here\n\nsecond paragraph follows\n\nthird"
        chunks = notes_index._split_chunks(content)
        assert len(chunks) == 3
        assert chunks[0][1].startswith("first")
        assert chunks[1][1].startswith("second")
        assert chunks[2][1].startswith("third")

    def test_line_numbers_track_offsets(self):
        content = "alpha\n\nbeta\n\ngamma"
        chunks = notes_index._split_chunks(content)
        assert chunks[0][0] == 1
        assert chunks[1][0] == 3
        assert chunks[2][0] == 5

    def test_drops_pure_whitespace(self):
        content = "real\n\n\n\nreal again"
        chunks = notes_index._split_chunks(content)
        assert all(t.strip() for _, t in chunks)


class TestIndexBuild:
    def test_index_file_populates_chunks(self, stub_index, notes_dir):
        path = notes_dir / "boiler.md"
        path.write_text(
            "# Heating boiler notes\n\n"
            "Vaillant ecoTEC, last serviced March 2026.\n\n"
            "Next service due in July 2026.\n"
        )
        stub_index.index_file(path)
        # Header survives MIN_CHUNK_CHARS=12, plus two body paragraphs.
        assert len(stub_index._chunks) == 3
        files = {c.file for c in stub_index._chunks}
        assert "boiler.md" in files

    def test_re_indexing_drops_stale_chunks(self, stub_index, notes_dir):
        path = notes_dir / "shopping.md"
        path.write_text(
            "# Shopping list reminder\n\nremember to buy milk\n\nremember to buy bread\n"
        )
        stub_index.index_file(path)
        initial_texts = {c.text for c in stub_index._chunks}
        assert any("milk" in t for t in initial_texts)
        assert any("bread" in t for t in initial_texts)

        path.write_text("# Shopping list reminder\n\nonly one combined paragraph now please\n")
        stub_index.index_file(path)
        new_texts = {c.text for c in stub_index._chunks}
        assert not any("milk" in t for t in new_texts)
        assert not any("bread" in t for t in new_texts)
        assert any("combined paragraph" in t for t in new_texts)

    def test_deleting_file_purges_chunks(self, stub_index, notes_dir):
        path = notes_dir / "ephemeral.md"
        path.write_text("# Ephemeral notes\n\nplaceholder body text here\n")
        stub_index.index_file(path)
        assert any(c.file == "ephemeral.md" for c in stub_index._chunks)
        path.unlink()
        stub_index.index_file(path)
        assert not any(c.file == "ephemeral.md" for c in stub_index._chunks)


class TestScan:
    def test_picks_up_files_not_yet_indexed(self, stub_index, notes_dir):
        (notes_dir / "a.md").write_text("# Alpha notes\n\nfirst paragraph body\n")
        (notes_dir / "b.md").write_text("# Beta notes\n\nsecond paragraph body\n")
        stub_index.scan()
        files = {c.file for c in stub_index._chunks}
        assert "a.md" in files
        assert "b.md" in files

    def test_skips_unchanged_files(self, stub_index, notes_dir):
        path = notes_dir / "stable.md"
        path.write_text("# Stable notes\n\nbody contents one here\n")
        stub_index.scan()
        chunks_first = list(stub_index._chunks)
        stub_index.scan()
        chunks_second = list(stub_index._chunks)
        # Same chunk identities, no churn
        assert [c.text for c in chunks_first] == [c.text for c in chunks_second]

    def test_drops_files_that_disappear(self, stub_index, notes_dir):
        path = notes_dir / "vanishing.md"
        path.write_text("# Gone soon notes\n\ncontent that will vanish\n")
        stub_index.scan()
        assert any(c.file == "vanishing.md" for c in stub_index._chunks)
        path.unlink()
        stub_index.scan()
        assert not any(c.file == "vanishing.md" for c in stub_index._chunks)


class TestPersistence:
    def test_roundtrip_through_disk(self, stub_index, notes_dir, tmp_path):
        path = notes_dir / "persisted.md"
        path.write_text("# Persisted\n\nimportant paragraph here\n")
        stub_index.index_file(path)
        assert stub_index._npy_path.exists()
        assert stub_index._meta_path.exists()

        # Build a fresh index over the same files, restore from disk.
        fresh = notes_index.NotesIndex(
            notes_root=notes_dir,
            index_path=tmp_path / "notes_index",
            spoken_filter=notes._to_spoken,
        )
        fresh._model = stub_index._model  # share stub encoder
        fresh._restore()
        fresh._loaded = True
        assert len(fresh._chunks) == len(stub_index._chunks)
        assert [c.text for c in fresh._chunks] == [c.text for c in stub_index._chunks]


class TestSearch:
    def test_returns_relevant_chunks_first(self, stub_index, notes_dir):
        (notes_dir / "boiler.md").write_text(
            "# Boiler\n\nVaillant ecoTEC heating unit installed 2019\n"
        )
        (notes_dir / "shopping.md").write_text("# Shopping\n\nmilk eggs bread\n")
        stub_index.scan()
        results = stub_index.search("Vaillant ecoTEC heating", k=2)
        assert results
        top_file = results[0][1].file
        assert top_file == "boiler.md"

    def test_empty_index_returns_no_results(self, stub_index):
        results = stub_index.search("anything")
        assert results == []


class TestWriteHook:
    """Verify the notes.py write tools trigger _after_write → indexing."""

    def test_write_note_indexes_immediately(self, stub_index, monkeypatch, notes_dir):
        # _after_write runs in a background thread; patch it to run synchronously
        # so the assertion doesn't race the thread.
        monkeypatch.setattr(notes, "_after_write", lambda path: stub_index.index_file(path))
        notes.write_note("Heating", "boiler is a Vaillant ecoTEC")
        files = {c.file for c in stub_index._chunks}
        assert "heating.md" in files

    def test_search_semantic_finds_recent_writes(self, stub_index, monkeypatch, notes_dir):
        monkeypatch.setattr(notes, "_get_index", lambda: stub_index)
        notes.write_note("Heating", "Vaillant ecoTEC heating")
        notes.write_note("Shopping", "milk bread eggs")
        # Loosen threshold for the stub encoder's smaller similarity range
        monkeypatch.setattr(notes, "SEMANTIC_MIN_SCORE", 0.0)
        # search_notes is now the hybrid entry point (search_notes_semantic is
        # just an alias); the semantic pass runs through the stub index.
        result = notes.search_notes("Vaillant ecoTEC heating")
        assert "heating" in result.lower()
