"""Embedding index for semantic note search.

Loaded lazily — the embedding model and any persisted index only spin up
when the user actually writes a note or runs a semantic search. Updates
happen synchronously from the write hooks in `notes.py` so the index is
always consistent with disk; for a typical personal-note count this is a
sub-second cost. A `threading.Lock` protects the in-memory chunk list
because writes can land from the assistant's per-turn worker thread.

Persistence: `data/notes_index.npy` (float32 N×D embeddings) + a JSON
sidecar with `{mtimes: {file: float}, chunks: [{file, line, text}]}`
where the chunk list order matches the .npy row order.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Picked in the planning step: 384-dim, retrieval-tuned, English-focused.
# Outperforms MiniLM on most retrieval benchmarks for ~40MB extra download.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE-v1.5 is trained to prepend this instruction to the *query* only (never
# the documents). It measurably lifts recall for short topical queries — the
# common case here ("Sydney to Perth route") — so `search()` adds it at encode
# time. Index-time chunk embeddings are left bare.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Bumped whenever the *embedding input* recipe changes (e.g. prepending the
# note title to each chunk). A persisted index built under an older recipe is
# discarded on restore so the whole store is re-embedded under the new one —
# otherwise mtime-diffing would keep stale embeddings forever.
INDEX_VERSION = 2

# Embeddings reach the SLM through a TTS-bound assistant — top-3 is enough
# breadth for spoken summaries without bloating replies.
DEFAULT_TOP_K = 3
# Skip chunks too short to carry retrievable signal. Avoids polluting the
# index with single-bullet "x" entries.
MIN_CHUNK_CHARS = 12

_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


@dataclass
class Chunk:
    """A single indexable unit: a paragraph from a note, plus its location."""

    file: str  # path relative to notes_root, posix-style
    line: int  # 1-indexed line where the paragraph starts (for citation)
    text: str  # spoken-mode text fed to both the embedder and the SLM
    embedding: np.ndarray  # (D,) float32, L2-normalized


def _split_chunks(content: str) -> List[Tuple[int, str]]:
    """Split markdown into (start_line, text) paragraphs.

    Yields paragraphs that are long enough to be meaningful. Markdown
    markers are kept here — `_to_spoken` is applied later before embedding
    so the line offsets in this step match the on-disk file.
    """
    chunks: List[Tuple[int, str]] = []
    cursor_line = 1
    # Walk paragraph-by-paragraph but track line numbers against the raw
    # content so we can cite a file location back to the user.
    pos = 0
    for match in _PARA_SPLIT_RE.finditer(content):
        chunk_text = content[pos : match.start()]
        if chunk_text.strip():
            chunks.append((cursor_line, chunk_text.strip()))
        cursor_line += content[pos : match.end()].count("\n")
        pos = match.end()
    tail = content[pos:]
    if tail.strip():
        chunks.append((cursor_line, tail.strip()))
    return chunks


class NotesIndex:
    """Lazy-loaded sentence-transformer index over the notes folder."""

    def __init__(
        self,
        notes_root: Path,
        index_path: Path,
        spoken_filter,
    ):
        """
        Args:
            notes_root: absolute path to the notes directory.
            index_path: where to persist the .npy / .json pair (stem).
            spoken_filter: callable str->str applied to chunk text before
                embedding. Reuses `tools.notes._to_spoken` so markdown
                markers don't pollute embeddings or spoken hits.
        """
        self._notes_root = notes_root
        self._npy_path = index_path.with_suffix(".npy")
        self._meta_path = index_path.with_suffix(".json")
        self._spoken_filter = spoken_filter

        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._chunks: List[Chunk] = []
        self._mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lazy load — model + persisted index
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the embedding model and any persisted index (idempotent)."""
        if self._loaded:
            return
        # Imports are deferred so users without the `notes:` block never
        # pay the sentence-transformers startup cost.
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {EMBED_MODEL_NAME} on {device}...")
        self._model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

        if self._npy_path.exists() and self._meta_path.exists():
            try:
                self._restore()
                logger.info(
                    f"Restored notes index: {len(self._chunks)} chunks, {len(self._mtimes)} files"
                )
            except Exception as e:
                logger.error(f"Failed to restore index ({e}); starting fresh")
                self._chunks = []
                self._mtimes = {}

        self._loaded = True

    def _restore(self) -> None:
        embeddings = np.load(self._npy_path)
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        version = meta.get("version", 1)
        if version != INDEX_VERSION:
            # Embedding recipe changed under us — force a full re-embed by
            # refusing the stale index (the caller resets chunks/mtimes).
            raise ValueError(f"Index version {version} != {INDEX_VERSION}; rebuilding")
        chunk_meta = meta.get("chunks", [])
        if len(chunk_meta) != embeddings.shape[0]:
            raise ValueError(
                f"Index inconsistent: {embeddings.shape[0]} embeddings vs {len(chunk_meta)} chunks"
            )
        self._chunks = [
            Chunk(
                file=m["file"],
                line=m["line"],
                text=m["text"],
                embedding=embeddings[i],
            )
            for i, m in enumerate(chunk_meta)
        ]
        self._mtimes = meta.get("mtimes", {})

    def _save(self) -> None:
        if not self._chunks:
            # Clean up the on-disk files if we've emptied the index.
            for p in (self._npy_path, self._meta_path):
                if p.exists():
                    p.unlink()
            return
        embeddings = np.stack([c.embedding for c in self._chunks]).astype(np.float32)
        chunk_meta = [{"file": c.file, "line": c.line, "text": c.text} for c in self._chunks]
        self._npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._npy_path, embeddings)
        self._meta_path.write_text(
            json.dumps(
                {
                    "version": INDEX_VERSION,
                    "mtimes": self._mtimes,
                    "chunks": chunk_meta,
                }
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self._notes_root).as_posix()

    def _embed(self, texts: List[str]) -> np.ndarray:
        # `normalize_embeddings=True` so search can use plain dot product.
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _build_chunks(self, path: Path) -> List[Chunk]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.error(f"Failed to read {path} for indexing: {e}")
            return []
        raw_chunks = _split_chunks(content)
        # Strip markdown markers — what we embed should match the text the
        # SLM/TTS will eventually surface, otherwise spoken output looks
        # nothing like what got matched.
        cleaned: List[Tuple[int, str]] = []
        for line, text in raw_chunks:
            spoken = self._spoken_filter(text).strip()
            if len(spoken) >= MIN_CHUNK_CHARS:
                cleaned.append((line, spoken))
        if not cleaned:
            return []
        # Prepend the note title to the *embedded* text (not the stored/spoken
        # text). A body paragraph that never repeats the note's subject still
        # embeds near a topical query ("Sydney to Perth route") because the
        # title rides along — a big lift for "what does my note about X say".
        title = path.stem.replace("-", " ").strip()
        embed_inputs = [f"{title}: {text}" if title else text for _, text in cleaned]
        embeddings = self._embed(embed_inputs)
        rel = self._rel(path)
        return [
            Chunk(file=rel, line=line, text=text, embedding=embeddings[i])
            for i, (line, text) in enumerate(cleaned)
        ]

    def index_file(self, path: Path) -> None:
        """Drop+rebuild this file's chunks. Called from write hooks."""
        with self._lock:
            self._ensure_loaded()
            rel = self._rel(path)
            if not path.exists():
                # File was deleted — strip its chunks from the index.
                self._chunks = [c for c in self._chunks if c.file != rel]
                self._mtimes.pop(rel, None)
                self._save()
                return
            self._chunks = [c for c in self._chunks if c.file != rel]
            self._chunks.extend(self._build_chunks(path))
            self._mtimes[rel] = path.stat().st_mtime
            self._save()

    def scan(self) -> None:
        """Walk the notes folder, embed anything new or mtime-stale.

        Cheap when nothing changed (just stats files). Runs on every
        `search()` so a user editing notes outside the assistant still
        gets fresh results next query.
        """
        with self._lock:
            self._ensure_loaded()
            seen = set()
            dirty = False
            for path in sorted(self._notes_root.rglob("*.md")):
                rel = self._rel(path)
                seen.add(rel)
                mtime = path.stat().st_mtime
                if abs(self._mtimes.get(rel, 0.0) - mtime) < 1e-3:
                    continue
                self._chunks = [c for c in self._chunks if c.file != rel]
                self._chunks.extend(self._build_chunks(path))
                self._mtimes[rel] = mtime
                dirty = True
            # Drop files that vanished from disk.
            for rel in list(self._mtimes):
                if rel not in seen:
                    self._chunks = [c for c in self._chunks if c.file != rel]
                    self._mtimes.pop(rel, None)
                    dirty = True
            if dirty:
                logger.info(
                    f"Notes index updated: {len(self._chunks)} chunks across "
                    f"{len(self._mtimes)} files"
                )
                self._save()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[float, Chunk]]:
        """Return top-k (score, chunk) pairs for the query, scanning first."""
        self.scan()
        with self._lock:
            if not self._chunks:
                return []
            query_emb = self._embed([QUERY_INSTRUCTION + query])[0]
            matrix = np.stack([c.embedding for c in self._chunks])
            scores = matrix @ query_emb
            top_idx = np.argsort(-scores)[:k]
            return [(float(scores[i]), self._chunks[i]) for i in top_idx]
