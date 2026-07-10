"""Shared sentence-embedding model singleton.

Loading a `SentenceTransformer` costs real startup time and ~130MB
resident memory, so every feature that wants semantic similarity (notes
search, Spotify mood/genre playlist matching, ...) shares this one model
instance instead of each loading its own copy. Deferred import — nothing
here pays the `sentence-transformers` cost until `get_model()`/`embed()`
is actually called.
"""

from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# 384-dim, retrieval-tuned, English-focused. Outperforms MiniLM on most
# retrieval benchmarks for ~40MB extra download — picked for notes search,
# reused wherever else short-text semantic similarity is useful.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE-v1.5 is trained to prepend this instruction to the *query* only
# (never the documents/candidates being searched against). Measurably
# lifts recall for short topical queries.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_lock = threading.Lock()
_model = None


def get_model():
    """Return the shared SentenceTransformer, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {EMBED_MODEL_NAME} on {device}...")
        _model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        return _model


def embed(texts: List[str], query: bool = False) -> np.ndarray:
    """Embed a batch of texts, L2-normalized so cosine similarity is a dot product.

    Pass `query=True` only for the search query itself, not the candidates
    being searched against — see `QUERY_INSTRUCTION`.
    """
    model = get_model()
    if query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
