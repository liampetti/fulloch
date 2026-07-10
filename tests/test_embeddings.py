"""Tests for core/embeddings.py's shared embedding-model singleton.

The real model (BAAI/bge-small-en-v1.5) is too heavy for a unit test, so
`get_model()`'s deferred `torch`/`sentence_transformers` imports are stubbed
via sys.modules where the loading path itself is under test.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    import core.embeddings as embeddings

    embeddings._model = None
    yield
    embeddings._model = None


class _FakeSentenceTransformer:
    calls = 0

    def __init__(self, name, device=None):
        _FakeSentenceTransformer.calls += 1
        self.name = name
        self.device = device

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return np.ones((len(texts), 4), dtype=np.float32)


def _stub_heavy_deps(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    fake_st_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)


def test_get_model_loads_once_and_caches(monkeypatch):
    import core.embeddings as embeddings

    _stub_heavy_deps(monkeypatch)
    _FakeSentenceTransformer.calls = 0

    first = embeddings.get_model()
    second = embeddings.get_model()
    assert first is second
    assert _FakeSentenceTransformer.calls == 1


def test_embed_applies_query_instruction_only_when_requested(monkeypatch):
    import core.embeddings as embeddings

    seen = {}

    class _RecordingEncoder(_FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            seen["texts"] = list(texts)
            return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(embeddings, "_model", _RecordingEncoder("x"))

    embeddings.embed(["hello"], query=False)
    assert seen["texts"] == ["hello"]

    embeddings.embed(["hello"], query=True)
    assert seen["texts"] == [embeddings.QUERY_INSTRUCTION + "hello"]


def test_embed_returns_normalized_float32_array():
    import core.embeddings as embeddings

    embeddings._model = _FakeSentenceTransformer("x")
    out = embeddings.embed(["a", "b"])
    assert out.dtype == np.float32
    assert out.shape == (2, 4)
