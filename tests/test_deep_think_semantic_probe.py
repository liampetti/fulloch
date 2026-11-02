"""The semantic probe proposes pairs only and keeps original fixtures intact."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.probe_deep_think_semantics import adversarial_scores, candidate_pairs, probe_fixture


def test_similar_records_remain_separate_candidates_not_merged():
    ids = ["price-42", "price-425", "same-source", "unrelated"]
    sources = ["source-a", "source-b", "source-a", "source-c"]
    vectors = np.array([[1, 0], [0.99, 0.01], [1, 0], [0, 1]], dtype=np.float32)
    result = candidate_pairs(ids, sources, vectors, threshold=0.9)
    pairs = {(row["left"], row["right"]) for row in result["top_candidates"]}
    assert ("price-42", "price-425") in pairs  # Similarity does not validate the values.
    assert ("price-42", "same-source") not in pairs
    assert ids == ["price-42", "price-425", "same-source", "unrelated"]


@pytest.mark.parametrize("vectors", [np.array([[float("nan"), 0]]), np.array([1, 2])])
def test_probe_rejects_invalid_vectors(vectors):
    with pytest.raises(ValueError):
        candidate_pairs(["one"], ["source"], vectors)


def test_probe_fixture_is_read_only_and_singletons_are_not_pairs():
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    path = Path(__file__).parent / "deep-think/synthetic-finance.evidence.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = probe_fixture(path, Encoder())
    assert result["embedded_cards"] == 1
    assert result["candidate_count"] == 0
    assert result["question_top_cards"]
    assert all(row["card_id"] == "artifact-001-finance-1" for row in result["question_top_cards"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_adversarial_examples_do_not_become_equivalence_verdicts():
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    results = adversarial_scores(Encoder())
    assert {row["domain"] for row in results} == {"finance", "travel", "home_assistant"}
    assert all(row["score"] == 1.0 and row["left"] != row["right"] for row in results)
