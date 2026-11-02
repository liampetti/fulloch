"""Offline, CPU-only semantic navigation probe; never changes evidence or reports.

Example: python -m scripts.probe_deep_think_semantics tests/deep-think/*.evidence.json
Scores are candidates for manual review, not evidence of equivalence or support.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core.embeddings import EMBED_MODEL_NAME, QUERY_INSTRUCTION
from core.thinking_fact_cards import report_card_ledger
from core.thinking_replay import load_replay


def candidate_pairs(ids, source_ids, vectors, *, threshold=0.85, limit=10):
    """Return high-scoring cross-source pairs without grouping or modifying cards."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(ids) != len(source_ids) or vectors.ndim != 2 or len(ids) != len(vectors):
        raise ValueError("IDs, source IDs and vectors must have the same length")
    if len(set(ids)) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Card IDs must be unique and vectors must be finite")
    if not -1 <= threshold <= 1 or limit < 1:
        raise ValueError("Invalid threshold or limit")
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] != 0)
    scores = normalized @ normalized.T
    pairs = []
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            if norms[left] == 0 or norms[right] == 0:
                continue
            if source_ids[left] and source_ids[left] == source_ids[right]:
                continue
            score = float(scores[left, right])
            if score >= threshold:
                pairs.append({"left": ids[left], "right": ids[right], "score": round(score, 4)})
    pairs.sort(key=lambda row: (-row["score"], row["left"], row["right"]))
    return {"candidate_count": len(pairs), "top_candidates": pairs[:limit]}


def probe_fixture(path, model, *, threshold=0.85, limit=10, max_chars=500):
    """Compare short retained cards and question navigation using a supplied encoder."""
    job = load_replay(path)
    cards = report_card_ledger(job.fact_cards)
    rows = []
    for card in cards:
        text = str(card.get("statement") or "").strip()
        if not text and card.get("fields"):
            text = json.dumps(card["fields"], ensure_ascii=False, sort_keys=True)
        if 10 <= len(text) <= max_chars:
            source_ids = card.get("source_ids") or []
            source = ((card.get("provenance") or {}).get("source_id") or card.get("source_url")
                      or (source_ids[0] if source_ids else ""))
            rows.append((card["id"], str(source), text))
    result = {"fixture": Path(path).name, "eligible_cards": len(cards),
              "embedded_cards": len(rows), "omitted_cards": len(cards) - len(rows),
              "max_card_chars": max_chars, "threshold": threshold}
    if not rows:
        return {**result, "candidate_count": 0, "top_candidates": [], "question_top_cards": []}
    vectors = model.encode([row[2] for row in rows], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    result.update(candidate_pairs([row[0] for row in rows], [row[1] for row in rows], vectors,
                                  threshold=threshold, limit=limit))
    questions = [(item.get("identifier"), item.get("question")) for item in job.evidence_requirements
                 if isinstance(item.get("question"), str) and item["question"].strip()]
    result["question_top_cards"] = []
    if questions:
        q_vectors = model.encode([QUERY_INSTRUCTION + question for _, question in questions],
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        for (identifier, _), scores in zip(questions, q_vectors @ vectors.T, strict=True):
            top = int(np.argmax(scores))
            result["question_top_cards"].append({
                "question_id": identifier, "card_id": rows[top][0], "score": round(float(scores[top]), 4),
            })
    return result


def adversarial_scores(model):
    """Compare fabricated similar statements with materially different values."""
    examples = {
        "finance": ("Example Corporation TEST quoted at USD 42.50 on 2026-10-29 09:59 UTC.",
                    "Example Corporation TEST quoted at USD 425.00 on 2026-10-29 09:59 UTC."),
        "travel": ("Return flight AAA to BBB outbound 17 January 2027, return 3 March 2027, EUR 700.",
                   "Return flight AAA to BBB outbound 17 January 2027, return 3 March 2027, EUR 7000."),
        "home_assistant": ("Living room temperature history, high 20.0 °C and low 18.5 °C.",
                           "Living room temperature history, high 18.5 °C and low 20.0 °C."),
    }
    rows = []
    for domain, (left, right) in examples.items():
        vectors = model.encode([left, right], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        rows.append({"domain": domain, "left": left, "right": right,
                     "score": round(float(vectors[0] @ vectors[1]), 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--adversarial", action="store_true", help="Include fabricated conflicting values.")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    # Avoid GPU contention and implicit downloads during a local experiment.
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu", local_files_only=True)
    results = [probe_fixture(path, model, threshold=args.threshold, limit=args.limit) for path in args.fixtures]
    output = {"model": EMBED_MODEL_NAME, "device": "cpu", "candidate_only": True, "results": results}
    if args.adversarial:
        output["synthetic_adversarial_pairs"] = adversarial_scores(model)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
