"""Cross-chapter passage dedup before synthesis."""

from core.thinking_worker import (
    _bound_cited_summary,
    _dedup_chapter_passages,
    _reconcile_summary_with_findings,
)


def test_identical_passage_dropped_from_later_chapter():
    seen: set[str] = set()
    first = "The race is at the Circuit on Sunday. <!-- cards: c1 -->"
    second = "The race is at the Circuit on Sunday. <!-- cards: c2 -->"
    # First chapter keeps the passage and registers it.
    kept_first = _dedup_chapter_passages(seen, first)
    assert kept_first == first
    # Second chapter's identical prose (different marker) is dropped.
    kept_second = _dedup_chapter_passages(seen, second)
    assert kept_second == ""


def test_distinct_passage_kept():
    seen: set[str] = set()
    _dedup_chapter_passages(seen, "The race is on Sunday. <!-- cards: c1 -->")
    kept = _dedup_chapter_passages(seen, "The weather is dry. <!-- cards: c2 -->")
    assert kept == "The weather is dry. <!-- cards: c2 -->"


def test_same_prose_different_markers_dedups():
    seen: set[str] = set()
    _dedup_chapter_passages(seen, "Acme was quoted at $41.17. <!-- cards: a,b -->")
    kept = _dedup_chapter_passages(seen, "Acme was quoted at $41.17. <!-- cards: c -->")
    assert kept == ""


def test_near_duplicate_not_dropped():
    seen: set[str] = set()
    _dedup_chapter_passages(seen, "The race is on Sunday morning. <!-- cards: c1 -->")
    # One word differs: verbatim-only dedup keeps it.
    kept = _dedup_chapter_passages(seen, "The race is on Monday morning. <!-- cards: c2 -->")
    assert kept == "The race is on Monday morning. <!-- cards: c2 -->"


def test_case_and_whitespace_insensitive():
    seen: set[str] = set()
    _dedup_chapter_passages(seen, "The  race is   on Sunday. <!-- cards: c1 -->")
    kept = _dedup_chapter_passages(seen, "the race is on Sunday. <!-- cards: c2 -->")
    assert kept == ""


def test_mixed_body_drops_only_seen_passage():
    seen: set[str] = set()
    _dedup_chapter_passages(seen, "Known fact one. <!-- cards: c1 -->")
    body = (
        "Known fact one. <!-- cards: c1 -->\n\n"
        "A fresh finding with new detail. <!-- cards: c2 -->"
    )
    kept = _dedup_chapter_passages(seen, body)
    assert "Known fact one" not in kept
    assert "A fresh finding with new detail. <!-- cards: c2 -->" in kept


def test_empty_seen_keeps_everything():
    seen: set[str] = set()
    body = "First. <!-- cards: a -->\n\nSecond. <!-- cards: b -->"
    kept = _dedup_chapter_passages(seen, body)
    assert kept == body


def test_long_summary_keeps_whole_cited_passages_only():
    passages = [f"Finding {number}. <!-- cards: c{number} -->" for number in range(6)]
    summary = "\n\n".join(passages)
    bounded = _bound_cited_summary(summary)
    assert bounded == "\n\n".join(passages[:4])
    assert _bound_cited_summary(passages[0]) == passages[0]
    assert _bound_cited_summary("No cited passages.") == "No cited passages."


def test_reconciled_summary_keeps_original_support_marker():
    requirements = [{"importance": "high", "state": "resolved", "evidence_ids": ["card-1"]}]
    cards = [{"id": "card-1", "statement": "The quote was USD 41.17."}]
    result = _reconcile_summary_with_findings(
        "The USD 41.17 quote is not verified.", requirements, cards, {"card-1"},
    )
    assert result == "The quote was USD 41.17. <!-- cards: card-1 -->"
