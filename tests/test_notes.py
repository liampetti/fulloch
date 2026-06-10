"""Tests for the markdown notes tool module.

The notes module reads its base directory from config at import time, so the
fixture monkeypatches `notes.NOTES_DIR` (and `notes.DAILY_SUBDIR`) onto a
fresh tmp_path per test rather than re-importing the module.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import notes  # noqa: E402


class _StubHit:
    """Minimal stand-in for a NotesIndex Chunk in semantic-search results."""
    def __init__(self, file):
        self.file = file


class _StubIndex:
    """Fake embedding index: returns whatever (score, hit) pairs it's given."""
    def __init__(self, results=None):
        self._results = results or []

    def search(self, query, k=3):
        return self._results[:k]


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    """Point the notes module at a fresh tmp directory.

    `_after_write` and `_get_index` are stubbed so these tests don't trigger
    the real sentence-transformers model load — the indexing hook is exercised
    by test_notes_index.py. The default stub index returns no hits; tests that
    exercise the semantic fallback re-patch `_get_index` with seeded results.
    """
    base = tmp_path / "notes"
    base.mkdir()
    monkeypatch.setattr(notes, "NOTES_DIR", base)
    monkeypatch.setattr(notes, "DAILY_SUBDIR", "daily")
    monkeypatch.setattr(notes, "_after_write", lambda _path: None)
    monkeypatch.setattr(notes, "_get_index", lambda: _StubIndex())
    (base / "daily").mkdir()
    return base


class TestSlugify:
    def test_lowercases_and_hyphenates(self):
        assert notes._slugify("My Boiler Note") == "my-boiler-note"

    def test_strips_punctuation(self):
        assert notes._slugify("Shopping list!! (urgent)") == "shopping-list-urgent"

    def test_collapses_whitespace(self):
        assert notes._slugify("  multiple   spaces  ") == "multiple-spaces"

    def test_empty_falls_back(self):
        assert notes._slugify("") == "note"
        assert notes._slugify("!!!") == "note"


class TestWriteRead:
    def test_write_creates_file_with_title_header(self, notes_dir):
        result = notes.write_note("Shopping", "buy milk")
        assert "Saved" in result
        path = notes_dir / "shopping.md"
        assert path.exists()
        body = path.read_text()
        assert body.startswith("# Shopping")
        assert "buy milk" in body

    def test_write_appends_when_note_exists(self, notes_dir):
        notes.write_note("Shopping", "buy milk")
        result = notes.write_note("Shopping", "buy bread")
        assert "Added" in result
        # Both the original and the new content are present
        body = (notes_dir / "shopping.md").read_text()
        assert "buy milk" in body
        assert "buy bread" in body

    def test_read_finds_by_exact_title(self, notes_dir):
        notes.write_note("Boiler", "Vaillant ecoTEC")
        result = notes.read_note("Boiler")
        assert "Vaillant ecoTEC" in result
        # Markdown header is stripped for spoken output
        assert "#" not in result

    def test_read_fuzzy_match_on_substring(self, notes_dir):
        notes.write_note("Vaillant Boiler Notes", "service due July")
        result = notes.read_note("boiler")
        assert "service due July" in result

    def test_read_missing_note(self, notes_dir):
        result = notes.read_note("nonexistent")
        assert "couldn't find" in result.lower()

    def test_read_falls_back_to_semantic_on_topic_query(self, notes_dir, monkeypatch):
        # Title shares no literal overlap with the query, so _find_note misses
        # and the semantic index resolves it instead.
        notes.write_note(
            "climate-living-advice-australia",
            "Tasmania is the top pick for future climate resilience.",
        )
        monkeypatch.setattr(
            notes, "_get_index",
            lambda: _StubIndex([(0.62, _StubHit("climate-living-advice-australia.md"))]),
        )
        result = notes.read_note("climate change in Australia")
        assert "Tasmania" in result

    def test_semantic_fallback_respects_score_threshold(self, notes_dir, monkeypatch):
        notes.write_note("some-note", "body text here")
        below = notes.SEMANTIC_MIN_SCORE - 0.05
        monkeypatch.setattr(
            notes, "_get_index",
            lambda: _StubIndex([(below, _StubHit("some-note.md"))]),
        )
        result = notes.read_note("totally unrelated topic")
        assert "couldn't find" in result.lower()


class TestAppend:
    def test_append_adds_to_existing_note(self, notes_dir):
        notes.write_note("Shopping", "buy milk")
        result = notes.append_to_note("Shopping", "buy bread")
        assert "Added" in result
        body = (notes_dir / "shopping.md").read_text()
        assert "buy milk" in body
        assert "buy bread" in body

    def test_append_refuses_when_not_found(self, notes_dir):
        result = notes.append_to_note("ghost", "anything")
        assert "couldn't find" in result.lower()

    def test_append_falls_back_to_semantic_on_topic_query(self, notes_dir, monkeypatch):
        # Title shares no literal overlap with the query, so _find_note misses
        # and the semantic index resolves the target note to append to.
        notes.write_note(
            "climate-living-advice-australia",
            "Tasmania is the top pick for future climate resilience.",
        )
        monkeypatch.setattr(
            notes, "_get_index",
            lambda: _StubIndex([(0.62, _StubHit("climate-living-advice-australia.md"))]),
        )
        result = notes.append_to_note("climate change in Australia", "Darwin is tropical.")
        assert "Added" in result
        body = (notes_dir / "climate-living-advice-australia.md").read_text()
        assert "Darwin is tropical." in body

    def test_append_to_today_creates_dated_file(self, notes_dir):
        result = notes.append_to_today("Took the bins out")
        assert "Logged" in result
        daily_files = list((notes_dir / "daily").glob("*.md"))
        assert len(daily_files) == 1
        body = daily_files[0].read_text()
        assert "Took the bins out" in body
        # Header includes the date
        assert body.startswith("#")

    def test_append_to_today_reuses_existing_file(self, notes_dir):
        notes.append_to_today("entry one")
        notes.append_to_today("entry two")
        daily_files = list((notes_dir / "daily").glob("*.md"))
        assert len(daily_files) == 1
        body = daily_files[0].read_text()
        assert "entry one" in body
        assert "entry two" in body

    def test_read_today_reads_back_logged_entries(self, notes_dir):
        notes.append_to_today("Took the bins out")
        notes.append_to_today("Called the plumber")
        result = notes.read_today()
        assert "Took the bins out" in result
        assert "Called the plumber" in result
        # Markdown markers stripped for TTS — no bullet/hash characters.
        assert "#" not in result and "- " not in result

    def test_read_today_when_nothing_logged(self, notes_dir):
        result = notes.read_today()
        assert "don't have any notes logged today" in result.lower()

    def test_read_today_ignores_non_date_arg_and_uses_today(self, notes_dir):
        """The SLM may pass the literal word 'today' — fall back to today's file."""
        notes.append_to_today("real entry")
        result = notes.read_today("today")
        assert "real entry" in result

    def test_read_today_reads_specific_past_date(self, notes_dir):
        (notes_dir / "daily" / "2026-06-08.md").write_text(
            "# Monday 08 June 2026\n\n- 09:00 yesterday entry\n"
        )
        result = notes.read_today("2026-06-08")
        assert "yesterday entry" in result
        assert "on 2026-06-08" in result

    def test_read_today_rejects_path_traversal_date(self, notes_dir):
        """A non-date arg can't escape the daily folder; it resolves to today."""
        result = notes.read_today("../../facts")
        assert "don't have any notes logged today" in result.lower()


class TestSearch:
    def test_search_finds_matches_across_notes(self, notes_dir):
        notes.write_note("Boiler", "Vaillant ecoTEC, last serviced March")
        notes.write_note("Shopping", "milk, bread, eggs")
        result = notes.search_notes("vaillant")
        assert "Vaillant" in result or "vaillant" in result.lower()
        assert "boiler" in result.lower()

    def test_search_case_insensitive(self, notes_dir):
        notes.write_note("Recipes", "Pasta carbonara")
        result = notes.search_notes("PASTA")
        assert "carbonara" in result.lower()

    def test_search_no_match(self, notes_dir):
        notes.write_note("Boiler", "Vaillant")
        result = notes.search_notes("submarine")
        assert "didn't find" in result.lower()

    def test_search_empty_query(self, notes_dir):
        result = notes.search_notes("")
        assert "something to search" in result.lower()


class TestRememberFact:
    def test_remembers_with_date_stamp(self, notes_dir):
        result = notes.remember_fact("The boiler is a Vaillant ecoTEC")
        assert "remember" in result.lower()
        facts = (notes_dir / "facts.md").read_text()
        assert "Vaillant ecoTEC" in facts
        # Date stamp in [YYYY-MM-DD] form
        import re
        assert re.search(r'\[\d{4}-\d{2}-\d{2}\]', facts)

    def test_appends_multiple_facts(self, notes_dir):
        notes.remember_fact("fact one")
        notes.remember_fact("fact two")
        facts = (notes_dir / "facts.md").read_text()
        assert "fact one" in facts
        assert "fact two" in facts


class TestList:
    def test_list_empty(self, notes_dir):
        result = notes.list_notes()
        assert "don't have any" in result.lower()

    def test_list_returns_titles(self, notes_dir):
        notes.write_note("Boiler", "x")
        notes.write_note("Shopping List", "y")
        result = notes.list_notes()
        assert "boiler" in result.lower()
        assert "shopping list" in result.lower()


class TestRecallFacts:
    def test_returns_empty_when_facts_file_missing(self, notes_dir):
        assert notes.recall_facts() == ""

    def test_returns_empty_when_only_header_present(self, notes_dir):
        (notes_dir / "facts.md").write_text("# Long-term facts\n\n")
        assert notes.recall_facts() == ""

    def test_returns_known_facts_block_when_populated(self, notes_dir):
        notes.remember_fact("The boiler is a Vaillant ecoTEC")
        notes.remember_fact("Wife is allergic to peanuts")
        block = notes.recall_facts()
        assert block.startswith("## Known facts about the user")
        assert "Vaillant ecoTEC" in block
        assert "peanuts" in block
        # Header is stripped — we already supply our own framing
        assert "# Long-term facts" not in block

    def test_skips_blank_lines(self, notes_dir):
        (notes_dir / "facts.md").write_text(
            "# Long-term facts\n\n\n- [2026-05-20] fact one\n\n\n"
        )
        block = notes.recall_facts()
        assert "fact one" in block
        # No empty lines between the block header and the bullet
        assert "\n\n\n" not in block


class TestFactsCrud:
    def test_list_facts_empty_when_no_file(self, notes_dir):
        assert notes.list_facts() == []

    def test_list_facts_parses_each_line(self, notes_dir):
        notes.remember_fact("fact one")
        notes.remember_fact("fact two")
        facts = notes.list_facts()
        assert len(facts) == 2
        assert facts[0]["index"] == 0
        assert facts[0]["text"] == "fact one"
        assert facts[1]["index"] == 1
        assert facts[1]["text"] == "fact two"
        # Date stamps preserved
        import re
        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', facts[0]["date"])

    def test_list_facts_skips_non_bullet_lines(self, notes_dir):
        (notes_dir / "facts.md").write_text(
            "# Long-term facts\n\nsome stray prose\n- [2026-05-20] real fact\n"
        )
        facts = notes.list_facts()
        assert len(facts) == 1
        assert facts[0]["text"] == "real fact"

    def test_update_fact_replaces_text_keeps_date(self, notes_dir):
        notes.remember_fact("original text")
        facts = notes.list_facts()
        original_date = facts[0]["date"]
        assert notes.update_fact(0, "corrected text") is True
        facts = notes.list_facts()
        assert facts[0]["text"] == "corrected text"
        assert facts[0]["date"] == original_date

    def test_update_fact_rejects_empty(self, notes_dir):
        notes.remember_fact("a fact")
        assert notes.update_fact(0, "   ") is False
        assert notes.list_facts()[0]["text"] == "a fact"

    def test_update_fact_returns_false_on_bad_index(self, notes_dir):
        notes.remember_fact("only fact")
        assert notes.update_fact(5, "anything") is False

    def test_update_fact_returns_false_when_no_file(self, notes_dir):
        assert notes.update_fact(0, "anything") is False

    def test_delete_fact_removes_line(self, notes_dir):
        notes.remember_fact("keep me")
        notes.remember_fact("delete me")
        notes.remember_fact("also keep")
        assert notes.delete_fact(1) is True
        remaining = [f["text"] for f in notes.list_facts()]
        assert remaining == ["keep me", "also keep"]

    def test_delete_fact_bad_index(self, notes_dir):
        notes.remember_fact("only fact")
        assert notes.delete_fact(7) is False
        assert len(notes.list_facts()) == 1

    def test_delete_fact_preserves_header(self, notes_dir):
        notes.remember_fact("fact one")
        notes.delete_fact(0)
        body = (notes_dir / "facts.md").read_text()
        # Header line still present after a delete
        assert body.startswith("# Long-term facts")

    def test_atomic_write_leaves_no_tmp_files(self, notes_dir):
        notes.remember_fact("a")
        notes.remember_fact("b")
        notes.update_fact(0, "a-edited")
        notes.delete_fact(1)
        tmps = list(notes_dir.glob(".facts-*.tmp"))
        assert tmps == []


class TestWarmIndex:
    def test_calls_scan(self, notes_dir, monkeypatch):
        class FakeIndex:
            def __init__(self):
                self.scanned = False

            def scan(self):
                self.scanned = True

        fake = FakeIndex()
        monkeypatch.setattr(notes, "_get_index", lambda: fake)
        assert notes.warm_index() is True
        assert fake.scanned is True

    def test_swallows_scan_errors(self, notes_dir, monkeypatch):
        class BrokenIndex:
            def scan(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(notes, "_get_index", lambda: BrokenIndex())
        # Should log and return False, not propagate.
        assert notes.warm_index() is False


class TestSpokenStripping:
    def test_strips_headers_bullets_emphasis(self):
        md = "# Title\n\n- item one\n- *important* thing\n\n`code`"
        out = notes._to_spoken(md)
        assert "#" not in out
        assert "*" not in out
        assert "`" not in out
        assert "Title" in out
        assert "item one" in out
        assert "important thing" in out


class TestNoteFilesCrud:
    def test_list_note_files_returns_name_and_title(self, notes_dir):
        notes.write_note("Boiler", "x")
        notes.write_note("Shopping List", "y")
        files = notes.list_note_files()
        names = {f["name"] for f in files}
        titles = {f["title"] for f in files}
        assert names == {"boiler", "shopping-list"}
        assert titles == {"boiler", "shopping list"}

    def test_list_note_files_excludes_facts(self, notes_dir):
        notes.write_note("Boiler", "x")
        notes.remember_fact("a secret fact")
        names = {f["name"] for f in notes.list_note_files()}
        assert "facts" not in names
        assert names == {"boiler"}

    def test_read_note_file_returns_raw_markdown(self, notes_dir):
        notes.write_note("Boiler", "Vaillant ecoTEC")
        raw = notes.read_note_file("boiler")
        assert raw.startswith("# Boiler")
        assert "Vaillant ecoTEC" in raw

    def test_read_note_file_missing_returns_none(self, notes_dir):
        assert notes.read_note_file("ghost") is None

    def test_read_note_file_refuses_facts(self, notes_dir):
        notes.remember_fact("a secret fact")
        assert notes.read_note_file("facts") is None

    def test_save_note_file_overwrites_existing(self, notes_dir):
        notes.write_note("Boiler", "old")
        assert notes.save_note_file("boiler", "# Boiler\n\nbrand new body\n") is True
        raw = (notes_dir / "boiler.md").read_text()
        assert "brand new body" in raw
        assert "old" not in raw

    def test_save_note_file_appends_trailing_newline(self, notes_dir):
        notes.write_note("Boiler", "x")
        notes.save_note_file("boiler", "no trailing newline")
        assert (notes_dir / "boiler.md").read_text().endswith("\n")

    def test_save_note_file_refuses_missing(self, notes_dir):
        assert notes.save_note_file("ghost", "anything") is False
        assert not (notes_dir / "ghost.md").exists()

    def test_save_note_file_refuses_facts(self, notes_dir):
        notes.remember_fact("a secret fact")
        assert notes.save_note_file("facts", "wiped") is False
        assert "secret fact" in (notes_dir / "facts.md").read_text()

    def test_save_note_file_rejects_path_traversal(self, notes_dir, tmp_path):
        outside = tmp_path / "outside.md"
        outside.write_text("# Outside\n")
        assert notes.save_note_file("../outside", "hacked") is False
        assert outside.read_text() == "# Outside\n"

    def test_save_note_file_leaves_no_tmp_files(self, notes_dir):
        notes.write_note("Boiler", "x")
        notes.save_note_file("boiler", "edited")
        assert list(notes_dir.glob(".note-*.tmp")) == []

    def test_round_trips_a_daily_note(self, notes_dir):
        notes.append_to_today("logged something")
        files = notes.list_note_files()
        daily = next(f for f in files if f["name"].startswith("daily/"))
        raw = notes.read_note_file(daily["name"])
        assert "logged something" in raw
        assert notes.save_note_file(daily["name"], raw + "- another line\n") is True
        assert "another line" in notes.read_note_file(daily["name"])
