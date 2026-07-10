"""Tests for utils/reply_stream.py — A1a's incremental reply parser.

Feeds `ReplyStreamParser` arbitrary delta splits of the
`{"reply": "..."}` / `{"actions": [...]}` grammar (agent.gbnf) and asserts on
the clause fragments it emits. Deltas are split at pathological points on
purpose (mid-escape, mid-unicode-escape, one character at a time) since a
real SLM's token boundaries don't respect JSON syntax.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.reply_stream import ReplyStreamParser  # noqa: E402


def _feed_all(parser: ReplyStreamParser, deltas) -> list:
    out = []
    for d in deltas:
        out.extend(parser.feed(d))
    return out


class TestActionsBranch:
    def test_emits_nothing_and_marks_done(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"actions": [{"intent": "x", "args": []}]}'])
        assert out == []
        assert p.branch == "actions"
        assert p.done is True

    def test_actions_branch_detected_from_partial_prefix(self):
        p = ReplyStreamParser()
        assert p.feed('{"acti') == []
        assert p.branch is None  # not yet disambiguated
        assert p.feed('ons": []}') == []
        assert p.branch == "actions"


class TestReplyBranchBasic:
    def test_single_clause_flushed_on_closing_quote(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "Done"}'])
        assert out == ["Done"]
        assert p.branch == "reply"
        assert p.done is True

    def test_multi_clause_split_on_punctuation(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "Hello there. How are you?"}'])
        assert out == ["Hello there.", "How are you?"]

    def test_clause_emitted_as_soon_as_boundary_completes(self):
        # The first clause should surface the moment its trailing space
        # arrives — before the closing quote, and before the rest of the
        # reply has even streamed in.
        p = ReplyStreamParser()
        assert p.feed('{"reply": "First clause. ') == ["First clause."]
        assert p.feed('Second clause."}') == ["Second clause."]

    def test_one_character_at_a_time(self):
        p = ReplyStreamParser()
        text = '{"reply": "Hi there. Bye now."}'
        out = _feed_all(p, list(text))
        assert out == ["Hi there.", "Bye now."]

    def test_no_trailing_punctuation_still_flushed_on_close(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "no terminal punctuation"}'])
        assert out == ["no terminal punctuation"]


class TestWhitespaceTolerance:
    def test_space_after_brace(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{ "reply": "hi"}'])
        assert p.branch == "reply"
        assert out == ["hi"]

    def test_newline_and_indent_after_brace(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{\n  "reply": "hi"}'])
        assert p.branch == "reply"
        assert out == ["hi"]

    def test_extra_space_before_colon_and_quote(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply"  :   "hi"}'])
        assert out == ["hi"]


class TestEscapeSequences:
    def test_escaped_quote(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "she said \\"hi\\""}'])
        assert out == ['she said "hi"']

    def test_escaped_backslash(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "C:\\\\path"}'])
        assert out == ["C:\\path"]

    def test_escaped_newline_does_not_itself_split_clauses(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "line one\\nline two"}'])
        assert out == ["line one\nline two"]

    def test_unicode_escape(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "caf\\u00e9"}'])
        assert out == ["café"]

    def test_escape_sequence_split_across_deltas(self):
        # The backslash and its code character arrive in separate deltas —
        # a real SLM's token boundaries don't respect JSON escape pairs.
        # (":" is itself a clause delimiter, hence the two-fragment split.)
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "quote: \\', '"', ' end"}'])
        assert out == ["quote:", '" end']

    def test_unicode_escape_split_across_deltas(self):
        p = ReplyStreamParser()
        out = _feed_all(p, ['{"reply": "caf\\u00', "e9", '"}'])
        assert out == ["café"]

    def test_unicode_escape_split_one_hex_digit_at_a_time(self):
        p = ReplyStreamParser()
        deltas = ['{"reply": "x\\u'] + list("00e9") + ['"}']
        out = _feed_all(p, deltas)
        assert out == ["xé"]


class TestMalformedFallsBackCleanly:
    def test_unrecognisable_prefix_marks_unknown_and_done(self):
        p = ReplyStreamParser()
        # Never becomes a prefix of either {"actions" or {"reply".
        out = p.feed("not json at all, just prose way past the probe limit")
        assert out == []
        assert p.branch == "unknown"
        assert p.done is True

    def test_further_feeds_after_unknown_are_inert(self):
        p = ReplyStreamParser()
        p.feed("definitely not json " * 3)
        assert p.done is True
        assert p.feed('{"reply": "hi"}') == []
        assert p.branch == "unknown"


class TestFinalizeMidStream:
    def test_finalize_flushes_buffered_incomplete_clause(self):
        # Generation stopped (e.g. cancelled) before the closing quote.
        p = ReplyStreamParser()
        assert p.feed('{"reply": "still talking') == []
        out = p.finalize()
        assert out == ["still talking"]
        assert p.done is True

    def test_finalize_after_a_complete_clause_only_flushes_the_tail(self):
        p = ReplyStreamParser()
        assert p.feed('{"reply": "Done. Still going') == ["Done."]
        assert p.finalize() == ["Still going"]

    def test_finalize_on_actions_branch_is_a_noop(self):
        p = ReplyStreamParser()
        p.feed('{"actions": [')
        assert p.finalize() == []

    def test_finalize_before_branch_resolved_is_a_noop(self):
        p = ReplyStreamParser()
        p.feed('{"re')
        assert p.finalize() == []

    def test_finalize_is_idempotent(self):
        p = ReplyStreamParser()
        p.feed('{"reply": "hi there')
        assert p.finalize() == ["hi there"]
        assert p.finalize() == []

    def test_feed_after_done_is_inert(self):
        p = ReplyStreamParser()
        _feed_all(p, ['{"reply": "Done."}'])
        assert p.done is True
        assert p.feed(" more text") == []


class TestNoSinkNoIoPureFunction:
    def test_empty_delta_is_a_noop(self):
        p = ReplyStreamParser()
        assert p.feed("") == []
        assert p.branch is None

    def test_two_independent_parsers_do_not_share_state(self):
        p1, p2 = ReplyStreamParser(), ReplyStreamParser()
        p1.feed('{"reply": "hello')
        assert p2.branch is None
        assert p2._pending == ""
