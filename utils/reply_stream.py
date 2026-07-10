"""A1a: incremental parser over SLM text deltas for the `{"reply": "..."}` /
`{"actions": [...]}` grammar (`data/models/grammars/agent.gbnf`).

`ReplyStreamParser` consumes deltas as `core.slm._call_slm` streams them and,
once the emission has committed to the `reply` branch, incrementally
JSON-unescapes the string value and emits clause-level fragments (using the
same split as the overlap-synthesis TTS backends — see
`core.text_utils.split_clauses`) as soon as each one is complete. The
`actions` branch — and anything that doesn't look like either — emits
nothing; the caller proceeds exactly as it does today (wait for the full
emission).

Pure function of the deltas fed in: no I/O, no threading, no knowledge of
TTS/SLM internals — that wiring (A1b/A1c) consumes the fragments this module
produces. `agent.gbnf`'s `reply` branch is grammar-terminal (mutually
exclusive with `actions`, capped at one string field), so any fragment this
parser emits is safe to speak immediately.
"""

import re
from typing import Optional

from core.text_utils import CLAUSE_SPLIT

# Matches `{"reply"` or `{"actions"` at the start of the buffered prefix,
# tolerant of the grammar's `ws` rule (`data/models/grammars/agent.gbnf`:
# nothing, a single space, or a newline + up to 20 tabs/spaces) around the
# brace — and deliberately a little more permissive (`\s*`) so a remote,
# grammar-less backend's formatting still matches.
_PREFIX_RE = re.compile(r'^\s*\{\s*"(actions|reply)"')
# After the branch is known to be `reply`, matches through `: ws "` up to
# (not including) the string body's first character.
_TO_STRING_RE = re.compile(r'^\s*:\s*"')
# Longest either target key could plausibly take to disambiguate ({"actions"
# is 10 chars, plus a few chars of tolerated whitespace/formatting). Past
# this with no match, the emission isn't going to become well-formed —
# give up rather than buffer forever.
_MAX_PREFIX_PROBE = 24

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

# Internal states.
_PREFIX = "prefix"
_TO_STRING = "to_string"
_IN_STRING = "in_string"


class ReplyStreamParser:
    """Feed text deltas in; get back newly-complete clause fragments.

    `branch` is `None` until enough of the prefix has arrived to tell
    `{"reply"` from `{"actions"` apart (or `"unknown"` if the input can never
    become either — see `_MAX_PREFIX_PROBE`). `done` is True once no further
    fragments will ever be produced (branch resolved to `actions`/`unknown`,
    or the reply string's closing quote was reached, or `finalize()` was
    called).
    """

    def __init__(self) -> None:
        self.branch: Optional[str] = None
        self.done: bool = False
        self._state = _PREFIX
        self._buf = ""
        self._pending = ""  # unescaped reply text not yet flushed as a fragment
        # None, "backslash" (just saw `\`, next char picks the escape), or
        # ("unicode", partial_hex) mid `\uXXXX` (may span multiple feed() calls).
        self._escape_state = None

    def feed(self, delta: str) -> list:
        """Advance the parser with the next chunk of streamed text.

        Returns any clause fragments that became complete as a result — an
        empty list most of the time (still buffering the prefix, still mid
        clause, or the `actions` branch, which never emits).
        """
        if self.done or not delta:
            return []
        self._buf += delta
        out: list = []
        progressed = True
        while progressed:
            progressed = False
            if self._state == _PREFIX:
                progressed = self._advance_prefix()
            elif self._state == _TO_STRING:
                progressed = self._advance_to_string()
            elif self._state == _IN_STRING:
                consumed, fragments = self._consume_string_chars()
                out.extend(fragments)
                progressed = consumed
        return out

    def finalize(self) -> list:
        """Flush any buffered-but-incomplete text as a final fragment.

        For when the stream ends (or is cancelled) before the closing quote
        arrives — the caller still gets whatever was spoken so far instead of
        silently dropping it. No-op (returns `[]`) once already `done`, or if
        the branch never resolved to `reply`.
        """
        if self.done:
            return []
        self.done = True
        if self.branch != "reply" or not self._pending:
            return []
        out = [p.strip() for p in CLAUSE_SPLIT.split(self._pending) if p.strip()]
        self._pending = ""
        return out

    # --- internal state transitions -----------------------------------

    def _advance_prefix(self) -> bool:
        m = _PREFIX_RE.match(self._buf)
        if m:
            self.branch = m.group(1)
            self._buf = self._buf[m.end() :]
            if self.branch == "actions":
                self.done = True
                self._buf = ""
                return False
            self._state = _TO_STRING
            return True
        if len(self._buf) >= _MAX_PREFIX_PROBE:
            self.branch = "unknown"
            self.done = True
            self._buf = ""
        return False

    def _advance_to_string(self) -> bool:
        m = _TO_STRING_RE.match(self._buf)
        if m:
            self._buf = self._buf[m.end() :]
            self._state = _IN_STRING
            return True
        return False

    def _consume_string_chars(self):
        fragments: list = []
        n = len(self._buf)
        i = 0
        consumed_any = False
        while i < n:
            ch = self._buf[i]
            if self._escape_state == "backslash":
                if ch == "u":
                    self._escape_state = ("unicode", "")
                elif ch in _SIMPLE_ESCAPES:
                    self._pending += _SIMPLE_ESCAPES[ch]
                    self._escape_state = None
                else:
                    # Not valid per the grammar — pass the character through
                    # literally rather than dropping it.
                    self._pending += ch
                    self._escape_state = None
                i += 1
                consumed_any = True
                continue
            if isinstance(self._escape_state, tuple):
                hex_digits = self._escape_state[1] + ch
                i += 1
                consumed_any = True
                if len(hex_digits) == 4:
                    self._pending += chr(int(hex_digits, 16))
                    self._escape_state = None
                else:
                    self._escape_state = ("unicode", hex_digits)
                continue
            if ch == "\\":
                self._escape_state = "backslash"
                i += 1
                consumed_any = True
                continue
            if ch == '"':
                # Unescaped quote — end of the JSON string value.
                self._buf = self._buf[i + 1 :]
                self.done = True
                fragments.extend(self._flush(final=True))
                return True, fragments
            self._pending += ch
            i += 1
            consumed_any = True
        self._buf = self._buf[i:]
        fragments.extend(self._flush(final=False))
        return consumed_any, fragments

    def _flush(self, final: bool) -> list:
        parts = CLAUSE_SPLIT.split(self._pending)
        if final:
            out = [p.strip() for p in parts if p.strip()]
            self._pending = ""
            return out
        if len(parts) <= 1:
            return []
        *complete, tail = parts
        self._pending = tail
        return [p.strip() for p in complete if p.strip()]
