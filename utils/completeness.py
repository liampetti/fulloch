"""Heuristics for deciding whether a *partial* transcript is safe to act on
before the speaker has actually finished (the early-endpoint / speculation path).

Two questions live here:

- `is_complete(text)` — is this partial *syntactically closed*, or does it end on
  a word that invites continuation ("turn off the …", "play some music and …")?
  Used to gate the free-form (SLM-bound) path: commit early only when the user
  has plausibly finished a clause.
- `SPECULATION_UNSAFE_INTENTS` — the `catchAll` intents whose cost-of-being-wrong
  is too high to ever fire on a partial (security, or a jarring/committing side
  effect). Everything else is reversible or read-only and may commit early.

Pure functions, no model / audio / config imports — trivially testable.
"""

import re
from typing import Iterable

# `catchAll` intents we must NOT act on speculatively, even when the partial
# looks complete: acting on a misheard early partial is expensive or unsafe.
#   - ha_lock / ha_unlock : security — a wrong early unlock is unacceptable.
#   - play_song           : wrong media mid-sentence is jarring and not silent.
#   - start_countdown     : commits a timer that then has to be cancelled.
# Reversible (lights, colour, volume, toggle) and read-only (time, list timers,
# summaries, skip/resume/pause) intents are absent — they're safe to commit on a
# regex match. A free-form query (no regex match) is gated by `is_complete`.
SPECULATION_UNSAFE_INTENTS: frozenset[str] = frozenset(
    {
        "ha_lock",
        "ha_unlock",
        "play_song",
        "start_countdown",
    }
)

# Words that, when they END a partial utterance, signal the speaker is mid-clause
# and about to say more — so the partial is NOT yet complete. Conjunctions,
# prepositions, articles, possessives, and on/off/up/down particles (a trailing
# "turn off" is almost always followed by its target; the *complete* reversed
# form "turn the lights off" matches a `catchAll` regex and bypasses this gate).
# Fillers ("um", "uh", "so") and a trailing "to"/"and" are the strongest signals.
_CONTINUATION_WORDS: frozenset[str] = frozenset(
    {
        # conjunctions / connectives
        "and",
        "or",
        "but",
        "then",
        "plus",
        "also",
        "with",
        "without",
        # prepositions
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "by",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "as",
        "like",
        "than",
        # articles / determiners / possessives
        "the",
        "a",
        "an",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "some",
        "any",
        "this",
        "that",
        "these",
        "those",
        # particles that usually precede a target
        "up",
        "down",
        "off",
        # copulas / auxiliaries left dangling
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "will",
        "would",
        "could",
        "should",
        "can",
        "do",
        "does",
        "did",
        # fillers
        "um",
        "uh",
        "er",
        "hmm",
        "so",
        "well",
    }
)

# Request endings that need an object even though their final word is not a
# conventional conjunction/preposition. This stays deliberately small so
# concise commands such as "stop" and "what time" remain responsive.
_INCOMPLETE_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("can", "you"),
    ("could", "you"),
    ("would", "you"),
    ("read",),
    ("tell",),
    ("show",),
    ("bring",),
    ("find",),
    ("search",),
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def is_complete(text: str, continuation_words: Iterable[str] = _CONTINUATION_WORDS) -> bool:
    """True if `text` reads as a finished clause rather than a mid-utterance fragment.

    Conservative: an empty/blank string is *not* complete (nothing to act on),
    and a partial whose last word invites continuation ("...and", "...the",
    "...turn off") is held back so the full hard endpoint can capture the rest.
    Trailing punctuation is ignored, so "lights." and "lights" both judge on
    "lights".
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return False
    if words[-1] in set(continuation_words):
        return False
    return not any(
        len(words) >= len(suffix) and tuple(words[-len(suffix) :]) == suffix
        for suffix in _INCOMPLETE_SUFFIXES
    )


def should_commit_provisional(prompt: str, catch_result) -> bool:
    """Decide whether a *provisional* (soft-endpointed) partial may commit the
    turn early, given the prompt and its `intent_catch.catchAll` result.

    `catch_result` is `catchAll`'s return: a string (no regex match → free-form,
    SLM-bound) or a dict (`{"actions": [...]}` / `{"reply": ...}`).

    - Regex-matched **unsafe** intent (`SPECULATION_UNSAFE_INTENTS`) → never
      commit early; wait for the hard endpoint.
    - Regex-matched **safe** intent (lights, volume, time, …) → commit only if
      syntactically complete. This preserves quick commands but holds a broad
      regex match on a mid-sentence request such as "can you read".
    - No regex match (free-form) → commit only if syntactically complete.

    Kept here (not in `intent_catch`) so the policy table lives next to
    `SPECULATION_UNSAFE_INTENTS`; the caller passes the `catchAll` result so this
    module stays import-light.
    """
    if isinstance(catch_result, dict):
        actions = catch_result.get("actions") or []
        intents = {a.get("intent") for a in actions}
        if intents & SPECULATION_UNSAFE_INTENTS:
            return False
        return is_complete(prompt)
    # Free-form string → SLM. Only commit once the clause reads as finished.
    return is_complete(prompt)
