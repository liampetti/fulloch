"""Heuristics for deciding whether a *partial* transcript is safe to act on
before the speaker has actually finished (the early-endpoint / speculation path).

Two questions live here:

 - `is_complete(text)` — is this partial *syntactically closed*, or does it end on
   a word that invites continuation ("turn off the …", "play some music and …")?
   Used to gate deterministic intent matches: commit early only when the user has
   plausibly finished a clause.
- `SPECULATION_UNSAFE_INTENTS` — the `catchAll` intents whose cost-of-being-wrong
  is too high to ever fire on a partial (security, or a jarring/committing side
  effect). Everything else is reversible or read-only and may commit early.

Pure functions, no model / audio / config imports — trivially testable.
"""

import re
from typing import Iterable

# Intents not eligible for immediate speculation. A narrow subset may be
# admitted after a separately verified sustained pause (see below).
#   - ha_lock / ha_unlock : security — a wrong early unlock is unacceptable.
#   - play_song           : wrong media mid-sentence is jarring and not silent.
#   - start_countdown     : commits a timer that then has to be cancelled.
# Reversible (lights, colour, volume, toggle) and read-only (time, list timers,
 # summaries, skip/resume/pause) intents are absent — they're safe to commit on a
 # deterministic regex match.
SPECULATION_UNSAFE_INTENTS: frozenset[str] = frozenset(
    {
        "ha_lock",
        "ha_unlock",
        "ha_open_cover",
        "ha_close_cover",
        "ha_set_cover_position",
        "ha_set_climate",
        "ha_hassil_area",
        "play_song",
        "start_countdown",
        "cancel_timer",
        "send_satellite_message",
        "deep_think",
    }
)

# Only explicit reversible or read-only operations may run on a rolling
# transcript. New deterministic tools must not silently become speculative.
SPECULATION_SAFE_INTENTS: frozenset[str] = frozenset({
    "turn_on", "turn_off", "ha_set_brightness", "ha_adjust_light_brightness", "ha_set_color",
    "ha_volume_up", "ha_volume_down", "ha_volume_set", "ha_mute",
    "pause", "resume", "skip", "previous", "ha_stop_cover",
    "get_current_time", "get_weather_forecast", "get_exchange_rate",
    "get_timer_status", "get_entities_in_area_state",
})

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

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


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


def should_commit_provisional(prompt: str, catch_result, *, guarded_pause: bool = False) -> bool:
    """Decide whether a *provisional* (soft-endpointed) partial may commit the
    turn early, given the prompt and its `intent_catch.catchAll` result.

    `catch_result` is `catchAll`'s return: a string (no regex match → free-form,
    SLM-bound) or a dict (`{"actions": [...]}` / `{"reply": ...}`).

    - Higher-risk intents wait for the hard endpoint. A timer, supported area
      action or climate setting can commit after a verified guarded pause.
    - Regex-matched **safe** intent (lights, volume, time, …) → commit only if
      syntactically complete. This preserves quick commands but holds a broad
      regex match on a mid-sentence request such as "can you read".
    - No regex match or a reply-only result → wait for the hard endpoint. A soft
      snapshot can end at a natural clause boundary while the speaker continues,
      and free-form turns do not have a deterministic action to safely execute.

    Kept here (not in `intent_catch`) so the policy table lives next to
    `SPECULATION_UNSAFE_INTENTS`; the caller passes the `catchAll` result so this
    module stays import-light.
    """
    if isinstance(catch_result, dict):
        actions = catch_result.get("actions") or []
        if actions:
            # A single, explicitly scoped light-area operation is as reversible
            # as a named-light operation. Other area domains still need a pause.
            if len(actions) == 1 and actions[0].get("intent") == "ha_hassil_area":
                args = actions[0].get("args") or []
                if (isinstance(args, list) and len(args) == 5
                        and args[0] == "area" and isinstance(args[1], str) and args[1].strip()
                        and args[2] == "light" and args[3] in {"on", "off", "brightness", "brightness_step"}):
                    return getattr(catch_result, "complete_followup", False) or is_complete(prompt) or bool(
                        re.search(r"\b\w+\s+(?:on|off)\s*[.!?]*$", prompt, re.I)
                        and not re.search(r"\b(?:turn|switch)\s+(?:on|off)\s*[.!?]*$", prompt, re.I)
                    )
            intents = {a.get("intent") for a in actions}
            guarded_intents = {"ha_hassil_area", "ha_set_climate", "start_countdown"}
            if not intents <= (SPECULATION_SAFE_INTENTS | (guarded_intents if guarded_pause else set())):
                return False
            if intents & guarded_intents:
                if len(actions) != 1:
                    return False
                action = actions[0]
                args = action.get("args") or []
                if action.get("intent") == "ha_hassil_area":
                    if (not isinstance(args, list) or len(args) < 4
                            or args[2] not in {"light", "switch", "fan"}
                            or args[3] not in {"on", "off", "brightness"}):
                        return False
                elif action.get("intent") == "start_countdown":
                    # Bare numeric durations may be followed by a unit. HassIL
                    # normalizes fully parsed durations to explicit seconds.
                    if (not isinstance(args, list) or not args or not isinstance(args[0], str)
                            or not re.search(r"\b(?:seconds?|minutes?|hours?)\b", args[0], re.I)):
                        return False
            # HassIL may resolve a generic turn-on to a scene or script. Those
            # can perform arbitrary side effects and must wait for the endpoint.
            for action in actions:
                args = action.get("args") or []
                if action.get("intent") in {"turn_on", "turn_off"} and isinstance(args, list) and args:
                    if isinstance(args[0], str) and args[0].startswith(("scene.", "script.")):
                        return False
            if getattr(catch_result, "complete_followup", False) or is_complete(prompt):
                return True
            # In reversed English commands the particle closes the command,
            # rather than introducing a missing target ("turn on ...").
            return bool(
                len(actions) == 1 and actions[0].get("intent") in {"turn_on", "turn_off"}
                and re.search(r"\b\w+\s+(?:on|off)\s*[.!?]*$", prompt, re.I)
                and not re.search(r"\b(?:turn|switch)\s+(?:on|off)\s*[.!?]*$", prompt, re.I)
            )
    # Free-form and reply-only paths need the full utterance. In particular,
    # Parakeet can snapshot after a short pause in "can you read me ..." or
    # conversational feedback before the speaker finishes the thought.
    return False
