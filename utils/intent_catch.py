"""Regex fast-path for common voice intents (play / stop / time / timer).

Returns an agent emission dict `{"actions": [{"intent": ..., "args": ...}]}`
from `catchAll` when matched, or the original string unchanged. Individual
`extract_*` helpers remain importable for tests.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _match(label: str, pattern: re.Pattern, command: str):
    match = pattern.search(command)
    if match:
        logger.debug(f"Caught {label} match: {match}")
    return match


# "play" is an ordinary English word ("...games I can play with my daughter"),
# so this MUST anchor to the start of the command — an unanchored search grabs
# a mid-sentence "play" and routes a web-search request to play_song. Only an
# optional polite prefix may precede the verb ("please play the Beatles").
_PLAY_RE = re.compile(
    r'^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+|'
    r'i\s+want\s+you\s+to\s+|i\s+want\s+to\s+|i\s+wanna\s+|go\s+ahead\s+and\s+)*'
    r'play\s+(.+)$',
    re.IGNORECASE,
)
_STOP_RE = re.compile(r'^\s*(stop|pause|halt)\b', re.IGNORECASE)
_SKIP_RE = re.compile(r'^\s*skip\b', re.IGNORECASE)
_RESUME_RE = re.compile(r'^\s*resume\b', re.IGNORECASE)
# Leading context is fine ("do you know what time is it"), so the start stays
# unanchored — but the phrase must END the question (bar a few benign tail
# adverbs / trailing punctuation). Without the end-anchor, "what time is it
# GOING TO BE ON" matched the "what time is it" substring and answered with the
# current wall-clock time instead of letting the SLM resolve "it" (an event's
# broadcast time) from context. Same guard rejects "what's the time IN LONDON".
_TIME_QUERY_RE = re.compile(
    r"(?:what time is it|what'?s the time|what time it is)"
    r"(?:\s+(?:right\s+now|now|currently|today|please|exactly))?"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)
# Anchored like _PLAY_RE so a mid-sentence "set a timer for the eggs" mention
# can't hijack the turn.
_TIMER_RE = re.compile(
    r'^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)*'
    r'(?:start|set)\s+(?:a\s+)?(?:timer|time)\s+(?:for\s+)?(.+?)(?:\s+please)?$',
    re.IGNORECASE,
)
_LIST_TIMERS_RE = re.compile(
    r'^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)*'
    r'get\s+(?:a\s+)?(?:timers|timer|time)(?:\s+(.*?))?(?:\s+status)?$',
    re.IGNORECASE,
)


def extract_after_play(command: str) -> Optional[str]:
    m = _match("Play", _PLAY_RE, command)
    return m.group(1).strip() if m else None


def extract_stop(command: str) -> Optional[bool]:
    return True if _match("Stop", _STOP_RE, command) else None


def extract_skip(command: str) -> Optional[bool]:
    return True if _match("Skip", _SKIP_RE, command) else None


def extract_resume(command: str) -> Optional[bool]:
    return True if _match("Resume", _RESUME_RE, command) else None


def has_time_query(command: str) -> Optional[bool]:
    return True if _match("Time", _TIME_QUERY_RE, command) else None


def extract_timer(command: str) -> Optional[str]:
    m = _match("Timer", _TIMER_RE, command)
    return m.group(1).strip() if m else None


def list_timers(command: str) -> Optional[bool]:
    return True if _match("List Timers", _LIST_TIMERS_RE, command) else None


# --- Light / switch / device control fast-path ----------------------------
# Maps common smart-home commands straight to HA tool intents, skipping the
# SLM agent call entirely (the example "set downstairs office lights to a
# hundred percent" cost a full ~5s round-trip through the agent loop before
# this existed). These are deliberately allowed to over-match a little: HA's
# tools resolve the entity, and a name that doesn't resolve returns a
# "Reactive question:" sentinel, so the agent loop replans into the SLM as a
# fallback. The regex only has to win the common case to pay off.
#
# Compound utterances ("turn on the lamp and play music") and anaphora ("turn
# it off") are bailed on so the SLM can handle them properly.

_COMPOUND_RE = re.compile(r"\b(?:and|then)\b", re.IGNORECASE)

# Entity references too vague to resolve to a single entity_id — let the SLM
# decide (it may expand "everything" into several actions).
_VAGUE_ENTITIES = frozenset({
    "it", "that", "this", "them", "those", "these", "everything", "all",
})

_PERCENT_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_PERCENT_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _has_compound(command: str) -> bool:
    return bool(_COMPOUND_RE.search(command))


def _is_vague_entity(entity: str) -> bool:
    return entity.lower().strip() in _VAGUE_ENTITIES


def _parse_percent(text: str) -> Optional[int]:
    """Parse a spoken or digit brightness ('a hundred', 'fifty', '75') to 0-100.

    Returns None if no number is recognised, so the caller falls through to the
    SLM rather than guessing.
    """
    text = text.strip().lower()
    m = re.search(r"\d{1,3}", text)
    if m:
        return max(0, min(100, int(m.group())))

    text = text.replace("-", " ")
    if "hundred" in text or text in ("full", "max", "maximum", "all the way"):
        return 100
    if text == "half":
        return 50
    if text in ("zero", "none"):
        return 0

    total = None
    for tok in text.split():
        if tok in _PERCENT_TENS:
            total = (total or 0) + _PERCENT_TENS[tok]
        elif tok in _PERCENT_ONES:
            total = (total or 0) + _PERCENT_ONES[tok]
    if total is None:
        return None
    return max(0, min(100, total))


# "set the lights to 50 percent" / "dim downstairs office to a hundred percent".
# Trailing "percent"/"%" is required so the rule stays scoped to brightness and
# doesn't swallow unrelated "set X to Y" phrasing.
_LIGHT_BRIGHTNESS_RE = re.compile(
    r"^\s*(?:please\s+)?(?:set|change|put|turn|adjust|dim|brighten)\s+"
    r"(?:the\s+)?(.+?)\s+(?:to|at|down\s+to|up\s+to)\s+"
    r"(.+?)\s*(?:percent|%)\s*$",
    re.IGNORECASE,
)

# "turn on the kitchen lights" and the reversed "turn the kitchen lights on".
_TURN_ONOFF_RE = re.compile(
    r"^\s*(?:please\s+)?turn\s+(on|off)\s+(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)
_TURN_ONOFF_REV_RE = re.compile(
    r"^\s*(?:please\s+)?turn\s+(?:the\s+)?(.+?)\s+(on|off)\s*$",
    re.IGNORECASE,
)

_TOGGLE_RE = re.compile(
    r"^\s*(?:please\s+)?toggle\s+(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)

# "dim/brighten" with no explicit percentage → fixed levels (dim=30, full=100).
# The percentage form ("dim X to 20 percent") is handled earlier by
# _LIGHT_BRIGHTNESS_RE, which is listed first. \b after the verb stops "dimitri"
# from matching. Filler ("the", "lights", "in the") is peeled off the remainder
# so "dim the lights in the downstairs office" → entity "downstairs office".
_DIM_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)*"
    r"(dim|brighten)\b(.*)$",
    re.IGNORECASE,
)
_DIM_FILLER_RE = re.compile(
    r"^(?:the\s+)?(?:lights?\s+|lamps?\s+)?(?:in\s+(?:the\s+)?)?",
    re.IGNORECASE,
)
_DIM_LEVELS = {"dim": 30, "brighten": 100}


def extract_light_brightness(command: str) -> Optional[tuple]:
    """Return (entity, brightness_pct) for a brightness command, else None."""
    if _has_compound(command):
        return None
    m = _match("Brightness", _LIGHT_BRIGHTNESS_RE, command)
    if not m:
        return None
    entity = m.group(1).strip()
    pct = _parse_percent(m.group(2))
    if not entity or pct is None or _is_vague_entity(entity):
        return None
    # "set the volume to 50 percent" also matches — but that's a media_player
    # action, not a light. Defer it to the SLM (→ ha_volume_set).
    if re.search(r"\b(?:volume|sound)\b", entity, re.IGNORECASE):
        return None
    return (entity, pct)


def extract_turn_onoff(command: str) -> Optional[tuple]:
    """Return ('on'|'off', entity) for a turn on/off command, else None."""
    if _has_compound(command):
        return None
    m = _match("Turn on/off", _TURN_ONOFF_RE, command)
    if m:
        state, entity = m.group(1).lower(), m.group(2).strip()
    else:
        m = _match("Turn on/off (reversed)", _TURN_ONOFF_REV_RE, command)
        if not m:
            return None
        entity, state = m.group(1).strip(), m.group(2).lower()
    if not entity or _is_vague_entity(entity):
        return None
    return (state, entity)


def extract_dim_brighten(command: str) -> Optional[tuple]:
    """Return (entity, brightness_pct) for a bare 'dim'/'brighten', else None.

    Maps to fixed levels (dim=30, brighten=100). A bare "dim the lights" with
    no room targets the "lights" group.
    """
    if _has_compound(command):
        return None
    m = _match("Dim/Brighten", _DIM_RE, command)
    if not m:
        return None
    action = m.group(1).lower()
    entity = _DIM_FILLER_RE.sub("", m.group(2).strip(), count=1).strip()
    if not entity:
        entity = "lights"
    if _is_vague_entity(entity):
        return None
    return (entity, _DIM_LEVELS[action])


def extract_toggle(command: str) -> Optional[str]:
    """Return the entity for a toggle command, else None."""
    if _has_compound(command):
        return None
    m = _match("Toggle", _TOGGLE_RE, command)
    if not m:
        return None
    entity = m.group(1).strip()
    if not entity or _is_vague_entity(entity):
        return None
    return entity


# "lock the front door" / "unlock the back door" → ha_lock / ha_unlock.
_LOCK_RE = re.compile(
    r"^\s*(?:please\s+)?(lock|unlock)\s+(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)

# Leading particles mark a phrasal/idiomatic verb ("lock in", "lock down")
# rather than a real device name — device names never start with these, so a
# match here is an idiom, not a command.
_VERB_PARTICLES = frozenset({
    "in", "up", "down", "into", "onto", "out", "off", "on", "away",
})


def _starts_with_particle(entity: str) -> bool:
    first = entity.lower().split(None, 1)[0] if entity.split() else ""
    return first in _VERB_PARTICLES


def extract_lock(command: str) -> Optional[tuple]:
    """Return ('lock'|'unlock', entity), else None."""
    if _has_compound(command):
        return None
    m = _match("Lock", _LOCK_RE, command)
    if not m:
        return None
    action, entity = m.group(1).lower(), m.group(2).strip()
    if not entity or _is_vague_entity(entity) or _starts_with_particle(entity):
        return None
    return (action, entity)


# "open/close/shut the blinds/garage" → ha_open_cover / ha_close_cover.
# "open"/"close" are generic verbs, so this rule (unlike turn on/off) requires a
# cover-ish noun in the entity — otherwise "open Spotify" would needlessly
# bounce off HA before replanning.
_COVER_KEYWORDS = frozenset({
    "blind", "blinds", "curtain", "curtains", "garage", "shutter", "shutters",
    "shade", "shades", "awning", "awnings", "cover", "covers", "gate", "gates",
    "door", "doors",
})
_COVER_RE = re.compile(
    r"^\s*(?:please\s+)?(open|close|shut)\s+(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)


def extract_cover(command: str) -> Optional[tuple]:
    """Return ('open'|'close', entity) for a cover command, else None."""
    if _has_compound(command):
        return None
    m = _match("Cover", _COVER_RE, command)
    if not m:
        return None
    action = m.group(1).lower()
    entity = m.group(2).strip()
    if not entity or _is_vague_entity(entity):
        return None
    if not (set(entity.lower().split()) & _COVER_KEYWORDS):
        return None
    return ("open" if action == "open" else "close", entity)


# "set/make/turn the lights red" → ha_set_color. Gated on a known colour word
# (mirrors home_assistant._COLOR_MAP) so it never fires on unrelated "set X"
# phrasing. Multi-word colours are listed first so the alternation prefers them.
# RGB phrasing ("set the lights to 255,0,0") is left to the SLM.
_COLOR_NAMES = (
    "warm white", "cool white", "white", "red", "green", "blue", "yellow",
    "orange", "purple", "pink", "cyan", "magenta",
)
_COLOR_RE = re.compile(
    r"^\s*(?:please\s+)?(?:set|change|make|turn)\s+(?:the\s+)?(.+?)\s+"
    r"(?:to\s+|into\s+)?(" + "|".join(_COLOR_NAMES) + r")\s*$",
    re.IGNORECASE,
)


def extract_color(command: str) -> Optional[tuple]:
    """Return (entity, color) for a colour command, else None."""
    if _has_compound(command):
        return None
    m = _match("Color", _COLOR_RE, command)
    if not m:
        return None
    entity = m.group(1).strip()
    color = m.group(2).lower()
    if not entity or _is_vague_entity(entity):
        return None
    return (entity, color)


# "louder" / "volume up" / "turn the volume up" → ha_volume_up (and the
# symmetric down forms). No entity — the volume tools default to AVR→TV.
_VOLUME_UP_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"louder"
    r"|volume\s+up"
    r"|turn\s+up(?:\s+the)?\s+(?:volume|sound)"
    r"|turn\s+(?:it|the\s+(?:volume|sound))\s+up"
    r")\s*$",
    re.IGNORECASE,
)
_VOLUME_DOWN_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"quieter|softer"
    r"|volume\s+down"
    r"|turn\s+down(?:\s+the)?\s+(?:volume|sound)"
    r"|turn\s+(?:it|the\s+(?:volume|sound))\s+down"
    r")\s*$",
    re.IGNORECASE,
)


def extract_volume_up(command: str) -> Optional[bool]:
    return True if _match("Volume up", _VOLUME_UP_RE, command) else None


def extract_volume_down(command: str) -> Optional[bool]:
    return True if _match("Volume down", _VOLUME_DOWN_RE, command) else None


# Match explicit asks for deeper thought, not casual uses like "I think it's
# raining". Anchored to start of utterance and requires the topic preposition
# (about/on/over/through) so "consider" alone won't catch "consider it done".
_DEEP_THINK_PATTERNS = [
    re.compile(
        r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?"
        r"(?:think|consider|reflect|ponder)\s+"
        r"(?:carefully\s+|deeply\s+|hard\s+)?"
        r"(?:about|on|over|through)\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:let\s+me|let\'?s)\s+think\s+"
        r"(?:about\s+|through\s+|over\s+)(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?what\s+do\s+you\s+think\s+about\s+(.+)$",
        re.IGNORECASE,
    ),
]


def extract_deep_think(command: str) -> Optional[str]:
    """Return the topic if `command` is asking for thinking-mode reasoning."""
    for pattern in _DEEP_THINK_PATTERNS:
        match = pattern.search(command)
        if not match:
            continue
        topic = match.group(1).strip(' .,?!').strip()
        if topic:
            logger.debug(f"Caught Deep Think Match: {topic}")
            return topic
    return None


# Anchored asks to surface the partial thinking from a cancelled thinking turn,
# so passing mentions ("we discussed this so far") don't trigger. The bare
# "summarise" branch is end-anchored (\s*$) so it only catches the standalone
# command ("summarise", "summarise so far") — not "summarise today's news" /
# "summarise this email", which must fall through to the agent's web/reply path.
_SUMMARIZE_THINKING_PATTERNS = [
    re.compile(
        r"^\s*(?:please\s+)?summari[sz]e"
        r"(?:\s+(?:your\s+thoughts"
        r"|what\s+you(?:'?ve|\s+have)\s+(?:got|been\s+thinking)"
        r"|what\s+you(?:'?ve|\s+have)\s+(?:got\s+)?so\s+far))?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:so\s+)?what(?:'?s)?\s+(?:have\s+you\s+got|do\s+you\s+have|"
        r"are\s+you\s+thinking|have\s+you\s+been\s+thinking)"
        r"(?:\s+so\s+far|\s+now)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:give|tell)\s+me\s+(?:your\s+thoughts|"
        r"what\s+you('?ve|\s+have)\s+got)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:stop\s+thinking|give\s+up\s+thinking|"
        r"that('?s|\s+is)\s+enough)",
        re.IGNORECASE,
    ),
]


def extract_summarize_thinking(command: str) -> Optional[bool]:
    for pattern in _SUMMARIZE_THINKING_PATTERNS:
        if pattern.search(command):
            logger.debug(f"Caught Summarize Thinking match: {command!r}")
            return True
    return None


# Deleting or editing notes is deliberately NOT a capability — the voice
# assistant can only add to notes (there is no delete/edit tool). Without this
# guard, an asked-to-delete SLM emits a free-text reply claiming it removed the
# note when it did nothing, leaving the user to believe data is gone when it
# isn't. Catch the explicit phrasings deterministically and short-circuit to a
# fixed refusal *before* the SLM runs, so it can't confabulate. Requires both a
# removal/edit verb near the start AND a note-domain object ("note", "fact",
# "journal"), so "turn off the lights" or "add a note to delete the config
# tomorrow" don't trip it. A false refusal is recoverable (the user rephrases);
# a false claim of deletion is silent and corrosive — so we err toward refusing.
_NOTE_DELETE_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"will\s+you\s+|i\s+(?:want|need|'?d\s+like)\s+you\s+to\s+|just\s+)*"
    r"(?:delete|remove|erase|clear|wipe|scrub|undo|forget|"
    r"get\s+rid\s+of|take\s+out|edit|change|modify|update|rewrite|amend)\b"
    r".*\b(?:notes?|facts?|journal)\b",
    re.IGNORECASE | re.DOTALL,
)

_NOTE_DELETE_REPLY = (
    "I can't delete or edit notes by voice. You can remove or change them "
    "from the dashboard's Notes or Facts tab."
)


def extract_note_delete(command: str) -> Optional[bool]:
    """True if `command` asks to delete or edit a note/fact (an unsupported op)."""
    return True if _match("Note delete", _NOTE_DELETE_RE, command) else None


# Each rule: (extractor, intent_dict_builder). The builder receives the
# extractor's return value (string or True). Order matters — summarize_thinking
# is listed before deep_think so "summarise what you've been thinking" doesn't
# slip into the deep_think 'think about' family.
_INTENT_RULES = [
    # Smart-home control first — these skip the SLM entirely (HA resolves the
    # entity; a miss replans into the agent), so let them win the common case.
    (extract_light_brightness,
     lambda v: {"intent": "ha_set_brightness", "args": [v[0], v[1]]}),
    (extract_dim_brighten,
     lambda v: {"intent": "ha_set_brightness", "args": [v[0], v[1]]}),
    (extract_color,
     lambda v: {"intent": "ha_set_color", "args": [v[0], v[1]]}),
    (extract_volume_up, lambda _: {"intent": "ha_volume_up", "args": []}),
    (extract_volume_down, lambda _: {"intent": "ha_volume_down", "args": []}),
    (extract_lock,
     lambda v: {"intent": "ha_lock" if v[0] == "lock" else "ha_unlock",
                "args": [v[1]]}),
    (extract_cover,
     lambda v: {"intent": "ha_open_cover" if v[0] == "open" else "ha_close_cover",
                "args": [v[1]]}),
    (extract_turn_onoff,
     lambda v: {"intent": "turn_on" if v[0] == "on" else "turn_off",
                "args": [v[1]]}),
    (extract_toggle, lambda v: {"intent": "toggle", "args": [v]}),
    (extract_after_play, lambda v: {"intent": "play_song", "args": [v]}),
    (extract_stop, lambda _: {"intent": "pause", "args": []}),
    (has_time_query, lambda _: {"intent": "get_time", "args": []}),
    (extract_skip, lambda _: {"intent": "skip", "args": []}),
    (extract_resume, lambda _: {"intent": "resume", "args": []}),
    (extract_timer, lambda v: {"intent": "start_countdown", "args": [v]}),
    (list_timers, lambda _: {"intent": "list_timers", "args": []}),
    (extract_summarize_thinking, lambda _: {"intent": "summarize_thinking", "args": []}),
    (extract_deep_think, lambda v: {"intent": "deep_think", "args": [v]}),
]


def catchAll(user_message: str):
    """Run regex catchers in order. On match, return an agent emission
    `{"actions": [{"intent": ..., "args": ...}]}`; otherwise return the
    original message string unchanged for the SLM to handle.
    """
    # Deletion/editing of notes isn't supported — refuse deterministically
    # before the SLM can claim it did it. Returns a `reply` emission, so the
    # agent loop speaks the refusal directly without an SLM call or any tool.
    if extract_note_delete(user_message):
        logger.debug("Caught note delete/edit request; refusing")
        return {"reply": _NOTE_DELETE_REPLY}
    for extract, build in _INTENT_RULES:
        value = extract(user_message)
        if value is not None:
            return {"actions": [build(value)]}
    return user_message
