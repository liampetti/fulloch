"""Regex fast-path for common voice intents (play / stop / time / timer).

Returns an agent emission dict `{"actions": [{"intent": ..., "args": ...}]}`
from `catchAll` when matched, or the original string unchanged. Individual
`extract_*` helpers remain importable for tests.
"""

import logging
import re
from typing import Optional

from utils.voice_fastpath import fastpath_pattern as _pattern

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
_PLAY_RE = _pattern("play")
_STOP_RE = _pattern("stop")
_SKIP_RE = _pattern("skip")
_RESUME_RE = _pattern("resume")
# Leading context is fine ("do you know what time is it"), so the start stays
# unanchored — but the phrase must END the question (bar a few benign tail
# adverbs / trailing punctuation). Without the end-anchor, "what time is it
# GOING TO BE ON" matched the "what time is it" substring and answered with the
# current wall-clock time instead of letting the SLM resolve "it" (an event's
# broadcast time) from context. Same guard rejects "what's the time IN LONDON".
_TIME_QUERY_RE = _pattern("time")
_DATE_QUERY_RE = _pattern("date")
_OUTPUT_LANGUAGES = {
    "english": "en", "englisch": "en", "anglais": "en",
    "french": "fr", "französisch": "fr", "français": "fr",
    "german": "de", "deutsch": "de", "allemand": "de",
    "spanish": "es", "spanisch": "es", "espagnol": "es", "español": "es",
    "italian": "it", "italiano": "it", "portuguese": "pt", "português": "pt",
    "brazilian portuguese": "pt-BR", "dutch": "nl", "japanese": "ja", "chinese": "zh",
    "arabic": "ar", "swedish": "sv", "norwegian": "no", "danish": "da",
}
_OUTPUT_LANGUAGE_RE = _pattern("output_language")
_TRANSLATE_VERB_RE = _pattern("translate_verb")
_OUTPUT_VERB_RE = _pattern("output_verb")
_MEDIA_REQUEST_RE = _pattern("media_request")


def requested_output_language(command: str) -> Optional[str]:
    """Recognize a bounded explicit output language, not a media-language filter."""
    match = _OUTPUT_LANGUAGE_RE.search(command)
    if not match:
        return None
    if _MEDIA_REQUEST_RE.search(command) and not _OUTPUT_VERB_RE.search(command):
        return None
    return _OUTPUT_LANGUAGES[match.group("language").casefold()]


def requests_translation(command: str) -> bool:
    return bool(requested_output_language(command) or _TRANSLATE_VERB_RE.search(command))


def _english_clock_emission(command: str):
    from utils.hassil_fastpath import FastPathEmission

    intent = "HassGetCurrentTime" if _TIME_QUERY_RE.search(command) else (
        "HassGetCurrentDate" if _DATE_QUERY_RE.search(command) else None
    )
    if intent:
        return FastPathEmission({"intent": "get_current_time", "args": []}, "en", intent, {})
    return None


# The direct weather path is intentionally limited to generic, default-location
# forecasts. Locations, dates, and conditional questions need the agent to set
# tool arguments or resolve context, so they must not match this rule.
_WEATHER_FORECAST_RE = _pattern("weather")
# Anchored like _PLAY_RE so a mid-sentence "set a timer for the eggs" mention
# can't hijack the turn.
_TIMER_RE = _pattern("timer")
_CANCEL_TIMER_RE = _pattern("cancel_timer")
_SATELLITE_MESSAGE_RE = _pattern("satellite_message")
_LIST_TIMERS_RE = _pattern("list_timers")


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


def has_default_weather_forecast(command: str) -> Optional[bool]:
    return True if _match("Weather forecast", _WEATHER_FORECAST_RE, command) else None


def extract_timer(command: str) -> Optional[str]:
    m = _match("Timer", _TIMER_RE, command)
    return m.group(1).strip() if m else None


def extract_cancel_timer(command: str) -> Optional[tuple[Optional[str]]]:
    """Extract a direct timer cancellation, optionally with its dashboard ID."""
    m = _match("Cancel timer", _CANCEL_TIMER_RE, command)
    return (m.group("timer_id"),) if m else None


def list_timers(command: str) -> Optional[bool]:
    return True if _match("List Timers", _LIST_TIMERS_RE, command) else None


def extract_satellite_message(command: str) -> Optional[tuple[str, str]]:
    """Return `(target, message)` for a direct satellite announcement."""
    match = _match("Satellite message", _SATELLITE_MESSAGE_RE, command)
    if not match:
        return None
    target = _pattern("satellite_target_article").sub("", match.group("target").strip())
    message = match.group("message").strip().rstrip(".!?")
    return (target, message) if target and message else None


_CONTEXTUAL_WEB_SEARCH_RE = _pattern("contextual_web")


def is_contextual_web_search_request(command: str) -> bool:
    """Whether a topic-less web-search request must use conversation context."""
    return bool(_match("Contextual web search", _CONTEXTUAL_WEB_SEARCH_RE, command))


# A status question about all lights in an area needs live state reads, not the
# inventory-only `list_entities_in_area` flow. Keep this narrow so open-ended
# room questions still reach the agent.
_AREA_LIGHT_STATE_RE = _pattern("area_light_state")


def extract_area_light_state(command: str) -> Optional[tuple[str, str]]:
    """Return `(area, state)` for 'which lights are on upstairs' queries."""
    match = _match("Area light state", _AREA_LIGHT_STATE_RE, command)
    if not match:
        return None
    return match.group("area").strip(), match.group("state").lower()


# --- Light / switch / device control fast-path ----------------------------
# Maps common smart-home commands straight to HA tool intents, skipping the
# SLM agent call entirely (the example "set downstairs office lights to a
# hundred percent" cost a full ~5s round-trip through the agent loop before
# this existed). These are deliberately allowed to over-match a little: HA's
# tools resolve the entity, and a name that doesn't resolve returns a
# "Reactive question:" sentinel, so the agent loop replans into the SLM as a
# fallback. The regex only has to win the common case to pay off.
#
# Anaphora ("turn it off") is bailed on so the SLM can resolve it from history.
# A later generic compound pass combines only clauses that each independently
# match one of these deterministic rules.

_COMPOUND_RE = _pattern("compound")
_COMPOUND_SPLIT_RE = _pattern("compound_split")
_LEADING_DISCOURSE_RE = _pattern("leading_discourse")
_COMPOUND_ANAPHORA_RE = _pattern("compound_anaphora")

# Entity references too vague to resolve to a single entity_id — let the SLM
# decide (it may expand "everything" into several actions).
_VAGUE_ENTITIES = frozenset(
    {
        "it",
        "that",
        "this",
        "them",
        "those",
        "these",
        "everything",
        "all",
    }
)

# Anaphoric determiners that, when they *lead* a multi-word entity ("those
# lights", "that lamp"), mark a back-reference to something said earlier — a
# real device alias never starts with one. Without this, "turn those lights
# off" captures the literal entity "those lights", which HA can't resolve;
# bailing here routes the turn to the SLM, which resolves the reference from
# conversation history instead. (Single-token forms like "it"/"them" are
# already caught by _VAGUE_ENTITIES above.)
_ANAPHORA_DETERMINERS = frozenset({"it", "that", "this", "them", "those", "these"})

_PERCENT_ONES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_PERCENT_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _has_compound(command: str) -> bool:
    return bool(_COMPOUND_RE.search(command))


def _is_vague_entity(entity: str) -> bool:
    e = entity.lower().strip()
    if e in _VAGUE_ENTITIES:
        return True
    # Leading anaphor ("those lights", "that lamp") → needs history to resolve.
    first = e.split(None, 1)[0] if e.split() else ""
    return first in _ANAPHORA_DETERMINERS


def _parse_percent(text: str) -> Optional[int]:
    """Parse a spoken or digit brightness ('a hundred', 'fifty', '75') to 0-100.

    Returns None if no number is recognised, so the caller falls through to the
    SLM rather than guessing.
    """
    text = text.strip().lower()
    m = _pattern("percentage_digits").search(text)
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
_LIGHT_BRIGHTNESS_RE = _pattern("light_brightness")

# "turn on the kitchen lights" and the reversed "turn the kitchen lights on".
_TURN_ONOFF_RE = _pattern("turn_onoff")
# Trailing "again" sits after the state ("turn the kitchen lights off again");
# a "back" re-do adverb sits before it ("...lights back off") and is swept into
# the entity capture (stripped by _REDO_ADVERB_RE in extract_turn_onoff).
_TURN_ONOFF_REV_RE = _pattern("turn_onoff_reversed")

# Re-do adverbs that trail a turn on/off command ("turn the lights back off",
# "turn it off again"). The reversed pattern captures these into the entity
# ("dining room lights back"), which HA can't resolve — strip them so the real
# entity survives.
_REDO_ADVERB_RE = _pattern("redo_adverb")

_TOGGLE_RE = _pattern("toggle")

# "dim/brighten" with no explicit percentage → fixed levels (dim=30, full=100).
# The percentage form ("dim X to 20 percent") is handled earlier by
# _LIGHT_BRIGHTNESS_RE, which is listed first. \b after the verb stops "dimitri"
# from matching. Filler ("the", "lights", "in the") is peeled off the remainder
# so "dim the lights in the downstairs office" → entity "downstairs office".
_DIM_RE = _pattern("dim")
_DIM_FILLER_RE = _pattern("dim_filler")
_DIM_LEVELS = {"dim": 30, "brighten": 100, "brighton": 100}


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
    if _pattern("volume_or_sound").search(entity):
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
    # "turn the lights back off" / "turn it off again" — drop the re-do adverb
    # the pattern swept into the entity before resolving / vague-checking.
    entity = _REDO_ADVERB_RE.sub("", entity).strip()
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
_LOCK_RE = _pattern("lock")

# Leading particles mark a phrasal/idiomatic verb ("lock in", "lock down")
# rather than a real device name — device names never start with these, so a
# match here is an idiom, not a command.
_VERB_PARTICLES = frozenset(
    {
        "in",
        "up",
        "down",
        "into",
        "onto",
        "out",
        "off",
        "on",
        "away",
    }
)


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
_COVER_KEYWORDS = frozenset(
    {
        "blind",
        "blinds",
        "curtain",
        "curtains",
        "garage",
        "shutter",
        "shutters",
        "shade",
        "shades",
        "awning",
        "awnings",
        "cover",
        "covers",
        "gate",
        "gates",
        "door",
        "doors",
    }
)
_COVER_RE = _pattern("cover")


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
    "warm white",
    "cool white",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "cyan",
    "magenta",
)
_COLOR_RE = _pattern("color")


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
# symmetric down forms). With no target, the volume tools prefer configured
# Spotify playback, then AVR, then TV.
_VOLUME_UP_RE = _pattern("volume_up")
_VOLUME_UP_TARGET_RE = _pattern("volume_up_target")
_VOLUME_DOWN_RE = _pattern("volume_down")
_VOLUME_DOWN_TARGET_RE = _pattern("volume_down_target")


def extract_volume_up(command: str) -> Optional[bool]:
    return True if _match("Volume up", _VOLUME_UP_RE, command) else None


def extract_volume_up_target(command: str) -> Optional[str]:
    m = _match("Volume up target", _VOLUME_UP_TARGET_RE, command)
    return m.group(1).strip() if m else None


def extract_volume_down(command: str) -> Optional[bool]:
    return True if _match("Volume down", _VOLUME_DOWN_RE, command) else None


def extract_volume_down_target(command: str) -> Optional[str]:
    m = _match("Volume down target", _VOLUME_DOWN_TARGET_RE, command)
    return m.group(1).strip() if m else None


_CURRENCY_ALIASES = {
    "aud": "AUD", "australian dollar": "AUD", "australian dollars": "AUD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "usd": "USD", "us dollar": "USD", "us dollars": "USD", "dollar": "USD", "dollars": "USD",
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "british pound": "GBP", "british pounds": "GBP",
    "jpy": "JPY", "yen": "JPY",
}
_CURRENCY_CONVERSION_RE = _pattern("currency")


def extract_currency_conversion(command: str) -> Optional[tuple[str, str, float | None]]:
    """Route simple spoken FX conversions before a numeric amount looks like a ticker."""
    match = _match("Currency conversion", _CURRENCY_CONVERSION_RE, command)
    if not match:
        return None
    amount, base, quote = match.groups()
    base_code = _CURRENCY_ALIASES.get(base.strip().lower())
    quote_code = _CURRENCY_ALIASES.get(quote.strip().lower())
    if base_code is None or quote_code is None:
        return None
    return base_code, quote_code, float(amount) if amount is not None else None


# Match explicit asks for deeper thought, not casual uses like "I think it's
# raining". Anchored to start of utterance and requires the topic preposition
# (about/on/over/through) so "consider" alone won't catch "consider it done".
_DEEP_THINK_PATTERNS = [
    _pattern("deep_think_1"),
    _pattern("deep_think_2"),
    _pattern("deep_think_3"),
]


def extract_deep_think(command: str) -> Optional[str]:
    """Return the topic if `command` is asking for thinking-mode reasoning."""
    for pattern in _DEEP_THINK_PATTERNS:
        match = pattern.search(command)
        if not match:
            continue
        topic = match.group(1).strip(" .,?!").strip()
        if topic:
            logger.debug(f"Caught Deep Think Match: {topic}")
            return topic
    return None


# Anchored asks to surface the partial thinking from a cancelled thinking turn,
# so passing mentions ("we discussed this so far") don't trigger. The bare
# "summarise" branch is end-anchored (\s*$) so it only catches the standalone
# command ("summarise", "summarise so far") — not "summarise today's news" /
# "summarise this email", which must fall through to the agent's web/reply path.
# When Obsidian edit/delete mode is off, deletion and editing are deliberately
# not voice capabilities. Catch explicit phrasings before the SLM can claim it
# acted when nothing happened. When the user has explicitly enabled that mode,
# let active-note/cursor requests reach the guarded editor tools instead.
_NOTE_DELETE_RE = _pattern("note_delete", re.IGNORECASE | re.DOTALL)

_NOTE_DELETE_REPLY = (
    "I can't delete or edit notes by voice. You can remove or change them "
    "from the dashboard's Notes or Facts tab."
)


def extract_note_delete(command: str) -> Optional[bool]:
    """True if `command` asks to delete or edit a note/fact (an unsupported op)."""
    return True if _match("Note delete", _NOTE_DELETE_RE, command) else None


def _obsidian_edit_enabled() -> bool:
    """Read the live editor-access switch without importing tools at startup."""
    from tools.notes import _obsidian_edit_allowed

    return _obsidian_edit_allowed()


# The YAML chooses the ordering and canonical operations; Python retains a
# bounded registry of extractors and argument adapters, never arbitrary tools.
_ROUTE_ADAPTERS = {
    "area_light_state": (extract_area_light_state, "area_light_state", lambda v: {"intent": "get_entities_in_area_state", "args": [v[0], "light", v[1]]}),
    "light_brightness": (extract_light_brightness, "brightness", lambda v: {"intent": "ha_set_brightness", "args": [v[0], v[1]]}),
    "dim_brighten": (extract_dim_brighten, "brightness", lambda v: {"intent": "ha_set_brightness", "args": [v[0], v[1]]}),
    "color": (extract_color, "color", lambda v: {"intent": "ha_set_color", "args": [v[0], v[1]]}),
    "volume_up_target": (extract_volume_up_target, "volume_up_target", lambda v: {"intent": "ha_volume_up", "args": [v]}),
    "volume_up": (extract_volume_up, "volume_up", lambda _: {"intent": "ha_volume_up", "args": []}),
    "volume_down_target": (extract_volume_down_target, "volume_down_target", lambda v: {"intent": "ha_volume_down", "args": [v]}),
    "volume_down": (extract_volume_down, "volume_down", lambda _: {"intent": "ha_volume_down", "args": []}),
    "lock": (extract_lock, "lock", lambda v: {"intent": "ha_lock" if v[0] == "lock" else "ha_unlock", "args": [v[1]]}),
    "cover": (extract_cover, "cover", lambda v: {"intent": "ha_open_cover" if v[0] == "open" else "ha_close_cover", "args": [v[1]]}),
    "turn_onoff": (extract_turn_onoff, "turn_onoff", lambda v: {"intent": "turn_on" if v[0] == "on" else "turn_off", "args": [v[1]]}),
    "toggle": (extract_toggle, "toggle", lambda v: {"intent": "toggle", "args": [v]}),
    "play": (extract_after_play, "play", lambda v: {"intent": "play_song", "args": [v]}),
    "satellite_message": (extract_satellite_message, "satellite_message", lambda v: {"intent": "send_satellite_message", "args": [v[0], v[1]]}),
    "stop": (extract_stop, "pause", lambda _: {"intent": "pause", "args": []}),
    "time": (has_time_query, "time", lambda _: {"intent": "get_current_time", "args": []}),
    "weather": (has_default_weather_forecast, "weather", lambda _: {"intent": "get_weather_forecast", "args": []}),
    "skip": (extract_skip, "skip", lambda _: {"intent": "skip", "args": []}),
    "resume": (extract_resume, "resume", lambda _: {"intent": "resume", "args": []}),
    "currency": (extract_currency_conversion, "currency", lambda v: {"intent": "get_exchange_rate", "args": [v[0], v[1], v[2]]}),
    "timer": (extract_timer, "timer", lambda v: {"intent": "start_countdown", "args": [v]}),
    "cancel_timer": (extract_cancel_timer, "cancel_timer", lambda v: {"intent": "cancel_timer", "args": [v[0]] if v[0] else []}),
    "list_timers": (list_timers, "timer_status", lambda _: {"intent": "get_timer_status", "args": []}),
    "deep_think": (extract_deep_think, "deep_think", lambda v: {"intent": "deep_think", "args": [v]}),
}


def _configured_routes():
    from utils.voice_fastpath import fastpath_section

    routes = fastpath_section("routes") or []
    result = []
    for route in routes:
        adapter = _ROUTE_ADAPTERS.get(route.get("extractor")) if isinstance(route, dict) else None
        if adapter is None or route.get("operation") != adapter[1]:
            logger.warning("Ignoring invalid fast-path route: %r", route)
            continue
        result.append((adapter[0], adapter[2]))
    return result


_INTENT_RULES = _configured_routes()


def extract_compound_actions(command: str, rules=None) -> Optional[list[dict]]:
    """Return ordered actions when every conjunction clause is independently safe.

    Reusing the single-command rules keeps compound handling conservative: any
    contextual, ambiguous, or unsupported clause falls through to the agent.
    """
    clauses = _COMPOUND_SPLIT_RE.split(command.strip())
    if (
        len(clauses) not in (2, 3)
        or any(not clause.strip() or _COMPOUND_ANAPHORA_RE.search(clause) for clause in clauses)
    ):
        return None

    clauses[0] = _LEADING_DISCOURSE_RE.sub("", clauses[0]).strip()
    actions = []
    for clause in clauses:
        for extract, build in _INTENT_RULES if rules is None else rules:
            value = extract(clause)
            if value is not None:
                actions.append(build(value))
                break
        else:
            return None
    return actions


# Preserve local/system and media rules while replacing HA device regexes in
# the live router. The legacy public catchAll remains for isolated callers.
_HA_DEVICE_EXTRACTORS = frozenset({
    extract_light_brightness, extract_dim_brighten, extract_color,
    extract_lock, extract_cover, extract_turn_onoff, extract_toggle,
})
_LOCAL_RULES = [(extract, build) for extract, build in _INTENT_RULES if extract not in _HA_DEVICE_EXTRACTORS]


def catch_with_hassil(user_message: str, matcher, *, voice=False, context=None, ha_area=None):
    """Prefer local rules, then try the multilingual device matcher."""
    if extract_note_delete(user_message) and not _obsidian_edit_enabled():
        return {"reply": _NOTE_DELETE_REPLY}
    if requests_translation(user_message):
        return user_message
    voice_rules = getattr(matcher, "voice_fastpath", None) if voice else None
    if voice_rules and (followup := voice_rules.followup(user_message, context, ha_area)):
        return followup
    if voice_rules and (command := voice_rules.command(user_message, ha_area, matcher)):
        return command
    if actions := extract_compound_actions(user_message, _LOCAL_RULES):
        return {"actions": actions}
    if emission := _english_clock_emission(user_message):
        return emission
    for extract, build in _LOCAL_RULES:
        value = extract(user_message)
        if value is not None:
            return {"actions": [build(value)]}
    exact = matcher.match(user_message)
    if exact:
        return exact
    if voice_rules:
        return voice_rules.corrected_match(user_message, matcher, area=ha_area) or user_message
    return user_message


def catchAll(user_message: str):
    """Run regex catchers in order. On match, return an agent emission
    `{"actions": [{"intent": ..., "args": ...}]}`; otherwise return the
    original message string unchanged for the SLM to handle.
    """
    # Keep the deterministic refusal unless the user explicitly enabled the
    # active-note editor controls in the dashboard.
    if extract_note_delete(user_message) and not _obsidian_edit_enabled():
        logger.debug("Caught note delete/edit request; refusing")
        return {"reply": _NOTE_DELETE_REPLY}
    if requests_translation(user_message):
        return user_message
    if actions := extract_compound_actions(user_message):
        return {"actions": actions}
    if emission := _english_clock_emission(user_message):
        return emission
    for extract, build in _INTENT_RULES:
        value = extract(user_message)
        if value is not None:
            return {"actions": [build(value)]}
    return user_message
