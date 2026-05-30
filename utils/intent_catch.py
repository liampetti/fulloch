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


_PLAY_RE = re.compile(r'play\s+(.+)$', re.IGNORECASE)
_STOP_RE = re.compile(r'^\s*(stop|pause|halt)\b', re.IGNORECASE)
_SKIP_RE = re.compile(r'^\s*skip\b', re.IGNORECASE)
_RESUME_RE = re.compile(r'^\s*resume\b', re.IGNORECASE)
_TIME_QUERY_RE = re.compile(r"(what time is it|what'?s the time|what time it is)", re.IGNORECASE)
_TIMER_RE = re.compile(
    r'(?:start|set)\s+(?:a\s+)?(?:timer|time)\s+(?:for\s+)?(.+?)(?:\s+please)?$',
    re.IGNORECASE,
)
_LIST_TIMERS_RE = re.compile(
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
# so passing mentions ("we discussed this so far") don't trigger.
_SUMMARIZE_THINKING_PATTERNS = [
    re.compile(
        r"^\s*(?:please\s+)?summari[sz]e"
        r"(?:\s+(?:your\s+thoughts|what\s+you('?ve|\s+have)\s+got|so\s+far))?",
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


# Each rule: (extractor, intent_dict_builder). The builder receives the
# extractor's return value (string or True). Order matters — summarize_thinking
# is listed before deep_think so "summarise what you've been thinking" doesn't
# slip into the deep_think 'think about' family.
_INTENT_RULES = [
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
    for extract, build in _INTENT_RULES:
        value = extract(user_message)
        if value is not None:
            return {"actions": [build(value)]}
    return user_message
