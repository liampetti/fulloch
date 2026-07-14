"""Validated Higgs delivery controls and dashboard display cleanup."""

import re

_TOKEN = re.compile(r"<\|(?P<category>emotion|style|sfx|prosody):(?P<value>[a-z_]+)\|>")
_ANY_TOKEN = re.compile(r"<\|[^|>]+\|>")
_ALLOWED = {
    "emotion": frozenset("elation amusement enthusiasm determination pride contentment affection relief contemplation confusion surprise awe longing arousal anger fear disgust bitterness sadness shame helplessness".split()),
    "style": frozenset(("singing", "shouting", "whispering")),
    "sfx": frozenset("cough laughter crying screaming burping humming sigh sniff sneeze".split()),
    "prosody": frozenset("speed_very_slow speed_slow speed_fast speed_very_fast pitch_low pitch_high pause long_pause expressive_high expressive_low".split()),
}

_DELIVERY_REQUESTS = (
    (re.compile(r"\bwhisper\b", re.I), "<|style:whispering|>"),
    (re.compile(r"\b(?:laugh|laughing)\b", re.I), "<|sfx:laughter|>Haha "),
    (re.compile(r"\b(?:sing|singing)\b", re.I), "<|style:singing|>"),
    (re.compile(r"\b(?:shout|yell)\b", re.I), "<|style:shouting|>"),
    (re.compile(r"\b(?:slowly|calmly)\b", re.I), "<|prosody:speed_slow|>"),
    (re.compile(r"\bquietly\b", re.I), "<|style:whispering|>"),
)
_CONTINUATION = re.compile(r"^\s*(?:continue|read more|keep (?:going|whispering|reading))\b", re.I)
_SPEECH_VERB = re.compile(r"\b(?:say|tell|read|speak|repeat)\b", re.I)
_WHISPER_PREFIX = re.compile(r"^\s*(?:please\s+)?whisper(?:\s+to\s+me)?\s+(?P<body>.+?)\s*$", re.I)
_PREFIX_DELIVERY = re.compile(
    r"\b(?P<modifier>quickly|fast|slowly|calmly|quietly)\s+(?=(?:say|tell|read|speak|repeat)\b)",
    re.I,
)
_SUFFIX_DELIVERY = re.compile(
    r"\s+(?P<modifier>quickly|fast|slowly|calmly|quietly|in\s+a\s+whisper)\s*(?P<punct>[.!?]?)\s*$",
    re.I,
)
_EXTRACTED_DELIVERY = {
    "quickly": "<|prosody:speed_fast|>",
    "fast": "<|prosody:speed_fast|>",
    "slowly": "<|prosody:speed_slow|>",
    "calmly": "<|prosody:speed_slow|>",
    "quietly": "<|style:whispering|>",
    "in a whisper": "<|style:whispering|>",
}


def extract_delivery_request(text: str, previous: str = "") -> tuple[str, str]:
    """Strip unambiguous Higgs delivery wording before agent planning.

    Delivery modifiers are only removed when they qualify a speech/readback
    verb. This keeps ordinary content such as "what happened quickly after"
    intact while allowing a small model to focus on the requested action.
    """
    if _CONTINUATION.search(text):
        return text, previous

    whisper = _WHISPER_PREFIX.match(text)
    if whisper:
        return whisper.group("body"), "<|style:whispering|>"

    prefix = _PREFIX_DELIVERY.search(text)
    if prefix:
        cleaned = (text[:prefix.start()] + text[prefix.end():]).strip()
        return cleaned, _EXTRACTED_DELIVERY[prefix.group("modifier").lower()]

    suffix = _SUFFIX_DELIVERY.search(text)
    if suffix and _SPEECH_VERB.search(text[:suffix.start()]):
        cleaned = (text[:suffix.start()].rstrip() + suffix.group("punct")).strip()
        if cleaned:
            return cleaned, _EXTRACTED_DELIVERY[suffix.group("modifier").lower()]

    return text, delivery_for_request(text)


def delivery_for_request(text: str, previous: str = "") -> str:
    """Return an explicit delivery request, retaining it for continuations."""
    for pattern, tag in _DELIVERY_REQUESTS:
        if pattern.search(text):
            return tag
    return previous if previous and _CONTINUATION.search(text) else ""


def apply_delivery(text: str, delivery: str) -> str:
    """Prefix one validated per-turn delivery directive to final speech only."""
    return f"{delivery}{text}" if delivery and not text.lstrip().startswith("<|") else text




def sanitize_for_higgs(text: str, max_tags: int = 8) -> str:
    """Keep only supported tags, with a conservative per-reply limit."""
    count = 0

    def keep(match: re.Match) -> str:
        nonlocal count
        if match.group("value") not in _ALLOWED[match.group("category")] or count >= max_tags:
            return ""
        count += 1
        return match.group(0)

    def remove_or_keep(match: re.Match) -> str:
        valid = _TOKEN.fullmatch(match.group(0))
        return keep(valid) if valid else ""

    return _ANY_TOKEN.sub(remove_or_keep, text).strip()


def split_leading_higgs_controls(text: str) -> tuple[str, str]:
    """Return valid leading tags separately from the text they control."""
    tags = []
    remaining = text.lstrip()
    while match := _TOKEN.match(remaining):
        if match.group("value") not in _ALLOWED[match.group("category")]:
            break
        tags.append(match.group(0))
        remaining = remaining[match.end():].lstrip()
    return "".join(tags), remaining


def strip_higgs_controls(text: str) -> str:
    """Return natural user-visible text without delivery instructions."""
    return _ANY_TOKEN.sub("", text).replace("  ", " ").strip()
