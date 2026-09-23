"""Validated English Breeze TTS 2 audio tags and delivery helpers.

Breeze's English language contract uses parenthesised audio tags.  Keep this
separate from Higgs's ``<|category:value|>`` controls: the two syntaxes are
model-specific and must never be passed to the other model.
"""

import re

_AUDIO_TAGS = frozenset(
    {
        "laughs", "chuckles", "giggles",
        "crying", "sobs", "whimpers", "groans", "moans",
        "sighs", "gasps", "inhales", "exhales", "breathing heavily",
        "whispers", "shouts", "screams", "singing", "humming", "stutters", "pause",
        "clears throat", "coughs", "sniffs",
        "smacks lips", "clicks tongue",
        "yawns", "sneezes", "hiccups", "burps", "gulps", "gags",
        "grunts", "scoffs", "snorts",
    }
)
_TAG = re.compile(r"\((?P<tag>[a-z]+(?: [a-z]+)?)\)", re.I)
_CONTINUATION = re.compile(r"^\s*(?:continue|read more|keep (?:going|whispering|reading))\b", re.I)
_SPEECH_VERB = re.compile(r"\b(?:say|tell|read|speak|repeat)\b", re.I)
_WHISPER_PREFIX = re.compile(r"^\s*(?:please\s+)?whisper(?:\s+to\s+me)?\s+(?P<body>.+?)\s*$", re.I)
_PREFIX_DELIVERY = re.compile(r"\b(?P<modifier>quietly)\s+(?=(?:say|tell|read|speak|repeat)\b)", re.I)
_SUFFIX_DELIVERY = re.compile(r"\s+(?P<modifier>quietly|in\s+a\s+whisper)\s*(?P<punct>[.!?]?)\s*$", re.I)

_DELIVERY_REQUESTS = (
    (re.compile(r"\bwhisper\b", re.I), "(whispers)"),
    (re.compile(r"\b(?:laugh|laughing)\b", re.I), "(laughs) Haha"),
    (re.compile(r"\b(?:sing|singing)\b", re.I), "(singing)"),
    (re.compile(r"\b(?:shout|yell)\b", re.I), "(shouts)"),
    (re.compile(r"\bquietly\b", re.I), "(whispers)"),
)


def sanitize_for_breeze(text: str, max_tags: int = 8) -> str:
    """Keep only documented English Breeze audio tags, with a reply cap."""
    count = 0

    def keep(match: re.Match) -> str:
        nonlocal count
        tag = match.group("tag").lower()
        if tag not in _AUDIO_TAGS or count >= max_tags:
            return ""
        count += 1
        return f"({tag})"

    return " ".join(_TAG.sub(keep, text).split())


def strip_breeze_tags(text: str) -> str:
    """Return dashboard text without any parenthesised Breeze control tag."""
    return _TAG.sub("", text).replace("  ", " ").strip()


def delivery_for_request(text: str, previous: str = "") -> str:
    """Return a supported explicit delivery request for this turn."""
    for pattern, tag in _DELIVERY_REQUESTS:
        if pattern.search(text):
            return tag
    return previous if previous and _CONTINUATION.search(text) else ""


def extract_delivery_request(text: str, previous: str = "") -> tuple[str, str]:
    """Remove unambiguous whisper wording before agent planning.

    Breeze has no tag for rate changes, so unlike Higgs we leave slow/fast
    wording with the agent rather than incorrectly translating it to an audio
    event. Quiet delivery maps to the documented ``(whispers)`` tag.
    """
    if _CONTINUATION.search(text):
        return text, previous
    whisper = _WHISPER_PREFIX.match(text)
    if whisper:
        return whisper.group("body"), "(whispers)"
    prefix = _PREFIX_DELIVERY.search(text)
    if prefix:
        return (text[:prefix.start()] + text[prefix.end():]).strip(), "(whispers)"
    suffix = _SUFFIX_DELIVERY.search(text)
    if suffix and _SPEECH_VERB.search(text[:suffix.start()]):
        cleaned = (text[:suffix.start()].rstrip() + suffix.group("punct")).strip()
        if cleaned:
            return cleaned, "(whispers)"
    return text, delivery_for_request(text, previous)


def apply_delivery(text: str, delivery: str) -> str:
    """Prefix one validated Breeze event to final speech only."""
    return f"{delivery} {text}" if delivery and not text.lstrip().startswith("(") else text
