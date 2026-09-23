"""Strict parsing for short, spoken relative durations."""

import re
from typing import Optional

from word2number import w2n

_NUMBER_WORD = (
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand)\b"
)
_COMPONENT_RE = re.compile(
    rf"(?P<value>\d+|{_NUMBER_WORD}(?:[ -]{_NUMBER_WORD}){{0,3}})\s+"
    r"(?P<unit>seconds?|minutes?|hours?)\b",
    re.I,
)
_CONNECTOR_RE = re.compile(r"(?:\s|,|\band\b)+", re.I)
_ALLOWED_OUTSIDE_RE = re.compile(r"(?:\s|[,.;]|\b(?:in|for)\b)+", re.I)
_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}


def duration_components(text: str) -> Optional[list[tuple[int, str, str]]]:
    """Extract contiguous duration components from text.

    Components may be joined only with natural list separators, preventing an
    unrelated number from silently becoming part of a timer.
    """
    matches = list(_COMPONENT_RE.finditer(text))
    if not matches:
        return None
    for left, right in zip(matches, matches[1:], strict=False):
        if not _CONNECTOR_RE.fullmatch(text[left.end():right.start()]):
            return None
    components = []
    for match in matches:
        raw_value = match.group("value")
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = int(w2n.word_to_num(raw_value.replace("-", " ")))
            except ValueError:
                return None
        if value <= 0:
            return None
        components.append((value, match.group("unit").lower(), match.group(0)))
    return components


def parse_duration(text: str) -> int:
    """Parse a complete duration phrase into seconds.

    Supports compounds such as ``one minute and thirty-five seconds`` and
    rejects extra words rather than dropping a duration component.
    """
    stripped = text.strip()
    if re.fullmatch(r"\d+", stripped):
        seconds = int(stripped)
        if seconds > 0:
            return seconds
        raise ValueError("No valid duration value found")
    components = duration_components(text)
    if not components:
        raise ValueError("No valid duration value found")
    matches = list(_COMPONENT_RE.finditer(text))
    outside = text[:matches[0].start()] + text[matches[-1].end():]
    if outside and not _ALLOWED_OUTSIDE_RE.fullmatch(outside):
        raise ValueError(f"Invalid duration {text!r}")
    return sum(value * _UNIT_SECONDS[unit.rstrip("s")] for value, unit, _ in components)


def normalise_duration(components: list[tuple[int, str, str]]) -> str:
    """Return a model-safe numeric duration while retaining all units."""
    return " ".join(f"{value} {unit}" for value, unit, _ in components)
