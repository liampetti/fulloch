"""Small, side-effect-free coercions shared by routing and tools."""

import re
from numbers import Real

from word2number import w2n

_NUMBER_WORDS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:[ -](?:zero|one|two|three|"
    r"four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred))*\b",
    re.I,
)


def parse_percentage(value: object) -> int:
    """Return a clamped percentage from a number or natural-language value."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return max(0, min(100, round(value)))
    if not isinstance(value, str):
        raise ValueError("percentage must be a number or percentage phrase")

    text = value.strip().lower().replace("%", " percent")
    if text in {"full", "maximum", "max", "all the way"}:
        return 100
    match = re.search(r"\b\d{1,3}\b", text)
    if match:
        return max(0, min(100, int(match.group())))
    words = _NUMBER_WORDS.search(text)
    if words is None:
        raise ValueError(f"could not parse percentage {value!r}")
    try:
        return max(0, min(100, int(w2n.word_to_num(words.group(0).replace("-", " ")))))
    except ValueError as exc:
        raise ValueError(f"could not parse percentage {value!r}") from exc
