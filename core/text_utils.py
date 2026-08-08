"""Shared text cleanup for the TTS pipeline."""

import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "☀-⛿"  # misc symbols
    "✀-➿"  # dingbats
    "]+",
    flags=re.UNICODE,
)

# Dots in abbreviations are sentence boundaries to `split_sentences` and
# `split_clauses`, so use spaces to make these news acronyms unambiguously
# letter-by-letter without fragmenting speech.
_TTS_ACRONYMS = frozenset({
    "WA", "NSW", "QLD", "VIC", "TAS", "SA", "NT", "ACT",
    "NRL", "AFL", "NBA", "NFL", "MLB", "UFC", "FIFA",
    "NY", "UK", "US", "USA", "NZ", "EU",
})
_TTS_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(sorted(_TTS_ACRONYMS, key=len, reverse=True)) + r")\b"
)
_SEMICOLON_RE = re.compile(r"\s*;\s*")

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((1_000_000_000_000, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
_NUMBER_BODY = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_NUMBER = rf"-?{_NUMBER_BODY}(?:\.\d+)?"
_BOUNDARY_AFTER = r"(?!(?:[\w/:-]|\.\d))"
_CURRENCY_RE = re.compile(
    rf"(?<![\w./:-])(?P<symbol>[$£€])\s*(?P<number>-?{_NUMBER_BODY})(?:\.(?P<cents>\d{{1,2}}))?{_BOUNDARY_AFTER}"
)
_PERCENT_RE = re.compile(rf"(?<![\w./:-])(?P<number>{_NUMBER})\s*%{_BOUNDARY_AFTER}")
_UNIT_NAMES = {
    "°c": "degree Celsius", "°f": "degree Fahrenheit", "celsius": "degree Celsius",
    "fahrenheit": "degree Fahrenheit", "kg": "kilogram", "g": "gram", "mg": "milligram",
    "lb": "pound", "lbs": "pound", "oz": "ounce", "km": "kilometre", "m": "metre",
    "cm": "centimetre", "mm": "millimetre", "mi": "mile", "ft": "foot", "in": "inch",
    "l": "litre", "ml": "millilitre", "gal": "gallon", "mph": "mile per hour",
    "kph": "kilometre per hour", "km/h": "kilometre per hour", "m/s": "metre per second",
}
_UNIT_RE = re.compile(
    rf"(?<![\w./:-])(?P<number>{_NUMBER})\s*(?P<unit>"
    + "|".join(re.escape(unit) for unit in sorted(_UNIT_NAMES, key=len, reverse=True))
    + rf"){_BOUNDARY_AFTER}",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(rf"(?<![\w./:-])(?P<number>{_NUMBER}){_BOUNDARY_AFTER}")

_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
# Gemma 4 wraps its reasoning in a "thought" channel: `<|channel>thought ... <channel|>`.
# Its template also emits an empty `<|channel>thought\n<channel|>` ghost block even
# when thinking is off, so a stray one can surface in output — strip both. A lone
# unmatched closing `<channel|>` is mopped up too.
_GEMMA_THINK_PATTERN = re.compile(r"<\|channel>thought.*?<channel\|>", flags=re.DOTALL)
_GEMMA_CHANNEL_STRAY = re.compile(r"</?\|?channel\|?>")

# Split on whitespace that immediately follows a sentence terminator.
# Lookbehind keeps the punctuation with the preceding sentence so each
# returned string is itself a valid utterance.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Break at sentence + clause punctuation, keeping the delimiter on the left
# part. Shared by every TTS backend that overlaps synthesis with playback
# (core/tts_onnx.py, core/tts_crispasr.py — non-autoregressive/chunked
# backends where one fragment plays while the next renders) and by
# utils/reply_stream.py's incremental reply parser (A1a), which emits
# clause-level fragments as the SLM streams so TTS can start on the first
# clause instead of waiting for the whole reply.
CLAUSE_SPLIT = re.compile(r"(?<=[.!?,;:])\s+")


def split_clauses(text: str):
    """Yield clause/sentence fragments of `text`, dropping whitespace-only
    pieces. One fragment per clause so the first can start playing/rendering
    while the rest streams in or renders."""
    for fragment in CLAUSE_SPLIT.split(text.strip()):
        fragment = fragment.strip()
        if fragment:
            yield fragment


def clean_for_tts(text: str, strip_think: bool = True) -> str:
    """Prepare raw SLM output for speech."""
    text = text.replace('"', "").replace("*", "")
    if strip_think:
        text = _THINK_PATTERN.sub("", text)
        text = _GEMMA_THINK_PATTERN.sub("", text)
        text = _GEMMA_CHANNEL_STRAY.sub("", text).strip()
    text = _EMOJI_PATTERN.sub("", text)
    text = _SEMICOLON_RE.sub(". ", text)
    return _TTS_ACRONYM_RE.sub(lambda m: " ".join(m.group(1)), text)


def _integer_words(number: int) -> str:
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, remainder = divmod(number, 10)
        return _TENS[tens] if not remainder else f"{_TENS[tens]}-{_ONES[remainder]}"
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        prefix = f"{_ONES[hundreds]} hundred"
        return prefix if not remainder else f"{prefix} {_integer_words(remainder)}"
    for scale, name in _SCALES:
        if number >= scale:
            major, remainder = divmod(number, scale)
            prefix = f"{_integer_words(major)} {name}"
            return prefix if not remainder else f"{prefix} {_integer_words(remainder)}"
    return str(number)


def _number_words(value: str) -> str:
    """Speak a decimal digit-by-digit after the point, preserving precision."""
    negative = value.startswith("-")
    value = value.removeprefix("-").replace(",", "")
    whole, dot, fraction = value.partition(".")
    spoken = _integer_words(int(whole))
    if dot:
        spoken += " point " + " ".join(_ONES[int(digit)] for digit in fraction)
    return f"minus {spoken}" if negative else spoken


def _pluralize_unit(unit: str, value: str) -> str:
    singular = abs(float(value.replace(",", ""))) == 1
    if singular:
        return unit
    if unit == "foot":
        return "feet"
    if unit == "inch":
        return "inches"
    if unit.startswith("degree "):
        return f"degrees {unit.removeprefix('degree ')}"
    if unit.endswith("per hour") or unit.endswith("per second"):
        return unit.replace("mile per", "miles per").replace("kilometre per", "kilometres per").replace("metre per", "metres per")
    return f"{unit}s"


def spoken_for_tts(text: str) -> str:
    """Render common numeric notation into speech without changing display text.

    This intentionally skips dates, times, versions, paths, IP addresses, and
    identifier-like strings. Those forms have a digit next to a protected
    separator, while standalone quantities are safe to verbalise.
    """
    def currency(match: re.Match) -> str:
        number = match.group("number")
        cents = match.group("cents")
        singular = abs(int(number.replace(",", ""))) == 1
        symbol = match.group("symbol")
        units = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}[symbol]
        result = f"{_number_words(number)} {units[0] if singular else units[0] + 's'}"
        if cents and int(cents):
            cents_value = str(int(cents))
            cent_name = units[1] if int(cents_value) == 1 else units[1] + ("" if units[1] == "pence" else "s")
            result += f" and {_number_words(cents_value)} {cent_name}"
        return result

    def percent(match: re.Match) -> str:
        return f"{_number_words(match.group('number'))} percent"

    def quantity(match: re.Match) -> str:
        number = match.group("number")
        unit = _UNIT_NAMES[match.group("unit").lower()]
        return f"{_number_words(number)} {_pluralize_unit(unit, number)}"

    text = _CURRENCY_RE.sub(currency, text)
    text = _PERCENT_RE.sub(percent, text)
    text = _UNIT_RE.sub(quantity, text)
    return _NUMBER_RE.sub(lambda match: _number_words(match.group("number")), text)


def split_sentences(text: str) -> list:
    """Split `text` into sentences on `.`, `!`, `?` (punctuation preserved).

    Returns at least one element if `text` has any content. Edge cases
    like "Mr. Smith" will mis-split — fine for the assistant's own
    short outputs where abbreviations don't appear.
    """
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
