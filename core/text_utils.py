"""Shared text cleanup for the TTS pipeline."""

import re

from utils.number_words import number_words

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
_CLOCK_RE = re.compile(
    r"(?<![\w./:-])(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    r"(?![\w/:]|\.\d|-\d)(?P<suffix>\s*(?:a\.?m\.?|p\.?m\.?|uhr)(?!\w))?",
    re.IGNORECASE,
)
_CLOCK_CONTEXT = {
    "en": re.compile(r"\b(?:at|it(?:'s| is)|time(?:'s| is))\s*$", re.IGNORECASE),
    "de": re.compile(r"\b(?:um|es ist|uhrzeit(?: ist)?)\s*$", re.IGNORECASE),
    "fr": re.compile(r"(?:\b(?:il est|heure(?: est)?)|\bà)\s*$", re.IGNORECASE),
}

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


def _spoken_clock_times(text: str, language: str) -> str:
    """Verbalize unambiguous clock notation, not arbitrary colon-separated values."""
    base = (language or "en").replace("_", "-").split("-", 1)[0].lower()
    if base not in _CLOCK_CONTEXT:
        return text

    def clock(match: re.Match) -> str:
        hour, minute = int(match.group("hour")), int(match.group("minute"))
        suffix = (match.group("suffix") or "").strip()
        suffix_kind = suffix.lower().replace(".", "")
        if hour > 23 or minute > 59 or (suffix_kind in {"am", "pm"} and not 1 <= hour <= 12):
            return match.group()
        if suffix_kind == "uhr" and base != "de":
            return match.group()
        if suffix_kind in {"am", "pm"} and base != "en":
            return match.group()
        # Bare clock strings are useful to TTS callers, but embedded notation
        # needs a clock cue to avoid silently changing scores, ratios and paths.
        bare = not text[:match.start()].strip() and not text[match.end():].strip(" .!?")
        if not (bare or suffix or _CLOCK_CONTEXT[base].search(text[:match.start()])):
            return match.group()
        hour_words = number_words(hour, language)
        minute_words = number_words(minute, language)
        if hour_words == str(hour) or minute_words == str(minute):
            return match.group()
        if base == "en":
            if minute == 0:
                result = hour_words if suffix else f"{hour_words} o'clock"
            elif minute < 10:
                result = f"{hour_words} oh {minute_words}"
            else:
                result = f"{hour_words} {minute_words}"
            return f"{result} {suffix}" if suffix else result
        if base == "de":
            return f"{hour_words} Uhr" + (f" {minute_words}" if minute else "")
        # French hour counts are feminine: une heure, vingt et une heures.
        if hour % 10 == 1 and hour != 11:
            hour_words = hour_words[:-2] + "une" if hour_words.endswith("un") else hour_words
        return f"{hour_words} {'heure' if hour == 1 else 'heures'}" + (f" {minute_words}" if minute else "")

    return _CLOCK_RE.sub(clock, text)


def spoken_for_tts(text: str, language: str = "en") -> str:
    """Render common numeric notation into speech without changing display text.

    Recognizable clock times are verbalized for English, German and French.
    Dates, versions, paths, IP addresses and ambiguous identifiers are retained.
    """
    text = _spoken_clock_times(text, language)
    if language.replace("_", "-").split("-", 1)[0].lower() != "en":
        # Keep symbols and units as written: their grammatical forms depend on
        # the language. Do not interpret comma decimals as English thousands.
        from babel import Locale
        from babel.core import UnknownLocaleError
        from babel.numbers import (
            NumberFormatError,
            get_decimal_symbol,
            get_group_symbol,
            parse_decimal,
        )

        try:
            locale = Locale.parse(language.replace("-", "_"))
            decimal = get_decimal_symbol(locale)
            group = get_group_symbol(locale)
        except (UnknownLocaleError, ValueError, TypeError):
            return text
        body = rf"\d+(?:{re.escape(decimal)}\d+)?"
        if group in {",", "."} and group != decimal:
            body = rf"(?:\d{{1,3}}(?:{re.escape(group)}\d{{3}})+(?:{re.escape(decimal)}\d+)?|{body})"
        pattern = re.compile(rf"(?<![\w./:-])(?P<number>-?{body})(?![\w/:-]|\.\d|,\d)")

        def localized_number(match: re.Match) -> str:
            original = match.group("number")
            try:
                value = parse_decimal(original, locale=locale)
            except NumberFormatError:
                return original
            converted = number_words(str(value), language)
            return original if converted == str(value) else converted

        return pattern.sub(localized_number, text)

    def _number_words(value: str) -> str:
        return number_words(value.replace(",", ""), language)

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
