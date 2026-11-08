"""Locale-aware numeric speech, with a safe fallback for unsupported languages."""

from decimal import Decimal, InvalidOperation

from num2words import num2words


def number_words(value: str | int, language: str = "en", *, ordinal: bool = False) -> str:
    """Convert a canonical decimal or integer, retaining digits if unsupported.

    num2words understands locale variants with underscores and falls back to
    their base language. Never silently substitute English for an unknown tag.
    """
    original = str(value)
    try:
        number = Decimal(original)
        parts = (language or "en").replace("-", "_").split("_")
        tag = "_".join([parts[0].lower(), *(part.upper() for part in parts[1:])])
        return num2words(number, lang=tag,
                         to="ordinal" if ordinal else "cardinal")
    except (NotImplementedError, ValueError, TypeError, InvalidOperation, OverflowError):
        return original
