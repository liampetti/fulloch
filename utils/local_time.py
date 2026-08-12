"""Application-local time boundary.

Use this module for user-facing calendar dates and wall-clock timestamps rather
than ``datetime.now()`` or ``date.today()``. The timezone comes from
``general.timezone`` and falls back to UTC until initial setup chooses one.
"""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_tz: ZoneInfo | None = None


def is_valid_tz(name: str | None) -> bool:
    """Whether ``name`` is an IANA timezone, or is intentionally unset."""
    if name is None or not str(name).strip():
        return True
    try:
        ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, KeyError):
        return False
    return True


def set_tz(name: str | None) -> bool:
    """Set the application timezone, returning ``False`` for an invalid name."""
    global _tz
    if name is None or not str(name).strip():
        _tz = None
        return True
    try:
        _tz = ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, KeyError):
        return False
    return True


def get_tz() -> datetime.tzinfo:
    return _tz if _tz is not None else datetime.timezone.utc


def now() -> datetime.datetime:
    return datetime.datetime.now(tz=get_tz())


def today() -> datetime.date:
    return now().date()
