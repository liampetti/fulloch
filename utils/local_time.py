"""Central timezone — all datetime.now() calls in this project use this module.

The timezone is set once at startup from config (general.timezone) and updated
live whenever the browser POSTs /setup/timezone. Falls back to UTC when no
timezone has been configured yet.
"""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_tz: ZoneInfo | None = None


def set_tz(name: str) -> None:
    global _tz
    try:
        _tz = ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        pass


def get_tz() -> datetime.tzinfo:
    return _tz if _tz is not None else datetime.timezone.utc


def now() -> datetime.datetime:
    return datetime.datetime.now(tz=get_tz())


def today() -> datetime.date:
    return now().date()
