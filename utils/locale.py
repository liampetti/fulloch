"""Household locale defaults inferred from the configured IANA timezone."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from babel.core import get_global
from babel.numbers import get_territory_currencies

from utils import local_time


@dataclass(frozen=True)
class HouseholdLocale:
    """Regional defaults suitable for presentation, not a precise address."""

    timezone: str
    territory: str | None
    currency: str
    distance_unit: str
    temperature_unit: str
    volume_system: str


def _timezone_name() -> str:
    tz = local_time.get_tz()
    return str(getattr(tz, "key", None) or tz)


@lru_cache(maxsize=512)
def _territory_for_timezone(timezone: str) -> str | None:
    """Resolve an IANA timezone to its sole CLDR territory, if any."""
    territories = [
        territory
        for territory, zones in get_global("territory_zones").items()
        if territory not in {"001", "ZZ"} and timezone in zones
    ]
    return territories[0] if len(territories) == 1 else None


def _currency_for_territory(territory: str | None) -> str | None:
    if territory is None:
        return None
    currencies = get_territory_currencies(territory, tender=True)
    return currencies[0] if currencies else None


def household_locale() -> HouseholdLocale:
    """Return presentation defaults inferred from the active household timezone."""
    timezone = _timezone_name()
    territory = _territory_for_timezone(timezone)
    currency = _currency_for_territory(territory) or "USD"
    if territory == "US":
        return HouseholdLocale(timezone, territory, currency, "miles", "fahrenheit", "us")
    if territory == "GB":
        return HouseholdLocale(timezone, territory, currency, "miles", "celsius", "imperial")
    return HouseholdLocale(timezone, territory, currency, "kilometres", "celsius", "metric")
