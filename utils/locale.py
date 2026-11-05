"""Household locale defaults inferred from the configured IANA timezone."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from babel import Locale
from babel.core import get_global
from babel.numbers import get_territory_currencies

from utils import local_time

# Sentence directories in the pinned OHF-Voice/intents 2025.6.23 release.
# Coverage varies by language; regional codes use canonical casing here.
_INTENT_LANGUAGE_CODES = (
    "af ar bg bn ca cs da de-CH de el en es et eu fa fi fr ga gl gu he hi hr hu "
    "id is it ja ka kn ko kw lb lt lv ml mn mr ms nb ne nl pl pt-BR pt ro ru "
    "sk sl sr-Latn sr sv sw te th tr uk ur vi zh-CN zh-HK zh-TW"
).split()


def intent_languages() -> dict[str, str]:
    """Return upstream sentence language codes with native and English names."""
    names = {}
    for code in _INTENT_LANGUAGE_CODES:
        locale = Locale.parse(code, sep="-")
        native = locale.get_display_name(locale).capitalize()
        english = locale.get_display_name("en").capitalize()
        names[code] = native if native.casefold() == english.casefold() else f"{native} ({english})"
    return names


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
