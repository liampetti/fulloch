"""Timezone-derived household locale defaults."""
from zoneinfo import ZoneInfo

import pytest

import utils.locale as locale


def test_australian_timezone_supplies_currency_and_units(monkeypatch):
    monkeypatch.setattr(locale.local_time, "get_tz", lambda: ZoneInfo("Australia/Sydney"))

    result = locale.household_locale()

    assert result.territory == "AU"
    assert result.currency == "AUD"
    assert result.distance_unit == "kilometres"
    assert result.temperature_unit == "celsius"


def test_us_timezone_uses_us_preferences(monkeypatch):
    monkeypatch.setattr(locale.local_time, "get_tz", lambda: ZoneInfo("America/Chicago"))

    result = locale.household_locale()

    assert result.territory == "US"
    assert result.currency == "USD"
    assert result.distance_unit == "miles"
    assert result.temperature_unit == "fahrenheit"
    assert result.volume_system == "us"


@pytest.mark.parametrize(
    ("timezone", "territory", "currency"),
    [("America/Mexico_City", "MX", "MXN"), ("Europe/Oslo", "NO", "NOK"), ("Europe/Zurich", "CH", "CHF")],
)
def test_currency_comes_from_babel_cldr_territory_data(monkeypatch, timezone, territory, currency):
    monkeypatch.setattr(locale.local_time, "get_tz", lambda: ZoneInfo(timezone))

    result = locale.household_locale()

    assert result.territory == territory
    assert result.currency == currency
