"""Isolated shared-client state for HA domain tests.

The production tool wrappers, resolvers and HTTP helpers still run. Tests seed
a loaded cache explicitly, then replace only the service boundaries they need.
Lazy-loading tests manage the unloaded state themselves.
"""

import pytest

from tools import ha_client as client


@pytest.fixture(autouse=True)
def loaded_ha(monkeypatch):
    monkeypatch.setattr(client, "_loaded", True)
    for name in ("_ENTITY_ALIASES", "_ENTITY_ALIASES_MULTI", "_AREA_MAP", "_FLOOR_MAP"):
        monkeypatch.setattr(client, name, {})
    for name in (
        "SPOTIFY_ENTITY",
        "TV_ENTITY",
        "AVR_ENTITY",
        "CALENDAR_ENTITY",
        "TODO_ENTITY",
    ):
        monkeypatch.setattr(client, name, None)
    monkeypatch.setattr(client, "_DEFAULT_WEATHER_ENTITY", "weather.home")
    monkeypatch.setattr(client, "_DENIED_ENTITIES", frozenset())
