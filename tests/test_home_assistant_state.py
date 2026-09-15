"""HA state behavior against production implementations."""

import datetime
import json
from unittest.mock import MagicMock, patch

from tests.ha_fixtures import loaded_ha  # noqa: F401
from tools import ha_client as client
from tools import ha_state as state_module


def _history_response(states):
    """Build a mock /api/history/period response: [[state, ...]]."""
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: [states]
    return resp


def _patch_history(states, entity_id="light.dining_room", friendly="Dining Room Lights"):
    """Seed the real resolver and return endpoint-shaped HTTP responses."""
    current_response = MagicMock()
    current_response.json.return_value = {"state": "off", "attributes": {}}

    def get(url, **kwargs):
        if "/api/history/period/" in url:
            assert kwargs["params"]["filter_entity_id"] == entity_id
            return _history_response(states)
        assert url.endswith(f"/api/states/{entity_id}")
        return current_response

    return [
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_loaded", True),
        patch.object(client, "_ENTITY_ALIASES", {friendly.lower(): entity_id}),
        patch.object(client, "_ENTITY_ALIASES_MULTI", {friendly.lower(): [entity_id]}),
        patch("tools.ha_client.requests.get", side_effect=get),
    ]


def test_humanize_condition_maps_ha_slugs_to_speech():
    from tools.ha_state import _humanize_condition

    # The no-separator slug a bare "-"→" " swap can't fix (and that the CPU TTS
    # front-end silently drops).
    assert _humanize_condition("partlycloudy") == "partly cloudy"
    # Valid-but-wrong words no splitter could fix.
    assert _humanize_condition("exceptional") == "severe weather"
    assert _humanize_condition("windy-variant") == "windy"
    assert _humanize_condition("lightning-rainy") == "thunderstorms"
    # Case/whitespace tolerant; empty → empty.
    assert _humanize_condition(" PartlyCloudy ") == "partly cloudy"
    assert _humanize_condition(None) == ""
    # Unknown slug falls back to a hyphen swap rather than dropping it.
    assert _humanize_condition("some-new-state") == "some new state"


def test_weather_forecast_uses_default_days_for_invalid_count(monkeypatch):
    """A malformed model argument must not prevent the current forecast."""
    from tools import ha_client as client

    monkeypatch.setattr(client, "HA_TOKEN", "token")
    monkeypatch.setattr(client, "_DEFAULT_WEATHER_ENTITY", "weather.home")
    monkeypatch.setattr(
        client,
        "_get_state",
        lambda _entity: {"state": "cloudy", "attributes": {"temperature": 8}},
    )
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(Exception())
    )

    result = state_module.get_weather_forecast(days="days")

    assert "currently cloudy at 8 degrees Celsius" in result


def test_weather_forecast_includes_todays_low_and_high(monkeypatch):
    """Forecast dates must be evaluated in Home Assistant's local timezone."""
    from tools import ha_client as client

    monkeypatch.setattr(client, "HA_TOKEN", "token")
    monkeypatch.setattr(client, "_DEFAULT_WEATHER_ENTITY", "weather.home")
    monkeypatch.setattr(client._local_tz, "today", lambda: datetime.date(2026, 8, 13))
    monkeypatch.setattr(client._local_tz, "get_tz", lambda: datetime.timezone.utc)
    monkeypatch.setattr(
        client,
        "_get_state",
        lambda _entity: {"state": "cloudy", "attributes": {"temperature": 14}},
    )
    response = MagicMock()
    response.json.return_value = {
        "service_response": {
            "weather.home": {
                "forecast": [
                    {
                        "datetime": "2026-08-13T00:00:00+00:00",
                        "condition": "cloudy",
                        "templow": 12,
                        "temperature": 18,
                    }
                ]
            }
        }
    }
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: response)

    result = state_module.get_weather_forecast(days=1)

    assert "Today cloudy 12 to 18 degrees Celsius" in result


def test_weather_forecast_attaches_dashboard_artifact(monkeypatch):
    """The text response remains usable while the dashboard gets typed data."""
    from tools import ha_client as client

    monkeypatch.setattr(client, "HA_TOKEN", "token")
    monkeypatch.setattr(client, "_DEFAULT_WEATHER_ENTITY", "weather.home")
    monkeypatch.setattr(client._local_tz, "today", lambda: datetime.date(2026, 8, 13))
    monkeypatch.setattr(client._local_tz, "get_tz", lambda: datetime.timezone.utc)
    monkeypatch.setattr(
        client,
        "_get_state",
        lambda _entity: {
            "state": "partlycloudy",
            "attributes": {"temperature": 14, "temperature_unit": "°C"},
        },
    )
    response = MagicMock()
    response.json.return_value = {
        "service_response": {
            "weather.home": {
                "forecast": [
                    {
                        "datetime": "2026-08-13T00:00:00+00:00",
                        "condition": "rainy",
                        "templow": 11,
                        "temperature": 17,
                        "precipitation_probability": 75,
                    }
                ]
            }
        }
    }
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: response)

    result = state_module.get_weather_forecast(days=1)

    assert result.artifact == {
        "type": "weather",
        "title": "home",
        "unit": "degrees Celsius",
        "current": {"condition": "partly cloudy", "temperature": 14},
        "forecast": [
            {
                "date": "2026-08-13",
                "label": "Today",
                "condition": "rainy",
                "low": 11,
                "high": 17,
                "precipitation_probability": 75,
            }
        ],
    }


def test_get_temperature_resolves_collided_climate_over_light():
    """A climate entity named 'Upstairs' that lost the first-wins alias key to
    a light of the same name is still found for a temperature lookup."""
    # light.upstairs won the single map; both share the "upstairs" name.
    aliases = {"upstairs": "light.upstairs"}
    multi = {"upstairs": ["light.upstairs", "climate.living"]}
    climate_state = {
        "entity_id": "climate.living",
        "state": "fan_only",
        "attributes": {"friendly_name": "Upstairs", "current_temperature": 18.3},
    }

    def fake_get_state(entity_id):
        return climate_state if entity_id == "climate.living" else None

    with (
        patch("tools.ha_client._ENTITY_ALIASES", aliases),
        patch("tools.ha_client._ENTITY_ALIASES_MULTI", multi),
        patch("tools.ha_client._get_state", side_effect=fake_get_state),
    ):
        from tools.ha_client import _resolve_entity
        from tools.ha_state import get_temperature

        # Variant resolver recovers the climate entity despite the light winning.
        assert _resolve_entity("upstairs", domain="climate") == "climate.living"
        result = get_temperature("upstairs")
        # Reads the climate temp, preserves its decimal precision, and speaks
        # as "upstairs" (not the slug "living").
        assert "18.3" in result and "upstairs" in result.lower()
        assert "living" not in result.lower()


def test_get_temperature_reports_climate_target_when_different():
    """A climate zone's target/setpoint should surface alongside the current
    reading, e.g. answering "what's it set to?" without a second tool."""
    from tools.ha_state import get_temperature

    state = {
        "entity_id": "climate.upstairs",
        "state": "heat",
        "attributes": {
            "friendly_name": "Upstairs",
            "current_temperature": 18.3,
            "temperature": 21.0,
        },
    }
    with (
        patch("tools.ha_client._resolve_with_variants", return_value="climate.upstairs"),
        patch("tools.ha_client._get_state", return_value=state),
    ):
        result = get_temperature("upstairs")

    assert "18.3" in result
    assert "21" in result


def test_get_temperature_omits_target_when_equal_to_current():
    from tools.ha_state import get_temperature

    state = {
        "entity_id": "climate.upstairs",
        "state": "heat",
        "attributes": {
            "friendly_name": "Upstairs",
            "current_temperature": 21.0,
            "temperature": 21.0,
        },
    }
    with (
        patch("tools.ha_client._resolve_with_variants", return_value="climate.upstairs"),
        patch("tools.ha_client._get_state", return_value=state),
    ):
        result = get_temperature("upstairs")

    assert result.lower() == "upstairs is 21 degrees celsius"
    assert "set to" not in result


def test_get_temperature_sensor_ignores_temperature_attr_as_target():
    """A plain sensor's `temperature` attribute (if present) isn't a
    thermostat setpoint — only climate.* entities get the "set to" phrasing."""
    from tools.ha_state import get_temperature

    state = {
        "entity_id": "sensor.upstairs_temperature",
        "state": "18.3",
        "attributes": {"friendly_name": "Upstairs Temperature", "temperature": 21.0},
    }
    with (
        patch(
            "tools.ha_client._resolve_with_variants",
            return_value="sensor.upstairs_temperature",
        ),
        patch("tools.ha_client._get_state", return_value=state),
    ):
        result = get_temperature("upstairs")

    assert "set to" not in result


def test_get_entity_state_reports_humidity_battery_and_position():

    state = {
        "state": "on",
        "attributes": {
            "friendly_name": "Upstairs Sensor",
            "humidity": 45,
            "battery_level": 20,
            "current_position": 60,
            "hvac_action": "heating",
        },
    }
    with (
        patch("tools.ha_client._resolve_entity", return_value="sensor.upstairs"),
        patch("tools.ha_client._get_state", return_value=state),
    ):
        result = state_module.get_entity_state("upstairs")

    assert "humidity: 45%" in result
    assert "battery: 20%" in result
    assert "position: 60% open" in result
    assert "hvac action: heating" in result
    assert result.artifact == {
        "type": "entity_status",
        "title": "Upstairs Sensor",
        "domain": "sensor",
        "state": "on",
        "details": [
            {"label": "Humidity", "value": "45%"},
            {"label": "Open", "value": "60%"},
            {"label": "Battery", "value": "20%"},
            {"label": "Activity", "value": "heating"},
        ],
    }


def test_get_media_player_state_attaches_media_artifact():

    state = {
        "state": "playing",
        "attributes": {
            "friendly_name": "Kitchen Speaker",
            "media_title": "Teardrop",
            "media_artist": "Massive Attack",
            "volume_level": 0.42,
        },
    }
    with (
        patch("tools.ha_client._resolve_entity", return_value="media_player.kitchen"),
        patch("tools.ha_client._get_state", return_value=state),
    ):
        result = state_module.get_entity_state("kitchen speaker")

    assert "Kitchen Speaker is playing" in result
    assert result.artifact == {
        "type": "media",
        "title": "Teardrop",
        "artist": "Massive Attack",
        "player": "Kitchen Speaker",
        "state": "playing",
        "volume": 42,
        "artwork_url": "/media-artwork/media_player.kitchen",
    }


def test_home_overview_groups_live_voice_enabled_states(monkeypatch):
    from tools import ha_client as client

    states = [
        {"entity_id": "light.kitchen", "state": "on", "attributes": {"friendly_name": "Kitchen"}},
        {
            "entity_id": "binary_sensor.window",
            "state": "on",
            "attributes": {"friendly_name": "Study Window", "device_class": "window"},
        },
        {
            "entity_id": "lock.front_door",
            "state": "unlocked",
            "attributes": {"friendly_name": "Front Door"},
        },
        {
            "entity_id": "media_player.office",
            "state": "playing",
            "attributes": {"friendly_name": "Office Speaker"},
        },
        {"entity_id": "light.private", "state": "on", "attributes": {"friendly_name": "Private"}},
    ]
    response = MagicMock()
    response.json.return_value = states
    monkeypatch.setattr(client, "HA_TOKEN", "token")
    monkeypatch.setattr(client, "_loaded", True)
    monkeypatch.setattr(client, "_DENIED_ENTITIES", frozenset({"light.private"}))
    monkeypatch.setattr(client.requests, "get", lambda *args, **kwargs: response)

    result = state_module.get_home_overview()

    assert result == "Home overview: 1 lights on, 1 open, 1 unlocked, 1 active."
    assert result.artifact == {
        "type": "home_overview",
        "groups": [
            {"label": "Lights on", "kind": "lights", "count": 1, "entities": ["Kitchen"]},
            {"label": "Open", "kind": "openings", "count": 1, "entities": ["Study Window"]},
            {"label": "Unlocked", "kind": "locks", "count": 1, "entities": ["Front Door"]},
            {"label": "Active", "kind": "active", "count": 1, "entities": ["Office Speaker"]},
        ],
    }


def test_energy_overview_attaches_current_readings_and_history(monkeypatch):
    from tools import ha_client as client

    states = [
        {
            "entity_id": "sensor.grid_power",
            "state": "850",
            "attributes": {
                "friendly_name": "Grid power",
                "device_class": "power",
                "unit_of_measurement": "W",
            },
        },
        {
            "entity_id": "sensor.solar_power",
            "state": "1200",
            "attributes": {
                "friendly_name": "Solar",
                "device_class": "power",
                "unit_of_measurement": "W",
            },
        },
        {
            "entity_id": "sensor.powerwall_battery",
            "state": "76",
            "attributes": {
                "friendly_name": "Home battery",
                "device_class": "battery",
                "unit_of_measurement": "%",
            },
        },
    ]
    snapshot = MagicMock()
    snapshot.json.return_value = states
    history = MagicMock()
    history.json.return_value = [
        [
            {"state": "800", "last_changed": "2026-08-31T10:00:00+00:00"},
            {"state": "850", "last_changed": "2026-08-31T11:00:00+00:00"},
        ]
    ]
    monkeypatch.setattr(client, "HA_TOKEN", "token")
    monkeypatch.setattr(client, "_loaded", True)
    monkeypatch.setattr(
        client.requests, "get", lambda url, **kwargs: history if "/history/" in url else snapshot
    )

    result = state_module.get_energy_overview()

    assert result == "Energy overview: Grid power is 850 W, Solar is 1200 W, Home battery is 76 %."
    assert result.artifact == {
        "type": "energy",
        "metrics": [
            {"kind": "consumption", "label": "Grid power", "value": 850.0, "unit": "W"},
            {"kind": "solar", "label": "Solar", "value": 1200.0, "unit": "W"},
            {"kind": "battery", "label": "Home battery", "value": 76.0, "unit": "%"},
        ],
        "history": [
            {"time": "2026-08-31T10:00:00+00:00", "value": 800.0},
            {"time": "2026-08-31T11:00:00+00:00", "value": 850.0},
        ],
        "history_unit": "W",
    }


def test_security_overview_reports_only_constrained_entity_summaries(monkeypatch):
    from tools import ha_client as client

    states = [
        {
            "entity_id": "lock.front_door",
            "state": "unlocked",
            "attributes": {"friendly_name": "Front Door"},
        },
        {
            "entity_id": "binary_sensor.window",
            "state": "on",
            "attributes": {"friendly_name": "Study Window", "device_class": "window"},
        },
        {
            "entity_id": "camera.driveway",
            "state": "idle",
            "attributes": {"friendly_name": "Driveway"},
        },
        {
            "entity_id": "camera.backyard",
            "state": "unavailable",
            "attributes": {"friendly_name": "Backyard"},
        },
    ]
    response = MagicMock()
    response.json.return_value = states
    monkeypatch.setattr(client, "HA_TOKEN", "token")
    monkeypatch.setattr(client, "_loaded", True)
    monkeypatch.setattr(client.requests, "get", lambda *args, **kwargs: response)

    result = state_module.get_security_overview()

    assert result == "Security attention: 1 open entries, 1 unlocked locks, 1 cameras unavailable."
    assert result.artifact == {
        "type": "security",
        "status": "attention",
        "groups": [
            {"label": "Open entries", "kind": "openings", "count": 1, "entities": ["Study Window"]},
            {"label": "Unlocked", "kind": "locks", "count": 1, "entities": ["Front Door"]},
            {"label": "Cameras online", "kind": "cameras", "count": 1, "entities": ["Driveway"]},
            {
                "label": "Cameras unavailable",
                "kind": "camera_issues",
                "count": 1,
                "entities": ["Backyard"],
            },
        ],
    }


def test_entity_history_no_longer_accepts_state_arg():
    """The state= pre-filter was removed — the agent now distills the list via a
    composing replan (intents.LOOKUP_TOOLS), so the tool no longer takes state."""
    import inspect

    params = inspect.signature(state_module.get_entity_history).parameters
    assert "state" not in params


def test_entity_history_returns_full_change_list_for_the_agent():
    """The tool returns the raw state-change list; the agent loop composes the
    spoken answer from it (see intents.is_lookup)."""
    import contextlib

    states = [
        {"state": "on", "last_changed": "2026-06-24T19:30:00+00:00"},
        {"state": "off", "last_changed": "2026-06-24T23:00:00+00:00"},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _patch_history(states):
            stack.enter_context(cm)
        result = state_module.get_entity_history("dining room lights")
    assert "History for" in result
    assert ": on" in result and ": off" in result


def test_temperature_history_attaches_chart_artifact():
    import contextlib

    states = [
        {"state": "18.5", "last_changed": "2026-06-24T19:30:00+00:00"},
        {"state": "20", "last_changed": "2026-06-24T23:00:00+00:00"},
    ]
    current = {
        "state": "20.5",
        "attributes": {"temperature_unit": "°C", "current_temperature": 20.5, "temperature": 21},
    }
    with contextlib.ExitStack() as stack:
        for cm in _patch_history(states, "climate.dining_room", "Dining Room Temperature"):
            stack.enter_context(cm)
        stack.enter_context(patch("tools.ha_client._get_state", return_value=current))
        result = state_module.get_entity_history("dining room temperature")

    assert result.artifact == {
        "type": "temperature_history",
        "title": "dining room temperature",
        "unit": "°C",
        "current": 20.5,
        "target": 21.0,
        "min": 18.5,
        "max": 20.0,
        "points": [
            {"time": "2026-06-24T19:30:00+00:00", "value": 18.5},
            {"time": "2026-06-24T23:00:00+00:00", "value": 20.0},
        ],
    }


def test_light_history_attaches_state_and_brightness_artifact():
    import contextlib

    states = [
        {
            "state": "on",
            "last_changed": "2026-06-24T19:30:00+00:00",
            "attributes": {"brightness": 64},
        },
        {"state": "off", "last_changed": "2026-06-24T23:00:00+00:00", "attributes": {}},
    ]
    current = {"state": "off", "attributes": {"brightness": 128}}
    with contextlib.ExitStack() as stack:
        for cm in _patch_history(states):
            stack.enter_context(cm)
        stack.enter_context(patch("tools.ha_client._get_state", return_value=current))
        result = state_module.get_entity_history("dining room lights")

    assert result.artifact == {
        "type": "light_history",
        "title": "dining room lights",
        "state": "off",
        "brightness": 50,
        "points": [
            {"time": "2026-06-24T19:30:00+00:00", "on": True, "brightness": 25},
            {"time": "2026-06-24T23:00:00+00:00", "on": False, "brightness": None},
        ],
    }


def test_entity_history_uses_ha_local_date_for_relative_labels():
    import contextlib

    from tools import ha_client as client

    states = [{"state": "on", "last_changed": "2026-06-25T00:30:00+00:00"}]
    with contextlib.ExitStack() as stack:
        for cm in _patch_history(states):
            stack.enter_context(cm)
        stack.enter_context(
            patch.object(client._local_tz, "today", return_value=datetime.date(2026, 6, 25))
        )
        stack.enter_context(
            patch.object(client._local_tz, "get_tz", return_value=datetime.timezone.utc)
        )
        result = state_module.get_entity_history("dining room lights")

    assert "today at 12:30 AM" in result


def test_conversation_history_sorts_normalised_timestamps(monkeypatch):
    from tools import ha_client as client

    monkeypatch.setattr(client, "HA_TOKEN", "tok")
    monkeypatch.setattr(
        state_module,
        "_fetch_history_states",
        lambda entity, _start, _end: (
            [{"state": "question", "last_changed": "2026-06-24T10:00:00+10:00"}]
            if entity.endswith("utterance")
            else [{"state": "answer", "last_changed": "2026-06-24T00:30:00+00:00"}]
        ),
    )
    monkeypatch.setattr(client._local_tz, "today", lambda: datetime.date(2026, 6, 24))
    monkeypatch.setattr(client._local_tz, "get_tz", lambda: datetime.timezone.utc)
    monkeypatch.setattr(
        client._local_tz,
        "now",
        lambda: datetime.datetime(2026, 6, 25, tzinfo=datetime.timezone.utc),
    )

    result = state_module.get_conversation_history("2026-06-24")

    assert result.index("You: question") < result.index("Fulloch: answer")


def test_get_entity_state_not_found_is_reactive():
    """A miss must be a `Reactive question:` sentinel, not a plain apology —
    otherwise a batch of alternate-name guesses in one turn gets every failed
    guess joined verbatim into the spoken reply alongside a successful one."""

    with (
        patch("tools.ha_client._resolve_entity", return_value="climate.nope"),
        patch("tools.ha_client._get_state", return_value=None),
    ):
        result = state_module.get_entity_state("downstairs thermostat")

    assert result.startswith("Reactive question:")


def test_list_entities_in_floor_aggregates_its_areas():
    from tools import ha_client as client

    responses = iter(
        [
            json.dumps(["upstairs_bathroom", "main_bedroom"]),
            json.dumps(["light.bathroom"]),
            json.dumps(["light.bedroom", "sensor.bedroom_temperature"]),
        ]
    )
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_AREA_MAP", {"upstairs_bathroom": "Upstairs Bathroom"}),
        patch.object(client, "_FLOOR_MAP", {"upstairs": "Upstairs"}),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch("tools.ha_client._render_template", side_effect=responses),
        patch("tools.ha_client._friendly_for", side_effect=lambda entity_id: entity_id),
    ):
        result = state_module.list_entities_in_area("upstairs")

    assert result == "Upstairs has: light.bathroom, light.bedroom, sensor.bedroom_temperature"


def test_list_entities_in_area_filters_domain_and_denylist():
    from tools import ha_client as client

    entity_ids = ["light.downstairs_office", "light.downstairs_hallway", "switch.downstairs_fan"]
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_AREA_MAP", {"downstairs": "Downstairs"}),
        patch.object(client, "_DENIED_ENTITIES", frozenset({"light.downstairs_hallway"})),
        patch("tools.ha_client._resolve_area", return_value="downstairs"),
        patch("tools.ha_client._render_template", return_value=json.dumps(entity_ids)),
    ):
        result = state_module.list_entities_in_area("downstairs", "light")

    assert "downstairs office" in result
    assert "downstairs hallway" not in result  # deny-listed
    assert "downstairs fan" not in result  # wrong domain


def test_list_entities_in_area_unknown_area_is_reactive():
    from tools import ha_client as client

    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch("tools.ha_client._resolve_area", return_value=None),
    ):
        result = state_module.list_entities_in_area("nonexistent zone")

    assert result.startswith("Reactive question:")


def test_get_entities_in_area_state_filters_to_requested_state(monkeypatch):
    from tools import ha_client as client

    states = {
        "light.office_main": {
            "state": "on",
            "attributes": {"friendly_name": "Office Main", "brightness": 128},
        },
        "light.office_lamp": {"state": "off", "attributes": {"friendly_name": "Office Lamp"}},
    }
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_AREA_MAP", {"office": "Office"}),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch("tools.ha_client._resolve_area", return_value="office"),
        patch("tools.ha_client._area_entities", return_value=list(states)),
        patch("tools.ha_client._get_state", side_effect=states.get),
    ):
        result = state_module.get_entities_in_area_state("office", "light", "on")

    assert result == "Office Main is on, brightness: 50%"
    assert result.artifact == {
        "type": "entity_status",
        "title": "Office",
        "entities": [
            {
                "type": "entity_status",
                "title": "Office Main",
                "domain": "light",
                "state": "on",
                "details": [{"label": "Brightness", "value": "50%"}],
            }
        ],
    }
