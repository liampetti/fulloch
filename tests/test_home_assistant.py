"""Logic-layer tests for tools/home_assistant.py.

The HTTP wrappers themselves aren't tested (matches the repo pattern of
not unit-testing thin REST wrappers). What is tested: fallback chains,
friendly-name role resolution, and date windowing. Temperature is now
passed through to HA without clamping — HA's per-entity min_temp /
max_temp attributes enforce safe bounds.
"""

import datetime
import json
from unittest.mock import MagicMock, patch


def test_import_does_no_network_load_lazily(monkeypatch):
    """Importing the module must not fetch from HA; the load is lazy + one-shot.

    Guards the regression where importing tools.home_assistant connected to HA
    (default URL + credentials.json token) at import time, re-enabling HA via a stray import.
    """
    import tools.home_assistant as ha

    # No eager load happened at import (token-free test env).
    assert ha._loaded is False

    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return {"kitchen": "light.kitchen"}, {"kitchen": ["light.kitchen"]}

    # Patch every global _ensure_loaded writes, so the module is restored after.
    for _name, _val in (
        ("_loaded", False),
        ("HA_TOKEN", "tok"),
        ("_ENTITY_ALIASES", {}),
        ("_ENTITY_ALIASES_MULTI", {}),
        ("_AREA_MAP", {}),
        ("_FLOOR_MAP", {}),
        ("_DEFAULT_WEATHER_ENTITY", None),
        ("SPOTIFY_ENTITY", None),
        ("TV_ENTITY", None),
        ("AVR_ENTITY", None),
        ("CALENDAR_ENTITY", None),
        ("TODO_ENTITY", None),
    ):
        monkeypatch.setattr(ha, _name, _val)
    monkeypatch.setattr(ha, "_fetch_entity_aliases", fake_fetch)
    monkeypatch.setattr(ha, "_fetch_area_map", lambda: {})
    monkeypatch.setattr(ha, "_fetch_floor_map", lambda: {})

    assert calls["n"] == 0  # nothing fetched yet
    ha._ensure_loaded()
    assert calls["n"] == 1  # first use loads
    assert ha._ENTITY_ALIASES == {"kitchen": "light.kitchen"}
    ha._ensure_loaded()
    assert calls["n"] == 1  # idempotent — no refetch


def test_ensure_loaded_is_noop_without_token(monkeypatch):
    """No token → nothing to fetch, and patched globals are left untouched."""
    import tools.home_assistant as ha

    monkeypatch.setattr(ha, "_loaded", False)
    monkeypatch.setattr(ha, "HA_TOKEN", "")
    monkeypatch.setattr(ha, "SPOTIFY_ENTITY", "media_player.spotify")
    called = {"n": 0}
    monkeypatch.setattr(
        ha, "_fetch_entity_aliases", lambda: called.__setitem__("n", called["n"] + 1) or ({}, {})
    )

    ha._ensure_loaded()
    assert called["n"] == 0  # never fetched
    assert ha.SPOTIFY_ENTITY == "media_player.spotify"  # patch not clobbered


def test_set_climate_passes_temperature_through():
    """No application-level clamp — HA enforces its own min/max bounds."""
    with (
        patch("tools.home_assistant._resolve_entity", return_value="climate.office"),
        patch("tools.home_assistant._call_service") as call,
    ):
        from tools.home_assistant import set_climate

        set_climate("office", 21)
        sent = call.call_args.args[3]
        assert sent["temperature"] == 21


def test_humanize_condition_maps_ha_slugs_to_speech():
    from tools.home_assistant import _humanize_condition

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
    import tools.home_assistant as ha

    monkeypatch.setattr(ha, "HA_TOKEN", "token")
    monkeypatch.setattr(ha, "_DEFAULT_WEATHER_ENTITY", "weather.home")
    monkeypatch.setattr(
        ha,
        "_get_state",
        lambda _entity: {"state": "cloudy", "attributes": {"temperature": 8}},
    )
    monkeypatch.setattr(ha.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(Exception()))

    result = ha.get_weather_forecast(days="days")

    assert "currently cloudy at 8 degrees Celsius" in result


def test_weather_forecast_includes_todays_low_and_high(monkeypatch):
    """Forecast dates must be evaluated in Home Assistant's local timezone."""
    import tools.home_assistant as ha

    monkeypatch.setattr(ha, "HA_TOKEN", "token")
    monkeypatch.setattr(ha, "_DEFAULT_WEATHER_ENTITY", "weather.home")
    monkeypatch.setattr(ha._local_tz, "today", lambda: datetime.date(2026, 8, 13))
    monkeypatch.setattr(ha._local_tz, "get_tz", lambda: datetime.timezone.utc)
    monkeypatch.setattr(
        ha,
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
    monkeypatch.setattr(ha.requests, "post", lambda *args, **kwargs: response)

    result = ha.get_weather_forecast(days=1)

    assert "Today cloudy 12 to 18 degrees Celsius" in result


def test_calendar_window_today_is_midnight_to_midnight():
    from tools.home_assistant import _calendar_window

    start, end = _calendar_window("today", now=datetime.datetime(2026, 5, 21, 14, 30))
    assert start == "2026-05-21T00:00:00"
    assert end == "2026-05-22T00:00:00"


def test_calendar_window_tomorrow_is_one_day_after():
    from tools.home_assistant import _calendar_window

    start, end = _calendar_window("tomorrow", now=datetime.datetime(2026, 5, 21, 14, 30))
    assert start == "2026-05-22T00:00:00"
    assert end == "2026-05-23T00:00:00"


def test_calendar_window_week_is_seven_days_from_today():
    from tools.home_assistant import _calendar_window

    start, end = _calendar_window("this_week", now=datetime.datetime(2026, 5, 21, 14, 30))
    assert start == "2026-05-21T00:00:00"
    assert end == "2026-05-28T00:00:00"


def test_calendar_window_specific_iso_date_is_single_day():
    from tools.home_assistant import _calendar_window

    start, end = _calendar_window("2026-06-26", now=datetime.datetime(2026, 6, 18, 14, 30))
    assert start == "2026-06-26T00:00:00"
    assert end == "2026-06-27T00:00:00"


def test_calendar_window_unrecognised_string_defaults_to_today():
    from tools.home_assistant import _calendar_window

    start, end = _calendar_window("sometime", now=datetime.datetime(2026, 6, 18, 14, 30))
    assert start == "2026-06-18T00:00:00"
    assert end == "2026-06-19T00:00:00"


def test_whats_on_reads_both_primary_and_reminder_calendars():
    """Events Fulloch writes to its reminder calendar must surface in whats_on
    even when the autodetected read calendar differs (#2 regression)."""
    import tools.home_assistant as ha

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-06-26T09:00:00", "summary": "Standup"},
            ]
        },
        "calendar.fulloch": {
            "events": [
                {"start": "2026-06-26T12:00:00", "summary": "Australia vs Paraguay"},
            ]
        },
    }
    with (
        patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(ha, "_reminder_calendar_entity", return_value="calendar.fulloch"),
        patch.object(ha, "_call_service_with_response", return_value=response) as call,
    ):
        out = ha._ha_get_events("2026-06-26")

    # Both calendars were queried in one call.
    assert call.call_args.args[2]["entity_id"] == ["calendar.primary", "calendar.fulloch"]
    # The reminder-calendar event is present in the spoken summary.
    assert "Australia vs Paraguay" in out
    assert "Standup" in out


def test_whats_on_dedupes_when_read_and_reminder_calendars_match():
    import tools.home_assistant as ha

    response = {"calendar.primary": {"events": []}}
    with (
        patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(ha, "_reminder_calendar_entity", return_value="calendar.primary"),
        patch.object(ha, "_call_service_with_response", return_value=response) as call,
    ):
        ha._ha_get_events("today")
    assert call.call_args.args[2]["entity_id"] == ["calendar.primary"]


def test_parse_lookback_days_units():
    from tools.home_assistant import _parse_lookback_days

    assert _parse_lookback_days("30d") == 30
    assert _parse_lookback_days("2w") == 14
    assert _parse_lookback_days("6m") == 180
    assert _parse_lookback_days("1y") == 365
    assert _parse_lookback_days("10") == 10  # bare number -> days
    assert _parse_lookback_days("garbage") == 30  # falls back to default


def test_relative_day_phrase():
    from tools.home_assistant import _relative_day_phrase

    now = datetime.datetime(2026, 6, 18, 9, 0)  # Thursday
    assert _relative_day_phrase(now, now) == "today"
    assert _relative_day_phrase(now + datetime.timedelta(days=1), now) == "tomorrow"
    assert _relative_day_phrase(now - datetime.timedelta(days=1), now) == "yesterday"
    assert _relative_day_phrase(now + datetime.timedelta(days=3), now) == "this Sunday"
    assert _relative_day_phrase(now + datetime.timedelta(days=10), now) == "next Sunday"
    assert _relative_day_phrase(now - datetime.timedelta(days=3), now) == "last Monday"
    # Far outside the near-date window: full calendar date, not a bare weekday
    # (a bare weekday is ambiguous across a multi-week search window).
    far = _relative_day_phrase(now - datetime.timedelta(days=25), now)
    assert far == "Sunday, May 24"


def test_when_is_it_on_matches_by_substring():
    import tools.home_assistant as ha

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-05-20T09:00:00", "summary": "Dentist appointment"},
                {"start": "2026-06-26T09:00:00", "summary": "Standup"},
            ]
        },
    }
    with (
        patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(ha, "_reminder_calendar_entity", return_value=None),
        patch.object(ha, "_call_service_with_response", return_value=response),
    ):
        out = ha._ha_get_events_name("dentist", "30d")

    assert "Dentist appointment" in out
    assert "Standup" not in out


def test_when_is_it_on_no_match_is_spoken_directly():
    import tools.home_assistant as ha

    response = {"calendar.primary": {"events": []}}
    with (
        patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(ha, "_reminder_calendar_entity", return_value=None),
        patch.object(ha, "_call_service_with_response", return_value=response),
    ):
        out = ha._ha_get_events_name("dentist", "30d")

    assert "couldn't find" in out
    assert not out.startswith("Reactive question:")


def test_when_is_it_on_no_calendar_configured():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "CALENDAR_ENTITY", None),
        patch.object(ha, "_reminder_calendar_entity", return_value=None),
    ):
        out = ha._ha_get_events_name("dentist", "30d")

    assert out == "No calendar is configured in Home Assistant."


def test_calendar_normalises_timed_event():
    import tools.home_assistant as ha

    ha_event = {"start": "2026-05-25T10:00:00+10:00", "end": "...", "summary": "Dentist"}
    with patch.object(ha._local_tz, "get_tz", return_value=datetime.timezone.utc):
        norm = ha._normalise_ha_event(ha_event)
    assert norm == {"start": "2026-05-25T00:00:00+00:00", "summary": "Dentist", "all_day": False}


def test_calendar_normalises_all_day_event():
    from tools.home_assistant import _normalise_ha_event

    ha_event = {"start": "2026-05-26", "end": "2026-05-27", "summary": "Public holiday"}
    norm = _normalise_ha_event(ha_event)
    assert norm == {"start": "2026-05-26", "summary": "Public holiday", "all_day": True}


def test_named_calendar_event_normalises_timezone_before_relative_date():
    import tools.home_assistant as ha

    response = {
        "calendar.primary": {
            "events": [{"start": "2026-05-25T10:00:00+10:00", "summary": "Dentist"}]
        }
    }
    now = datetime.datetime(2026, 5, 24, 12, tzinfo=datetime.timezone.utc)
    with (
        patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(ha, "_reminder_calendar_entity", return_value=None),
        patch.object(ha, "_call_service_with_response", return_value=response),
        patch.object(ha._local_tz, "now", return_value=now),
        patch.object(ha._local_tz, "get_tz", return_value=datetime.timezone.utc),
    ):
        out = ha._ha_get_events_name("dentist")

    assert out == "Dentist is at 12:00 AM tomorrow."


def test_calendar_events_sort_by_normalised_timestamp():
    import tools.home_assistant as ha

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-06-26T01:00:00+00:00", "summary": "Second"},
                {"start": "2026-06-26T10:00:00+10:00", "summary": "First"},
            ]
        }
    }
    with (
        patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(ha, "_reminder_calendar_entity", return_value=None),
        patch.object(ha, "_call_service_with_response", return_value=response),
        patch.object(ha._local_tz, "get_tz", return_value=datetime.timezone.utc),
    ):
        out = ha._ha_get_events("2026-06-26")

    assert out.index("First") < out.index("Second")


# ---------------------------------------------------------------------------
# Role-entity auto-detection.
# ---------------------------------------------------------------------------


def _patch_aliases(aliases: dict):
    """Helper: patch the module-level alias maps with a fresh dict.

    Patches both the first-wins single map and the collision multimap (derived
    one-entity-per-name) so resolution paths that consult either stay in sync.
    """
    multi = {k: [v] for k, v in aliases.items()}
    return patch.multiple(
        "tools.home_assistant",
        _ENTITY_ALIASES=aliases,
        _ENTITY_ALIASES_MULTI=multi,
    )


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
        patch("tools.home_assistant._ENTITY_ALIASES", aliases),
        patch("tools.home_assistant._ENTITY_ALIASES_MULTI", multi),
        patch("tools.home_assistant._get_state", side_effect=fake_get_state),
    ):
        from tools.home_assistant import _resolve_entity, get_temperature

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
    from tools.home_assistant import get_temperature

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
        patch("tools.home_assistant._resolve_with_variants", return_value="climate.upstairs"),
        patch("tools.home_assistant._get_state", return_value=state),
    ):
        result = get_temperature("upstairs")

    assert "18.3" in result
    assert "21" in result


def test_get_temperature_omits_target_when_equal_to_current():
    from tools.home_assistant import get_temperature

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
        patch("tools.home_assistant._resolve_with_variants", return_value="climate.upstairs"),
        patch("tools.home_assistant._get_state", return_value=state),
    ):
        result = get_temperature("upstairs")

    assert result.lower() == "upstairs is 21 degrees celsius"
    assert "set to" not in result


def test_get_temperature_sensor_ignores_temperature_attr_as_target():
    """A plain sensor's `temperature` attribute (if present) isn't a
    thermostat setpoint — only climate.* entities get the "set to" phrasing."""
    from tools.home_assistant import get_temperature

    state = {
        "entity_id": "sensor.upstairs_temperature",
        "state": "18.3",
        "attributes": {"friendly_name": "Upstairs Temperature", "temperature": 21.0},
    }
    with (
        patch(
            "tools.home_assistant._resolve_with_variants",
            return_value="sensor.upstairs_temperature",
        ),
        patch("tools.home_assistant._get_state", return_value=state),
    ):
        result = get_temperature("upstairs")

    assert "set to" not in result


def test_open_cover_uses_cover_domain_for_cover_entity():
    import tools.home_assistant as ha

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch("tools.home_assistant._resolve_entity", return_value="cover.garage"),
        patch("tools.home_assistant.requests.post", return_value=resp) as post,
    ):
        ha.open_cover("garage")

    url = post.call_args.args[0]
    assert "/services/cover/open_cover" in url


def test_open_cover_uses_valve_domain_for_valve_entity():
    """A valve.* entity gets valve.open_valve, not cover.open_cover — same
    voice verb ("open the valve"), different HA domain/service."""
    import tools.home_assistant as ha

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch("tools.home_assistant._resolve_entity", return_value="valve.main_water"),
        patch("tools.home_assistant.requests.post", return_value=resp) as post,
    ):
        result = ha.open_cover("main water valve")

    url = post.call_args.args[0]
    assert "/services/valve/open_valve" in url
    assert "Opened" in result


def test_set_cover_position_clamps_and_targets_valve_service():
    import tools.home_assistant as ha

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch("tools.home_assistant._resolve_entity", return_value="valve.main_water"),
        patch("tools.home_assistant.requests.post", return_value=resp) as post,
    ):
        ha.set_cover_position("main water valve", 150)

    url = post.call_args.args[0]
    payload = post.call_args.kwargs["json"]
    assert "/services/valve/set_valve_position" in url
    assert payload["position"] == 100


def test_ha_vacuum_dispatches_known_action():
    import tools.home_assistant as ha

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch("tools.home_assistant._resolve_entity", return_value="vacuum.roomba"),
        patch("tools.home_assistant.requests.post", return_value=resp) as post,
    ):
        result = ha.ha_vacuum("roomba", "dock")

    url = post.call_args.args[0]
    assert "/services/vacuum/return_to_base" in url
    assert "dock" in result.lower()


def test_ha_vacuum_rejects_unknown_action():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch("tools.home_assistant._resolve_entity", return_value="vacuum.roomba"),
        patch("tools.home_assistant.requests.post") as post,
    ):
        result = ha.ha_vacuum("roomba", "levitate")

    assert "don't know how to" in result.lower()
    post.assert_not_called()


def test_get_entity_state_reports_humidity_battery_and_position():
    import tools.home_assistant as ha

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
        patch("tools.home_assistant._resolve_entity", return_value="sensor.upstairs"),
        patch("tools.home_assistant._get_state", return_value=state),
    ):
        result = ha.get_entity_state("upstairs")

    assert "humidity: 45%" in result
    assert "battery: 20%" in result
    assert "position: 60% open" in result
    assert "hvac action: heating" in result


def test_complete_todo_item_matches_by_substring():
    import tools.home_assistant as ha

    items = [
        {"summary": "Buy milk", "uid": "1"},
        {"summary": "Call the plumber", "uid": "2"},
    ]
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "TODO_ENTITY", "todo.shopping_list"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch(
            "tools.home_assistant._call_service_with_response",
            return_value={"todo.shopping_list": {"items": items}},
        ),
        patch("tools.home_assistant.requests.post", return_value=resp) as post,
    ):
        result = ha.complete_todo_item("milk")

    payload = post.call_args.kwargs["json"]
    assert payload["item"] == "Buy milk"
    assert payload["status"] == "completed"
    assert "Buy milk" in result


def test_complete_todo_item_no_match_is_spoken_directly():
    import tools.home_assistant as ha

    items = [{"summary": "Buy milk", "uid": "1"}]
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "TODO_ENTITY", "todo.shopping_list"),
        patch(
            "tools.home_assistant._call_service_with_response",
            return_value={"todo.shopping_list": {"items": items}},
        ),
    ):
        result = ha.complete_todo_item("dentist appointment")

    assert "couldn't find" in result.lower()


def test_resolve_entity_without_domain_keeps_first_wins():
    """No domain hint → first registration order entity (unchanged behaviour)."""
    with patch(
        "tools.home_assistant._ENTITY_ALIASES_MULTI",
        {"upstairs": ["light.upstairs", "climate.living"]},
    ):
        from tools.home_assistant import _resolve_entity

        assert _resolve_entity("upstairs") == "light.upstairs"


def test_autodetect_spotify_uses_configured_entity():
    """No autodetection — the configured friendly name resolves via the alias map."""
    with (
        _patch_aliases({"sonos living room": "media_player.sonos_living_room"}),
        patch("tools.home_assistant.HA_CONFIG", {"spotify_entity": "Sonos Living Room"}),
    ):
        from tools.home_assistant import _autodetect_spotify_entity

        assert _autodetect_spotify_entity() == "media_player.sonos_living_room"


def test_autodetect_spotify_returns_none_with_no_match():
    with (
        _patch_aliases({"kitchen speaker": "media_player.kitchen"}),
        patch("tools.home_assistant.HA_CONFIG", {}),
    ):
        from tools.home_assistant import _autodetect_spotify_entity

        assert _autodetect_spotify_entity() is None


def test_autodetect_tv_matches_underscore_token():
    with (
        _patch_aliases(
            {
                "living room tv": "media_player.living_room_tv",
                "spotify": "media_player.spotify_alice",
            }
        ),
        patch("tools.home_assistant.HA_CONFIG", {}),
    ):
        from tools.home_assistant import _autodetect_tv_entity

        assert _autodetect_tv_entity() == "media_player.living_room_tv"


def test_autodetect_tv_does_not_steal_spotify():
    """Even if no TV exists, the spotify entity must not be picked as TV."""
    with (
        _patch_aliases({"spotify": "media_player.spotify_alice"}),
        patch("tools.home_assistant.HA_CONFIG", {}),
    ):
        from tools.home_assistant import _autodetect_tv_entity

        assert _autodetect_tv_entity() is None


def test_autodetect_avr_matches_pioneer_keyword():
    with (
        _patch_aliases(
            {
                "kitchen speaker": "media_player.kitchen",
                "pioneer avr": "media_player.pioneer_avr",
            }
        ),
        patch("tools.home_assistant.HA_CONFIG", {}),
    ):
        from tools.home_assistant import _autodetect_avr_entity

        assert _autodetect_avr_entity() == "media_player.pioneer_avr"


def test_autodetect_avr_matches_receiver_keyword_in_friendly_name():
    with (
        _patch_aliases({"living room receiver": "media_player.lounge_av"}),
        patch("tools.home_assistant.HA_CONFIG", {}),
    ):
        from tools.home_assistant import _autodetect_avr_entity

        assert _autodetect_avr_entity() == "media_player.lounge_av"


def test_autodetect_calendar_prefers_primary():
    with (
        _patch_aliases(
            {
                "work": "calendar.work",
                "primary": "calendar.primary",
                "personal": "calendar.personal",
            }
        ),
        patch("tools.home_assistant.HA_CONFIG", {}),
    ):
        from tools.home_assistant import _autodetect_calendar_entity

        assert _autodetect_calendar_entity() == "calendar.primary"


def test_autodetect_calendar_falls_back_to_first_when_no_primary():
    with _patch_aliases({"work": "calendar.work"}), patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_calendar_entity

        assert _autodetect_calendar_entity() == "calendar.work"


# ---------------------------------------------------------------------------
# Voice deny-list — dashboard-managed, file-backed. Entities switched off in
# the dashboard's Entities tab can't be voice-controlled, but stay usable
# in the dashboard. Edits are live (no restart) and persisted to JSON.
# ---------------------------------------------------------------------------


def test_call_service_refuses_denied_entity():
    """A deny-listed entity_id is refused before any HTTP call."""
    import tools.home_assistant as ha

    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset({"lock.front_door"})),
        patch("tools.home_assistant.requests.post") as post,
    ):
        result = ha._call_service("lock", "unlock", "lock.front_door")
        assert "voice control" in result.lower()
        post.assert_not_called()


def test_call_service_allows_non_denied_entity():
    """A normal entity still calls the service (deny-list doesn't over-block)."""
    import tools.home_assistant as ha

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_DENIED_ENTITIES", frozenset({"lock.front_door"})),
        patch("tools.home_assistant.requests.post", return_value=resp) as post,
    ):
        ha._call_service("light", "turn_on", "light.kitchen", success_message="ok")
        post.assert_called_once()


def test_set_entity_denied_persists_and_takes_effect(tmp_path):
    """Toggling deny mutates the live set, persists JSON, and round-trips on load."""
    import tools.home_assistant as ha

    path = str(tmp_path / "voice_denylist.json")
    with (
        patch.object(ha, "_DENYLIST_PATH", path),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
    ):
        ha.set_entity_denied("lock.front_door", True)
        # Live: the in-memory set updated immediately, no restart.
        assert "lock.front_door" in ha.get_denylist()
        # Persisted: the JSON file reflects it and reloads identically.
        assert ha._load_denylist() == frozenset({"lock.front_door"})
        # Toggling back off removes it.
        ha.set_entity_denied("lock.front_door", False)
        assert "lock.front_door" not in ha.get_denylist()
        assert ha._load_denylist() == frozenset()


def test_load_denylist_missing_file_is_empty(tmp_path):
    """No persisted file → nothing blocked (feature is a no-op until used)."""
    import tools.home_assistant as ha

    with patch.object(ha, "_DENYLIST_PATH", str(tmp_path / "absent.json")):
        assert ha._load_denylist() == frozenset()


def test_load_denylist_ignores_malformed(tmp_path):
    """A corrupt deny-list file fails safe to empty rather than crashing."""
    import tools.home_assistant as ha

    path = tmp_path / "voice_denylist.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with patch.object(ha, "_DENYLIST_PATH", str(path)):
        assert ha._load_denylist() == frozenset()


def test_list_entities_reports_deny_state():
    """list_entities surfaces every entity with its allow/deny flag, sorted."""
    import tools.home_assistant as ha

    with (
        _patch_aliases(
            {
                "kitchen": "light.kitchen",
                "front door": "lock.front_door",
            }
        ),
        patch.object(ha, "_DENIED_ENTITIES", frozenset({"lock.front_door"})),
    ):
        entities = ha.list_entities()
    by_id = {e["entity_id"]: e for e in entities}
    assert by_id["lock.front_door"]["denied"] is True
    assert by_id["light.kitchen"]["denied"] is False
    assert by_id["light.kitchen"]["domain"] == "light"
    # Deny-listed entities stay listed so they can be re-enabled.
    assert "lock.front_door" in by_id


def _history_response(states):
    """Build a mock /api/history/period response: [[state, ...]]."""
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: [states]
    return resp


def _patch_history(ha, states):
    """Stub entity resolution + the HA history GET for get_entity_history tests."""
    return [
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_loaded", True),
        patch("tools.home_assistant._resolve_entity", return_value="light.dining_room"),
        patch("tools.home_assistant._friendly_for", return_value="Dining Room Lights"),
        patch("tools.home_assistant.requests.get", return_value=_history_response(states)),
    ]


def test_entity_history_no_longer_accepts_state_arg():
    """The state= pre-filter was removed — the agent now distills the list via a
    composing replan (intents.LOOKUP_TOOLS), so the tool no longer takes state."""
    import inspect

    import tools.home_assistant as ha

    params = inspect.signature(ha.get_entity_history).parameters
    assert "state" not in params


def test_entity_history_returns_full_change_list_for_the_agent():
    """The tool returns the raw state-change list; the agent loop composes the
    spoken answer from it (see intents.is_lookup)."""
    import contextlib

    import tools.home_assistant as ha

    states = [
        {"state": "on", "last_changed": "2026-06-24T19:30:00+00:00"},
        {"state": "off", "last_changed": "2026-06-24T23:00:00+00:00"},
    ]
    with contextlib.ExitStack() as stack:
        for cm in _patch_history(ha, states):
            stack.enter_context(cm)
        result = ha.get_entity_history("dining room lights")
    assert "History for" in result
    assert ": on" in result and ": off" in result


def test_entity_history_uses_ha_local_date_for_relative_labels():
    import contextlib

    import tools.home_assistant as ha

    states = [{"state": "on", "last_changed": "2026-06-25T00:30:00+00:00"}]
    with contextlib.ExitStack() as stack:
        for cm in _patch_history(ha, states):
            stack.enter_context(cm)
        stack.enter_context(patch.object(ha._local_tz, "today", return_value=datetime.date(2026, 6, 25)))
        stack.enter_context(patch.object(ha._local_tz, "get_tz", return_value=datetime.timezone.utc))
        result = ha.get_entity_history("dining room lights")

    assert "today at 12:30 AM" in result


def test_conversation_history_sorts_normalised_timestamps(monkeypatch):
    import tools.home_assistant as ha

    monkeypatch.setattr(ha, "HA_TOKEN", "tok")
    monkeypatch.setattr(
        ha,
        "_fetch_history_states",
        lambda entity, _start, _end: (
            [{"state": "question", "last_changed": "2026-06-24T10:00:00+10:00"}]
            if entity.endswith("utterance")
            else [{"state": "answer", "last_changed": "2026-06-24T00:30:00+00:00"}]
        ),
    )
    monkeypatch.setattr(ha._local_tz, "today", lambda: datetime.date(2026, 6, 24))
    monkeypatch.setattr(ha._local_tz, "get_tz", lambda: datetime.timezone.utc)
    monkeypatch.setattr(
        ha._local_tz,
        "now",
        lambda: datetime.datetime(2026, 6, 25, tzinfo=datetime.timezone.utc),
    )

    result = ha.get_conversation_history("2026-06-24")

    assert result.index("You: question") < result.index("Fulloch: answer")


def test_get_entity_state_not_found_is_reactive():
    """A miss must be a `Reactive question:` sentinel, not a plain apology —
    otherwise a batch of alternate-name guesses in one turn gets every failed
    guess joined verbatim into the spoken reply alongside a successful one."""
    import tools.home_assistant as ha

    with (
        patch("tools.home_assistant._resolve_entity", return_value="climate.nope"),
        patch("tools.home_assistant._get_state", return_value=None),
    ):
        result = ha.get_entity_state("downstairs thermostat")

    assert result.startswith("Reactive question:")


def test_resolve_area_matches_by_display_name():
    import tools.home_assistant as ha

    with patch.object(ha, "_AREA_MAP", {"downstairs": "Downstairs", "office": "Office"}):
        assert ha._resolve_area("downstairs") == "downstairs"
        assert ha._resolve_area("the office") == "office"
        assert ha._resolve_area("upstairs") is None


def test_floor_name_does_not_fuzzy_match_child_area():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "_AREA_MAP", {"upstairs_bathroom": "Upstairs Bathroom"}),
        patch.object(ha, "_FLOOR_MAP", {"upstairs": "Upstairs"}),
    ):
        assert ha._resolve_area("upstairs") is None
        assert ha._resolve_floor("upstairs") == "upstairs"


def test_list_entities_in_floor_aggregates_its_areas():
    import tools.home_assistant as ha

    responses = iter(
        [
            json.dumps(["upstairs_bathroom", "main_bedroom"]),
            json.dumps(["light.bathroom"]),
            json.dumps(["light.bedroom", "sensor.bedroom_temperature"]),
        ]
    )
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_AREA_MAP", {"upstairs_bathroom": "Upstairs Bathroom"}),
        patch.object(ha, "_FLOOR_MAP", {"upstairs": "Upstairs"}),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch("tools.home_assistant._render_template", side_effect=responses),
        patch("tools.home_assistant._friendly_for", side_effect=lambda entity_id: entity_id),
    ):
        result = ha.list_entities_in_area("upstairs")

    assert result == "Upstairs has: light.bathroom, light.bedroom, sensor.bedroom_temperature"


def test_media_target_prefers_spotify_and_resolves_a_room_player():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "_loaded", True),
        patch.object(ha, "SPOTIFY_ENTITY", "media_player.sonos"),
        patch.object(ha, "AVR_ENTITY", "media_player.avr"),
        patch.object(ha, "TV_ENTITY", "media_player.tv"),
        patch.object(ha, "_resolve_area", return_value="living_room"),
        patch.object(
            ha,
            "_area_entities",
            return_value=["media_player.tv", "media_player.sonos"],
        ),
    ):
        assert ha._media_target(None) == "media_player.sonos"
        assert ha._media_target("living room") == "media_player.sonos"
        assert ha._media_target("living room", prefer_spotify=False) == "media_player.tv"


def test_list_entities_in_area_filters_domain_and_denylist():
    import tools.home_assistant as ha

    entity_ids = ["light.downstairs_office", "light.downstairs_hallway", "switch.downstairs_fan"]
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_AREA_MAP", {"downstairs": "Downstairs"}),
        patch.object(ha, "_DENIED_ENTITIES", frozenset({"light.downstairs_hallway"})),
        patch("tools.home_assistant._resolve_area", return_value="downstairs"),
        patch("tools.home_assistant._render_template", return_value=json.dumps(entity_ids)),
    ):
        result = ha.list_entities_in_area("downstairs", "light")

    assert "downstairs office" in result
    assert "downstairs hallway" not in result  # deny-listed
    assert "downstairs fan" not in result  # wrong domain


def test_list_entities_in_area_unknown_area_is_reactive():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch("tools.home_assistant._resolve_area", return_value=None),
    ):
        result = ha.list_entities_in_area("nonexistent zone")

    assert result.startswith("Reactive question:")


def test_get_entities_in_area_state_filters_to_requested_state(monkeypatch):
    import tools.home_assistant as ha

    states = {
        "light.office_main": {"state": "on", "attributes": {"friendly_name": "Office Main", "brightness": 128}},
        "light.office_lamp": {"state": "off", "attributes": {"friendly_name": "Office Lamp"}},
    }
    with (
        patch.object(ha, "HA_TOKEN", "tok"),
        patch.object(ha, "_AREA_MAP", {"office": "Office"}),
        patch.object(ha, "_DENIED_ENTITIES", frozenset()),
        patch("tools.home_assistant._resolve_area", return_value="office"),
        patch("tools.home_assistant._area_entities", return_value=list(states)),
        patch("tools.home_assistant._get_state", side_effect=states.get),
    ):
        result = ha.get_entities_in_area_state("office", "light", "on")

    assert result == "Office Main is on, brightness: 50%"


# --- Spotify Connect fallback for transport controls (pause/resume/skip/previous) ---
# Covers the Spotify-only-no-HA setup: no media_player entity resolves via
# HA, so these fall back to direct Spotify Connect instead of a dead
# "I don't know which player" response.


def test_spotify_transport_fallback_skipped_when_spotify_not_configured():
    import tools.home_assistant as ha

    with patch.object(ha, "config", {}):
        assert ha._spotify_transport_fallback("pause") is None


def test_spotify_transport_fallback_none_without_spotify_credentials():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=None),
    ):
        assert ha._spotify_transport_fallback("pause") is None


def test_spotify_transport_fallback_pause_calls_spotify_connect():
    import tools.home_assistant as ha

    sp = MagicMock()
    with (
        patch.object(ha, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = ha._spotify_transport_fallback("pause")
        assert result == "Spotify paused"
        sp.pause_playback.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_resume_calls_spotify_connect():
    import tools.home_assistant as ha

    sp = MagicMock()
    with (
        patch.object(ha, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = ha._spotify_transport_fallback("resume")
        assert result == "Spotify resumed"
        sp.start_playback.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_skip_calls_spotify_connect():
    import tools.home_assistant as ha

    sp = MagicMock()
    with (
        patch.object(ha, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = ha._spotify_transport_fallback("skip")
        assert result == "Skipped to the next track on Spotify"
        sp.next_track.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_previous_calls_spotify_connect():
    import tools.home_assistant as ha

    sp = MagicMock()
    with (
        patch.object(ha, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = ha._spotify_transport_fallback("previous")
        assert result == "Back a track on Spotify"
        sp.previous_track.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_returns_friendly_error_on_failure():
    import tools.home_assistant as ha

    sp = MagicMock()
    sp.pause_playback.side_effect = Exception("no active device")
    with (
        patch.object(ha, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value=None),
    ):
        result = ha._spotify_transport_fallback("pause")
        assert result == "Couldn't control Spotify — no active device found."


def test_pause_falls_back_to_spotify_when_no_ha_target_resolves():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "SPOTIFY_ENTITY", None),
        patch.object(ha, "AVR_ENTITY", None),
        patch.object(ha, "TV_ENTITY", None),
        patch.object(ha, "_spotify_transport_fallback", return_value="Spotify paused") as fallback,
    ):
        assert ha.pause() == "Spotify paused"
        fallback.assert_called_once_with("pause")


def test_pause_keeps_original_message_when_spotify_fallback_unavailable():
    import tools.home_assistant as ha

    with (
        patch.object(ha, "SPOTIFY_ENTITY", None),
        patch.object(ha, "AVR_ENTITY", None),
        patch.object(ha, "TV_ENTITY", None),
        patch.object(ha, "_spotify_transport_fallback", return_value=None),
    ):
        assert ha.pause() == "I don't know which player to pause."


# --- HA @tool registration gated on "home_assistant" being configured ---
# Guards the leak where tools/spotify.py's module-level `import
# tools.home_assistant` (needed for its area-resolution helpers) used to
# register every HA tool into the SLM's tool registry even when
# home_assistant wasn't configured at all.


def test_ha_tool_decorator_skips_registration_when_not_configured():
    import tools.home_assistant as ha
    from tools.tool_registry import tool_registry

    probe_name = "_test_probe_unconfigured"
    with patch.object(ha, "config", {}):

        @ha.tool(name=probe_name)
        def probe():
            return "ok"

    try:
        assert probe_name not in tool_registry._tools
        assert probe_name not in tool_registry._schemas
        assert probe() == "ok"  # still a fully working plain function
    finally:
        tool_registry._tools.pop(probe_name, None)
        tool_registry._schemas.pop(probe_name, None)


def test_ha_tool_decorator_registers_when_configured():
    import tools.home_assistant as ha
    from tools.tool_registry import tool_registry

    probe_name = "_test_probe_configured"
    with patch.object(ha, "config", {"home_assistant": {}}):

        @ha.tool(name=probe_name)
        def probe():
            return "ok"

    try:
        assert probe_name in tool_registry._tools
        assert probe_name in tool_registry._schemas
    finally:
        tool_registry._tools.pop(probe_name, None)
        tool_registry._schemas.pop(probe_name, None)
