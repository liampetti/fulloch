"""HA entry-point composition; domain behavior lives in responsibility modules."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("configured", [False, True], ids=["spotify-only", "ha-and-spotify"])
def test_fresh_import_is_network_free_and_registration_is_gated(tmp_path, configured):
    config_path = tmp_path / "config.yml"
    config_path.write_text("spotify: {}\n" + ("home_assistant: {}\n" if configured else ""))
    code = """
import json
from unittest.mock import patch

with patch("requests.sessions.Session.request") as request:
    import tools.home_assistant as ha
    from tools import ha_calendar, ha_client, ha_devices, ha_media, ha_state, spotify
    from tools.tool_registry import tool_registry
    request.assert_not_called()

assert ha_client.HA_TOKEN == "import-test-token"
assert ha_client._loaded is False
assert ha_calendar.client is ha_devices.client is ha_media.client is ha_state.client is spotify.ha is ha_client
assert ha.turn_on is ha_devices.turn_on
assert ha.get_entity_history is ha_state.get_entity_history
assert ha.get_upcoming_events is ha_calendar.get_upcoming_events
assert ha._reminder_calendar_entity is ha_calendar._reminder_calendar_entity
assert ha.HA_CONFIG is ha_client.HA_CONFIG
assert ha.set_entity_denied is ha_client.set_entity_denied
registered = {name: fn for name, fn in tool_registry._tools.items() if fn.__module__.startswith("tools.ha_")}
for fn in registered.values():
    assert getattr(ha, fn.__name__) is fn
if registered:
    assert tool_registry.get_tool("stop") is ha.pause
    assert tool_registry.get_tool("weather") is ha.get_weather_forecast
print(json.dumps(sorted(registered)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "FULLOCH_CONFIG_PATH": str(config_path),
            "FULLOCH_DENYLIST_PATH": str(tmp_path / "denylist.json"),
            "HA_TOKEN": "import-test-token",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    expected = {
        "turn_on",
        "turn_off",
        "toggle",
        "ha_set_brightness",
        "ha_set_color",
        "ha_volume_set",
        "ha_volume_up",
        "ha_volume_down",
        "ha_select_source",
        "pause",
        "resume",
        "skip",
        "previous",
        "ha_mute",
        "get_temperature",
        "get_home_overview",
        "get_energy_overview",
        "get_security_overview",
        "get_entity_state",
        "list_entities_in_area",
        "get_entities_in_area_state",
        "ha_service",
        "ha_set_climate",
        "ha_lock",
        "ha_unlock",
        "ha_open_cover",
        "ha_close_cover",
        "ha_stop_cover",
        "ha_set_cover_position",
        "ha_set_fan_speed",
        "ha_vacuum",
        "ha_run_script",
        "ha_activate_scene",
        "whats_on",
        "find_calendar_event",
        "create_calendar_event",
        "add_todo_item",
        "get_todo_items",
        "complete_todo_item",
        "get_weather_forecast",
        "get_entity_history",
        "get_conversation_history",
    }
    assert set(json.loads(result.stdout)) == (expected if configured else set())
