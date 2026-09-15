"""HA devices behavior against production implementations."""

from unittest.mock import MagicMock, patch

from tests.ha_fixtures import loaded_ha  # noqa: F401
from tools import ha_client as client
from tools import ha_devices as devices


def test_set_climate_passes_temperature_through():
    """No application-level clamp — HA enforces its own min/max bounds."""
    with (
        patch("tools.ha_client._resolve_entity", return_value="climate.office"),
        patch("tools.ha_client._call_service") as call,
    ):
        from tools.ha_devices import set_climate

        set_climate("office", 21)
        sent = call.call_args.args[3]
        assert sent["temperature"] == 21


def test_open_cover_uses_cover_domain_for_cover_entity():

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch("tools.ha_client._resolve_entity", return_value="cover.garage"),
        patch("tools.ha_client.requests.post", return_value=resp) as post,
    ):
        devices.open_cover("garage")

    url = post.call_args.args[0]
    assert "/services/cover/open_cover" in url


def test_open_cover_uses_valve_domain_for_valve_entity():
    """A valve.* entity gets valve.open_valve, not cover.open_cover — same
    voice verb ("open the valve"), different HA domain/service."""

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch("tools.ha_client._resolve_entity", return_value="valve.main_water"),
        patch("tools.ha_client.requests.post", return_value=resp) as post,
    ):
        result = devices.open_cover("main water valve")

    url = post.call_args.args[0]
    assert "/services/valve/open_valve" in url
    assert "Opened" in result


def test_set_cover_position_clamps_and_targets_valve_service():

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch("tools.ha_client._resolve_entity", return_value="valve.main_water"),
        patch("tools.ha_client.requests.post", return_value=resp) as post,
    ):
        devices.set_cover_position("main water valve", 150)

    url = post.call_args.args[0]
    payload = post.call_args.kwargs["json"]
    assert "/services/valve/set_valve_position" in url
    assert payload["position"] == 100


def test_ha_vacuum_dispatches_known_action():

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch("tools.ha_client._resolve_entity", return_value="vacuum.roomba"),
        patch("tools.ha_client.requests.post", return_value=resp) as post,
    ):
        result = devices.ha_vacuum("roomba", "dock")

    url = post.call_args.args[0]
    assert "/services/vacuum/return_to_base" in url
    assert "dock" in result.lower()


def test_ha_vacuum_rejects_unknown_action():

    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch("tools.ha_client._resolve_entity", return_value="vacuum.roomba"),
        patch("tools.ha_client.requests.post") as post,
    ):
        result = devices.ha_vacuum("roomba", "levitate")

    assert "don't know how to" in result.lower()
    post.assert_not_called()
