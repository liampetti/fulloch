"""A bare "turn off the lights" (no room named, no "all"/"every" qualifier)
from a satellite with a configured `ha_area` resolves to that room's lights
specifically — deliberately scoped to lights only.
Explicit room names and "all"/"every" always bypass this and use the
pre-existing resolution unchanged.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestIsBareLightPhrase:
    @pytest.mark.parametrize(
        "phrase",
        ["lights", "light", "lamps", "lamp"],
    )
    def test_bare_light_phrases(self, phrase):
        import tools.home_assistant as ha

        # _is_bare_light_phrase takes the already filler-stripped key (the
        # stripping happens in _bare_light_area_entities before calling it,
        # same as _resolve_entity/_resolve_area's own filler-stripping) —
        # "the lights" is covered as an integration case below instead.
        assert ha._is_bare_light_phrase(phrase) is True

    @pytest.mark.parametrize(
        "phrase",
        [
            "kitchen lights",
            "downstairs office lamp",
            "all lights",
            "all the lights",
            "every light",
            "everything",
            "",
            "the door",
            "fan",
        ],
    )
    def test_not_bare_light_phrases(self, phrase):
        import tools.home_assistant as ha

        assert ha._is_bare_light_phrase(phrase) is False


class TestCurrentSatelliteHaArea:
    def test_no_live_assistant_returns_none(self):
        import tools.home_assistant as ha
        from core.satellite_context import set_current_assistant

        set_current_assistant(None)
        assert ha._current_satellite_ha_area() is None

    def test_no_satellite_id_set_returns_none(self):
        import tools.home_assistant as ha
        from core.satellite_context import set_current_assistant

        fake_assistant = MagicMock()
        fake_assistant.satellites = {}
        set_current_assistant(fake_assistant)
        try:
            assert ha._current_satellite_ha_area() is None
        finally:
            set_current_assistant(None)

    def test_returns_the_calling_satellites_area(self):
        import tools.home_assistant as ha
        from core.satellite import SatelliteSession
        from core.satellite_context import current_satellite_id, set_current_assistant

        fake_assistant = MagicMock()
        fake_assistant.satellites = {
            "sat-a": SatelliteSession(id="sat-a", chunk_q=None, ha_area="kitchen")
        }
        set_current_assistant(fake_assistant)
        token = current_satellite_id.set("sat-a")
        try:
            assert ha._current_satellite_ha_area() == "kitchen"
        finally:
            current_satellite_id.reset(token)
            set_current_assistant(None)


@pytest.fixture
def satellite_with_area():
    """Registers a fake live assistant + satellite context with ha_area set
    to 'kitchen', for the duration of the test."""
    from core.satellite import SatelliteSession
    from core.satellite_context import current_satellite_id, set_current_assistant

    fake_assistant = MagicMock()
    fake_assistant.satellites = {
        "sat-a": SatelliteSession(id="sat-a", chunk_q=None, ha_area="kitchen")
    }
    set_current_assistant(fake_assistant)
    token = current_satellite_id.set("sat-a")
    try:
        yield fake_assistant
    finally:
        current_satellite_id.reset(token)
        set_current_assistant(None)


class TestBareLightAreaEntities:
    def test_resolves_area_lights_excluding_denied(self, satellite_with_area):
        import tools.home_assistant as ha

        entities = ["light.kitchen_ceiling", "light.kitchen_lamp", "switch.kitchen_kettle"]
        with (
            patch.object(ha, "_AREA_MAP", {"kitchen": "Kitchen"}),
            patch.object(ha, "_DENIED_ENTITIES", {"light.kitchen_lamp"}),
            patch.object(ha, "_render_template", return_value=json.dumps(entities)),
        ):
            result = ha._bare_light_area_entities("lights")

        assert result is not None
        light_ids, area_name = result
        assert light_ids == ["light.kitchen_ceiling"]  # lamp denied, switch not a light
        assert area_name == "Kitchen"

    def test_filler_words_are_stripped_before_matching(self, satellite_with_area):
        import tools.home_assistant as ha

        with (
            patch.object(ha, "_AREA_MAP", {"kitchen": "Kitchen"}),
            patch.object(ha, "_DENIED_ENTITIES", set()),
            patch.object(
                ha, "_render_template", return_value=json.dumps(["light.kitchen_ceiling"])
            ),
        ):
            result = ha._bare_light_area_entities("the lights")

        assert result == (["light.kitchen_ceiling"], "Kitchen")

    def test_explicit_room_bypasses_the_fallback(self, satellite_with_area):
        import tools.home_assistant as ha

        # "office lights" names a room explicitly — must not be scoped to
        # the satellite's own "kitchen" area.
        assert ha._bare_light_area_entities("office lights") is None

    def test_all_qualifier_bypasses_the_fallback(self, satellite_with_area):
        import tools.home_assistant as ha

        assert ha._bare_light_area_entities("all the lights") is None

    def test_no_satellite_area_configured_returns_none(self):
        import tools.home_assistant as ha
        from core.satellite_context import set_current_assistant

        set_current_assistant(None)
        assert ha._bare_light_area_entities("lights") is None

    def test_area_with_no_lights_returns_none(self, satellite_with_area):
        import tools.home_assistant as ha

        with (
            patch.object(ha, "_AREA_MAP", {"kitchen": "Kitchen"}),
            patch.object(ha, "_render_template", return_value=json.dumps(["switch.kettle"])),
        ):
            assert ha._bare_light_area_entities("lights") is None

    def test_ha_unreachable_returns_none(self, satellite_with_area):
        import tools.home_assistant as ha

        with (
            patch.object(ha, "_AREA_MAP", {"kitchen": "Kitchen"}),
            patch.object(ha, "_render_template", return_value=None),
        ):
            assert ha._bare_light_area_entities("lights") is None


class TestToolsUseAreaDefault:
    def test_turn_off_targets_every_light_in_the_area(self, satellite_with_area):
        import tools.home_assistant as ha

        with (
            patch.object(ha, "HA_TOKEN", "tok"),
            patch.object(ha, "_loaded", True),
            patch.object(ha, "_AREA_MAP", {"kitchen": "Kitchen"}),
            patch.object(
                ha,
                "_render_template",
                return_value=json.dumps(["light.kitchen_ceiling", "light.kitchen_lamp"]),
            ),
            patch.object(ha, "_DENIED_ENTITIES", set()),
            patch("tools.home_assistant.requests.post") as post,
        ):
            post.return_value = MagicMock(status_code=200)
            result = ha.turn_off("lights")

        assert "Kitchen" in result
        payload = post.call_args.kwargs["json"]
        assert set(payload["entity_id"]) == {"light.kitchen_ceiling", "light.kitchen_lamp"}

    def test_turn_off_explicit_room_uses_normal_resolution(self, satellite_with_area):
        import tools.home_assistant as ha

        with (
            patch.object(ha, "HA_TOKEN", "tok"),
            patch.object(ha, "_loaded", True),
            patch.object(ha, "_ENTITY_ALIASES", {"office lights": "light.office"}),
            patch("tools.home_assistant.requests.post") as post,
        ):
            post.return_value = MagicMock(status_code=200)
            ha.turn_off("office lights")

        payload = post.call_args.kwargs["json"]
        assert payload["entity_id"] == "light.office"

    def test_set_brightness_applies_to_every_light_in_the_area(self, satellite_with_area):
        import tools.home_assistant as ha

        with (
            patch.object(ha, "HA_TOKEN", "tok"),
            patch.object(ha, "_loaded", True),
            patch.object(ha, "_AREA_MAP", {"kitchen": "Kitchen"}),
            patch.object(
                ha, "_render_template", return_value=json.dumps(["light.kitchen_ceiling"])
            ),
            patch.object(ha, "_DENIED_ENTITIES", set()),
            patch("tools.home_assistant.requests.post") as post,
        ):
            post.return_value = MagicMock(status_code=200)
            result = ha.set_ha_brightness("lights", 40)

        assert "40 percent" in result
        payload = post.call_args.kwargs["json"]
        assert payload["entity_id"] == ["light.kitchen_ceiling"]
        assert payload["brightness"] == int(40 / 100 * 255)
