"""HA client behavior against production implementations."""

from unittest.mock import MagicMock, patch

from tests.ha_fixtures import loaded_ha  # noqa: F401
from tools import ha_client as client


def _patch_aliases(aliases: dict):
    """Helper: patch the module-level alias maps with a fresh dict.

    Patches both the first-wins single map and the collision multimap (derived
    one-entity-per-name) so resolution paths that consult either stay in sync.
    """
    multi = {k: [v] for k, v in aliases.items()}
    return patch.multiple(
        "tools.ha_client",
        _ENTITY_ALIASES=aliases,
        _ENTITY_ALIASES_MULTI=multi,
    )


def test_load_is_lazy_and_one_shot(monkeypatch):
    """The real loader populates its cache only once, on first use."""

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
        monkeypatch.setattr(client, _name, _val)
    monkeypatch.setattr(client, "_fetch_entity_aliases", fake_fetch)
    monkeypatch.setattr(client, "_fetch_area_map", lambda: {})
    monkeypatch.setattr(client, "_fetch_floor_map", lambda: {})

    assert calls["n"] == 0  # nothing fetched yet
    client._ensure_loaded()
    assert calls["n"] == 1  # first use loads
    assert client._ENTITY_ALIASES == {"kitchen": "light.kitchen"}
    client._ensure_loaded()
    assert calls["n"] == 1  # idempotent — no refetch


def test_ensure_loaded_is_noop_without_token(monkeypatch):
    """No token → nothing to fetch, and patched globals are left untouched."""

    monkeypatch.setattr(client, "_loaded", False)
    monkeypatch.setattr(client, "HA_TOKEN", "")
    monkeypatch.setattr(client, "SPOTIFY_ENTITY", "media_player.spotify")
    called = {"n": 0}
    monkeypatch.setattr(
        client,
        "_fetch_entity_aliases",
        lambda: called.__setitem__("n", called["n"] + 1) or ({}, {}),
    )

    client._ensure_loaded()
    assert called["n"] == 0  # never fetched
    assert client.SPOTIFY_ENTITY == "media_player.spotify"  # patch not clobbered


def test_resolve_entity_without_domain_keeps_first_wins():
    """No domain hint → first registration order entity (unchanged behaviour)."""
    with patch(
        "tools.ha_client._ENTITY_ALIASES_MULTI",
        {"upstairs": ["light.upstairs", "climate.living"]},
    ):
        from tools.ha_client import _resolve_entity

        assert _resolve_entity("upstairs") == "light.upstairs"


def test_autodetect_spotify_uses_configured_entity():
    """No autodetection — the configured friendly name resolves via the alias map."""
    with (
        _patch_aliases({"sonos living room": "media_player.sonos_living_room"}),
        patch("tools.ha_client.HA_CONFIG", {"spotify_entity": "Sonos Living Room"}),
    ):
        from tools.ha_client import _autodetect_spotify_entity

        assert _autodetect_spotify_entity() == "media_player.sonos_living_room"


def test_autodetect_spotify_returns_none_with_no_match():
    with (
        _patch_aliases({"kitchen speaker": "media_player.kitchen"}),
        patch("tools.ha_client.HA_CONFIG", {}),
    ):
        from tools.ha_client import _autodetect_spotify_entity

        assert _autodetect_spotify_entity() is None


def test_autodetect_tv_matches_underscore_token():
    with (
        _patch_aliases(
            {
                "living room tv": "media_player.living_room_tv",
                "spotify": "media_player.spotify_alice",
            }
        ),
        patch("tools.ha_client.HA_CONFIG", {}),
    ):
        from tools.ha_client import _autodetect_tv_entity

        assert _autodetect_tv_entity() == "media_player.living_room_tv"


def test_autodetect_tv_does_not_steal_spotify():
    """Even if no TV exists, the spotify entity must not be picked as TV."""
    with (
        _patch_aliases({"spotify": "media_player.spotify_alice"}),
        patch("tools.ha_client.HA_CONFIG", {}),
    ):
        from tools.ha_client import _autodetect_tv_entity

        assert _autodetect_tv_entity() is None


def test_autodetect_avr_matches_pioneer_keyword():
    with (
        _patch_aliases(
            {
                "kitchen speaker": "media_player.kitchen",
                "pioneer avr": "media_player.pioneer_avr",
            }
        ),
        patch("tools.ha_client.HA_CONFIG", {}),
    ):
        from tools.ha_client import _autodetect_avr_entity

        assert _autodetect_avr_entity() == "media_player.pioneer_avr"


def test_autodetect_avr_matches_receiver_keyword_in_friendly_name():
    with (
        _patch_aliases({"living room receiver": "media_player.lounge_av"}),
        patch("tools.ha_client.HA_CONFIG", {}),
    ):
        from tools.ha_client import _autodetect_avr_entity

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
        patch.object(client, "HA_CONFIG", {}),
    ):
        assert client._autodetect_calendar_entity() == "calendar.primary"


def test_autodetect_calendar_falls_back_to_first_when_no_primary():
    with _patch_aliases({"work": "calendar.work"}), patch.object(client, "HA_CONFIG", {}):
        assert client._autodetect_calendar_entity() == "calendar.work"


def test_call_service_refuses_denied_entity():
    """A deny-listed entity_id is refused before any HTTP call."""

    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_DENIED_ENTITIES", frozenset({"lock.front_door"})),
        patch("tools.ha_client.requests.post") as post,
    ):
        result = client._call_service("lock", "unlock", "lock.front_door")
        assert "voice control" in result.lower()
        post.assert_not_called()


def test_call_service_allows_non_denied_entity():
    """A normal entity still calls the service (deny-list doesn't over-block)."""

    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_DENIED_ENTITIES", frozenset({"lock.front_door"})),
        patch.object(client, "_get_state", return_value={"entity_id": "light.kitchen"}),
        patch("tools.ha_client.requests.post", return_value=resp) as post,
    ):
        client._call_service("light", "turn_on", "light.kitchen", success_message="ok")
        post.assert_called_once()


def test_call_service_does_not_send_unverified_target():
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_ENTITY_ALIASES", {}),
        patch.object(client, "_ENTITY_ALIASES_MULTI", {}),
        patch.object(client, "_get_state", return_value=None),
        patch.object(client, "_post") as post,
    ):
        result = client._call_service("cover", "open_cover", "cover.imaginary")
    assert result.startswith("Reactive question:")
    assert "No command was sent" in result
    post.assert_not_called()


def test_known_target_requires_no_preliminary_state_request():
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_ENTITY_ALIASES", {"lamp": "light.lamp"}),
        patch.object(client, "_get_state") as state,
        patch.object(client, "_post", return_value=MagicMock()) as post,
    ):
        assert client._call_service("light", "turn_on", "light.lamp") == "OK"
    state.assert_not_called()
    post.assert_called_once()


def test_service_data_cannot_replace_validated_target():
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "_post") as post,
    ):
        result = client._call_service(
            "light", "turn_on", "light.lamp", {"entity_id": "light.imaginary"}
        )
    assert result.startswith("Reactive question:")
    post.assert_not_called()


def test_set_entity_denied_persists_and_takes_effect(tmp_path):
    """Toggling deny mutates the live set, persists JSON, and round-trips on load."""

    path = str(tmp_path / "voice_denylist.json")
    with (
        patch.object(client, "_DENYLIST_PATH", path),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
    ):
        client.set_entity_denied("lock.front_door", True)
        # Live: the in-memory set updated immediately, no restart.
        assert "lock.front_door" in client.get_denylist()
        # Persisted: the JSON file reflects it and reloads identically.
        assert client._load_denylist() == frozenset({"lock.front_door"})
        # Toggling back off removes it.
        client.set_entity_denied("lock.front_door", False)
        assert "lock.front_door" not in client.get_denylist()
        assert client._load_denylist() == frozenset()


def test_load_denylist_missing_file_is_empty(tmp_path):
    """No persisted file → nothing blocked (feature is a no-op until used)."""

    with patch.object(client, "_DENYLIST_PATH", str(tmp_path / "absent.json")):
        assert client._load_denylist() == frozenset()


def test_load_denylist_ignores_malformed(tmp_path):
    """A corrupt deny-list file fails safe to empty rather than crashing."""

    path = tmp_path / "voice_denylist.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with patch.object(client, "_DENYLIST_PATH", str(path)):
        assert client._load_denylist() == frozenset()


def test_list_entities_reports_deny_state():
    """list_entities surfaces every entity with its allow/deny flag, sorted."""

    with (
        _patch_aliases(
            {
                "kitchen": "light.kitchen",
                "front door": "lock.front_door",
            }
        ),
        patch.object(client, "_DENIED_ENTITIES", frozenset({"lock.front_door"})),
    ):
        entities = client.list_entities()
    by_id = {e["entity_id"]: e for e in entities}
    assert by_id["lock.front_door"]["denied"] is True
    assert by_id["light.kitchen"]["denied"] is False
    assert by_id["light.kitchen"]["domain"] == "light"
    # Deny-listed entities stay listed so they can be re-enabled.
    assert "lock.front_door" in by_id


def test_resolve_area_matches_by_display_name():

    with patch.object(client, "_AREA_MAP", {"downstairs": "Downstairs", "office": "Office"}):
        assert client._resolve_area("downstairs") == "downstairs"
        assert client._resolve_area("the office") == "office"
        assert client._resolve_area("upstairs") is None


def test_floor_name_does_not_fuzzy_match_child_area():

    with (
        patch.object(client, "_AREA_MAP", {"upstairs_bathroom": "Upstairs Bathroom"}),
        patch.object(client, "_FLOOR_MAP", {"upstairs": "Upstairs"}),
    ):
        assert client._resolve_area("upstairs") is None
        assert client._resolve_floor("upstairs") == "upstairs"


def test_ha_tool_decorator_skips_registration_when_not_configured():
    from tools.tool_registry import tool_registry

    probe_name = "_test_probe_unconfigured"
    with patch.object(client, "config", {}):

        @client.tool(name=probe_name)
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
    from tools.tool_registry import tool_registry

    probe_name = "_test_probe_configured"
    with patch.object(client, "config", {"home_assistant": {}}):

        @client.tool(name=probe_name)
        def probe():
            return "ok"

    try:
        assert probe_name in tool_registry._tools
        assert probe_name in tool_registry._schemas
    finally:
        tool_registry._tools.pop(probe_name, None)
        tool_registry._schemas.pop(probe_name, None)
