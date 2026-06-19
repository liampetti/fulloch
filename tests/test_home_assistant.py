"""Logic-layer tests for tools/home_assistant.py.

The HTTP wrappers themselves aren't tested (matches the repo pattern of
not unit-testing thin REST wrappers). What is tested: fallback chains,
friendly-name role resolution, and date windowing. Temperature is now
passed through to HA without clamping — HA's per-entity min_temp /
max_temp attributes enforce safe bounds.
"""

import datetime
from unittest.mock import MagicMock, patch


def test_set_climate_passes_temperature_through():
    """No application-level clamp — HA enforces its own min/max bounds."""
    with patch("tools.home_assistant._resolve_entity", return_value="climate.office"), \
         patch("tools.home_assistant._call_service") as call:
        from tools.home_assistant import set_climate
        set_climate("office", 21)
        sent = call.call_args.args[3]
        assert sent["temperature"] == 21


def test_play_song_picks_playlist_when_playlists_match():
    """First call: search_playlists returns a result → playlist context plays."""
    with patch("tools.home_assistant.SPOTIFY_ENTITY", "media_player.spotify"), \
         patch("tools.home_assistant._resolve_entity", return_value="media_player.spotify"), \
         patch("tools.home_assistant._call_service_with_response") as search, \
         patch("tools.home_assistant._call_service") as play:
        search.side_effect = [
            {"result": {"items": [{"uri": "spotify:playlist:abc", "name": "Calm Evening"}]}},
        ]
        from tools.home_assistant import play_song
        play_song("calm evening")
        # Only one search call (playlists), not a tracks call
        assert search.call_count == 1
        assert search.call_args_list[0].args[1] == "search_playlists"
        # play_media was called with the playlist URI
        play.assert_called_once()
        args = play.call_args.args
        assert args[1] == "play_media"
        assert args[3]["media_content_id"] == "spotify:playlist:abc"
        assert args[3]["media_content_type"] == "playlist"


def test_play_song_falls_through_to_track_when_no_playlists():
    """Playlists empty → tracks searched → track URI played."""
    with patch("tools.home_assistant.SPOTIFY_ENTITY", "media_player.spotify"), \
         patch("tools.home_assistant._resolve_entity", return_value="media_player.spotify"), \
         patch("tools.home_assistant._call_service_with_response") as search, \
         patch("tools.home_assistant._call_service") as play:
        search.side_effect = [
            {"result": {"items": []}},
            {"result": {"items": [{"uri": "spotify:track:xyz", "name": "Wagon Wheel"}]}},
        ]
        from tools.home_assistant import play_song
        play_song("wagon wheel")
        assert search.call_count == 2
        assert search.call_args_list[1].args[1] == "search_tracks"
        play.assert_called_once()
        assert play.call_args.args[3]["media_content_id"] == "spotify:track:xyz"
        assert play.call_args.args[3]["media_content_type"] == "music"


def test_play_song_generic_fallback_when_no_results():
    """No SpotifyPlus results → generic media_player.play_media fallback."""
    with patch("tools.home_assistant.SPOTIFY_ENTITY", "media_player.spotify"), \
         patch("tools.home_assistant._resolve_entity", return_value="media_player.spotify"), \
         patch("tools.home_assistant._call_service_with_response") as search, \
         patch("tools.home_assistant._call_service") as play:
        search.side_effect = [
            {"result": {"items": []}},
            {"result": {"items": []}},
        ]
        from tools.home_assistant import play_song
        play_song("nothing matches this")
        play.assert_called_once()
        assert play.call_args.args[3]["media_content_id"] == "spotify:search:nothing matches this"


def test_play_song_returns_friendly_error_when_spotify_entity_unset():
    with patch("tools.home_assistant.SPOTIFY_ENTITY", None):
        from tools.home_assistant import play_song
        result = play_song("anything")
        assert "spotify" in result.lower()


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
        "calendar.primary": {"events": [
            {"start": "2026-06-26T09:00:00", "summary": "Standup"},
        ]},
        "calendar.fulloch": {"events": [
            {"start": "2026-06-26T12:00:00", "summary": "Australia vs Paraguay"},
        ]},
    }
    with patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"), \
         patch.object(ha, "_reminder_calendar_entity", return_value="calendar.fulloch"), \
         patch.object(ha, "_call_service_with_response", return_value=response) as call:
        out = ha._ha_get_events("2026-06-26")

    # Both calendars were queried in one call.
    assert call.call_args.args[2]["entity_id"] == ["calendar.primary", "calendar.fulloch"]
    # The reminder-calendar event is present in the spoken summary.
    assert "Australia vs Paraguay" in out
    assert "Standup" in out


def test_whats_on_dedupes_when_read_and_reminder_calendars_match():
    import tools.home_assistant as ha
    response = {"calendar.primary": {"events": []}}
    with patch.object(ha, "CALENDAR_ENTITY", "calendar.primary"), \
         patch.object(ha, "_reminder_calendar_entity", return_value="calendar.primary"), \
         patch.object(ha, "_call_service_with_response", return_value=response) as call:
        ha._ha_get_events("today")
    assert call.call_args.args[2]["entity_id"] == ["calendar.primary"]


def test_calendar_normalises_timed_event():
    from tools.home_assistant import _normalise_ha_event
    ha_event = {"start": "2026-05-25T10:00:00+10:00", "end": "...", "summary": "Dentist"}
    norm = _normalise_ha_event(ha_event)
    assert norm == {"start": "2026-05-25T10:00:00+10:00", "summary": "Dentist", "all_day": False}


def test_calendar_normalises_all_day_event():
    from tools.home_assistant import _normalise_ha_event
    ha_event = {"start": "2026-05-26", "end": "2026-05-27", "summary": "Public holiday"}
    norm = _normalise_ha_event(ha_event)
    assert norm == {"start": "2026-05-26", "summary": "Public holiday", "all_day": True}


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

    with patch("tools.home_assistant._ENTITY_ALIASES", aliases), \
         patch("tools.home_assistant._ENTITY_ALIASES_MULTI", multi), \
         patch("tools.home_assistant._get_state", side_effect=fake_get_state):
        from tools.home_assistant import _resolve_entity, get_temperature
        # Variant resolver recovers the climate entity despite the light winning.
        assert _resolve_entity("upstairs", domain="climate") == "climate.living"
        result = get_temperature("upstairs")
        # Reads the climate temp, and speaks as "upstairs" (not the slug "living").
        assert "18" in result and "upstairs" in result.lower()
        assert "living" not in result.lower()


def test_resolve_entity_without_domain_keeps_first_wins():
    """No domain hint → first registration order entity (unchanged behaviour)."""
    with patch("tools.home_assistant._ENTITY_ALIASES_MULTI",
               {"upstairs": ["light.upstairs", "climate.living"]}):
        from tools.home_assistant import _resolve_entity
        assert _resolve_entity("upstairs") == "light.upstairs"


def test_autodetect_spotify_picks_media_player_spotify_prefix():
    with _patch_aliases({
        "kitchen speaker": "media_player.kitchen",
        "spotify": "media_player.spotify_alice",
    }), patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_spotify_entity
        assert _autodetect_spotify_entity() == "media_player.spotify_alice"


def test_autodetect_spotify_returns_none_with_no_match():
    with _patch_aliases({"kitchen speaker": "media_player.kitchen"}), \
         patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_spotify_entity
        assert _autodetect_spotify_entity() is None


def test_autodetect_tv_matches_underscore_token():
    with _patch_aliases({
        "living room tv": "media_player.living_room_tv",
        "spotify": "media_player.spotify_alice",
    }), patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_tv_entity
        assert _autodetect_tv_entity() == "media_player.living_room_tv"


def test_autodetect_tv_does_not_steal_spotify():
    """Even if no TV exists, the spotify entity must not be picked as TV."""
    with _patch_aliases({"spotify": "media_player.spotify_alice"}), \
         patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_tv_entity
        assert _autodetect_tv_entity() is None


def test_autodetect_avr_matches_pioneer_keyword():
    with _patch_aliases({
        "kitchen speaker": "media_player.kitchen",
        "pioneer avr": "media_player.pioneer_avr",
    }), patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_avr_entity
        assert _autodetect_avr_entity() == "media_player.pioneer_avr"


def test_autodetect_avr_matches_receiver_keyword_in_friendly_name():
    with _patch_aliases({"living room receiver": "media_player.lounge_av"}), \
         patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_avr_entity
        assert _autodetect_avr_entity() == "media_player.lounge_av"


def test_autodetect_calendar_prefers_primary():
    with _patch_aliases({
        "work": "calendar.work",
        "primary": "calendar.primary",
        "personal": "calendar.personal",
    }), patch("tools.home_assistant.HA_CONFIG", {}):
        from tools.home_assistant import _autodetect_calendar_entity
        assert _autodetect_calendar_entity() == "calendar.primary"


def test_autodetect_calendar_falls_back_to_first_when_no_primary():
    with _patch_aliases({"work": "calendar.work"}), \
         patch("tools.home_assistant.HA_CONFIG", {}):
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
    with patch.object(ha, "HA_TOKEN", "tok"), \
         patch.object(ha, "_DENIED_ENTITIES", frozenset({"lock.front_door"})), \
         patch("tools.home_assistant.requests.post") as post:
        result = ha._call_service("lock", "unlock", "lock.front_door")
        assert "voice control" in result.lower()
        post.assert_not_called()


def test_call_service_allows_non_denied_entity():
    """A normal entity still calls the service (deny-list doesn't over-block)."""
    import tools.home_assistant as ha
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with patch.object(ha, "HA_TOKEN", "tok"), \
         patch.object(ha, "_DENIED_ENTITIES", frozenset({"lock.front_door"})), \
         patch("tools.home_assistant.requests.post", return_value=resp) as post:
        ha._call_service("light", "turn_on", "light.kitchen", success_message="ok")
        post.assert_called_once()


def test_set_entity_denied_persists_and_takes_effect(tmp_path):
    """Toggling deny mutates the live set, persists JSON, and round-trips on load."""
    import tools.home_assistant as ha
    path = str(tmp_path / "voice_denylist.json")
    with patch.object(ha, "_DENYLIST_PATH", path), \
         patch.object(ha, "_DENIED_ENTITIES", frozenset()):
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
    with _patch_aliases({
        "kitchen": "light.kitchen",
        "front door": "lock.front_door",
    }), patch.object(ha, "_DENIED_ENTITIES", frozenset({"lock.front_door"})):
        entities = ha.list_entities()
    by_id = {e["entity_id"]: e for e in entities}
    assert by_id["lock.front_door"]["denied"] is True
    assert by_id["light.kitchen"]["denied"] is False
    assert by_id["light.kitchen"]["domain"] == "light"
    # Deny-listed entities stay listed so they can be re-enabled.
    assert "lock.front_door" in by_id
