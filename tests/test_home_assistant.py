"""Logic-layer tests for tools/home_assistant.py.

The HTTP wrappers themselves aren't tested (matches the repo pattern of
not unit-testing thin REST wrappers). What is tested: fallback chains,
friendly-name role resolution, and date windowing. Temperature is now
passed through to HA without clamping — HA's per-entity min_temp /
max_temp attributes enforce safe bounds.
"""

import datetime
from unittest.mock import patch


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
            {"result": {"playlists": [{"uri": "spotify:playlist:abc", "name": "Calm Evening"}]}},
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
            {"result": {"playlists": []}},
            {"result": {"tracks": [{"uri": "spotify:track:xyz", "name": "Wagon Wheel"}]}},
        ]
        from tools.home_assistant import play_song
        play_song("wagon wheel")
        assert search.call_count == 2
        assert search.call_args_list[1].args[1] == "search_tracks"
        play.assert_called_once()
        assert play.call_args.args[3]["media_content_id"] == "spotify:track:xyz"
        assert play.call_args.args[3]["media_content_type"] == "music"


def test_play_song_polite_failure_when_no_results():
    """No playlists and no tracks → polite failure, no play_media call."""
    with patch("tools.home_assistant.SPOTIFY_ENTITY", "media_player.spotify"), \
         patch("tools.home_assistant._resolve_entity", return_value="media_player.spotify"), \
         patch("tools.home_assistant._call_service_with_response") as search, \
         patch("tools.home_assistant._call_service") as play:
        search.side_effect = [
            {"result": {"playlists": []}},
            {"result": {"tracks": []}},
        ]
        from tools.home_assistant import play_song
        result = play_song("nothing matches this")
        play.assert_not_called()
        assert "couldn't" in result.lower() or "no" in result.lower()


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
    """Helper: patch the module-level _ENTITY_ALIASES with a fresh dict."""
    return patch("tools.home_assistant._ENTITY_ALIASES", aliases)


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
