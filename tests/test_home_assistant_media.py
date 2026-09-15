"""HA media behavior against production implementations."""

from unittest.mock import MagicMock, patch

from tests.ha_fixtures import loaded_ha  # noqa: F401
from tools import ha_client as client
from tools import ha_media as media


def test_media_target_prefers_spotify_and_resolves_a_room_player():

    with (
        patch.object(client, "_loaded", True),
        patch.object(client, "SPOTIFY_ENTITY", "media_player.sonos"),
        patch.object(client, "AVR_ENTITY", "media_player.avr"),
        patch.object(client, "TV_ENTITY", "media_player.tv"),
        patch.object(client, "_resolve_area", return_value="living_room"),
        patch.object(
            client,
            "_area_entities",
            return_value=["media_player.tv", "media_player.sonos"],
        ),
    ):
        assert media._media_target(None) == "media_player.sonos"
        assert media._media_target("living room") == "media_player.sonos"
        assert media._media_target("living room", prefer_spotify=False) == "media_player.tv"


def test_spotify_transport_fallback_skipped_when_spotify_not_configured():

    with patch.object(client, "config", {}):
        assert media._spotify_transport_fallback("pause") is None


def test_spotify_transport_fallback_none_without_spotify_credentials():

    with (
        patch.object(client, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=None),
    ):
        assert media._spotify_transport_fallback("pause") is None


def test_spotify_transport_fallback_pause_calls_spotify_connect():

    sp = MagicMock()
    with (
        patch.object(client, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = media._spotify_transport_fallback("pause")
        assert result == "Spotify paused"
        sp.pause_playback.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_resume_calls_spotify_connect():

    sp = MagicMock()
    with (
        patch.object(client, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = media._spotify_transport_fallback("resume")
        assert result == "Spotify resumed"
        sp.start_playback.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_skip_calls_spotify_connect():

    sp = MagicMock()
    with (
        patch.object(client, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = media._spotify_transport_fallback("skip")
        assert result == "Skipped to the next track on Spotify"
        sp.next_track.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_previous_calls_spotify_connect():

    sp = MagicMock()
    with (
        patch.object(client, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value="device1"),
    ):
        result = media._spotify_transport_fallback("previous")
        assert result == "Back a track on Spotify"
        sp.previous_track.assert_called_once_with(device_id="device1")


def test_spotify_transport_fallback_returns_friendly_error_on_failure():

    sp = MagicMock()
    sp.pause_playback.side_effect = Exception("no active device")
    with (
        patch.object(client, "config", {"spotify": {}}),
        patch("tools.spotify._get_client", return_value=sp),
        patch("tools.spotify._get_active_device", return_value=None),
    ):
        result = media._spotify_transport_fallback("pause")
        assert result == "Couldn't control Spotify — no active device found."


def test_pause_falls_back_to_spotify_when_no_ha_target_resolves():

    with (
        patch.object(client, "SPOTIFY_ENTITY", None),
        patch.object(client, "AVR_ENTITY", None),
        patch.object(client, "TV_ENTITY", None),
        patch.object(
            media, "_spotify_transport_fallback", return_value="Spotify paused"
        ) as fallback,
    ):
        assert media.pause() == "Spotify paused"
        fallback.assert_called_once_with("pause")


def test_pause_keeps_original_message_when_spotify_fallback_unavailable():

    with (
        patch.object(client, "SPOTIFY_ENTITY", None),
        patch.object(client, "AVR_ENTITY", None),
        patch.object(client, "TV_ENTITY", None),
        patch.object(media, "_spotify_transport_fallback", return_value=None),
    ):
        assert media.pause() == "I don't know which player to pause."
