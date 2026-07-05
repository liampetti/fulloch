"""Spotify music control via the direct Spotify Web API (spotipy).

Loaded when the `spotify:` block is present in config.yml. Supersedes HA's
`play_song` (tools/home_assistant.py), which goes through the SpotifyPlus
HACS integration and has unreliable search — this talks to Spotify directly.
HA still owns `pause`/`resume`/`skip` (also routes to AVR/TV), so this module
only registers `play_song`.

Auth is a one-time manual step (no in-app OAuth callback): create a Spotify
app at https://developer.spotify.com/dashboard, then run
`scripts/spotify_auth.py` once on a machine with a browser — it walks through
the OAuth consent flow and writes `spotify_client_id` / `spotify_client_secret`
/ `spotify_refresh_token` into data/credentials.json. The access token is
refreshed from that on each use, so no browser round-trip happens at runtime.
"""
import difflib
import logging
import re
import time
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from server.credentials_store import get_credential

from ._config import config
from .tool_registry import tool

logger = logging.getLogger(__name__)

SPOTIFY_CONFIG = config.get("spotify", {})
REDIRECT_URI = SPOTIFY_CONFIG.get("redirect_uri", "http://127.0.0.1:8080/callback")
DEVICE_NAME = SPOTIFY_CONFIG.get("device_id")

SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
SIMILARITY_THRESHOLD = 0.6  # How similar a user query is to a playlist, track or album

_client: Optional[spotipy.Spotify] = None
_client_expiry: float = 0.0


def _get_client() -> Optional[spotipy.Spotify]:
    """Return a cached Spotify client, refreshing the access token as needed.

    Returns None if credentials aren't configured — never raises, so a
    missing/invalid setup degrades to a spoken error instead of a crash.
    """
    global _client, _client_expiry
    if _client is not None and time.time() < _client_expiry:
        return _client
    client_id = get_credential("spotify_client_id").strip()
    client_secret = get_credential("spotify_client_secret").strip()
    refresh_token = get_credential("spotify_refresh_token").strip()
    if not (client_id and client_secret and refresh_token):
        return None
    try:
        oauth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            open_browser=False,
        )
        token_info = oauth.refresh_access_token(refresh_token)
        _client = spotipy.Spotify(auth=token_info["access_token"])
        _client_expiry = time.time() + token_info.get("expires_in", 3600) - 60
        return _client
    except Exception:
        logger.exception("Spotify token refresh failed")
        _client = None
        return None


def _get_active_device(sp: spotipy.Spotify) -> Optional[str]:
    """Return the configured device's id, or the first available device."""
    devices = sp.devices().get("devices", [])
    if not devices:
        return None
    if DEVICE_NAME:
        for device in devices:
            if device.get("name") == DEVICE_NAME:
                return device["id"]
    return devices[0]["id"]


def _pause(sp: spotipy.Spotify) -> None:
    """Pause playback if currently playing (best-effort, ignores errors)."""
    try:
        playback = sp.current_playback()
        if playback and playback.get("is_playing"):
            sp.pause_playback()
    except Exception:
        logger.exception("Spotify pause-before-play failed")


@tool(
    name="play_song",
    description="Play a song by artist and title, or search for a song by query",
    aliases=["play", "play_music", "start_music"],
)
def play_song(artist_query: Optional[str] = None, song: Optional[str] = None) -> str:
    """Play a song or playlist on Spotify, prioritizing the user's playlist
    names, then songs in playlists, then a general search.

    Args:
        artist_query: Artist name, playlist name, or search query
        song: Song title (if two arguments are provided, one is artist and one is song title)
    """
    sp = _get_client()
    if sp is None:
        return (
            "Reactive question: Spotify isn't set up yet. Add spotify_client_id, "
            "spotify_client_secret and spotify_refresh_token to data/credentials.json."
        )

    try:
        if artist_query and re.search(r"\s+by\s+", artist_query):
            parts = artist_query.split(" by ")
            if len(parts) == 2:
                artist_query, song = parts[1].strip(), parts[0].strip()

        if (re.sub(r"[^A-Za-z]+", "", str(artist_query).lower()) == "music") or (artist_query is None):
            _pause(sp)
            sp.start_playback(device_id=_get_active_device(sp))
            return "Playing music on spotify"

        playlists = sp.current_user_playlists(limit=50)["items"]

        # 1. Check if query closely matches a playlist name.
        playlist_names = [pl["name"] for pl in playlists]
        matches = difflib.get_close_matches(artist_query, playlist_names, n=1, cutoff=SIMILARITY_THRESHOLD)
        if matches:
            for playlist in playlists:
                if playlist["name"] == matches[0]:
                    _pause(sp)
                    sp.start_playback(device_id=_get_active_device(sp), context_uri=playlist["uri"])
                    return f'Playing your playlist "{playlist["name"]}"'

        # 2. Search the user's top five playlists for the track.
        for playlist in playlists[:5]:
            results = sp.playlist_tracks(playlist["id"])
            for item in results["items"]:
                track = item["track"]
                if (song and song.lower() in track["name"].lower()) or (
                    artist_query and artist_query.lower() in track["artists"][0]["name"].lower()
                ):
                    _pause(sp)
                    sp.start_playback(device_id=_get_active_device(sp), uris=[track["uri"]])
                    return f'Playing {track["name"]} by {track["artists"][0]["name"]} from your playlist "{playlist["name"]}"'

        # 3. Fall back to a general search.
        if artist_query and song:
            results = sp.search(q=f"artist:{artist_query} track:{song}", type="track", limit=1)
        elif artist_query:
            results = sp.search(q=artist_query, type="track", limit=1)
        else:
            return "Please provide either artist and song, playlist name, or a search query"

        tracks = results.get("tracks", {}).get("items", [])
        uris = [track["uri"] for track in tracks if "uri" in track]
        if not uris:
            _pause(sp)
            sp.start_playback(device_id=_get_active_device(sp))
            return "No tracks found, starting playback"

        _pause(sp)
        sp.start_playback(device_id=_get_active_device(sp), uris=uris)
        track = tracks[0]
        return f'Playing {track["name"]} by {track["artists"][0]["name"]}'
    except Exception:
        logger.exception("Spotify play_song failed")
        return "Unable to play your request on Spotify"
