"""
Spotify music control tool 
"""
import os
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env vars

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

import asyncio
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import re
from typing import Optional
import difflib

from .pioneer_avr import setup_avr
from .tool_registry import tool, tool_registry

import logging

logger = logging.getLogger(__name__)

CREDS= config['spotify']

SCOPE = 'user-read-playback-state user-modify-playback-state user-read-currently-playing'

SIMILARITY_THRESHOLD = 0.6 # How similar a user query is to a playlist, track or album

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CREDS['client_id'],
    client_secret=CREDS['client_secret'],
    redirect_uri=CREDS['redirect_uri'],
    scope='user-read-playback-state user-modify-playback-state user-read-currently-playing'
))


def get_active_device():
    """Get the active Spotify device ID."""
    devices = sp.devices()
    for device in devices['devices']:
        if device['name'] == CREDS['device_id']:
            return device['id']
    return None


@tool(
    name="play_song",
    description="Play a song by artist and title, or search for a song by query",
    aliases=["play", "play_music", "start_music"]
)
def play_song(artist_query: Optional[str] = None, song: Optional[str] = None) -> str:
    """
    Play a song or playlist on Spotify, prioritizing user's playlist names, then songs in playlists,
    top artists, saved albums, and finally general search.

    Args:
        artist_query: Artist name, playlist name, or search query
        song: Song title (if two arguments are provided, one is artist and one is song title)

    Returns:
        Status message about the played song or playlist.
    """
    # First try to setup sound system if not already on
    if config['spotify']['use_avr']:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(setup_avr("Music"))
        else:
            loop.create_task(setup_avr("Music"))

    # Ensure arguments are separated
    if artist_query and re.search(r'\s+by\s+', artist_query):
        parts = artist_query.split(' by ')
        if len(parts) == 2:
            artist_query, song = parts[1].strip(), parts[0].strip()

    if (re.sub(r'[^A-Za-z]+', '', str(artist_query).lower()) == "music") or (artist_query is None):
        pause()
        sp.start_playback(device_id=get_active_device())
        return "Playing music on spotify"

    playlists = sp.current_user_playlists(limit=50)['items']

    # 1. Check if query closely matches a playlist name
    playlist_names = [pl['name'] for pl in playlists]
    # Find close matches to artist_query
    matches = difflib.get_close_matches(artist_query, playlist_names, n=1, cutoff=0.6)
    if matches:
        for playlist in playlists:
            if playlist['name'] == matches[0]:
                pause()
                sp.start_playback(device_id=get_active_device(), context_uri=playlist['uri'])
                return f"Playing your playlist \"{playlist['name']}\""

    # Search user's top five playlists for the track
    for playlist in playlists[:5]:
        results = sp.playlist_tracks(playlist['id'])
        for item in results['items']:
            track = item['track']
            if (song and song.lower() in track['name'].lower()) or \
               (artist_query and artist_query.lower() in track['artists'][0]['name'].lower()):
                pause()
                sp.start_playback(device_id=get_active_device(), uris=[track['uri']])
                return f"Playing {track['name']} by {track['artists'][0]['name']} from your playlist \"{playlist['name']}\""

    # Fallback to general search
    if artist_query and song:
        results = sp.search(q=f"artist:{artist_query} track:{song}", type='track', limit=1)
    elif artist_query:
        results = sp.search(q=artist_query, type='track', limit=1)
    else:
        return "Please provide either artist and song, playlist name, or a search query"

    try:
        tracks = results.get('tracks', {}).get('items', [])
        uris = [track['uri'] for track in tracks if 'uri' in track]

        if len(uris) == 0:
            pause()
            sp.start_playback(device_id=get_active_device())
            return "No tracks found, starting playback"
        else:
            pause()
            sp.start_playback(device_id=get_active_device(), uris=uris)
            track = tracks[0]
            return f"Playing {track['name']} by {track['artists'][0]['name']}"
    except Exception as e:
        return "Unable to play your request"


def is_playing() -> bool:
    """Check if currently playing music on Spotify."""
    playback = sp.current_playback()
    if playback and playback['is_playing']:
        return True
    return False

@tool(
    name="pause",
    description="Pause the currently playing music",
    aliases=["stop"]
)
def pause() -> str:
    """Pause the currently playing music on Spotify."""
    playback = sp.current_playback()
    if playback and playback['is_playing']:
        sp.pause_playback()
    return "Playback paused."


@tool(
    name="resume",
    description="Resume the currently paused music",
    aliases=["play", "unpause"]
)
def resume() -> str:
    """Resume the currently paused music on Spotify."""
    playback = sp.current_playback()
    if playback and not playback['is_playing']:
        sp.start_playback(device_id=get_active_device())
    return "Playback resumed."


@tool(
    name="skip",
    description="Skip to the next track",
    aliases=["next", "next_track"]
)
def skip() -> str:
    """Skip to the next track in the playlist."""
    sp.next_track()
    return "Skipped to next track."

if __name__ == "__main__":
    print("Spotify Music Controller")
    print(sp.current_playback())
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    result = tool_registry.execute_tool("play_song", kwargs={"artist_query": "old mervs cellphone"})
    print(f"Result: {result}")