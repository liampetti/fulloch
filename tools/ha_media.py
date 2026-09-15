"""Home Assistant media implementation.

Uses ha_client as the single owner of configuration and resolution state.
"""

import logging
from typing import Optional

from . import ha_client as client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# media_player wrappers — volume, source, transport.
# ---------------------------------------------------------------------------


@client.tool(
    name="ha_volume_set",
    description="Set the volume of a media player (TV, AVR, speakers) by percentage 0-100",
    aliases=["set_volume", "volume", "tv_volume", "volume_tv"],
)
def volume_set(entity: str, volume: int) -> str:
    """Set the volume of a media_player entity.

    Args:
        entity: Media player entity name or ID (e.g. 'living room tv').
        volume: Volume as a percentage 0-100.
    """
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to adjust."
    friendly = client._friendly_for(entity_id)

    pct = max(0, min(100, int(volume)))
    return client._call_service(
        "media_player",
        "volume_set",
        entity_id,
        {"volume_level": pct / 100},
        success_message=f"{friendly} volume {pct}",
    )


@client.tool(
    name="ha_volume_up",
    description="Increase the volume of a media player one step",
    aliases=["louder", "volume_up", "increase_volume"],
)
def volume_up(entity: Optional[str] = None) -> str:
    """Step the volume up on a media_player entity (default: Spotify, AVR, then TV)."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to turn up."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "volume_up",
        entity_id,
        success_message=f"{friendly} louder",
    )


@client.tool(
    name="ha_volume_down",
    description="Decrease the volume of a media player one step",
    aliases=["quieter", "volume_down", "decrease_volume"],
)
def volume_down(entity: Optional[str] = None) -> str:
    """Step the volume down on a media_player entity (default: Spotify, AVR, then TV)."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to turn down."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "volume_down",
        entity_id,
        success_message=f"{friendly} quieter",
    )


@client.tool(
    name="ha_select_source",
    description="Select the input source on a media player (AVR, TV)",
    aliases=["select_source", "set_input", "switch_input", "set_source"],
)
def select_source(source: str, entity: Optional[str] = None) -> str:
    """Select the input source on a media_player entity.

    Args:
        source: The source name as configured in the AVR/TV (e.g. 'HDMI 1', 'TV', 'Spotify').
        entity: Optional media_player entity. Defaults to AVR if configured, else TV.
    """
    entity_id = _media_target(entity, prefer_spotify=False)
    if entity_id is None:
        return "I don't know which device to switch."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "select_source",
        entity_id,
        {"source": source},
        success_message=f"{friendly} switched to {source}",
    )


def _media_target(entity: Optional[str], prefer_spotify: bool = True) -> Optional[str]:
    """Resolve a media player entity or Home Assistant room target."""
    client._ensure_loaded()
    if entity:
        area_id = client._resolve_area(entity)
        if area_id is not None:
            players = [
                eid
                for eid in client._area_entities(area_id, domain="media_player")
                if eid not in client._DENIED_ENTITIES
            ]
            if prefer_spotify and client.SPOTIFY_ENTITY in players:
                return client.SPOTIFY_ENTITY
            return players[0] if players else None
        return client._resolve_entity(entity, domain="media_player")
    target = (
        (client.SPOTIFY_ENTITY if prefer_spotify else None) or client.AVR_ENTITY or client.TV_ENTITY
    )
    if not target:
        return None
    return client._resolve_entity(target, domain="media_player")


def _spotify_transport_fallback(action: str) -> Optional[str]:
    """Fall back to direct Spotify Connect for a transport control when no
    HA media_player entity resolved — e.g. a Spotify-only setup with no
    `home_assistant:` block configured at all. Controls whatever device
    Spotify Connect currently considers active, since there's no HA
    area/entity to target without HA.

    Only attempted if `spotify:` is actually configured — checked before
    importing `tools.spotify` at all, so a user with neither integration
    configured never pays for (or leaks into the tool registry) a module
    they didn't ask for.

    Spotify depends only on ha_client, so this optional import has no reverse
    dependency on the media tools or their public entry point.

    Returns None only if Spotify itself isn't usable either (not
    configured, or no valid credentials) — callers should fall through to
    their own HA-specific "I don't know which player" message in that
    case. Otherwise always returns a string (success or a Spotify-specific
    failure message), which is strictly more informative than that.
    """
    if "spotify" not in client.config:
        return None
    try:
        import tools.spotify as spotify_tool
    except Exception:
        return None

    sp = spotify_tool._get_client()
    if sp is None:
        return None

    try:
        device_id = spotify_tool._get_active_device(sp)
        if action == "pause":
            sp.pause_playback(device_id=device_id)
            return "Spotify paused"
        if action == "resume":
            sp.start_playback(device_id=device_id)
            return "Spotify resumed"
        if action == "skip":
            sp.next_track(device_id=device_id)
            return "Skipped to the next track on Spotify"
        if action == "previous":
            sp.previous_track(device_id=device_id)
            return "Back a track on Spotify"
    except Exception:
        logger.exception(f"Spotify Connect transport fallback failed ({action})")
        return "Couldn't control Spotify — no active device found."
    return None


@client.tool(
    name="pause",
    description="Pause playback on the active media player",
    aliases=["stop", "halt", "pause_music"],
)
def pause(entity: Optional[str] = None) -> str:
    """Pause a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return _spotify_transport_fallback("pause") or "I don't know which player to pause."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "media_pause",
        entity_id,
        success_message=f"{friendly} paused",
    )


@client.tool(
    name="resume",
    description="Resume playback on the active media player",
    aliases=["unpause", "resume_music"],
)
def resume(entity: Optional[str] = None) -> str:
    """Resume a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return _spotify_transport_fallback("resume") or "I don't know which player to resume."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "media_play",
        entity_id,
        success_message=f"{friendly} resumed",
    )


@client.tool(
    name="skip",
    description="Skip to the next track on the active media player",
    aliases=["next", "next_track", "skip_track"],
)
def skip(entity: Optional[str] = None) -> str:
    """Skip a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return _spotify_transport_fallback("skip") or "I don't know which player to skip on."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "media_next_track",
        entity_id,
        success_message=f"{friendly} skipped",
    )


@client.tool(
    name="previous",
    description="Go back to the previous track on the active media player",
    aliases=["previous_track", "skip_back", "last_track"],
)
def previous(entity: Optional[str] = None) -> str:
    """Go back a track on a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return (
            _spotify_transport_fallback("previous") or "I don't know which player to skip back on."
        )
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "media_previous_track",
        entity_id,
        success_message=f"{friendly} back a track",
    )


@client.tool(
    name="ha_mute",
    description="Mute or unmute a media player (TV, AVR, speakers)",
    aliases=["mute", "unmute", "mute_tv", "silence_tv"],
)
def ha_mute(entity: Optional[str] = None, muted: bool = True) -> str:
    """Mute or unmute a media_player entity. Defaults to Spotify, AVR, then TV.

    Args:
        entity: Media player name, ID, or HA room. Omit to use the default player.
        muted: True to mute (default), False to unmute.
    """
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to mute."
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "media_player",
        "volume_mute",
        entity_id,
        {"is_volume_muted": bool(muted)},
        success_message=f"{friendly} {'muted' if muted else 'unmuted'}",
    )
