"""Home Assistant integration via REST API.

Loaded when the `home_assistant:` block is present in config.yml. Requires a
long-lived access token. Registers generic tool names like `turn_on` /
`turn_off` — if other integrations are added later, the tool registry's
first-wins collision rule applies.
"""

import datetime as _dt
import json
import logging
import os
import time
from typing import Optional

import requests

from core.datetime_utils import tts_friendly_event_summary

from ._config import config
from .tool_registry import tool

logger = logging.getLogger(__name__)

HA_CONFIG = config.get('home_assistant', {})

# Home Assistant connection settings
HA_URL = HA_CONFIG.get('url', 'http://localhost:8123')
HA_TOKEN = HA_CONFIG.get('token', '')
TIMEOUT = HA_CONFIG.get('timeout', 10)

# Role entities (SPOTIFY_ENTITY / TV_ENTITY / AVR_ENTITY / CALENDAR_ENTITY)
# are auto-detected from /api/states after the alias map is fetched.
# See `_autodetect_*_entity` helpers further down. If autodetect picks
# wrong (e.g. multiple TVs), rename the desired entity in the HA UI so
# it lands first in the alias map.

# Substrings used by AVR auto-detect, matched against friendly_name + entity_id.
_AVR_KEYWORDS = (
    "avr", "receiver", "pioneer", "onkyo", "denon", "yamaha", "marantz",
)

# Spoken color name → RGB. Hoisted out of `set_color` so the dict isn't
# rebuilt on every call.
_COLOR_MAP = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "pink": [255, 192, 203],
    "white": [255, 255, 255],
    "warm white": [255, 244, 229],
    "cool white": [255, 255, 255],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
}


def _get_headers() -> dict:
    """Return authorization headers for Home Assistant API."""
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


# Retry budget for the startup alias fetch. When HA and Fulloch boot in the
# same compose stack, HA's /api/states often isn't responsive for the first
# few seconds. Without retries we'd lock in an empty alias map and every
# tool would resolve names to raw strings for the rest of the session.
# Override with FULLOCH_HA_ALIAS_RETRIES=0 in tests to skip the loop.
_ALIAS_FETCH_RETRIES = int(os.environ.get("FULLOCH_HA_ALIAS_RETRIES", "15"))
_ALIAS_FETCH_BACKOFF_S = 2.0


def _fetch_entity_aliases() -> dict:
    """Fetch every entity's friendly_name from HA at module load.

    Returns a lowercased `friendly_name -> entity_id` map. On the first
    duplicate friendly_name the earlier entity_id wins (later ones are
    logged and skipped). Returns an empty dict if HA is unreachable or
    no token is configured — direct entity_ids still resolve via the
    `'.' in name` fallback in `_resolve_entity`.
    """
    if not HA_TOKEN:
        logger.warning("Home Assistant token not configured; entity aliases unavailable")
        return {}

    states = None
    for attempt in range(1, _ALIAS_FETCH_RETRIES + 1):
        try:
            response = requests.get(
                f"{HA_URL}/api/states", headers=_get_headers(), timeout=TIMEOUT,
            )
            response.raise_for_status()
            states = response.json()
            break
        except Exception as e:
            if attempt < _ALIAS_FETCH_RETRIES:
                logger.info(
                    f"HA aliases fetch attempt {attempt}/{_ALIAS_FETCH_RETRIES} "
                    f"failed ({e}); retrying in {_ALIAS_FETCH_BACKOFF_S}s"
                )
                time.sleep(_ALIAS_FETCH_BACKOFF_S)
            else:
                logger.warning(
                    f"Could not fetch entity aliases from {HA_URL} after "
                    f"{_ALIAS_FETCH_RETRIES} attempts: {e}"
                )
                return {}
    if states is None:
        return {}

    aliases: dict = {}
    for state in states:
        entity_id = state.get('entity_id')
        if not entity_id:
            continue
        friendly = state.get('attributes', {}).get('friendly_name')
        # Entities without a friendly_name still need to appear in the map
        # so the role-entity autodetect (`_autodetect_weather_entity` etc.)
        # can find them by domain prefix. Use the entity_id as the key —
        # safe because entity_ids contain a "." which real friendly_names
        # never do, so there's no risk of colliding with a spoken alias.
        key = (friendly or entity_id).lower()
        if key in aliases and aliases[key] != entity_id:
            logger.debug(
                f"Duplicate key '{key}': keeping "
                f"{aliases[key]}, ignoring {entity_id}"
            )
            continue
        aliases[key] = entity_id

    logger.info(f"Fetched {len(aliases)} entity aliases from Home Assistant")
    return aliases


_ENTITY_ALIASES = _fetch_entity_aliases()

# Trailing words a speaker is likely to add or drop when referring to a device
# (e.g. "downstairs office" vs "downstairs office lights").
_NAME_SUFFIXES = (
    "lights", "light", "lamp", "lamps", "bulb", "bulbs",
    "group", "switch", "switches", "fan", "fans",
)

# Filler words that ASR often picks up at the start of an entity name.
_LEADING_FILLERS = ("the ", "a ", "an ", "my ")


def _friendly_for(entity_id: str) -> str:
    """Return a human-readable name for an entity_id, suitable for TTS."""
    for friendly, eid in _ENTITY_ALIASES.items():
        if eid == entity_id:
            return friendly
    slug = entity_id.split(".", 1)[-1] if "." in entity_id else entity_id
    return slug.replace("_", " ")


def _domain_of(entity_id: str) -> str:
    """Domain prefix of an entity_id (`light.kitchen` → `light`).

    Falls back to the `homeassistant` catch-all domain when `entity_id`
    isn't a dotted id, so generic on/off/toggle still target something.
    """
    return entity_id.split(".")[0] if "." in entity_id else "homeassistant"


def _call_service(
    domain: str,
    service: str,
    entity_id: str,
    data: Optional[dict] = None,
    success_message: Optional[str] = None,
) -> str:
    """Call a Home Assistant service and return a TTS-friendly response."""
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    url = f"{HA_URL}/api/services/{domain}/{service}"
    payload = {"entity_id": entity_id}
    if data:
        payload.update(data)

    friendly = _friendly_for(entity_id)
    action = service.replace("_", " ")
    try:
        response = requests.post(url, headers=_get_headers(), json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return success_message if success_message is not None else "OK"
    except requests.exceptions.ConnectionError:
        return "I couldn't reach Home Assistant."
    except requests.exceptions.Timeout:
        return "Home Assistant didn't respond in time."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        logger.warning(
            f"HA {domain}.{service} on {entity_id} failed: {status} {e.response.text}"
        )
        if status in (400, 404):
            # 400/404 usually means the entity_id didn't resolve. Route through
            # the agent replan loop so the next agent call can try a different
            # name or ask the user to clarify, instead of speaking a dead-end
            # "Sorry, I couldn't find X" line directly.
            return (
                f"Reactive question: Couldn't find an entity matching "
                f"{entity_id!r}. Try a different name or be more specific."
            )
        return f"Couldn't {action} {friendly}."
    except Exception as e:
        logger.warning(f"HA {domain}.{service} on {entity_id} failed: {e}")
        return f"Couldn't {action} {friendly}."


def _call_service_with_response(
    domain: str,
    service: str,
    payload: dict,
    timeout: Optional[int] = None,
) -> Optional[dict]:
    """Call a HA service that returns data, parse the service_response.

    Use for services like `calendar.get_events` and `spotifyplus.search_*`
    that need ?return_response=true. Returns the `service_response` block
    of the JSON, or None on any error.
    """
    if not HA_TOKEN:
        return None
    url = f"{HA_URL}/api/services/{domain}/{service}?return_response=true"
    try:
        response = requests.post(
            url, headers=_get_headers(), json=payload,
            timeout=timeout if timeout is not None else TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("service_response")
    except Exception as e:
        logger.warning(f"HA {domain}.{service} (with response) failed: {e}")
        return None


def _get_state(entity_id: str) -> Optional[dict]:
    """Get the state of an entity from Home Assistant."""
    if not HA_TOKEN:
        return None

    url = f"{HA_URL}/api/states/{entity_id}"
    try:
        response = requests.get(url, headers=_get_headers(), timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _resolve_entity(name: str, domain: str = None) -> str:
    """Resolve a friendly name to an entity_id.

    Tries (in order): exact friendly_name match, the name with a trailing
    "lights"/"group"/etc. stripped, the name with each common suffix
    appended, and finally falls back to assuming `name` is already a
    valid entity_id or constructing one from `domain`.
    """
    key = name.lower().strip()
    for filler in _LEADING_FILLERS:
        if key.startswith(filler):
            key = key[len(filler):].strip()
            break

    if key in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[key]

    head, _, tail = key.rpartition(" ")
    if head and tail in _NAME_SUFFIXES and head in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[head]

    for suffix in _NAME_SUFFIXES:
        candidate = f"{key} {suffix}"
        if candidate in _ENTITY_ALIASES:
            return _ENTITY_ALIASES[candidate]

    # Token-superset fuzzy match. "downstairs office" matches
    # "Downstairs Office Ceiling Light" because every input token appears
    # in the alias's tokens. Pick the shortest matching alias (most specific
    # to the input) to break ties.
    input_tokens = set(key.split())
    if input_tokens:
        fuzzy = [
            (alias, eid) for alias, eid in _ENTITY_ALIASES.items()
            if input_tokens.issubset(set(alias.split()))
        ]
        if fuzzy:
            fuzzy.sort(key=lambda kv: len(kv[0]))
            chosen_alias, chosen_eid = fuzzy[0]
            logger.debug(
                f"Fuzzy-resolved {name!r} → {chosen_eid} via {chosen_alias!r}"
            )
            return chosen_eid

    if '.' in name:
        return name

    if domain:
        entity_name = key.replace(' ', '_')
        return f"{domain}.{entity_name}"

    return name


@tool(
    name="turn_on",
    description="Turn on a device, light, switch, or other Home Assistant entity",
    aliases=["ha_turn_on", "switch_on", "turn_on_device"]
)
def turn_on(entity: str, brightness: Optional[int] = None) -> str:
    """Turn on a Home Assistant entity.

    Args:
        entity: Entity name or ID (e.g., 'living room lights', 'light.living_room')
        brightness: Optional brightness percentage (0-100) for lights
    """
    entity_id = _resolve_entity(entity, domain="light")
    domain = _domain_of(entity_id)
    friendly = _friendly_for(entity_id)

    data = {}
    success = f"{friendly} on"
    if brightness is not None and domain == 'light':
        data['brightness'] = int((brightness / 100) * 255)
        success = f"{friendly} on at {brightness} percent"

    return _call_service(domain, "turn_on", entity_id, data if data else None, success)


@tool(
    name="turn_off",
    description="Turn off a device, light, switch, or other Home Assistant entity",
    aliases=["ha_turn_off", "switch_off", "turn_off_device"]
)
def turn_off(entity: str) -> str:
    """Turn off a Home Assistant entity.

    Args:
        entity: Entity name or ID (e.g., 'living room lights', 'light.living_room')
    """
    entity_id = _resolve_entity(entity)
    domain = _domain_of(entity_id)
    friendly = _friendly_for(entity_id)

    return _call_service(domain, "turn_off", entity_id, success_message=f"{friendly} off")


@tool(
    name="toggle",
    description="Toggle a Home Assistant entity on or off",
    aliases=["ha_toggle", "toggle_device"]
)
def toggle(entity: str) -> str:
    """Toggle a Home Assistant entity.

    Args:
        entity: Entity name or ID (e.g., 'living room lights', 'light.living_room')
    """
    entity_id = _resolve_entity(entity)
    domain = _domain_of(entity_id)
    friendly = _friendly_for(entity_id)

    return _call_service(domain, "toggle", entity_id, success_message=f"Toggled {friendly}")


@tool(
    name="ha_set_brightness",
    description="Set the brightness of a light in Home Assistant",
    aliases=["ha_brightness", "ha_dim_light"]
)
def set_ha_brightness(entity: str, brightness: int) -> str:
    """Set the brightness of a light.

    Args:
        entity: Light entity name or ID
        brightness: Brightness percentage (0-100)
    """
    entity_id = _resolve_entity(entity, domain="light")
    friendly = _friendly_for(entity_id)

    brightness = max(0, min(100, brightness))
    brightness_255 = int((brightness / 100) * 255)

    return _call_service(
        "light", "turn_on", entity_id, {"brightness": brightness_255},
        success_message=f"{friendly} at {brightness} percent",
    )


@tool(
    name="ha_set_color",
    description="Set the color of a light in Home Assistant using color name or RGB",
    aliases=["ha_color", "change_light_color"]
)
def set_color(entity: str, color: str) -> str:
    """Set the color of a light.

    Args:
        entity: Light entity name or ID
        color: Color name (red, green, blue, etc.) or RGB as 'r,g,b'
    """
    entity_id = _resolve_entity(entity, domain="light")
    color_lower = color.lower().strip()

    if color_lower in _COLOR_MAP:
        rgb = _COLOR_MAP[color_lower]
    elif ',' in color:
        try:
            rgb = [int(c.strip()) for c in color.split(',')]
            if len(rgb) != 3:
                return "I need three numbers for an RGB colour."
        except ValueError:
            return f"I couldn't read the colour {color}."
    else:
        return f"I don't know the colour {color}."

    friendly = _friendly_for(entity_id)
    return _call_service(
        "light", "turn_on", entity_id, {"rgb_color": rgb},
        success_message=f"{friendly} set to {color_lower}",
    )


# ---------------------------------------------------------------------------
# media_player wrappers — volume, source, transport.
# ---------------------------------------------------------------------------

@tool(
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
    entity_id = _resolve_entity(entity, domain="media_player")
    friendly = _friendly_for(entity_id)

    pct = max(0, min(100, int(volume)))
    return _call_service(
        "media_player", "volume_set", entity_id, {"volume_level": pct / 100},
        success_message=f"{friendly} volume {pct}",
    )


@tool(
    name="ha_volume_up",
    description="Increase the volume of a media player one step",
    aliases=["louder", "volume_up", "increase_volume"],
)
def volume_up(entity: Optional[str] = None) -> str:
    """Step the volume up on a media_player entity (default: AVR if configured, else TV)."""
    target = entity or AVR_ENTITY or TV_ENTITY
    if not target:
        return "I don't know which speakers to turn up."
    entity_id = _resolve_entity(target, domain="media_player")
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player", "volume_up", entity_id,
        success_message=f"{friendly} louder",
    )


@tool(
    name="ha_volume_down",
    description="Decrease the volume of a media player one step",
    aliases=["quieter", "volume_down", "decrease_volume"],
)
def volume_down(entity: Optional[str] = None) -> str:
    """Step the volume down on a media_player entity (default: AVR if configured, else TV)."""
    target = entity or AVR_ENTITY or TV_ENTITY
    if not target:
        return "I don't know which speakers to turn down."
    entity_id = _resolve_entity(target, domain="media_player")
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player", "volume_down", entity_id,
        success_message=f"{friendly} quieter",
    )


@tool(
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
    target = entity or AVR_ENTITY or TV_ENTITY
    if not target:
        return "I don't know which device to switch."
    entity_id = _resolve_entity(target, domain="media_player")
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player", "select_source", entity_id, {"source": source},
        success_message=f"{friendly} switched to {source}",
    )


def _media_target(entity: Optional[str]) -> Optional[str]:
    """Resolve the media_player entity for a transport action."""
    target = entity or SPOTIFY_ENTITY or AVR_ENTITY or TV_ENTITY
    if not target:
        return None
    return _resolve_entity(target, domain="media_player")


@tool(
    name="pause",
    description="Pause playback on the active media player",
    aliases=["stop", "halt", "pause_music"],
)
def pause(entity: Optional[str] = None) -> str:
    """Pause a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which player to pause."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player", "media_pause", entity_id,
        success_message=f"{friendly} paused",
    )


@tool(
    name="resume",
    description="Resume playback on the active media player",
    aliases=["unpause", "resume_music"],
)
def resume(entity: Optional[str] = None) -> str:
    """Resume a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which player to resume."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player", "media_play", entity_id,
        success_message=f"{friendly} resumed",
    )


@tool(
    name="skip",
    description="Skip to the next track on the active media player",
    aliases=["next", "next_track", "skip_track"],
)
def skip(entity: Optional[str] = None) -> str:
    """Skip a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which player to skip on."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player", "media_next_track", entity_id,
        success_message=f"{friendly} skipped",
    )


# ---------------------------------------------------------------------------
# Spotify search-and-play via the SpotifyPlus HACS integration.
# ---------------------------------------------------------------------------

def _spotify_search(kind: str, query: str) -> list[dict]:
    """Run a SpotifyPlus search and return the result items list (possibly empty).

    `kind` is "playlists" or "tracks". SpotifyPlus returns:
      {"user_profile": {...}, "result": {"items": [...], "total": N, ...}}
    The service parameter is `criteria` (not `query`).
    """
    service = "search_playlists" if kind == "playlists" else "search_tracks"
    response = _call_service_with_response(
        "spotifyplus", service, {"entity_id": SPOTIFY_ENTITY, "criteria": query},
    )
    if not response:
        return []
    return (response.get("result") or {}).get("items") or []


@tool(
    name="play_song",
    description="Search Spotify and play a song or playlist by name",
    aliases=["play", "play_music", "start_music"],
)
def play_song(query: str, artist: Optional[str] = None) -> str:
    """Search Spotify for a playlist (by name) then fall back to track search.

    Args:
        query: The spoken phrase, e.g. 'wagon wheel' or 'calm evening playlist'.
        artist: Optional artist refinement. Combined into the query when given.
    """
    if not SPOTIFY_ENTITY:
        return (
            "Reactive question: No Spotify media player was found in Home Assistant. "
            "Check that a Spotify or SpotifyPlus integration is set up, or add "
            "`spotify_entity:` to the home_assistant block in config.yml."
        )

    search_query = f"{query} {artist}".strip() if artist else query
    entity_id = _resolve_entity(SPOTIFY_ENTITY, domain="media_player")

    # SpotifyPlus path: search_playlists → search_tracks → play URI.
    # Falls back to the generic path if SpotifyPlus isn't installed (empty results).
    playlists = _spotify_search("playlists", search_query)
    if playlists:
        uri = playlists[0].get("uri")
        name = playlists[0].get("name", "playlist")
        if uri:
            return _call_service(
                "media_player", "play_media", entity_id,
                {"media_content_id": uri, "media_content_type": "playlist"},
                success_message=f"Playing {name}",
            )

    tracks = _spotify_search("tracks", search_query)
    if tracks:
        uri = tracks[0].get("uri")
        name = tracks[0].get("name", "your song")
        if uri:
            return _call_service(
                "media_player", "play_media", entity_id,
                {"media_content_id": uri, "media_content_type": "music"},
                success_message=f"Playing {name}",
            )

    # Generic fallback for the built-in HA Spotify integration (no search service).
    # Passes the search query directly as the media_content_id — some Spotify
    # integrations resolve "spotify:search:<query>" but results are not guaranteed.
    logger.info(
        "SpotifyPlus search returned no results; trying generic media_player.play_media"
    )
    return _call_service(
        "media_player", "play_media", entity_id,
        {"media_content_id": f"spotify:search:{search_query}", "media_content_type": "music"},
        success_message=f"Playing {search_query}",
    )


def _resolve_with_variants(entity: str, suffixes: tuple, domains: tuple) -> Optional[str]:
    """Try to resolve `entity` against the alias map using common suffix/domain variants.

    Returns the entity_id of the first alias match, or None if nothing hits.
    Used for sensor-style lookups where the spoken name ("upstairs") differs
    from the HA friendly name ("Upstairs Temperature").
    """
    key = entity.lower().strip()
    candidates = [key] + [f"{key} {s}" for s in suffixes]
    for candidate in candidates:
        if candidate in _ENTITY_ALIASES:
            eid = _ENTITY_ALIASES[candidate]
            if any(eid.startswith(f"{d}.") for d in domains):
                return eid
    return None


@tool(
    name="get_temperature",
    description="Get the current temperature from a Home Assistant thermostat, climate zone, or temperature sensor",
    aliases=["temperature", "check_temperature", "what_temperature", "how_warm", "how_cold"],
)
def get_temperature(entity: str) -> str:
    """Return the current temperature reading for a room or sensor.

    Args:
        entity: Room or sensor name (e.g., 'upstairs', 'living room',
            'office thermostat'). Tries climate zones first, then
            temperature sensors named '{entity} temperature'.
    """
    suffixes = ("temperature", "temp", "thermostat", "climate")
    entity_id = _resolve_with_variants(entity, suffixes, ("climate", "sensor"))
    if entity_id is None:
        # Last resort: trust the SLM and assume the name matches an entity_id
        entity_id = _resolve_entity(entity, domain="climate")

    state = _get_state(entity_id)
    if state is None:
        return f"Sorry, I couldn't find a temperature reading for {entity}."

    attrs = state.get("attributes", {})
    temp = attrs.get("current_temperature")
    if temp is None:
        temp = attrs.get("temperature")
    if temp is None:
        # Temperature sensors put the reading in `state`
        raw = state.get("state")
        if raw not in (None, "", "unknown", "unavailable"):
            try:
                temp = float(raw)
            except (ValueError, TypeError):
                temp = None

    friendly = _friendly_for(entity_id)
    if temp is None:
        return f"I couldn't read a temperature for {friendly}."

    unit = _temperature_unit(attrs)
    return f"{friendly} is {round(temp)} {unit}"


@tool(
    name="get_entity_state",
    description="Get the current state of a Home Assistant entity (on/off, sensor reading, etc.)",
    aliases=["ha_state", "check_state", "is_on"]
)
def get_entity_state(entity: str) -> str:
    """Get the current state of a Home Assistant entity.

    Args:
        entity: Entity name or ID
    """
    entity_id = _resolve_entity(entity)
    state = _get_state(entity_id)

    if state is None:
        return f"Sorry, I couldn't find {_friendly_for(entity_id)}."

    entity_state = state.get('state', 'unknown')
    friendly_name = state.get('attributes', {}).get('friendly_name', entity_id)

    # Include relevant attributes
    attrs = state.get('attributes', {})
    details = [f"{friendly_name} is {entity_state}"]

    if 'brightness' in attrs:
        brightness_pct = int((attrs['brightness'] / 255) * 100)
        details.append(f"brightness: {brightness_pct}%")
    if 'temperature' in attrs:
        details.append(f"temperature: {attrs['temperature']}°")
    if 'current_temperature' in attrs:
        details.append(f"current temperature: {attrs['current_temperature']}°")

    return ", ".join(details)


@tool(
    name="ha_service",
    description="Call any Home Assistant service with custom data",
    aliases=["call_service", "ha_call"]
)
def call_ha_service(domain: str, service: str, entity: str, data: Optional[str] = None) -> str:
    """Call any Home Assistant service.

    Args:
        domain: Service domain (e.g., 'light', 'switch', 'climate')
        service: Service name (e.g., 'turn_on', 'set_temperature')
        entity: Entity ID to target
        data: Optional JSON string with additional service data
    """
    entity_id = _resolve_entity(entity)

    extra_data = None
    if data:
        try:
            extra_data = json.loads(data)
        except json.JSONDecodeError:
            return "Sorry, the data for that service call wasn't valid."

    friendly = _friendly_for(entity_id)
    return _call_service(
        domain, service, entity_id, extra_data,
        success_message=f"{service.replace('_', ' ').capitalize()} {friendly}",
    )


@tool(
    name="ha_set_climate",
    description="Set the temperature of a climate/thermostat entity in Home Assistant",
    aliases=["ha_climate", "ha_thermostat"]
)
def set_climate(entity: str, temperature: float, hvac_mode: Optional[str] = None) -> str:
    """Set climate/thermostat temperature.

    Args:
        entity: Climate entity name or ID
        temperature: Target temperature
        hvac_mode: Optional HVAC mode (heat, cool, auto, off)
    """
    entity_id = _resolve_entity(entity, domain="climate")
    friendly = _friendly_for(entity_id)

    data = {"temperature": temperature}
    if hvac_mode:
        data["hvac_mode"] = hvac_mode.lower()

    return _call_service(
        "climate", "set_temperature", entity_id, data,
        success_message=f"{friendly} set to {temperature} degrees",
    )


@tool(
    name="ha_lock",
    description="Lock a lock entity in Home Assistant",
    aliases=["lock_door"]
)
def lock(entity: str) -> str:
    """Lock a lock entity.

    Args:
        entity: Lock entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="lock")
    friendly = _friendly_for(entity_id)
    return _call_service("lock", "lock", entity_id, success_message=f"Locked {friendly}")


@tool(
    name="ha_unlock",
    description="Unlock a lock entity in Home Assistant",
    aliases=["unlock_door"]
)
def unlock(entity: str) -> str:
    """Unlock a lock entity.

    Args:
        entity: Lock entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="lock")
    friendly = _friendly_for(entity_id)
    return _call_service("lock", "unlock", entity_id, success_message=f"Unlocked {friendly}")


@tool(
    name="ha_open_cover",
    description="Open a cover/blind/garage in Home Assistant",
    aliases=["ha_open", "open_blind", "open_garage"]
)
def open_cover(entity: str) -> str:
    """Open a cover entity (blinds, garage door, etc.).

    Args:
        entity: Cover entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="cover")
    friendly = _friendly_for(entity_id)
    return _call_service("cover", "open_cover", entity_id, success_message=f"Opened {friendly}")


@tool(
    name="ha_close_cover",
    description="Close a cover/blind/garage in Home Assistant",
    aliases=["ha_close", "close_blind", "close_garage"]
)
def close_cover(entity: str) -> str:
    """Close a cover entity (blinds, garage door, etc.).

    Args:
        entity: Cover entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="cover")
    friendly = _friendly_for(entity_id)
    return _call_service("cover", "close_cover", entity_id, success_message=f"Closed {friendly}")


@tool(
    name="ha_run_script",
    description="Run a Home Assistant script or automation",
    aliases=["ha_script", "run_automation"]
)
def run_script(script_name: str) -> str:
    """Run a Home Assistant script.

    Args:
        script_name: Script entity ID or name (e.g., 'script.bedtime' or 'bedtime')
    """
    entity_id = _resolve_entity(script_name, domain="script")
    friendly = _friendly_for(entity_id)
    return _call_service("script", "turn_on", entity_id, success_message=f"Running {friendly}")


@tool(
    name="ha_activate_scene",
    description="Activate a Home Assistant scene",
    aliases=["ha_scene", "set_scene"]
)
def activate_scene(scene_name: str) -> str:
    """Activate a Home Assistant scene.

    Args:
        scene_name: Scene entity ID or name (e.g., 'scene.movie_time' or 'movie time')
    """
    entity_id = _resolve_entity(scene_name, domain="scene")
    friendly = _friendly_for(entity_id)
    return _call_service("scene", "turn_on", entity_id, success_message=f"Scene {friendly} activated")


# ---------------------------------------------------------------------------
# Calendar — wraps the HA `calendar.get_events` service.
# ---------------------------------------------------------------------------

def _calendar_window(day: str, now: Optional[_dt.datetime] = None) -> tuple[str, str]:
    """Return (start_iso, end_iso) for a spoken day phrase.

    Args:
        day: "today", "tomorrow", or "this_week" (also accepts "week").
        now: Override for testing; defaults to datetime.now().
    """
    now = now or _dt.datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if day == "tomorrow":
        start = midnight + _dt.timedelta(days=1)
        end = start + _dt.timedelta(days=1)
    elif day in ("this_week", "week"):
        start = midnight
        end = start + _dt.timedelta(days=7)
    else:  # today (default)
        start = midnight
        end = start + _dt.timedelta(days=1)

    return start.isoformat(), end.isoformat()


def _normalise_ha_event(event: dict) -> dict:
    """Convert a HA calendar event into the shared event shape.

    HA all-day events have date-only `start` (no 'T'). Timed events have
    ISO 8601 datetime `start`.
    """
    start = event.get("start", "")
    return {
        "start": start,
        "summary": event.get("summary"),
        "all_day": "T" not in start,
    }


def _ha_get_events(day: str) -> str:
    """Common body for whats_on and its day-specific aliases."""
    if not CALENDAR_ENTITY:
        return "No calendar is configured in Home Assistant."

    start_iso, end_iso = _calendar_window(day)
    response = _call_service_with_response(
        "calendar", "get_events",
        {
            "entity_id": CALENDAR_ENTITY,
            "start_date_time": start_iso,
            "end_date_time": end_iso,
        },
    )
    if not response:
        return "I couldn't reach your calendar."

    raw_events = (response.get(CALENDAR_ENTITY) or {}).get("events") or []
    events = [_normalise_ha_event(e) for e in raw_events]
    summary = tts_friendly_event_summary(events)
    # Multi-event days benefit from agent summarisation/filtering; route
    # through the replan loop. The "no events" case stays as a direct
    # spoken result.
    if len(events) >= 2:
        return f"Reactive question: {summary}"
    return summary


@tool(
    name="whats_on",
    description="Get calendar events for a spoken time period",
    aliases=["calendar", "events", "schedule"],
)
def whats_on(day: str = "today") -> str:
    """Calendar events for today, tomorrow, or this week.

    Args:
        day: "today" (default), "tomorrow", or "this_week".
    """
    return _ha_get_events(day)


@tool(
    name="whats_on_today",
    description="Get today's calendar events",
    aliases=["todays_events", "todays_schedule"],
)
def whats_on_today() -> str:
    return _ha_get_events("today")


@tool(
    name="whats_on_tomorrow",
    description="Get tomorrow's calendar events",
    aliases=["tomorrows_events", "tomorrows_schedule"],
)
def whats_on_tomorrow() -> str:
    return _ha_get_events("tomorrow")


@tool(
    name="whats_on_this_week",
    description="Get this week's calendar events",
    aliases=["this_weeks_events", "weekly_events"],
)
def whats_on_this_week() -> str:
    return _ha_get_events("this_week")


# ---------------------------------------------------------------------------
# Calendar write — wraps the HA `calendar.create_event` service.
# Requires `home_assistant.calendar` in config.yml to name the target calendar.
# ---------------------------------------------------------------------------

def get_upcoming_events(window_seconds: int = 90) -> list[dict]:
    """Return events starting on the reminder calendar within the next `window_seconds`.

    Used by the Assistant reminder poll thread. Not exposed as a tool.
    Returns a list of {"summary": str, "start": str} dicts.
    Uses UTC-aware datetimes so HA receives unambiguous timestamps regardless
    of the container's local timezone.
    """
    calendar = _reminder_calendar_entity()
    if not calendar:
        return []
    import datetime as _dt2
    now = _dt2.datetime.now(_dt2.timezone.utc)
    window_end = now + _dt2.timedelta(seconds=window_seconds)
    response = _call_service_with_response(
        "calendar", "get_events",
        {
            "entity_id": calendar,
            "start_date_time": now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "end_date_time": window_end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        },
    )
    if not response:
        return []
    raw = (response.get(calendar) or {}).get("events") or []
    results = []
    for e in raw:
        summary = e.get("summary", "")
        start = e.get("start", "")
        if not summary:
            continue
        # Filter out events that have already started — HA returns currently-active
        # events (started but not ended) which we don't want to re-fire as reminders.
        # Allow a small grace window (30s) so a poll that fires just after the
        # event's start time doesn't miss it.
        try:
            start_dt = _dt2.datetime.fromisoformat(start.replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=_dt2.timezone.utc)
            grace = now - _dt2.timedelta(seconds=30)
            if start_dt < grace:
                continue
        except (ValueError, AttributeError):
            pass
        results.append({"summary": summary, "start": start})
    return results


_RECURRENCE_TO_RRULE = {
    "daily": "FREQ=DAILY",
    "weekly": "FREQ=WEEKLY",
    "monthly": "FREQ=MONTHLY",
    "weekdays": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
}


def _reminder_calendar_entity() -> Optional[str]:
    """Resolve the configured reminder calendar to an entity ID.

    Reads `home_assistant.calendar` from config (friendly name or direct
    entity_id). Returns None if not configured or not found in the alias map.
    """
    name = HA_CONFIG.get('calendar')
    if not name:
        return None
    if '.' in str(name):
        return name
    entity_id = _ENTITY_ALIASES.get(name.lower())
    if entity_id and entity_id.startswith('calendar.'):
        return entity_id
    logger.warning(f"Reminder calendar '{name}' not found in HA entity aliases")
    return None


@tool(
    name="create_calendar_event",
    description=(
        "Create a one-off or recurring calendar event in Home Assistant. "
        "Use for reminders and recurring events (bin night, appointments, etc.). "
        "Requires home_assistant.calendar to be configured."
    ),
    aliases=["add_reminder", "set_reminder", "create_reminder", "add_calendar_event"],
)
def create_calendar_event(
    summary: str,
    date: str,
    time: Optional[str] = None,
    end_time: Optional[str] = None,
    recurrence: str = "none",
) -> str:
    """Create a calendar event in the configured HA reminder calendar.

    Args:
        summary: Event title (e.g. "Bin night", "Dentist").
        date: ISO date string, e.g. "2026-06-05".
        time: Start time in HH:MM 24-hour format. Omit for all-day events.
        end_time: End time in HH:MM 24-hour format. Defaults to 1 hour after start.
        recurrence: "none" (default), "daily", "weekly", or "monthly".
    """
    calendar = _reminder_calendar_entity()
    if not calendar:
        return (
            "User question: No reminder calendar is configured in Fulloch. "
            "Would you like me to save this as a note instead?"
        )

    try:
        start_dt = _dt.date.fromisoformat(date)
    except ValueError:
        return "I couldn't parse that date — please provide it as YYYY-MM-DD."

    extra: dict = {"summary": summary}

    if time:
        try:
            hour, minute = (int(p) for p in time.split(":")[:2])
        except (ValueError, AttributeError):
            return "I couldn't parse that time — please use HH:MM format."
        start_datetime = _dt.datetime.combine(start_dt, _dt.time(hour, minute))
        if end_time:
            try:
                eh, em = (int(p) for p in end_time.split(":")[:2])
            except (ValueError, AttributeError):
                return "I couldn't parse the end time — please use HH:MM format."
            end_datetime = _dt.datetime.combine(start_dt, _dt.time(eh, em))
        else:
            end_datetime = start_datetime + _dt.timedelta(hours=1)
        extra["start_date_time"] = start_datetime.isoformat()
        extra["end_date_time"] = end_datetime.isoformat()
    else:
        extra["start_date"] = start_dt.isoformat()
        extra["end_date"] = (start_dt + _dt.timedelta(days=1)).isoformat()

    rrule = _RECURRENCE_TO_RRULE.get(recurrence.lower())
    if rrule:
        extra["rrule"] = rrule

    recurrence_label = f" ({recurrence})" if recurrence != "none" else ""
    time_label = f" at {time}" if time else ""
    success = (
        f"Added '{summary}'{time_label} on "
        f"{start_dt.strftime('%A %-d %B')}{recurrence_label}"
    )
    return _call_service("calendar", "create_event", calendar, extra, success_message=success)


def _autodetect_weather_entity() -> str:
    """Pick a default weather entity: first weather.* in aliases > weather.home."""
    for eid in _ENTITY_ALIASES.values():
        if eid.startswith("weather."):
            logger.info(f"Auto-selected weather entity: {eid}")
            return eid
    logger.warning(
        "No weather.* entity found in HA aliases — falling back to "
        "'weather.home'. Likely the HA aliases fetch hit no weather entity."
    )
    return "weather.home"


def _autodetect_spotify_entity() -> Optional[str]:
    """Pick a Spotify media_player entity.

    Priority:
      1. `home_assistant.spotify_entity` in config (friendly name or direct entity_id)
      2. First `media_player.spotify*` in the alias map (covers both the
         built-in Spotify integration and SpotifyPlus whose entity IDs are
         `media_player.spotifyplus_<username>`)
    """
    configured = HA_CONFIG.get("spotify_entity")
    if configured:
        eid = _ENTITY_ALIASES.get(str(configured).lower()) or str(configured)
        logger.info(f"Using configured Spotify entity: {eid}")
        return eid
    for eid in _ENTITY_ALIASES.values():
        if eid.startswith("media_player.spotify"):
            logger.info(f"Auto-selected Spotify entity: {eid}")
            return eid
    logger.warning(
        "No Spotify media_player found in HA alias map. "
        "Add `spotify_entity: <friendly name or entity_id>` under home_assistant: in config.yml."
    )
    return None


def _looks_like_tv(entity_id: str, friendly: str) -> bool:
    """Match `tv` as a whole token in entity_id or friendly name."""
    if "tv" in entity_id.split(".", 1)[1].split("_"):
        return True
    return (
        " tv" in friendly
        or friendly.endswith(" tv")
        or friendly.startswith("tv ")
    )


def _autodetect_tv_entity() -> Optional[str]:
    """Pick a TV media_player entity: first media_player.* matching 'tv' > None."""
    for friendly, eid in _ENTITY_ALIASES.items():
        if not eid.startswith("media_player."):
            continue
        if eid.startswith("media_player.spotify"):
            continue  # don't poach Spotify for the TV role
        if _looks_like_tv(eid, friendly):
            logger.info(f"Auto-selected TV entity: {eid}")
            return eid
    return None


def _autodetect_avr_entity() -> Optional[str]:
    """Pick an AVR media_player entity: first media_player.* matching an AVR keyword > None."""
    for friendly, eid in _ENTITY_ALIASES.items():
        if not eid.startswith("media_player."):
            continue
        if eid.startswith("media_player.spotify"):
            continue
        haystack = f"{eid} {friendly}".lower()
        if any(kw in haystack for kw in _AVR_KEYWORDS):
            logger.info(f"Auto-selected AVR entity: {eid}")
            return eid
    return None


def _autodetect_calendar_entity() -> Optional[str]:
    """Pick a calendar entity: calendar.primary > first calendar.* > None."""
    if "calendar.primary" in _ENTITY_ALIASES.values():
        logger.info("Auto-selected calendar entity: calendar.primary")
        return "calendar.primary"
    for eid in _ENTITY_ALIASES.values():
        if eid.startswith("calendar."):
            logger.info(f"Auto-selected calendar entity: {eid}")
            return eid
    return None


def _autodetect_todo_entity() -> Optional[str]:
    """Pick a todo entity.

    Priority:
      1. `home_assistant.todo_entity` in config (friendly name or direct entity_id)
      2. `todo.shopping_list` (HA's default shopping list)
      3. First `todo.*` entity in the alias map
    """
    configured = HA_CONFIG.get("todo_entity")
    if configured:
        eid = _ENTITY_ALIASES.get(str(configured).lower()) or str(configured)
        logger.info(f"Using configured todo entity: {eid}")
        return eid
    if "todo.shopping_list" in _ENTITY_ALIASES.values():
        logger.info("Auto-selected todo entity: todo.shopping_list")
        return "todo.shopping_list"
    for eid in _ENTITY_ALIASES.values():
        if eid.startswith("todo."):
            logger.info(f"Auto-selected todo entity: {eid}")
            return eid
    return None


_DEFAULT_WEATHER_ENTITY = _autodetect_weather_entity()
SPOTIFY_ENTITY = _autodetect_spotify_entity()
TV_ENTITY = _autodetect_tv_entity()
AVR_ENTITY = _autodetect_avr_entity()
CALENDAR_ENTITY = _autodetect_calendar_entity()
TODO_ENTITY = _autodetect_todo_entity()


# ---------------------------------------------------------------------------
# Todo / task lists — wraps HA `todo.add_item` and `todo.get_items`.
# ---------------------------------------------------------------------------

@tool(
    name="add_todo_item",
    description=(
        "Add an item to a Home Assistant todo or shopping list. "
        "Use for simple list items ('add eggs to the shopping list', 'add dentist to my tasks'). "
        "Use append_to_note instead when the item needs context or belongs in a note."
    ),
    aliases=["add_shopping_item", "add_task", "todo_add"],
)
def add_todo_item(item: str) -> str:
    """Add an item to the configured HA todo list.

    Args:
        item: The item text to add (e.g. "eggs", "call the plumber").
    """
    if not TODO_ENTITY:
        return (
            "User question: No todo list is configured in Home Assistant. "
            "Would you like me to add this to a note instead?"
        )
    return _call_service(
        "todo", "add_item", TODO_ENTITY,
        {"item": item},
        success_message=f"Added '{item}' to your list",
    )


@tool(
    name="get_todo_items",
    description="Read pending items from the Home Assistant todo or shopping list.",
    aliases=["read_todo", "show_todo", "shopping_list", "get_shopping_list", "read_shopping_list"],
)
def get_todo_items() -> str:
    """Return pending (incomplete) items from the configured HA todo list."""
    if not TODO_ENTITY:
        return (
            "User question: No todo list is configured in Home Assistant. "
            "Would you like me to check your notes instead?"
        )
    response = _call_service_with_response(
        "todo", "get_items",
        {"entity_id": TODO_ENTITY, "status": "needs_action"},
    )
    if not response:
        return "I couldn't reach your todo list."
    items = (response.get(TODO_ENTITY) or {}).get("items") or []
    if not items:
        return "Your list is empty."
    names = [i.get("summary", "") for i in items if i.get("summary")]
    if not names:
        return "Your list is empty."
    if len(names) == 1:
        return f"You have one item: {names[0]}."
    return f"You have {len(names)} items: {', '.join(names[:-1])}, and {names[-1]}."


def _temperature_unit(attrs: dict) -> str:
    """Map HA's temperature_unit attribute to a spoken phrase."""
    raw = (attrs.get("temperature_unit") or "").upper()
    return "degrees Fahrenheit" if "F" in raw else "degrees Celsius"


def _format_day(label: str, day: dict, unit: str) -> str:
    cond = (day.get("condition") or "").replace("-", " ").strip()
    hi = day.get("temperature")
    lo = day.get("templow")
    precip = day.get("precipitation_probability")

    bits = [f"{label} {cond}"] if cond else [label]
    if lo is not None and hi is not None:
        bits.append(f"{round(lo)} to {round(hi)} {unit}")
    elif hi is not None:
        bits.append(f"high {round(hi)} {unit}")
    if precip is not None:
        bits.append(f"{int(precip)} percent chance of rain")
    return " ".join(bits)


@tool(
    name="get_weather_forecast",
    description="Get the weather forecast for today and tomorrow",
    aliases=["weather", "forecast", "get_weather"],
)
def get_weather_forecast(location: Optional[str] = None) -> str:
    """Speak current conditions plus the next day or two from a HA weather entity.

    Args:
        location: Optional weather entity name. Defaults to the first
            `weather.*` entity in HA (or `weather.home` if none found).
    """
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    entity_id = _DEFAULT_WEATHER_ENTITY
    if location:
        candidate = _resolve_entity(location, domain="weather")
        if candidate.startswith("weather."):
            entity_id = candidate

    state = _get_state(entity_id)
    if state is None:
        # Route through the agent replan loop so the next call can compose a
        # user-facing reply instead of speaking a dead-end "Sorry" line.
        return (
            f"Reactive question: Weather entity {entity_id!r} not found in "
            f"Home Assistant. Tell the user which entity to configure, or "
            f"suggest renaming a `weather.*` entity so autodetect picks it up."
        )

    attrs = state.get("attributes", {})
    unit = _temperature_unit(attrs)
    current_cond = (state.get("state") or "unknown").replace("-", " ")
    current_temp = attrs.get("temperature")

    parts = [f"Forecast for {_friendly_for(entity_id).replace('forecast', '')}"]
    if current_temp is not None:
        parts.append(f"currently {current_cond} at {round(current_temp)} {unit}")
    else:
        parts.append(f"currently {current_cond}")

    try:
        response = requests.post(
            f"{HA_URL}/api/services/weather/get_forecasts?return_response=true",
            headers=_get_headers(),
            json={"entity_id": entity_id, "type": "daily"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        service_response = response.json().get("service_response") or {}
        forecast = (service_response.get(entity_id) or {}).get("forecast") or []
    except Exception as e:
        logger.warning(f"HA weather forecast for {entity_id} failed: {e}")
        forecast = []

    for label, day in zip(("Today", "Tomorrow"), forecast[:2]):
        parts.append(_format_day(label, day, unit))

    return ". ".join(parts) + "."
