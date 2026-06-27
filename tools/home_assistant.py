"""Home Assistant integration via REST API.

Loaded when the `home_assistant:` block is present in config.yml. Requires a
long-lived access token. Registers generic tool names like `turn_on` /
`turn_off` — if other integrations are added later, the tool registry's
first-wins collision rule applies.
"""

import datetime as _dt
import functools
import json
import logging
import os
import threading
import time
from collections import Counter
from typing import Optional

import requests

from core.datetime_utils import tts_friendly_event_summary
from core.url_utils import normalize_url

from ._config import config
from .tool_registry import tool as _register_tool

logger = logging.getLogger(__name__)


def tool(*dargs, **dkwargs):
    """`tool_registry.tool`, wrapped so the first call to any HA tool triggers
    the one-time entity-alias / role-entity load (see `_ensure_loaded`).

    Importing this module no longer connects to HA — the load is deferred to
    first use — so a stray import can never reach out to Home Assistant.
    """
    register = _register_tool(*dargs, **dkwargs)

    def decorate(fn):
        @functools.wraps(fn)  # preserves __name__/__doc__/signature for the registry
        def wrapper(*args, **kwargs):
            _ensure_loaded()
            return fn(*args, **kwargs)
        return register(wrapper)
    return decorate

HA_CONFIG = config.get('home_assistant', {})

# Home Assistant connection settings.
# The long-lived token (effectively root on your smart home) is read from the
# FULLOCH_HA_TOKEN environment variable first — set it in `.env` to keep it out
# of plaintext config.yml — and falls back to home_assistant.token in config.yml
# for backward compatibility.
# Normalise so a missing scheme / trailing slash doesn't break requests, e.g.
# "host:8123/" -> "http://host:8123" (not "host:8123//api/states").
HA_URL = normalize_url(HA_CONFIG.get('url', 'http://localhost:8123'))
HA_TOKEN = os.environ.get('FULLOCH_HA_TOKEN', '').strip() or HA_CONFIG.get('token', '')
TIMEOUT = HA_CONFIG.get('timeout', 10)

# Default lookback for get_entity_history when the agent gives no start date —
# covers a plain "when was X last on" in one call (HA recorder retention, ~10
# days, is the hard ceiling anyway).
HISTORY_DEFAULT_LOOKBACK_DAYS = 7

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


def _fetch_entity_aliases() -> tuple:
    """Fetch every entity's friendly_name from HA at module load.

    Returns `(aliases, aliases_multi)`:
      - `aliases`: lowercased `friendly_name -> entity_id`, first-duplicate-wins
        (later ones are logged and skipped). Used by the exact-match and
        autodetect paths.
      - `aliases_multi`: lowercased `friendly_name -> [entity_id, ...]` keeping
        EVERY entity that shares a name, in registration order. Lets a
        domain-scoped lookup recover a collided entity that the first-wins map
        dropped — e.g. a `climate.*` named "Upstairs" that lost the key to a
        `light.*` also named "Upstairs".

    Both are empty if HA is unreachable or no token is configured — direct
    entity_ids still resolve via the `'.' in name` fallback in `_resolve_entity`.
    """
    if not HA_TOKEN:
        logger.warning("Home Assistant token not configured; entity aliases unavailable")
        return {}, {}

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
                return {}, {}
    if states is None:
        return {}, {}

    aliases: dict = {}
    aliases_multi: dict = {}
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
        bucket = aliases_multi.setdefault(key, [])
        if entity_id not in bucket:
            bucket.append(entity_id)
        if key in aliases and aliases[key] != entity_id:
            logger.debug(
                f"Duplicate key '{key}': keeping "
                f"{aliases[key]}, ignoring {entity_id}"
            )
            continue
        aliases[key] = entity_id

    logger.info(f"Fetched {len(aliases)} entity aliases from Home Assistant")
    return aliases, aliases_multi


# Entity alias map + role entities are loaded lazily on first tool use (see
# `_ensure_loaded`), NOT at import — so importing this module performs no network
# I/O. They start empty/None and are populated once a tool actually runs.
_ENTITY_ALIASES: dict = {}
_ENTITY_ALIASES_MULTI: dict = {}
_loaded = False
_load_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Voice deny-list — entities the user has switched off for voice control via
# the Fulloch dashboard's Entities tab. Fulloch-owned state (not an HA label):
# stored as a JSON array of entity_ids and read live, so toggles take effect
# immediately with no restart and no polling thread. `_call_service` refuses a
# deny-listed entity outright, so locks/alarms can be controlled from the
# secure dashboard but never by voice.
#
# The set is rebound atomically on edit (build-new-then-assign) so a concurrent
# membership test on the turn thread always sees a complete set — no lock needed
# on the read path; `_denylist_lock` only serialises writers (file + rebind).
# ---------------------------------------------------------------------------
_DENYLIST_PATH = os.environ.get(
    "FULLOCH_DENYLIST_PATH", "data/voice_denylist.json"
)
_denylist_lock = threading.Lock()


def _load_denylist() -> frozenset:
    """Read the persisted deny-list. Empty (nothing blocked) if absent/invalid."""
    try:
        with open(_DENYLIST_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return frozenset()
    except Exception as e:
        logger.warning(f"Could not read voice deny-list {_DENYLIST_PATH}: {e}")
        return frozenset()
    if not isinstance(data, list):
        logger.warning(f"Voice deny-list {_DENYLIST_PATH} is not a list; ignoring")
        return frozenset()
    return frozenset(str(e) for e in data)


def _persist_denylist(entities) -> None:
    """Write the deny-list atomically (temp file + os.replace)."""
    directory = os.path.dirname(_DENYLIST_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{_DENYLIST_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(entities), f, indent=2)
    os.replace(tmp, _DENYLIST_PATH)


_DENIED_ENTITIES = _load_denylist()
if _DENIED_ENTITIES:
    logger.info(
        f"Voice deny-list: {len(_DENIED_ENTITIES)} entity(ies) blocked from "
        f"voice control"
    )


def get_denylist() -> set:
    """Return a copy of the entity_ids currently blocked from voice control."""
    return set(_DENIED_ENTITIES)


def set_entity_denied(entity_id: str, denied: bool) -> None:
    """Block (`denied=True`) or unblock an entity for voice control, and persist.

    Takes effect immediately for the next `_call_service` — the set is rebound
    atomically so the reading turn thread never sees a half-updated set.
    """
    global _DENIED_ENTITIES
    with _denylist_lock:
        current = set(_DENIED_ENTITIES)
        if denied:
            current.add(entity_id)
        else:
            current.discard(entity_id)
        _persist_denylist(current)
        _DENIED_ENTITIES = frozenset(current)
    logger.info(
        f"Voice {'blocked' if denied else 'allowed'}: {entity_id} "
        f"({len(_DENIED_ENTITIES)} blocked total)"
    )


def list_entities() -> list:
    """Every known HA entity with its voice allow/deny state, for the dashboard.

    Built from the full alias map (deny-listed entities are kept in the map so
    they remain visible — and re-enableable — in the dashboard). Sorted by
    domain then name.
    """
    _ensure_loaded()  # dashboard path — not a @tool, so load the map explicitly
    denied = _DENIED_ENTITIES
    seen: dict = {}
    for eids in _ENTITY_ALIASES_MULTI.values():
        for eid in eids:
            if eid not in seen:
                seen[eid] = _friendly_for(eid)
    out = [
        {
            "entity_id": eid,
            "name": friendly,
            "domain": _domain_of(eid),
            "denied": eid in denied,
        }
        for eid, friendly in seen.items()
    ]
    out.sort(key=lambda e: (e["domain"], e["name"].lower()))
    return out

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
    # Fall back to the collision multimap so an entity that lost its name to a
    # same-named sibling still speaks as that name ("upstairs") rather than its
    # entity_id slug ("living").
    for friendly, eids in _ENTITY_ALIASES_MULTI.items():
        if entity_id in eids:
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

    # Enforcement backstop: even if a deny-listed entity_id reaches here (e.g.
    # the SLM emitted it verbatim, bypassing the filtered alias map), refuse it
    # outright rather than replanning — we don't want the agent retrying under a
    # different name to slip the control through.
    if entity_id in _DENIED_ENTITIES:
        friendly = _friendly_for(entity_id)
        logger.info(f"Refused voice control of deny-listed entity {entity_id}")
        return f"Sorry, {friendly} isn't available for voice control."

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


def _pick_by_domain(entity_ids: list, domain: str = None) -> str:
    """Choose one entity_id from a name's collision bucket.

    When `domain` is given, prefer the first entity in that domain so a
    `climate.*` named "Upstairs" wins over a `light.*` of the same name for a
    temperature/climate lookup. Falls back to the first (registration order)
    when no domain is given or none matches.
    """
    if domain:
        prefix = f"{domain}."
        for eid in entity_ids:
            if eid.startswith(prefix):
                return eid
    return entity_ids[0]


def _resolve_entity(name: str, domain: str = None) -> str:
    """Resolve a friendly name to an entity_id.

    Tries (in order): exact friendly_name match (domain-preferred when the
    name collides across domains), the name with a trailing "lights"/"group"/
    etc. stripped, the name with each common suffix appended, and finally
    falls back to assuming `name` is already a valid entity_id or constructing
    one from `domain`.
    """
    key = name.lower().strip()
    for filler in _LEADING_FILLERS:
        if key.startswith(filler):
            key = key[len(filler):].strip()
            break

    if key in _ENTITY_ALIASES_MULTI:
        return _pick_by_domain(_ENTITY_ALIASES_MULTI[key], domain)
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
        # Consult the collision multimap so a domain-matching entity is found
        # even when the first-wins single map handed the name to another
        # domain (e.g. "upstairs" -> light.upstairs hiding climate.living).
        for eid in _ENTITY_ALIASES_MULTI.get(candidate, ()):
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
            # Replan rather than dead-end: a malformed JSON `data` arg is the
            # most fragile thing the SLM has to assemble here, so hand control
            # back so it can retry via a typed wrapper or apologise gracefully
            # (mirrors the other HA error paths' `Reactive question:` sentinel).
            return (
                f"Reactive question: The data for the {domain}.{service} call "
                f"wasn't valid JSON. Retry with a more specific tool if one "
                f"fits, otherwise tell the user you couldn't complete that."
            )

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
    """Return (start_iso, end_iso) for a spoken day phrase or ISO date.

    Args:
        day: "today", "tomorrow", "this_week" (also "week"), or a specific
            ISO date "YYYY-MM-DD" for a single-day window.
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
    else:
        # Specific ISO date ("2026-06-26") — single-day window. Falls back to
        # today for any unrecognised string (so a stray phrase never silently
        # queries the wrong arbitrary day; it just defaults to today).
        try:
            d = _dt.date.fromisoformat(str(day).strip())
            start = _dt.datetime.combine(d, _dt.time())
        except (ValueError, AttributeError):
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


def _read_calendars() -> list[str]:
    """Calendars whats_on reads from.

    The autodetected primary calendar PLUS the configured reminder calendar
    Fulloch writes to (`create_calendar_event`). Without the reminder
    calendar here, events Fulloch creates are invisible to its own
    `whats_on` whenever the write target differs from the autodetected read
    target (e.g. config `calendar: "Fulloch"` vs an auto-picked
    `calendar.primary`). Deduped, order-preserving.
    """
    cals: list[str] = []
    for c in (CALENDAR_ENTITY, _reminder_calendar_entity()):
        if c and c not in cals:
            cals.append(c)
    return cals


def _ha_get_events(day: str) -> str:
    """Common body for whats_on and its day-specific aliases."""
    calendars = _read_calendars()
    if not calendars:
        return "No calendar is configured in Home Assistant."

    start_iso, end_iso = _calendar_window(day)
    response = _call_service_with_response(
        "calendar", "get_events",
        {
            "entity_id": calendars,
            "start_date_time": start_iso,
            "end_date_time": end_iso,
        },
    )
    if not response:
        return "I couldn't reach your calendar."

    raw_events: list[dict] = []
    for cal in calendars:
        raw_events.extend((response.get(cal) or {}).get("events") or [])
    events = [_normalise_ha_event(e) for e in raw_events]
    # Merged calendars arrive grouped by source; sort so the spoken summary
    # reads chronologically. Date-only all-day starts sort before that day's
    # timed events (string compare: "2026-06-26" < "2026-06-26T12:00").
    events.sort(key=lambda e: e["start"])
    summary = tts_friendly_event_summary(events)
    # Multi-event days benefit from agent summarisation/filtering; route
    # through the replan loop. The "no events" case stays as a direct
    # spoken result.
    if len(events) >= 2:
        return f"Reactive question: {summary}"
    return summary


@tool(
    name="whats_on",
    description=(
        "Get calendar events for a time period. Pass 'today', 'tomorrow', "
        "'this_week', or a specific date as 'YYYY-MM-DD'."
    ),
    aliases=["calendar", "events", "schedule"],
)
def whats_on(day: str = "today") -> str:
    """Calendar events for today, tomorrow, this week, or a specific date.

    Args:
        day: "today" (default), "tomorrow", "this_week", or an ISO date
            "YYYY-MM-DD" for a single day.
    """
    return _ha_get_events(day)


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
    _ensure_loaded()  # reminder-poll path — not a @tool, so load the map explicitly
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
        "Create a one-off or recurring calendar event in Home Assistant "
        "(reminders, appointments, bin night). recurrence: weekly/daily/monthly "
        "or 'none'. Requires home_assistant.calendar to be configured."
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


# Role entities — populated by _ensure_loaded() on first tool use, not at import.
_DEFAULT_WEATHER_ENTITY = None
SPOTIFY_ENTITY = None
TV_ENTITY = None
AVR_ENTITY = None
CALENDAR_ENTITY = None
TODO_ENTITY = None


def _ensure_loaded() -> None:
    """Fetch the entity alias map + autodetect role entities, once, on first use.

    Deferred out of import time so importing this module has NO network side
    effects: a stray import can't connect to HA. Idempotent and thread-safe.
    With no token configured there's nothing to fetch, so it's a cheap no-op
    that stays re-checkable in case a token is set later.
    """
    global _loaded, _ENTITY_ALIASES, _ENTITY_ALIASES_MULTI
    global _DEFAULT_WEATHER_ENTITY, SPOTIFY_ENTITY, TV_ENTITY, AVR_ENTITY
    global CALENDAR_ENTITY, TODO_ENTITY
    if _loaded or not HA_TOKEN:
        return
    with _load_lock:
        if _loaded:
            return
        _ENTITY_ALIASES, _ENTITY_ALIASES_MULTI = _fetch_entity_aliases()
        _DEFAULT_WEATHER_ENTITY = _autodetect_weather_entity()
        SPOTIFY_ENTITY = _autodetect_spotify_entity()
        TV_ENTITY = _autodetect_tv_entity()
        AVR_ENTITY = _autodetect_avr_entity()
        CALENDAR_ENTITY = _autodetect_calendar_entity()
        TODO_ENTITY = _autodetect_todo_entity()
        _loaded = True


# ---------------------------------------------------------------------------
# Todo / task lists — wraps HA `todo.add_item` and `todo.get_items`.
# ---------------------------------------------------------------------------

@tool(
    name="add_todo_item",
    description=(
        "Add an item to a Home Assistant todo or shopping list ('add eggs'). "
        "Use append_to_note instead when the item needs context or belongs in "
        "a note."
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


# HA returns weather as machine condition slugs (homeassistant.components.weather):
# some are concatenated with no separator ("partlycloudy"), some are semantically
# off for speech ("exceptional", "windy-variant"). A bare "-"→" " swap can't fix
# the no-separator slug or the valid-but-wrong words — and an unsplit
# "partlycloudy" is silently *dropped* by the CPU TTS front-end (misaki has no
# entry for the run-on token), so the word just vanishes from the forecast. Map
# the closed set explicitly; unknown values fall back to a hyphen swap. Improves
# every TTS backend, not only the tiny one.
_WEATHER_CONDITIONS = {
    "clear-night": "clear",
    "cloudy": "cloudy",
    "exceptional": "severe weather",
    "fog": "foggy",
    "hail": "hail",
    "lightning": "thunderstorms",
    "lightning-rainy": "thunderstorms",
    "partlycloudy": "partly cloudy",
    "pouring": "heavy rain",
    "rainy": "rainy",
    "snowy": "snowy",
    "snowy-rainy": "sleet",
    "sunny": "sunny",
    "windy": "windy",
    "windy-variant": "windy",
}


def _humanize_condition(state: Optional[str]) -> str:
    """HA weather condition slug → spoken phrase (closed set; hyphen-swap fallback)."""
    slug = (state or "").strip().lower()
    if not slug:
        return ""
    return _WEATHER_CONDITIONS.get(slug, slug.replace("-", " "))


def _format_day(label: str, day: dict, unit: str) -> str:
    cond = _humanize_condition(day.get("condition"))
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


def _weather_history_summary(entity_id: str, start: _dt.date, days: int, unit: str) -> str:
    """Query HA state history for a weather entity and summarise past conditions."""
    start_dt = _dt.datetime(start.year, start.month, start.day, 0, 0, 0).astimezone()
    end_date = start + _dt.timedelta(days=days)
    end_dt = _dt.datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0).astimezone()
    now_dt = _dt.datetime.now().astimezone()
    if end_dt > now_dt:
        end_dt = now_dt
    try:
        resp = requests.get(
            f"{HA_URL}/api/history/period/{start_dt.isoformat()}",
            headers=_get_headers(),
            params={"end_time": end_dt.isoformat(), "filter_entity_id": entity_id},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        hist = (resp.json() or [[]])[0]
    except Exception as e:
        logger.warning(f"HA weather history for {entity_id} failed: {e}")
        return f"Reactive question: Could not fetch weather history for {entity_id!r}. Tell the user there was an error."

    if not hist:
        return f"No weather history recorded for {_friendly_for(entity_id)} on {start}."

    today = _dt.date.today()
    yesterday = today - _dt.timedelta(days=1)
    daily: dict = {}
    for s in hist:
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = _dt.datetime.fromisoformat(ts_str).astimezone()
            day = ts.date()
        except Exception:
            continue
        entry = daily.setdefault(day, {"conditions": [], "temps": []})
        cond = _humanize_condition(s.get("state"))
        if cond and cond not in ("unavailable", "unknown"):
            entry["conditions"].append(cond)
        temp = (s.get("attributes") or {}).get("temperature")
        if temp is not None:
            entry["temps"].append(float(temp))

    if not daily:
        return f"No usable weather data found for {start}."

    parts = [f"Weather history for {_friendly_for(entity_id).replace('forecast', '').strip()}"]
    for day in sorted(daily):
        data = daily[day]
        label = "Today" if day == today else "Yesterday" if day == yesterday else day.strftime("%A")
        cond = Counter(data["conditions"]).most_common(1)[0][0] if data["conditions"] else "unknown"
        bits = [f"{label} {cond}"]
        if data["temps"]:
            lo, hi = round(min(data["temps"])), round(max(data["temps"]))
            bits.append(f"{lo} to {hi} {unit}" if lo != hi else f"{lo} {unit}")
        parts.append(" ".join(bits))
    return ". ".join(parts) + "."


@tool(
    name="get_weather_forecast",
    description=(
        "Weather forecast from Home Assistant — always use this for weather, "
        "never web search. start_date is an ISO date (omit for today); days is "
        "1–7 consecutive days (default 2). Compute exact dates from the system "
        "prompt's current date."
    ),
    aliases=["weather", "forecast", "get_weather"],
)
def get_weather_forecast(
    start_date: Optional[str] = None,
    days: int = 2,
    location: Optional[str] = None,
) -> str:
    """Return N days of forecast beginning from start_date (default today).

    Args:
        start_date: First date to include, ISO format (YYYY-MM-DD). Defaults to today.
        days: Number of consecutive days to return (1–7, default 2).
        location: Optional weather entity name. Defaults to the first
            `weather.*` entity in HA (or `weather.home` if none found).
    """
    days = max(1, min(days, 7))
    today = _dt.date.today()
    if start_date:
        try:
            start = _dt.date.fromisoformat(start_date)
        except ValueError:
            start = today
    else:
        start = today

    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    entity_id = _DEFAULT_WEATHER_ENTITY
    if location:
        candidate = _resolve_entity(location, domain="weather")
        if candidate.startswith("weather."):
            entity_id = candidate

    state = _get_state(entity_id)
    if state is None:
        return (
            f"Reactive question: Weather entity {entity_id!r} not found in "
            f"Home Assistant. Tell the user which entity to configure, or "
            f"suggest renaming a `weather.*` entity so autodetect picks it up."
        )

    attrs = state.get("attributes", {})
    unit = _temperature_unit(attrs)

    if start < today:
        return _weather_history_summary(entity_id, start, days, unit)

    current_cond = _humanize_condition(state.get("state")) or "unknown"
    current_temp = attrs.get("temperature")

    parts = [f"Forecast for {_friendly_for(entity_id).replace('forecast', '')}"]
    if start == today:
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

    count = 0
    for day in forecast:
        raw_dt = day.get("datetime") or ""
        try:
            day_date = _dt.date.fromisoformat(raw_dt[:10])
        except ValueError:
            continue
        if day_date < start:
            continue
        if count >= days:
            break
        if day_date == today:
            label = "Today"
        elif day_date == today + _dt.timedelta(days=1):
            label = "Tomorrow"
        else:
            label = day_date.strftime("%A")
        parts.append(_format_day(label, day, unit))
        count += 1

    return ". ".join(parts) + "."


def _parse_history_start(start: str):
    """Parse a history `start` arg into (start_dt, default_end_dt, date_only).

    Accepts an ISO date ('2026-06-02' → whole-day window) or ISO datetime
    ('2026-06-02T12:00:00' → start+2h default window). Naive values are
    localised to the host timezone. Raises ValueError on an unparseable arg.
    """
    if "T" in start or " " in start:
        start_dt = _dt.datetime.fromisoformat(start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.astimezone()
        return start_dt, start_dt + _dt.timedelta(hours=2), False
    d = _dt.date.fromisoformat(start)
    start_dt = _dt.datetime(d.year, d.month, d.day, 0, 0, 0).astimezone()
    return start_dt, start_dt + _dt.timedelta(days=1) - _dt.timedelta(seconds=1), True


def _parse_history_end(end: Optional[str], default_end_dt):
    """Parse an optional history `end` arg; fall back to `default_end_dt`.

    A date-only end resolves to the end of that day. An unparseable end
    silently degrades to the default rather than failing the whole query.
    """
    if not end:
        return default_end_dt
    try:
        if "T" in end or " " in end:
            end_dt = _dt.datetime.fromisoformat(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.astimezone()
        else:
            d = _dt.date.fromisoformat(end)
            end_dt = _dt.datetime(d.year, d.month, d.day, 23, 59, 59).astimezone()
    except ValueError:
        return default_end_dt
    return end_dt


def _fetch_history_states(entity_id: str, start_dt, end_dt) -> list:
    """GET /api/history/period for one entity; return its list of state dicts.

    Returns an empty list when HA has no recorded changes in the window.
    Raises on a transport/HTTP error so the caller can surface a sentinel.
    """
    response = requests.get(
        f"{HA_URL}/api/history/period/{start_dt.isoformat()}",
        headers=_get_headers(),
        params={
            "end_time": end_dt.isoformat(),
            "filter_entity_id": entity_id,
            "minimal_response": "true",
            "no_attributes": "true",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    history = response.json()
    if not history or not history[0]:
        return []
    return history[0]


@tool(
    name="get_entity_history",
    description=(
        "When a Home Assistant entity changed state — any entity, plus "
        "Fulloch's own conversation sensors 'sensor.fulloch_last_utterance' "
        "(what was said) and 'sensor.fulloch_last_response' (what Fulloch "
        "replied). start/end take an ISO date (whole-day window) or datetime. "
        "Omit start for a recent-history question like 'when was X last on' — "
        "it defaults to the last 7 days. end defaults to end-of-day for a date "
        "start, or start+2h for a datetime start. Returns the state changes in "
        "the window; answer the user's actual question from them (e.g. for 'when "
        "did X last turn on' give just the most recent on)."
    ),
    aliases=["entity_history", "check_history", "light_history"],
)
def get_entity_history(
    entity: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> str:
    """Return state-change history for a HA entity over a time window.

    Returns the raw list of state changes; the agent loop hands it back for a
    composing replan (see ``intents.LOOKUP_TOOLS``) so a focused question like
    "when did the lights last turn on" is answered from the records rather than
    read aloud in full.

    Args:
        entity: Friendly name or entity_id.
        start: Start of the time window (ISO date or datetime). Omit (or pass
            empty) for a recent-history question — defaults to
            ``HISTORY_DEFAULT_LOOKBACK_DAYS`` ago through now, so a plain "when
            was X last on" resolves in one agent call without picking a date.
        end: End of the time window (ISO date or datetime). Defaults to end of day
            for a date-only start, or start+2h for a datetime start.
    """
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    entity_id = _resolve_entity(entity)
    # If resolution failed to produce a valid entity_id, try domain-specific fallbacks
    if "." not in entity_id:
        weather_terms = {"weather", "forecast", "temperature", "climate", "outdoor", "outside"}
        if any(t in entity.lower() for t in weather_terms):
            entity_id = _DEFAULT_WEATHER_ENTITY
        else:
            return (
                f"Reactive question: Could not resolve '{entity}' to a Home Assistant entity. "
                f"Ask the user for the exact entity ID (e.g. 'light.kitchen', 'sensor.outdoor_temperature')."
            )

    if not start or not start.strip():
        # No date given — default to a recent window so "when was X last on"
        # answers in one agent call instead of a replan just to pick a start.
        start_dt = (
            _dt.datetime.now() - _dt.timedelta(days=HISTORY_DEFAULT_LOOKBACK_DAYS)
        ).astimezone()
        default_end_dt = _dt.datetime.now().astimezone()
    else:
        try:
            if "T" in start or " " in start:
                start_dt = _dt.datetime.fromisoformat(start)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.astimezone()
                default_end_dt = start_dt + _dt.timedelta(hours=2)
            else:
                d = _dt.date.fromisoformat(start)
                start_dt = _dt.datetime(d.year, d.month, d.day, 0, 0, 0).astimezone()
                default_end_dt = start_dt + _dt.timedelta(days=1) - _dt.timedelta(seconds=1)
        except ValueError:
            return f"Reactive question: Could not parse start '{start}'. Ask the user to clarify the date."

    if end:
        try:
            if "T" in end or " " in end:
                end_dt = _dt.datetime.fromisoformat(end)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.astimezone()
            else:
                d = _dt.date.fromisoformat(end)
                end_dt = _dt.datetime(d.year, d.month, d.day, 23, 59, 59).astimezone()
        except ValueError:
            end_dt = default_end_dt
    else:
        end_dt = default_end_dt

    now = _dt.datetime.now().astimezone()
    if end_dt > now:
        end_dt = now

    try:
        response = requests.get(
            f"{HA_URL}/api/history/period/{start_dt.isoformat()}",
            headers=_get_headers(),
            params={
                "end_time": end_dt.isoformat(),
                "filter_entity_id": entity_id,
                "minimal_response": "true",
                "no_attributes": "true",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        history = response.json()
    except Exception as e:
        logger.warning(f"HA history query for {entity_id} failed: {e}")
        return f"Reactive question: Could not fetch history for '{entity}'. Tell the user there was an error."

    if not history or not history[0]:
        if start and start.strip():
            window = f"{start}" + (f" to {end}" if end else "")
            window = f"on {window}"
        else:
            window = f"in the last {HISTORY_DEFAULT_LOOKBACK_DAYS} days"
        return f"No recorded state changes for {_friendly_for(entity_id)} {window}."

    states = history[0]
    friendly = _friendly_for(entity_id)
    today = _dt.date.today()
    yesterday = today - _dt.timedelta(days=1)

    def _when(s) -> str:
        """A spoken 'today/yesterday/<weekday> at <time>' label for one state."""
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = _dt.datetime.fromisoformat(ts_str).astimezone()
            ts_date = ts.date()
            if ts_date == today:
                day_label = "today"
            elif ts_date == yesterday:
                day_label = "yesterday"
            else:
                day_label = ts.strftime("%A")
            return f"{day_label} at {ts.strftime('%I:%M %p').lstrip('0')}"
        except Exception:
            return ts_str

    MAX_RESULTS = 15
    truncated = len(states) > MAX_RESULTS
    if truncated:
        states = states[-MAX_RESULTS:]

    lines = [f"History for {friendly}"]
    for s in states:
        lines.append(f"{_when(s)}: {s.get('state', 'unknown')}")

    if truncated:
        lines.append(f"showing the last {MAX_RESULTS} changes only")

    return ". ".join(lines) + "."


@tool(
    name="get_conversation_history",
    description=(
        "Recall an earlier conversation — both the user's questions and your "
        "own replies, interleaved — when the turns are NOT already in the "
        "current chat history. start/end take an ISO date (whole day) or "
        "datetime; end defaults to end-of-day for a date start, or start+2h "
        "for a datetime start."
    ),
    aliases=["conversation_history", "recall_conversation", "what_did_we_discuss"],
)
def get_conversation_history(start: str, end: Optional[str] = None) -> str:
    """Interleave Fulloch's utterance + response sensors into a Q/A transcript.

    Returns a `Reactive question:` sentinel wrapping the transcript so the
    agent loop re-plans and SUMMARISES the topics rather than reading the
    raw timestamped list back line by line.
    """
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    try:
        start_dt, default_end_dt, _ = _parse_history_start(start)
    except ValueError:
        return f"Reactive question: Could not parse start '{start}'. Ask the user to clarify the date."

    end_dt = _parse_history_end(end, default_end_dt)
    now = _dt.datetime.now().astimezone()
    if end_dt > now:
        end_dt = now

    try:
        user_states = _fetch_history_states("sensor.fulloch_last_utterance", start_dt, end_dt)
        bot_states = _fetch_history_states("sensor.fulloch_last_response", start_dt, end_dt)
    except Exception as e:
        logger.warning(f"HA conversation history query failed: {e}")
        return "Reactive question: Could not fetch the conversation history. Tell the user there was an error."

    # Merge both sensors into a single timeline tagged by speaker, then sort
    # by timestamp so each user question sits next to the reply it drew.
    events = [(s, "You") for s in user_states] + [(s, "Fulloch") for s in bot_states]
    events.sort(key=lambda it: it[0].get("last_changed") or it[0].get("last_updated") or "")

    _SKIP = {"unknown", "unavailable", ""}
    today = _dt.date.today()
    yesterday = today - _dt.timedelta(days=1)
    lines: list = []
    for s, speaker in events:
        state_val = (s.get("state") or "").strip()
        if state_val.lower() in _SKIP:
            continue
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = _dt.datetime.fromisoformat(ts_str).astimezone()
            ts_date = ts.date()
            if ts_date == today:
                day_label = "today"
            elif ts_date == yesterday:
                day_label = "yesterday"
            else:
                day_label = ts.strftime("%A")
            time_label = ts.strftime("%I:%M %p").lstrip("0")
            lines.append(f"{day_label} at {time_label} — {speaker}: {state_val}")
        except Exception:
            lines.append(f"{speaker}: {state_val}")

    if not lines:
        window = f"{start}" + (f" to {end}" if end else "")
        return f"No recorded conversation for {window}."

    # Keep the most recent exchanges if the window is busy — the tail is what
    # "what did we talk about" usually means, and caps the replan payload.
    MAX_RESULTS = 30
    if len(lines) > MAX_RESULTS:
        lines = lines[-MAX_RESULTS:]

    transcript = "\n".join(lines)
    return (
        "Reactive question: Below is the earlier conversation transcript "
        "(the user's questions and your replies). Summarise for the user what "
        "was discussed, grouped by topic, in a sentence or two — do NOT read it "
        "back line by line, and do NOT re-research those topics with another "
        "tool.\n" + transcript
    )
