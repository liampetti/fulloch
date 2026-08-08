"""Home Assistant integration via REST API.

Loaded when the `home_assistant:` block is present in config.yml. Requires a
long-lived access token. Registers generic tool names like `turn_on` /
`turn_off` — if other integrations are added later, the tool registry's
first-wins collision rule applies.
"""

import datetime as _dt
import difflib
import functools
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from typing import Optional

import requests

import utils.local_time as _local_tz
from core.datetime_utils import tts_friendly_event_summary
from core.satellite_context import current_satellite_id, get_current_assistant
from core.url_utils import normalize_url

from ._config import config
from .tool_registry import tool as _register_tool

logger = logging.getLogger(__name__)


def tool(*dargs, **dkwargs):
    """`tool_registry.tool`, wrapped so the first call to any HA tool triggers
    the one-time entity-alias / role-entity load (see `_ensure_loaded`).

    Importing this module no longer connects to HA — the load is deferred to
    first use — so a stray import can never reach out to Home Assistant.

    Registration itself is gated on `"home_assistant" in config`. This
    module can get imported as a side effect even when HA isn't configured
    at all — `tools/spotify.py` imports it at module level to reuse its
    area-resolution/service-call helpers — so without this guard, every HA
    tool (turn_on, locks, climate, ...) would leak into the SLM's tool
    registry for a Spotify-only user, advertising dozens of tools that can
    only ever fail. The underlying function is still fully defined and
    directly callable either way (e.g. `tools/spotify.py`'s
    `_spotify_transport_fallback` calls `tools.spotify`'s own helpers, and
    HA's own tools call each other, as plain functions — never through the
    registry) — only the SLM-facing registration is skipped.
    """
    register = _register_tool(*dargs, **dkwargs)

    def decorate(fn):
        @functools.wraps(fn)  # preserves __name__/__doc__/signature for the registry
        def wrapper(*args, **kwargs):
            _ensure_loaded()
            return fn(*args, **kwargs)

        if "home_assistant" not in config:
            return wrapper
        return register(wrapper)

    return decorate


HA_CONFIG = config.get("home_assistant", {})

HA_URL = normalize_url(HA_CONFIG.get("url", "http://localhost:8123"))
HA_TOKEN = os.environ.get("HA_TOKEN", "").strip()
TIMEOUT = HA_CONFIG.get("timeout", 10)

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
    "avr",
    "receiver",
    "pioneer",
    "onkyo",
    "denon",
    "yamaha",
    "marantz",
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
                f"{HA_URL}/api/states",
                headers=_get_headers(),
                timeout=TIMEOUT,
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
        entity_id = state.get("entity_id")
        if not entity_id:
            continue
        friendly = state.get("attributes", {}).get("friendly_name")
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
            logger.debug(f"Duplicate key '{key}': keeping {aliases[key]}, ignoring {entity_id}")
            continue
        aliases[key] = entity_id

    logger.info(f"Fetched {len(aliases)} entity aliases from Home Assistant")
    return aliases, aliases_multi


# Entity alias map + role entities are loaded lazily on first tool use (see
# `_ensure_loaded`), NOT at import — so importing this module performs no network
# I/O. They start empty/None and are populated once a tool actually runs.
_ENTITY_ALIASES: dict = {}
_ENTITY_ALIASES_MULTI: dict = {}
# area_id -> display name, populated by `_fetch_area_map` in `_ensure_loaded`.
# HA's REST API has no area/entity/device registry endpoints, so this goes
# through `/api/template` (Jinja `areas()` / `area_name()` / `area_entities()`
# built-ins) rather than a dedicated registry fetch.
_AREA_MAP: dict = {}
# floor_id -> display name, populated alongside areas. Floors contain areas;
# they let users ask for an inventory or status of a whole storey.
_FLOOR_MAP: dict = {}
_loaded = False
_load_lock = threading.Lock()


def _render_template(template: str) -> Optional[str]:
    """Render a Jinja template server-side via HA's `/api/template` endpoint.

    Returns the rendered text, or None if HA is unreachable/unconfigured.
    """
    if not HA_TOKEN:
        return None
    url = f"{HA_URL}/api/template"
    try:
        response = requests.post(
            url,
            headers=_get_headers(),
            json={"template": template},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"HA template render failed: {e}")
        return None


def _fetch_area_map() -> dict:
    """Fetch `{area_id: display_name}` for every HA area via templates.

    Empty if HA is unreachable/unconfigured or has no areas defined — callers
    degrade to "I don't have area information" rather than failing.
    """
    ids_raw = _render_template("{{ areas() | list | tojson }}")
    if ids_raw is None:
        return {}
    try:
        area_ids = json.loads(ids_raw)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse HA areas() template response: {ids_raw!r}")
        return {}
    names_raw = _render_template("{{ areas() | map('area_name') | list | tojson }}")
    try:
        names = json.loads(names_raw) if names_raw else []
    except (ValueError, TypeError):
        names = []
    if len(names) != len(area_ids):
        names = area_ids
    logger.info(f"Fetched {len(area_ids)} areas from Home Assistant")
    return {area_id: (name or area_id) for area_id, name in zip(area_ids, names, strict=True)}


def _fetch_floor_map() -> dict:
    """Fetch `{floor_id: display_name}` for every HA floor via templates."""
    ids_raw = _render_template("{{ floors() | list | tojson }}")
    if ids_raw is None:
        return {}
    try:
        floor_ids = json.loads(ids_raw)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse HA floors() template response: {ids_raw!r}")
        return {}
    names_raw = _render_template("{{ floors() | map('floor_name') | list | tojson }}")
    try:
        names = json.loads(names_raw) if names_raw else []
    except (ValueError, TypeError):
        names = []
    if len(names) != len(floor_ids):
        names = floor_ids
    logger.info(f"Fetched {len(floor_ids)} floors from Home Assistant")
    return {floor_id: (name or floor_id) for floor_id, name in zip(floor_ids, names, strict=True)}


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
_DENYLIST_PATH = os.environ.get("FULLOCH_DENYLIST_PATH", "data/voice_denylist.json")
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
    logger.info(f"Voice deny-list: {len(_DENIED_ENTITIES)} entity(ies) blocked from voice control")


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


def list_areas() -> list:
    """Every known HA area as `{"id": area_id, "name": display_name}`, for the
    dashboard's browser-satellite area picker (6b). Sorted by name."""
    _ensure_loaded()  # dashboard path — not a @tool, so load the map explicitly
    out = [{"id": area_id, "name": name} for area_id, name in _AREA_MAP.items()]
    out.sort(key=lambda a: a["name"].lower())
    return out


# Trailing words a speaker is likely to add or drop when referring to a device
# (e.g. "downstairs office" vs "downstairs office lights").
_NAME_SUFFIXES = (
    "lights",
    "light",
    "lamp",
    "lamps",
    "bulb",
    "bulbs",
    "group",
    "switch",
    "switches",
    "fan",
    "fans",
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
    entity_id,
    data: Optional[dict] = None,
    success_message: Optional[str] = None,
) -> str:
    """Call a Home Assistant service and return a TTS-friendly response.

    `entity_id` is normally a single entity_id string, but may be a list —
    used by the per-satellite lights area-default (#14 6b) to target every
    light entity in a room with one service call. Callers passing a list
    must have already filtered out denylisted entities themselves (e.g. via
    `_bare_light_area_entities`) — the single-entity denylist backstop below
    only applies to the string case, and always passing `success_message`
    for the list case avoids needing a `_friendly_for` that understands lists.
    """
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    # Enforcement backstop: even if a deny-listed entity_id reaches here (e.g.
    # the SLM emitted it verbatim, bypassing the filtered alias map), refuse it
    # outright rather than replanning — we don't want the agent retrying under a
    # different name to slip the control through.
    if isinstance(entity_id, str) and entity_id in _DENIED_ENTITIES:
        friendly = _friendly_for(entity_id)
        logger.info(f"Refused voice control of deny-listed entity {entity_id}")
        return f"Sorry, {friendly} isn't available for voice control."

    url = f"{HA_URL}/api/services/{domain}/{service}"
    payload = {"entity_id": entity_id}
    if data:
        payload.update(data)

    friendly = _friendly_for(entity_id) if isinstance(entity_id, str) else "those"
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
        logger.warning(f"HA {domain}.{service} on {entity_id} failed: {status} {e.response.text}")
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

    Use for services like `calendar.get_events` that need ?return_response=true.
    Returns the `service_response` block of the JSON, or None on any error.
    """
    if not HA_TOKEN:
        return None
    url = f"{HA_URL}/api/services/{domain}/{service}?return_response=true"
    try:
        response = requests.post(
            url,
            headers=_get_headers(),
            json=payload,
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


_SINGULARIZE_SUFFIXES = frozenset({"shes", "ches", "xes", "zes", "sses"})


def _singularize(word: str) -> str:
    """Crude English singularizer for entity-name token matching.

    Handles the plural patterns actually found in home-automation entity
    names (simple -s, -es after sibilants, -ies, -ves) without a dependency
    on a full NLP stemmer.  Input shorter than 4 characters is left alone
    to avoid stripping real words like ``gas`` or ``bus``.
    """
    if len(word) <= 3:
        return word
    # -ies → -y  (berries → berry, batteries → battery)
    if word.endswith("ies"):
        return word[:-3] + "y"
    # -ves → -f (leaves → leaf, shelves → shelf)
    if word.endswith("ves"):
        return word[:-3] + "f"
    # -shes / -ches / -xes / -zes / -sses → strip "es"
    if word.endswith(tuple(_SINGULARIZE_SUFFIXES)):
        return word[:-2]
    # Simple -s, guarded against -ss and -us singulars
    if word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
        return word[:-1]
    return word


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
            key = key[len(filler) :].strip()
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
    #
    # Both sides are singularized so plurals match their singular forms
    # ("lights" ≈ "light", "switches" ≈ "switch", "berries" ≈ "berry").
    input_tokens = {_singularize(t) for t in key.split()}
    if input_tokens:
        fuzzy = [
            (alias, eid)
            for alias, eid in _ENTITY_ALIASES.items()
            if input_tokens.issubset({_singularize(t) for t in alias.split()})
        ]
        if fuzzy:
            fuzzy.sort(key=lambda kv: len(kv[0]))
            chosen_alias, chosen_eid = fuzzy[0]
            logger.debug(f"Fuzzy-resolved {name!r} → {chosen_eid} via {chosen_alias!r}")
            return chosen_eid

    if "." in name:
        return name

    if domain:
        entity_name = key.replace(" ", "_")
        return f"{domain}.{entity_name}"

    return name


def _resolve_area(name: str) -> Optional[str]:
    """Resolve a spoken room/zone name to an HA area_id via `_AREA_MAP`.

    Tries an exact match on area_id or display name first, then a
    token-superset fuzzy match on the display name (mirrors `_resolve_entity`).
    Returns None if nothing matches (no "assume it's an id" fallback — a bad
    area name has no service call to silently mis-target).
    """
    key = name.lower().strip()
    for filler in _LEADING_FILLERS:
        if key.startswith(filler):
            key = key[len(filler) :].strip()
            break

    for area_id, area_name in _AREA_MAP.items():
        if key == area_id.lower() or key == area_name.lower():
            return area_id

    # A floor name must not fuzzy-match one of its child areas. For example,
    # "upstairs" is a floor, not the "Upstairs Bathroom" area.
    for floor_id, floor_name in _FLOOR_MAP.items():
        if key == floor_id.lower() or key == floor_name.lower():
            return None

    input_tokens = {_singularize(t) for t in key.split()}
    if input_tokens:
        fuzzy = [
            (area_id, area_name)
            for area_id, area_name in _AREA_MAP.items()
            if input_tokens.issubset({_singularize(t) for t in area_name.lower().split()})
        ]
        if fuzzy:
            fuzzy.sort(key=lambda kv: len(kv[1]))
            return fuzzy[0][0]

    return None


def _resolve_floor(name: str) -> Optional[str]:
    """Resolve a spoken floor/storey name to an HA floor_id."""
    key = name.lower().strip()
    for filler in _LEADING_FILLERS:
        if key.startswith(filler):
            key = key[len(filler) :].strip()
            break

    for floor_id, floor_name in _FLOOR_MAP.items():
        if key == floor_id.lower() or key == floor_name.lower():
            return floor_id

    input_tokens = {_singularize(t) for t in key.split()}
    if input_tokens:
        fuzzy = [
            (floor_id, floor_name)
            for floor_id, floor_name in _FLOOR_MAP.items()
            if input_tokens.issubset({_singularize(t) for t in floor_name.lower().split()})
        ]
        if fuzzy:
            fuzzy.sort(key=lambda kv: len(kv[1]))
            return fuzzy[0][0]

    return None


# Per-satellite default HA area for lights (#14 6b). Deliberately scoped to
# lights only — locks/covers/fans/climate could theoretically want the same
# treatment later, but each needs its own bare-word recognition and, for
# locks especially, "the door" is often genuinely ambiguous even within one
# room. Extend one domain at a time if a real need shows up.
_LIGHT_WORDS = frozenset({"light", "lights", "lamp", "lamps"})
# A command naming "all"/"every" must never be scoped down to the satellite's
# own room — it's an explicit request for the whole house, and the existing
# alias/group resolution (e.g. a configured "all lights" group entity)
# already knows how to handle that; the area default must not intercept it.
_ALL_QUALIFIER_WORDS = frozenset({"all", "every", "everything"})


def _is_bare_light_phrase(key: str) -> bool:
    """True if `key` (already lowercased/filler-stripped) refers to lights in
    general with no specific room named and no "all"/"every" qualifier —
    e.g. "lights"/"lamps", but not "kitchen lights" or "all the lights". Only
    phrases like this are eligible for the satellite's default-area
    fallback; anything else (an explicit room, or an explicit "all") always
    goes through the pre-existing resolution unchanged.
    """
    tokens = key.split()
    if not tokens:
        return False
    if any(t in _ALL_QUALIFIER_WORDS for t in tokens):
        return False
    return all(t in _LIGHT_WORDS for t in tokens)


def _current_satellite_ha_area() -> Optional[str]:
    """The calling satellite's configured `ha_area`, or None if there isn't
    a live assistant, the satellite has disconnected, or no area is set."""
    assistant = get_current_assistant()
    if assistant is None:
        return None
    sid = current_satellite_id.get()
    if not sid:
        return None
    session = assistant.satellites.get(sid)
    if session is None:
        return None
    return session.ha_area


def _bare_light_area_entities(entity: str) -> Optional[tuple[list, str]]:
    """If `entity` is a bare lights phrase and the calling satellite has a
    configured `ha_area` containing at least one (non-denylisted) light
    entity, return `(light_entity_ids, area_display_name)`. Returns None
    when the fallback doesn't apply — no satellite area configured, HA
    unreachable, or the area has no lights — so the caller falls through to
    the pre-existing single-entity resolution instead.
    """
    key = entity.lower().strip()
    for filler in _LEADING_FILLERS:
        if key.startswith(filler):
            key = key[len(filler) :].strip()
            break
    if not _is_bare_light_phrase(key):
        return None

    ha_area = _current_satellite_ha_area()
    if not ha_area:
        return None
    area_id = _resolve_area(ha_area)
    if area_id is None:
        return None

    raw = _render_template(f"{{{{ area_entities({area_id!r}) | list | tojson }}}}")
    if raw is None:
        return None
    try:
        entity_ids = json.loads(raw)
    except (ValueError, TypeError):
        return None

    lights = [eid for eid in entity_ids if _domain_of(eid) == "light" and eid not in _DENIED_ENTITIES]
    if not lights:
        return None
    area_name = _AREA_MAP.get(area_id, ha_area)
    return lights, area_name


@tool(
    name="turn_on",
    description="Turn on a device, light, switch, or other Home Assistant entity",
    aliases=["ha_turn_on", "switch_on", "turn_on_device"],
)
def turn_on(entity: str, brightness: Optional[int] = None) -> str:
    """Turn on a Home Assistant entity.

    Args:
        entity: Entity name or ID (e.g., 'living room lights', 'light.living_room')
        brightness: Optional brightness percentage (0-100) for lights
    """
    area_lights = _bare_light_area_entities(entity)
    if area_lights is not None:
        light_ids, area_name = area_lights
        data = {}
        success = f"Lights on in {area_name}"
        if brightness is not None:
            data["brightness"] = int((brightness / 100) * 255)
            success = f"Lights on in {area_name} at {brightness} percent"
        return _call_service("light", "turn_on", light_ids, data if data else None, success)

    entity_id = _resolve_entity(entity, domain="light")
    domain = _domain_of(entity_id)
    friendly = _friendly_for(entity_id)

    data = {}
    success = f"{friendly} on"
    if brightness is not None and domain == "light":
        data["brightness"] = int((brightness / 100) * 255)
        success = f"{friendly} on at {brightness} percent"

    return _call_service(domain, "turn_on", entity_id, data if data else None, success)


@tool(
    name="turn_off",
    description="Turn off a device, light, switch, or other Home Assistant entity",
    aliases=["ha_turn_off", "switch_off", "turn_off_device"],
)
def turn_off(entity: str) -> str:
    """Turn off a Home Assistant entity.

    Args:
        entity: Entity name or ID (e.g., 'living room lights', 'light.living_room')
    """
    area_lights = _bare_light_area_entities(entity)
    if area_lights is not None:
        light_ids, area_name = area_lights
        return _call_service(
            "light", "turn_off", light_ids, success_message=f"Lights off in {area_name}"
        )

    entity_id = _resolve_entity(entity)
    domain = _domain_of(entity_id)
    friendly = _friendly_for(entity_id)

    return _call_service(domain, "turn_off", entity_id, success_message=f"{friendly} off")


@tool(
    name="toggle",
    description="Toggle a Home Assistant entity on or off",
    aliases=["ha_toggle", "toggle_device"],
)
def toggle(entity: str) -> str:
    """Toggle a Home Assistant entity.

    Args:
        entity: Entity name or ID (e.g., 'living room lights', 'light.living_room')
    """
    area_lights = _bare_light_area_entities(entity)
    if area_lights is not None:
        light_ids, area_name = area_lights
        return _call_service(
            "light", "toggle", light_ids, success_message=f"Toggled the lights in {area_name}"
        )

    entity_id = _resolve_entity(entity)
    domain = _domain_of(entity_id)
    friendly = _friendly_for(entity_id)

    return _call_service(domain, "toggle", entity_id, success_message=f"Toggled {friendly}")


@tool(
    name="ha_set_brightness",
    description="Set the brightness of a light in Home Assistant",
    aliases=["ha_brightness", "ha_dim_light"],
)
def set_ha_brightness(entity: str, brightness: int) -> str:
    """Set the brightness of a light.

    Args:
        entity: Light entity name or ID
        brightness: Brightness percentage (0-100)
    """
    brightness = max(0, min(100, brightness))
    brightness_255 = int((brightness / 100) * 255)

    area_lights = _bare_light_area_entities(entity)
    if area_lights is not None:
        light_ids, area_name = area_lights
        return _call_service(
            "light",
            "turn_on",
            light_ids,
            {"brightness": brightness_255},
            success_message=f"Lights in {area_name} at {brightness} percent",
        )

    entity_id = _resolve_entity(entity, domain="light")
    friendly = _friendly_for(entity_id)

    return _call_service(
        "light",
        "turn_on",
        entity_id,
        {"brightness": brightness_255},
        success_message=f"{friendly} at {brightness} percent",
    )


@tool(
    name="ha_set_color",
    description="Set the color of a light in Home Assistant using color name or RGB",
    aliases=["ha_color", "change_light_color"],
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
    elif "," in color:
        try:
            rgb = [int(c.strip()) for c in color.split(",")]
            if len(rgb) != 3:
                return "I need three numbers for an RGB colour."
        except ValueError:
            return f"I couldn't read the colour {color}."
    else:
        return f"I don't know the colour {color}."

    friendly = _friendly_for(entity_id)
    return _call_service(
        "light",
        "turn_on",
        entity_id,
        {"rgb_color": rgb},
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
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to adjust."
    friendly = _friendly_for(entity_id)

    pct = max(0, min(100, int(volume)))
    return _call_service(
        "media_player",
        "volume_set",
        entity_id,
        {"volume_level": pct / 100},
        success_message=f"{friendly} volume {pct}",
    )


@tool(
    name="ha_volume_up",
    description="Increase the volume of a media player one step",
    aliases=["louder", "volume_up", "increase_volume"],
)
def volume_up(entity: Optional[str] = None) -> str:
    """Step the volume up on a media_player entity (default: Spotify, AVR, then TV)."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to turn up."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "volume_up",
        entity_id,
        success_message=f"{friendly} louder",
    )


@tool(
    name="ha_volume_down",
    description="Decrease the volume of a media player one step",
    aliases=["quieter", "volume_down", "decrease_volume"],
)
def volume_down(entity: Optional[str] = None) -> str:
    """Step the volume down on a media_player entity (default: Spotify, AVR, then TV)."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return "I don't know which speakers to turn down."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "volume_down",
        entity_id,
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
    entity_id = _media_target(entity, prefer_spotify=False)
    if entity_id is None:
        return "I don't know which device to switch."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "select_source",
        entity_id,
        {"source": source},
        success_message=f"{friendly} switched to {source}",
    )


def _media_target(entity: Optional[str], prefer_spotify: bool = True) -> Optional[str]:
    """Resolve a media player entity or Home Assistant room target."""
    _ensure_loaded()
    if entity:
        area_id = _resolve_area(entity)
        if area_id is not None:
            players = [
                eid for eid in _area_entities(area_id, domain="media_player")
                if eid not in _DENIED_ENTITIES
            ]
            if prefer_spotify and SPOTIFY_ENTITY in players:
                return SPOTIFY_ENTITY
            return players[0] if players else None
        return _resolve_entity(entity, domain="media_player")
    target = (SPOTIFY_ENTITY if prefer_spotify else None) or AVR_ENTITY or TV_ENTITY
    if not target:
        return None
    return _resolve_entity(target, domain="media_player")


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

    The import itself is deferred and local to this function rather than a
    module-level import: `tools/spotify.py` already imports this module at
    its top level (a documented exception to the no-cross-tool-import
    rule, see CLAUDE.md) to reuse its area-resolution/service-call
    machinery, so a module-level import back here would form a cycle. By
    the time this function actually runs, both modules have long since
    finished loading, so the local import is just a cheap `sys.modules`
    lookup.

    Returns None only if Spotify itself isn't usable either (not
    configured, or no valid credentials) — callers should fall through to
    their own HA-specific "I don't know which player" message in that
    case. Otherwise always returns a string (success or a Spotify-specific
    failure message), which is strictly more informative than that.
    """
    if "spotify" not in config:
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


@tool(
    name="pause",
    description="Pause playback on the active media player",
    aliases=["stop", "halt", "pause_music"],
)
def pause(entity: Optional[str] = None) -> str:
    """Pause a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return _spotify_transport_fallback("pause") or "I don't know which player to pause."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "media_pause",
        entity_id,
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
        return _spotify_transport_fallback("resume") or "I don't know which player to resume."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "media_play",
        entity_id,
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
        return _spotify_transport_fallback("skip") or "I don't know which player to skip on."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "media_next_track",
        entity_id,
        success_message=f"{friendly} skipped",
    )


@tool(
    name="previous",
    description="Go back to the previous track on the active media player",
    aliases=["previous_track", "skip_back", "last_track"],
)
def previous(entity: Optional[str] = None) -> str:
    """Go back a track on a media_player entity. Defaults to Spotify, then AVR, then TV."""
    entity_id = _media_target(entity)
    if entity_id is None:
        return _spotify_transport_fallback("previous") or "I don't know which player to skip back on."
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "media_previous_track",
        entity_id,
        success_message=f"{friendly} back a track",
    )


@tool(
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
    friendly = _friendly_for(entity_id)
    return _call_service(
        "media_player",
        "volume_mute",
        entity_id,
        {"is_volume_muted": bool(muted)},
        success_message=f"{friendly} {'muted' if muted else 'unmuted'}",
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
    description=(
        "Get the current temperature from a Home Assistant thermostat, climate "
        "zone, or temperature sensor. For a thermostat/climate zone, also "
        "reports the target temperature it's set to, if different from the "
        "current reading."
    ),
    aliases=["temperature", "check_temperature", "what_temperature", "how_warm", "how_cold"],
)
def get_temperature(entity: str) -> str:
    """Return the current (and, for a climate zone, target) temperature for a room or sensor.

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
    current = attrs.get("current_temperature")
    # `temperature` is the thermostat's target/setpoint on a climate.* entity —
    # not meaningful as a "target" on a plain sensor, so only read it as one
    # for climate zones.
    target = attrs.get("temperature") if entity_id.startswith("climate.") else None
    temp = current if current is not None else target
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
    if current is not None and target is not None and current != target:
        return f"{friendly} is {_format_reading(current)} {unit}, set to {_format_reading(target)} {unit}"
    return f"{friendly} is {_format_reading(temp)} {unit}"


def _format_reading(value) -> str:
    """Keep HA sensor precision while dropping an unhelpful trailing `.0`."""
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return str(value)


def _format_entity_state(entity_id: str, state: dict) -> str:
    """Format an HA state response for a spoken live-status reply."""
    entity_state = state.get("state", "unknown")
    friendly_name = state.get("attributes", {}).get("friendly_name", entity_id)

    attrs = state.get("attributes", {})
    details = [f"{friendly_name} is {entity_state}"]

    if "brightness" in attrs and attrs["brightness"] is not None:
        brightness_pct = int((attrs["brightness"] / 255) * 100)
        details.append(f"brightness: {brightness_pct}%")
    if "temperature" in attrs:
        details.append(f"temperature: {attrs['temperature']}°")
    if "current_temperature" in attrs:
        details.append(f"current temperature: {attrs['current_temperature']}°")
    if "hvac_action" in attrs:
        details.append(f"hvac action: {attrs['hvac_action']}")
    if "humidity" in attrs:
        details.append(f"humidity: {attrs['humidity']}%")
    if "current_position" in attrs:
        details.append(f"position: {attrs['current_position']}% open")
    if "current_valve_position" in attrs:
        details.append(f"position: {attrs['current_valve_position']}% open")
    if "battery_level" in attrs:
        details.append(f"battery: {attrs['battery_level']}%")

    return ", ".join(details)


@tool(
    name="get_entity_state",
    description="Get the current state of a Home Assistant entity (on/off, sensor reading, etc.)",
    aliases=["ha_state", "check_state", "is_on"],
)
def get_entity_state(entity: str) -> str:
    """Get the current state of a Home Assistant entity.

    Args:
        entity: Entity name or ID
    """
    entity_id = _resolve_entity(entity)
    state = _get_state(entity_id)

    if state is None:
        # Reactive sentinel (not a plain string) so a multi-guess batch —
        # the SLM hedging with several candidate names for the same entity
        # in one turn — replans instead of having every failed guess joined
        # verbatim into the spoken reply alongside a successful one.
        return (
            f"Reactive question: Couldn't find an entity matching "
            f"{_friendly_for(entity_id)!r}. Try a different name or be more specific."
        )

    return _format_entity_state(entity_id, state)


def _area_entities(area_id: str, domain: Optional[str] = None) -> list[str]:
    """Return entity_ids HA has registered in an area, optionally domain-filtered.

    Empty list if HA is unreachable or the template response can't be parsed —
    callers degrade accordingly rather than raising. Shared by
    `list_entities_in_area` and `tools/spotify.py`'s room-targeted `play_song`
    dispatch (imported directly — see that module's docstring for why).
    """
    raw = _render_template(f"{{{{ area_entities({area_id!r}) | list | tojson }}}}")
    if raw is None:
        return []
    try:
        entity_ids = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if domain:
        domain_key = domain.lower().strip()
        entity_ids = [eid for eid in entity_ids if _domain_of(eid) == domain_key]
    return entity_ids


def _floor_entities(floor_id: str, domain: Optional[str] = None) -> list[str]:
    """Return entities from every area assigned to an HA floor."""
    raw = _render_template(f"{{{{ floor_areas({floor_id!r}) | list | tojson }}}}")
    if raw is None:
        return []
    try:
        area_ids = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return list(
        dict.fromkeys(
            entity_id
            for area_id in area_ids
            for entity_id in _area_entities(area_id, domain)
        )
    )


@tool(
    name="list_entities_in_area",
    description=(
        "List the Home Assistant entities in a room, area, zone, or floor (e.g. 'downstairs', "
        "'office', 'kitchen'), optionally filtered to a domain like 'light', "
        "'switch', 'sensor', or 'climate'. Use this for questions like 'what "
        "lights do we have downstairs' or 'what else is in the office' instead "
        "of guessing individual entity names."
    ),
    aliases=["list_area_entities", "ha_list_entities", "area_entities", "what_is_in"],
)
def list_entities_in_area(area: str, domain: Optional[str] = None) -> str:
    """List the entities Home Assistant has registered in an area or floor.

    Args:
        area: Room/area/zone name, e.g. 'downstairs', 'office', 'kitchen'.
        domain: Optional domain filter, e.g. 'light', 'switch', 'sensor', 'climate'.
    """
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    area_id = _resolve_area(area)
    floor_id = _resolve_floor(area) if area_id is None else None
    if area_id is None and floor_id is None:
        return (
            f"Reactive question: Couldn't find an area matching {area!r}. "
            f"Try a different room name or be more specific."
        )

    entity_ids = _area_entities(area_id, domain) if area_id else _floor_entities(floor_id, domain)
    entity_ids = [eid for eid in entity_ids if eid not in _DENIED_ENTITIES]

    area_name = _AREA_MAP.get(area_id, area) if area_id else _FLOOR_MAP.get(floor_id, area)
    if not entity_ids:
        scoped = f"{domain} " if domain else ""
        return f"I don't see any {scoped}entities in {area_name}."

    names = sorted({_friendly_for(eid) for eid in entity_ids}, key=str.lower)
    return f"{area_name} has: " + ", ".join(names)


@tool(
    name="get_entities_in_area_state",
    description=(
        "Get current states for entities in a room/area/zone, optionally limited "
        "to a domain and an on/off state. Use for status questions such as "
        "'which lights are on upstairs', not for discovering what devices exist."
    ),
    aliases=["get_area_states", "which_are_on", "area_status"],
)
def get_entities_in_area_state(
    area: str, domain: Optional[str] = None, only_state: Optional[str] = None
) -> str:
    """Get current state for every voice-enabled entity in an HA area."""
    if not HA_TOKEN:
        return "Home Assistant isn't set up."

    area_id = _resolve_area(area)
    floor_id = _resolve_floor(area) if area_id is None else None
    if area_id is None and floor_id is None:
        return (
            f"Reactive question: Couldn't find an area matching {area!r}. "
            "Try a different room name or be more specific."
        )

    entity_ids = [
        entity_id
        for entity_id in (
            _area_entities(area_id, domain) if area_id else _floor_entities(floor_id, domain)
        )
        if entity_id not in _DENIED_ENTITIES
    ]
    area_name = _AREA_MAP.get(area_id, area) if area_id else _FLOOR_MAP.get(floor_id, area)
    if not entity_ids:
        scoped = f"{domain} " if domain else ""
        return f"I don't see any {scoped}entities in {area_name}."

    state_filter = (only_state or "").lower().strip()
    if state_filter not in ("", "on", "off"):
        return "The state filter must be 'on' or 'off'."

    details = []
    for entity_id in entity_ids:
        state = _get_state(entity_id)
        if state is None:
            continue
        if state_filter and state.get("state") != state_filter:
            continue
        details.append(_format_entity_state(entity_id, state))

    if not details and state_filter:
        scoped = f"{domain}s" if domain else "entities"
        return f"No {scoped} are {state_filter} in {area_name}."
    if not details:
        return f"I couldn't read any entity states in {area_name}."
    return "; ".join(details)


@tool(
    name="ha_service",
    description="Call any Home Assistant service with custom data",
    aliases=["call_service", "ha_call"],
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
        domain,
        service,
        entity_id,
        extra_data,
        success_message=f"{service.replace('_', ' ').capitalize()} {friendly}",
    )


@tool(
    name="ha_set_climate",
    description="Set the temperature of a climate/thermostat entity in Home Assistant",
    aliases=["ha_climate", "ha_thermostat"],
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
        "climate",
        "set_temperature",
        entity_id,
        data,
        success_message=f"{friendly} set to {temperature} degrees",
    )


@tool(name="ha_lock", description="Lock a lock entity in Home Assistant", aliases=["lock_door"])
def lock(entity: str) -> str:
    """Lock a lock entity.

    Args:
        entity: Lock entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="lock")
    friendly = _friendly_for(entity_id)
    return _call_service("lock", "lock", entity_id, success_message=f"Locked {friendly}")


@tool(
    name="ha_unlock", description="Unlock a lock entity in Home Assistant", aliases=["unlock_door"]
)
def unlock(entity: str) -> str:
    """Unlock a lock entity.

    Args:
        entity: Lock entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="lock")
    friendly = _friendly_for(entity_id)
    return _call_service("lock", "unlock", entity_id, success_message=f"Unlocked {friendly}")


# `valve.*` entities (smart water/gas shutoffs) use their own domain and
# service names but the same open/close/stop/position shape as `cover.*`
# (blinds, garages) — one set of tools handles both rather than duplicating
# every verb under a second "valve" name.
_COVER_LIKE_SERVICES = {
    "cover": {"open": "open_cover", "close": "close_cover", "stop": "stop_cover", "set_position": "set_cover_position"},
    "valve": {"open": "open_valve", "close": "close_valve", "stop": "stop_valve", "set_position": "set_valve_position"},
}


def _cover_like_domain(entity_id: str) -> str:
    return "valve" if entity_id.startswith("valve.") else "cover"


@tool(
    name="ha_open_cover",
    description="Open a cover, blind, garage door, or valve in Home Assistant",
    aliases=["ha_open", "open_blind", "open_garage", "open_valve"],
)
def open_cover(entity: str) -> str:
    """Open a cover or valve entity (blinds, garage door, water/gas valve, etc.).

    Args:
        entity: Cover or valve entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = _friendly_for(entity_id)
    return _call_service(
        domain, _COVER_LIKE_SERVICES[domain]["open"], entity_id, success_message=f"Opened {friendly}"
    )


@tool(
    name="ha_close_cover",
    description="Close a cover, blind, garage door, or valve in Home Assistant",
    aliases=["ha_close", "close_blind", "close_garage", "close_valve"],
)
def close_cover(entity: str) -> str:
    """Close a cover or valve entity (blinds, garage door, water/gas valve, etc.).

    Args:
        entity: Cover or valve entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = _friendly_for(entity_id)
    return _call_service(
        domain, _COVER_LIKE_SERVICES[domain]["close"], entity_id, success_message=f"Closed {friendly}"
    )


@tool(
    name="ha_stop_cover",
    description="Stop a moving cover, blind, garage door, or valve in Home Assistant",
    aliases=["stop_cover", "halt_cover", "stop_blind", "stop_garage", "stop_valve"],
)
def stop_cover(entity: str) -> str:
    """Halt a cover or valve entity mid-travel.

    Args:
        entity: Cover or valve entity name or ID
    """
    entity_id = _resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = _friendly_for(entity_id)
    return _call_service(
        domain, _COVER_LIKE_SERVICES[domain]["stop"], entity_id, success_message=f"Stopped {friendly}"
    )


@tool(
    name="ha_set_cover_position",
    description=(
        "Set a cover, blind, or valve to a specific position in Home Assistant "
        "(0 = fully closed, 100 = fully open) — for a partial position like "
        "'halfway'; use ha_open_cover/ha_close_cover for a full open or close."
    ),
    aliases=["set_cover_position", "set_blind_position", "cover_position"],
)
def set_cover_position(entity: str, position: int) -> str:
    """Set a cover or valve entity to an exact position.

    Args:
        entity: Cover or valve entity name or ID
        position: Target position 0-100 (0 closed, 100 fully open)
    """
    entity_id = _resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = _friendly_for(entity_id)
    pct = max(0, min(100, int(position)))
    return _call_service(
        domain,
        _COVER_LIKE_SERVICES[domain]["set_position"],
        entity_id,
        {"position": pct},
        success_message=f"{friendly} set to {pct} percent open",
    )


@tool(
    name="ha_set_fan_speed",
    description="Set a fan's speed as a percentage (0-100) in Home Assistant",
    aliases=["set_fan_speed", "fan_speed"],
)
def set_fan_speed(entity: str, speed: int) -> str:
    """Set a fan entity's speed.

    Args:
        entity: Fan entity name or ID
        speed: Speed percentage 0-100 (0 turns the fan off)
    """
    entity_id = _resolve_entity(entity, domain="fan")
    friendly = _friendly_for(entity_id)
    pct = max(0, min(100, int(speed)))
    return _call_service(
        "fan",
        "set_percentage",
        entity_id,
        {"percentage": pct},
        success_message=f"{friendly} speed set to {pct} percent",
    )


# vacuum.* actions that share one entity/target — a single parameterised tool
# instead of five near-identical start/pause/stop/dock/locate tools.
_VACUUM_ACTIONS = {
    "start": ("start", "Started {friendly}"),
    "resume": ("start", "Resumed {friendly}"),
    "pause": ("pause", "Paused {friendly}"),
    "stop": ("stop", "Stopped {friendly}"),
    "dock": ("return_to_base", "Sending {friendly} to dock"),
    "return_to_base": ("return_to_base", "Sending {friendly} to dock"),
    "locate": ("locate", "Locating {friendly}"),
}


@tool(
    name="ha_vacuum",
    description=(
        "Control a robot vacuum in Home Assistant. action: 'start', 'pause', "
        "'stop', 'dock' (return to base/charging), or 'locate' (make it beep "
        "so it can be found)."
    ),
    aliases=["vacuum", "start_vacuum", "stop_vacuum", "dock_vacuum", "run_vacuum"],
)
def ha_vacuum(entity: str, action: str = "start") -> str:
    """Control a robot vacuum.

    Args:
        entity: Vacuum entity name or ID
        action: "start" (default), "pause", "stop", "dock", or "locate"
    """
    entity_id = _resolve_entity(entity, domain="vacuum")
    friendly = _friendly_for(entity_id)
    key = action.lower().strip().replace(" ", "_")
    if key not in _VACUUM_ACTIONS:
        return f"I don't know how to '{action}' a vacuum — try start, pause, stop, dock, or locate."
    service, template = _VACUUM_ACTIONS[key]
    return _call_service("vacuum", service, entity_id, success_message=template.format(friendly=friendly))


@tool(
    name="ha_run_script",
    description="Run a Home Assistant script or automation",
    aliases=["ha_script", "run_automation"],
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
    aliases=["ha_scene", "set_scene"],
)
def activate_scene(scene_name: str) -> str:
    """Activate a Home Assistant scene.

    Args:
        scene_name: Scene entity ID or name (e.g., 'scene.movie_time' or 'movie time')
    """
    entity_id = _resolve_entity(scene_name, domain="scene")
    friendly = _friendly_for(entity_id)
    return _call_service(
        "scene", "turn_on", entity_id, success_message=f"Scene {friendly} activated"
    )


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
        "calendar",
        "get_events",
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
        "Get calendar events. Without event_name: everything in a fixed window — "
        "pass 'today' (default), 'tomorrow', 'this_week', or a specific date as "
        "'YYYY-MM-DD'. With event_name: search for a specific named event across "
        "a wide date range (default 30 days both before and after today, widen "
        "with limit), forward or backward — use this whenever the user names an "
        "event and asks if/when it's coming up or was on, e.g. 'any concerts "
        "coming up?', 'when's the dentist?'."
    ),
    aliases=[
        "calendar",
        "events",
        "schedule",
        "find_event",
        "search_calendar",
        "when_was",
        "when_is_it_on",
    ],
)
def whats_on(day: str = "today", event_name: Optional[str] = None, limit: str = "30d") -> str:
    """Calendar events — a fixed-window listing, or a named-event search.

    Args:
        day: "today" (default), "tomorrow", "this_week", or an ISO date
            "YYYY-MM-DD" for a single day. Ignored when event_name is set.
        event_name: If set, search for this event by name instead of listing a
            fixed window, e.g. "dentist", "school play".
        limit: How far to search either side of today when event_name is set,
            e.g. "30d", "2w", "6m". Default "30d".
    """
    if event_name:
        return _ha_get_events_name(event_name, limit)
    return _ha_get_events(day)


def _parse_lookback_days(limit: str, default: int = 30) -> int:
    """Parse a compact duration like '30d' / '2w' / '6m' / '1y' into days.

    Bare numbers are treated as days. Falls back to `default` for anything
    unparseable so a malformed arg degrades gracefully instead of raising.
    """
    m = re.match(r"\s*(\d+)\s*([dwmy]?)", str(limit).lower())
    if not m:
        return default
    value = int(m.group(1))
    unit = m.group(2) or "d"
    return value * {"d": 1, "w": 7, "m": 30, "y": 365}[unit]


def _relative_day_phrase(start_dt: _dt.datetime, now: _dt.datetime) -> str:
    """Render a date as 'today' / 'tomorrow' / 'next Wednesday' / 'Wednesday,
    March 4' etc. — a bare weekday name is ambiguous across a multi-week
    search window (was it last Wednesday or three weeks ago?), so near dates
    get relative phrasing and far ones get a full calendar date."""
    delta_days = (start_dt.date() - now.date()).days
    if delta_days == 0:
        return "today"
    if delta_days == 1:
        return "tomorrow"
    if delta_days == -1:
        return "yesterday"
    weekday = start_dt.strftime("%A")
    if 2 <= delta_days <= 6:
        return f"this {weekday}"
    if 7 <= delta_days <= 13:
        return f"next {weekday}"
    if -6 <= delta_days <= -2:
        return f"last {weekday}"
    return start_dt.strftime("%A, %B %-d")


def _speak_matched_events(matches: list[dict], now: _dt.datetime) -> str:
    spoken = []
    for event in matches:
        start_dt = _dt.datetime.fromisoformat(event["start"])
        day = _relative_day_phrase(start_dt, now)
        summary = event.get("summary") or "an event"
        if event.get("all_day"):
            spoken.append(f"{summary} is all day {day}.")
        else:
            time_str = start_dt.strftime("%-I:%M %p")
            spoken.append(f"{summary} is at {time_str} {day}.")
    return " ".join(spoken)


def _ha_get_events_name(event_description: str, limit: str = "30d") -> str:
    """Search calendars for events whose summary matches `event_description`,
    within `limit` days either side of now (both past and future)."""
    calendars = _read_calendars()
    if not calendars:
        return "No calendar is configured in Home Assistant."

    days = _parse_lookback_days(limit)
    now = _dt.datetime.now()
    start_iso = (now - _dt.timedelta(days=days)).isoformat()
    end_iso = (now + _dt.timedelta(days=days)).isoformat()

    response = _call_service_with_response(
        "calendar",
        "get_events",
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

    query = event_description.lower().strip()
    matches = [e for e in events if query in (e.get("summary") or "").lower()]
    if not matches:
        names = [e.get("summary") or "" for e in events]
        close = difflib.get_close_matches(event_description, names, n=5, cutoff=0.5)
        matches = [e for e in events if (e.get("summary") or "") in close]

    if not matches:
        return (
            f"I couldn't find any events matching '{event_description}' in the "
            f"{days} days before or after today."
        )

    matches.sort(key=lambda e: e["start"])
    summary = _speak_matched_events(matches, now)
    if len(matches) >= 2:
        return f"Reactive question: {summary}"
    return summary


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
        "calendar",
        "get_events",
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
    name = HA_CONFIG.get("calendar")
    if not name:
        return None
    if "." in str(name):
        return name
    entity_id = _ENTITY_ALIASES.get(name.lower())
    if entity_id and entity_id.startswith("calendar."):
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
    success = f"Added '{summary}'{time_label} on {start_dt.strftime('%A %-d %B')}{recurrence_label}"
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
    """Pick the media_player entity used as the default music player.

    No autodetection — must be set explicitly via `home_assistant.spotify_entity`
    in config.yml (friendly name or direct entity_id). This is also the entity
    `tools/spotify.py`'s `play_song` dispatches to by default (see its
    `_resolve_media_targets`), so play dispatch and pause/resume/skip agree on
    the same speaker unless a voice command names somewhere else.
    """
    configured = HA_CONFIG.get("spotify_entity")
    if not configured:
        logger.warning(
            "No home_assistant.spotify_entity configured — pause/resume/skip/play_song "
            "have no default media player. Add `spotify_entity: <friendly name or "
            "entity_id>` under home_assistant: in config.yml."
        )
        return None
    eid = _ENTITY_ALIASES.get(str(configured).lower()) or str(configured)
    logger.info(f"Using configured Spotify entity: {eid}")
    return eid


def _looks_like_tv(entity_id: str, friendly: str) -> bool:
    """Match `tv` as a whole token in entity_id or friendly name."""
    if "tv" in entity_id.split(".", 1)[1].split("_"):
        return True
    return " tv" in friendly or friendly.endswith(" tv") or friendly.startswith("tv ")


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
    global _loaded, _ENTITY_ALIASES, _ENTITY_ALIASES_MULTI, _AREA_MAP, _FLOOR_MAP
    global _DEFAULT_WEATHER_ENTITY, SPOTIFY_ENTITY, TV_ENTITY, AVR_ENTITY
    global CALENDAR_ENTITY, TODO_ENTITY
    if _loaded or not HA_TOKEN:
        return
    with _load_lock:
        if _loaded:
            return
        _ENTITY_ALIASES, _ENTITY_ALIASES_MULTI = _fetch_entity_aliases()
        _AREA_MAP = _fetch_area_map()
        _FLOOR_MAP = _fetch_floor_map()
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
        "todo",
        "add_item",
        TODO_ENTITY,
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
    items = _fetch_pending_todo_items()
    if items is None:
        return "I couldn't reach your todo list."
    names = [i.get("summary", "") for i in items if i.get("summary")]
    if not names:
        return "Your list is empty."
    if len(names) == 1:
        return f"You have one item: {names[0]}."
    return f"You have {len(names)} items: {', '.join(names[:-1])}, and {names[-1]}."


def _fetch_pending_todo_items() -> Optional[list]:
    """Fetch not-yet-completed items from the configured HA todo list, or None on error."""
    response = _call_service_with_response(
        "todo",
        "get_items",
        {"entity_id": TODO_ENTITY, "status": "needs_action"},
    )
    if not response:
        return None
    return (response.get(TODO_ENTITY) or {}).get("items") or []


@tool(
    name="complete_todo_item",
    description="Mark an item as done on the Home Assistant todo or shopping list.",
    aliases=["check_off_todo", "mark_todo_done", "todo_complete", "complete_shopping_item"],
)
def complete_todo_item(item: str) -> str:
    """Check off a pending todo/shopping-list item by name.

    Args:
        item: The item text to mark done (fuzzy-matched against pending items).
    """
    if not TODO_ENTITY:
        return "User question: No todo list is configured in Home Assistant."
    items = _fetch_pending_todo_items()
    if items is None:
        return "I couldn't reach your todo list."
    if not items:
        return f"There's nothing pending on your list matching '{item}'."

    query = item.lower().strip()
    match = next((i for i in items if query in (i.get("summary") or "").lower()), None)
    if match is None:
        names = [i.get("summary") or "" for i in items]
        close = difflib.get_close_matches(item, names, n=1, cutoff=0.5)
        if close:
            match = next((i for i in items if (i.get("summary") or "") == close[0]), None)
    if match is None:
        return f"I couldn't find '{item}' on your list."

    summary = match.get("summary", item)
    return _call_service(
        "todo",
        "update_item",
        TODO_ENTITY,
        {"item": summary, "status": "completed"},
        success_message=f"Checked off '{summary}'",
    )


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
    start_dt = _dt.datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=_local_tz.get_tz())
    end_date = start + _dt.timedelta(days=days)
    end_dt = _dt.datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=_local_tz.get_tz())
    now_dt = _local_tz.now()
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

    today = _local_tz.today()
    yesterday = today - _dt.timedelta(days=1)
    daily: dict = {}
    for s in hist:
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = _dt.datetime.fromisoformat(ts_str).astimezone(_local_tz.get_tz())
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
    # Remote, grammar-less models occasionally label positional values (for
    # example, `"days", "1"`). Treat an invalid count as the documented default
    # rather than raising before Home Assistant can answer the request.
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 2
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
            start_dt = start_dt.astimezone(_local_tz.get_tz())
        return start_dt, start_dt + _dt.timedelta(hours=2), False
    d = _dt.date.fromisoformat(start)
    start_dt = _dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=_local_tz.get_tz())
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
                end_dt = end_dt.astimezone(_local_tz.get_tz())
        else:
            d = _dt.date.fromisoformat(end)
            end_dt = _dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=_local_tz.get_tz())
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
        start_dt = _local_tz.now() - _dt.timedelta(days=HISTORY_DEFAULT_LOOKBACK_DAYS)
        default_end_dt = _local_tz.now()
    else:
        try:
            if "T" in start or " " in start:
                start_dt = _dt.datetime.fromisoformat(start)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.astimezone(_local_tz.get_tz())
                default_end_dt = start_dt + _dt.timedelta(hours=2)
            else:
                d = _dt.date.fromisoformat(start)
                start_dt = _dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=_local_tz.get_tz())
                default_end_dt = start_dt + _dt.timedelta(days=1) - _dt.timedelta(seconds=1)
        except ValueError:
            return f"Reactive question: Could not parse start '{start}'. Ask the user to clarify the date."

    if end:
        try:
            if "T" in end or " " in end:
                end_dt = _dt.datetime.fromisoformat(end)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.astimezone(_local_tz.get_tz())
            else:
                d = _dt.date.fromisoformat(end)
                end_dt = _dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=_local_tz.get_tz())
        except ValueError:
            end_dt = default_end_dt
    else:
        end_dt = default_end_dt

    now = _local_tz.now()
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
            ts = _dt.datetime.fromisoformat(ts_str).astimezone(_local_tz.get_tz())
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
        return (
            f"Reactive question: Could not parse start '{start}'. Ask the user to clarify the date."
        )

    end_dt = _parse_history_end(end, default_end_dt)
    now = _local_tz.now()
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
    today = _local_tz.today()
    yesterday = today - _dt.timedelta(days=1)
    lines: list = []
    for s, speaker in events:
        state_val = (s.get("state") or "").strip()
        if state_val.lower() in _SKIP:
            continue
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = _dt.datetime.fromisoformat(ts_str).astimezone(_local_tz.get_tz())
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
