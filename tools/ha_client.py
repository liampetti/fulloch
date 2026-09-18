"""Home Assistant client implementation.

Owns the shared HTTP configuration, entity/area/floor caches and deny-list.
Domain modules depend on this layer; it never imports them.
"""

import datetime as _dt
import functools
import json
import logging
import os
import threading
import time
from typing import Optional

import requests

import utils.local_time as _local_tz
from core.url_utils import normalize_url

from ._config import config
from .tool_registry import tool as _register_tool

logger = logging.getLogger(__name__)


def tool(*dargs, **dkwargs):
    """Register configured HA tools and lazily load the shared cache on use.

    Unconfigured tools remain directly callable, but are not advertised in the
    registry. Importing the client or a domain module never connects to HA.
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


def _get_headers() -> dict:
    """Return authorization headers for Home Assistant API."""
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _get(path: str, **kwargs) -> requests.Response:
    """GET a relative HA API path using the current token and timeout.

    Callers interpret responses and handle errors according to their tool's
    existing contract. Looking up configuration here keeps hot token updates
    visible to every domain and to Spotify's HA hand-off.
    """
    kwargs.setdefault("timeout", TIMEOUT)
    return requests.get(f"{HA_URL}{path}", headers=_get_headers(), **kwargs)


def _post(path: str, **kwargs) -> requests.Response:
    """POST a relative HA API path with the shared HTTP configuration."""
    kwargs.setdefault("timeout", TIMEOUT)
    return requests.post(f"{HA_URL}{path}", headers=_get_headers(), **kwargs)


# Retry budget for the startup alias fetch. When HA and Fulloch boot in the
# same compose stack, HA's /api/states often isn't responsive for the first
# few seconds. Without retries we'd lock in an empty alias map and every
# tool would resolve names to raw strings for the rest of the session.
# Override with FULLOCH_HA_ALIAS_RETRIES=0 in tests to skip the loop.
_ALIAS_FETCH_RETRIES = int(os.environ.get("FULLOCH_HA_ALIAS_RETRIES", "15"))
_ALIAS_FETCH_BACKOFF_S = 2.0


def _fetch_entity_aliases() -> tuple:
    """Fetch every entity's friendly_name from HA on the first use.

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
            response = _get("/api/states")
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
    try:
        response = _post("/api/template", json={"template": template})
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

    if data and "entity_id" in data and data["entity_id"] != entity_id:
        return (
            "Reactive question: Service data cannot override the selected entity. "
            "No command was sent. Use the entity argument to select the target."
        )

    # Enforcement backstop: even if a deny-listed entity_id reaches here (e.g.
    # the SLM emitted it verbatim, bypassing the filtered alias map), refuse it
    # outright rather than replanning — we don't want the agent retrying under a
    # different name to slip the control through.
    if isinstance(entity_id, str) and entity_id in _DENIED_ENTITIES:
        friendly = _friendly_for(entity_id)
        logger.info(f"Refused voice control of deny-listed entity {entity_id}")
        return f"Sorry, {friendly} isn't available for voice control."

    # A syntactically plausible ID is not evidence that the target exists.
    # Known IDs cost no I/O; check cache misses against HA before a service can
    # return an HTTP success for a nonexistent target. A failed lookup is not
    # proof of absence (HA may be unreachable).
    known_ids = set(_ENTITY_ALIASES.values())
    for ids in _ENTITY_ALIASES_MULTI.values():
        known_ids.update(ids)
    targets = [entity_id] if isinstance(entity_id, str) else entity_id
    for target in targets:
        if target not in known_ids and _get_state(target) is None:
            return (
                f"Reactive question: I couldn't verify the Home Assistant target {target!r}. "
                "No command was sent. Ask the user to clarify the target or use a lookup."
            )

    path = f"/api/services/{domain}/{service}"
    payload = {"entity_id": entity_id}
    if data:
        payload.update(data)

    friendly = _friendly_for(entity_id) if isinstance(entity_id, str) else "those"
    action = service.replace("_", " ")
    try:
        response = _post(path, json=payload)
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
    path = f"/api/services/{domain}/{service}?return_response=true"
    try:
        response = _post(
            path,
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

    path = f"/api/states/{entity_id}"
    try:
        response = _get(path)
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
            entity_id for area_id in area_ids for entity_id in _area_entities(area_id, domain)
        )
    )


def _normalise_ha_timestamp(timestamp: str) -> _dt.datetime:
    """Parse an HA timestamp in the configured local timezone.

    Home Assistant commonly returns UTC timestamps, but integrations may return
    other offsets or naive local datetimes. Normalising at the boundary keeps
    every subsequent comparison, sort, and spoken date in one timezone.
    """
    parsed = _dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    local_tz = _local_tz.get_tz()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(local_tz)


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
