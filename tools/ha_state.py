"""Home Assistant state implementation.

Uses ha_client as the single owner of configuration and resolution state.
"""

import datetime as _dt
import logging
import math
from collections import Counter
from typing import Optional

import utils.local_time as _local_tz
from core.satellite_context import current_response_language
from utils.response_language import response as localized_response

from . import ha_client as client
from .tool_registry import ArtifactText

logger = logging.getLogger(__name__)

# Covers a plain "when was X last on" in one lookup (within HA retention).
HISTORY_DEFAULT_LOOKBACK_DAYS = 7


@client.tool(
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
    entity_id = client._resolve_with_variants(entity, suffixes, ("climate", "sensor"))
    if entity_id is None:
        # Last resort: trust the SLM and assume the name matches an entity_id
        entity_id = client._resolve_entity(entity, domain="climate")

    state = client._get_state(entity_id)
    if state is None:
        return localized_response("temperature_missing", target=entity)

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

    friendly = client._friendly_for(entity_id)
    if temp is None:
        return localized_response("temperature_unreadable", target=friendly)

    unit = _spoken_temperature_unit(attrs)
    if current is not None and target is not None and current != target:
        return localized_response("temperature_with_target", target=friendly, value=_format_reading(current),
                                  setpoint=_format_reading(target), unit=unit)
    return localized_response("temperature_current", target=friendly, value=_format_reading(temp), unit=unit)


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
    known_states = {"on", "off", "open", "closed", "unlocked", "locked", "unknown", "unavailable"}
    spoken_state = localized_response(f"state_{entity_state}") if isinstance(entity_state, str) and entity_state in known_states else entity_state
    details = [localized_response("entity_state", target=friendly_name, state=spoken_state)]

    if "brightness" in attrs and attrs["brightness"] is not None:
        brightness_pct = int((attrs["brightness"] / 255) * 100)
        details.append(localized_response("entity_brightness", value=brightness_pct))
    if "temperature" in attrs:
        details.append(localized_response("entity_temperature", value=attrs['temperature']))
    if "current_temperature" in attrs:
        details.append(localized_response("entity_current_temperature", value=attrs['current_temperature']))
    if "hvac_action" in attrs:
        details.append(localized_response("entity_hvac_action", value=attrs['hvac_action']))
    if "humidity" in attrs:
        details.append(localized_response("entity_humidity", value=attrs['humidity']))
    if "current_position" in attrs:
        details.append(localized_response("entity_position", value=attrs['current_position']))
    if "current_valve_position" in attrs:
        details.append(localized_response("entity_position", value=attrs['current_valve_position']))
    if "battery_level" in attrs:
        details.append(localized_response("entity_battery", value=attrs['battery_level']))

    return ", ".join(details)


def _spoken_domain(domain: str) -> str:
    if current_response_language.get() == "en":
        return domain
    known = {"light", "switch", "sensor", "climate", "fan", "cover", "lock", "media_player", "binary_sensor"}
    return localized_response(f"domain_{domain}") if domain in known else domain


def _entity_status_artifact(entity_id: str, state: dict) -> dict:
    """Build the compact, dashboard-only companion to a state observation."""
    attrs = state.get("attributes", {})
    details = []
    brightness = attrs.get("brightness")
    if isinstance(brightness, (int, float)):
        details.append({"label": "Brightness", "value": f"{round(brightness / 255 * 100)}%"})
    rgb = attrs.get("rgb_color")
    if (
        isinstance(rgb, (list, tuple))
        and len(rgb) == 3
        and all(isinstance(value, (int, float)) and 0 <= value <= 255 for value in rgb)
    ):
        details.append(
            {"label": "Colour", "value": "#%02x%02x%02x" % tuple(round(value) for value in rgb)}
        )
    for attribute, label, suffix in (
        ("temperature", "Target", "°"),
        ("current_temperature", "Current", "°"),
        ("humidity", "Humidity", "%"),
        ("current_position", "Open", "%"),
        ("current_valve_position", "Open", "%"),
        ("battery_level", "Battery", "%"),
        ("hvac_action", "Activity", ""),
    ):
        value = attrs.get(attribute)
        if value is not None:
            details.append({"label": label, "value": f"{value}{suffix}"})
    return {
        "type": "entity_status",
        "title": str(attrs.get("friendly_name") or entity_id),
        "domain": client._domain_of(entity_id),
        "state": str(state.get("state", "unknown")),
        "details": details[:8],
    }


def _media_artifact(entity_id: str, state: dict) -> dict:
    """Build media-player data for the dashboard without exposing HA URLs."""
    attrs = state.get("attributes", {})
    volume = attrs.get("volume_level")
    try:
        volume = round(float(volume) * 100)
    except (TypeError, ValueError):
        volume = None
    return {
        "type": "media",
        "title": str(attrs.get("media_title") or attrs.get("friendly_name") or entity_id),
        "artist": str(attrs.get("media_artist") or attrs.get("app_name") or ""),
        "player": str(attrs.get("friendly_name") or entity_id),
        "state": str(state.get("state", "unknown")),
        "volume": max(0, min(100, volume)) if volume is not None else None,
        "artwork_url": f"/media-artwork/{entity_id}",
    }


def _overview_group(label: str, kind: str, entities: list[str]) -> dict | None:
    if not entities:
        return None
    return {"label": label, "kind": kind, "count": len(entities), "entities": entities[:8]}


@client.tool(
    name="get_home_overview",
    description=(
        "Get a live overview of the home: lights on, open doors/windows/covers, "
        "unlocked locks, and active devices."
    ),
    aliases=["home_overview", "home_status", "house_status", "overview"],
)
def get_home_overview() -> str:
    """Summarise voice-enabled live Home Assistant states by safety-relevant group."""
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")
    try:
        response = client._get("/api/states")
        response.raise_for_status()
        states = response.json()
    except Exception as exc:
        logger.warning("HA home overview failed: %s", exc)
        return localized_response("ha_unreachable", "error")
    if not isinstance(states, list):
        return localized_response("overview_unreadable")

    groups = {"lights": [], "openings": [], "locks": [], "active": []}
    inactive = {"off", "idle", "standby", "unavailable", "unknown"}
    for item in states:
        entity_id = item.get("entity_id") or ""
        if entity_id in client._DENIED_ENTITIES:
            continue
        domain = client._domain_of(entity_id)
        state = str(item.get("state") or "").lower()
        attrs = item.get("attributes") or {}
        name = str(attrs.get("friendly_name") or client._friendly_for(entity_id))
        if domain == "light" and state == "on":
            groups["lights"].append(name)
        elif domain == "lock" and state == "unlocked":
            groups["locks"].append(name)
        elif domain == "cover" and state in {"open", "opening"}:
            groups["openings"].append(name)
        elif (
            domain == "binary_sensor"
            and attrs.get("device_class") in {"door", "window", "opening"}
            and state == "on"
        ):
            groups["openings"].append(name)
        elif domain in {"switch", "fan", "vacuum", "media_player"} and state not in inactive:
            groups["active"].append(name)

    artifact_groups = [
        group
        for group in (
            _overview_group("Lights on", "lights", groups["lights"]),
            _overview_group("Open", "openings", groups["openings"]),
            _overview_group("Unlocked", "locks", groups["locks"]),
            _overview_group("Active", "active", groups["active"]),
        )
        if group is not None
    ]
    if not artifact_groups:
        return ArtifactText(
            localized_response("overview_settled"), {"type": "home_overview", "groups": []}
        )
    spoken = ", ".join(localized_response(f"overview_{group['kind']}", count=group["count"])
                       for group in artifact_groups)
    return ArtifactText(
        localized_response("overview_summary", groups=spoken), {"type": "home_overview", "groups": artifact_groups}
    )


def _energy_sensor_kind(entity_id: str, attrs: dict) -> str | None:
    """Classify common HA power/energy sensors without relying on one vendor."""
    device_class = str(attrs.get("device_class") or "").lower()
    name = f"{entity_id} {attrs.get('friendly_name', '')}".lower()
    if device_class == "battery" and any(
        token in name for token in ("solar", "battery", "powerwall", "storage")
    ):
        return "battery"
    if device_class not in {"power", "energy"}:
        return None
    if any(token in name for token in ("solar", "pv", "photovoltaic", "production", "generation")):
        return "solar"
    return "consumption"


def _energy_history(entity_id: str) -> list[dict]:
    """Fetch a small 24-hour numeric series for the primary consumption sensor."""
    end = _local_tz.now()
    start = end - _dt.timedelta(days=1)
    try:
        response = client._get(
            f"/api/history/period/{start.isoformat()}",
            params={
                "end_time": end.isoformat(),
                "filter_entity_id": entity_id,
                "minimal_response": "true",
                "no_attributes": "true",
            },
        )
        response.raise_for_status()
        records = (response.json() or [[]])[0]
    except Exception as exc:
        logger.warning("HA energy history failed for %s: %s", entity_id, exc)
        return []
    points = []
    for record in records:
        try:
            value = float(record.get("state"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        timestamp = record.get("last_changed") or record.get("last_updated")
        if isinstance(timestamp, str) and timestamp:
            points.append({"time": timestamp, "value": round(value, 2)})
    if len(points) > 36:
        stride = (len(points) - 1) / 35
        points = [points[round(index * stride)] for index in range(36)]
    return points


@client.tool(
    name="get_energy_overview",
    description="Get current Home Assistant energy readings for consumption, solar production, and home battery, with recent consumption history when available.",
    aliases=["energy", "energy_overview", "solar_status", "power_usage"],
)
def get_energy_overview() -> str:
    """Read standard HA energy sensors and return only readings that exist."""
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")
    try:
        response = client._get("/api/states")
        response.raise_for_status()
        states = response.json()
    except Exception as exc:
        logger.warning("HA energy overview failed: %s", exc)
        return localized_response("ha_unreachable", "error")
    metrics = {}
    for item in states if isinstance(states, list) else []:
        entity_id = item.get("entity_id") or ""
        attrs = item.get("attributes") or {}
        kind = _energy_sensor_kind(entity_id, attrs)
        if kind is None or kind in metrics:
            continue
        try:
            value = float(item.get("state"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        metrics[kind] = {
            "kind": kind,
            "label": str(attrs.get("friendly_name") or client._friendly_for(entity_id)),
            "value": round(value, 2),
            "unit": str(attrs.get("unit_of_measurement") or ""),
            "entity_id": entity_id,
        }
    if not metrics:
        return localized_response("energy_no_sensors")
    consumption = metrics.get("consumption")
    history = _energy_history(consumption["entity_id"]) if consumption else []
    artifact_metrics = [
        {key: value for key, value in metric.items() if key != "entity_id"}
        for metric in metrics.values()
    ]
    bits = [
        localized_response("energy_reading", label=metric['label'], value=f"{metric['value']:g}", unit=metric['unit']).strip()
        for metric in metrics.values()
    ]
    return ArtifactText(
        localized_response("energy_overview", readings=", ".join(bits)),
        {
            "type": "energy",
            "metrics": artifact_metrics,
            "history": history,
            "history_unit": consumption["unit"] if consumption else "",
        },
    )


@client.tool(
    name="get_security_overview",
    description="Get a constrained Home Assistant security summary of unlocked locks, open doors/windows, and camera availability. Read-only: never exposes camera feeds.",
    aliases=["security", "security_overview", "home_security"],
)
def get_security_overview() -> str:
    """Read safety-relevant state only, keeping camera details deliberately minimal."""
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")
    try:
        response = client._get("/api/states")
        response.raise_for_status()
        states = response.json()
    except Exception as exc:
        logger.warning("HA security overview failed: %s", exc)
        return localized_response("ha_unreachable", "error")
    findings = {"openings": [], "locks": [], "cameras": [], "camera_issues": []}
    for item in states if isinstance(states, list) else []:
        entity_id = item.get("entity_id") or ""
        if entity_id in client._DENIED_ENTITIES:
            continue
        attrs = item.get("attributes") or {}
        domain = client._domain_of(entity_id)
        state = str(item.get("state") or "").lower()
        name = str(attrs.get("friendly_name") or client._friendly_for(entity_id))
        if domain == "lock" and state == "unlocked":
            findings["locks"].append(name)
        elif (
            domain == "cover"
            and attrs.get("device_class") in {"door", "garage"}
            and state in {"open", "opening"}
        ):
            findings["openings"].append(name)
        elif (
            domain == "binary_sensor"
            and attrs.get("device_class") in {"door", "window", "opening"}
            and state == "on"
        ):
            findings["openings"].append(name)
        elif domain == "camera":
            (
                findings["camera_issues"]
                if state in {"unavailable", "unknown"}
                else findings["cameras"]
            ).append(name)
    groups = [
        group
        for group in (
            _overview_group("Open entries", "openings", findings["openings"]),
            _overview_group("Unlocked", "locks", findings["locks"]),
            _overview_group("Cameras online", "cameras", findings["cameras"]),
            _overview_group("Cameras unavailable", "camera_issues", findings["camera_issues"]),
        )
        if group is not None
    ]
    attention = bool(findings["openings"] or findings["locks"] or findings["camera_issues"])
    artifact = {
        "type": "security",
        "status": "attention" if attention else "secure",
        "groups": groups,
    }
    if not attention:
        camera_text = localized_response("security_cameras_online", count=len(findings['cameras'])) if findings["cameras"] else ""
        return ArtifactText(
            localized_response("security_settled") + camera_text, artifact
        )
    bits = []
    if findings["openings"]:
        bits.append(localized_response("security_open_entries", count=len(findings['openings'])))
    if findings["locks"]:
        bits.append(localized_response("security_unlocked_locks", count=len(findings['locks'])))
    if findings["camera_issues"]:
        bits.append(localized_response("security_cameras_unavailable", count=len(findings['camera_issues'])))
    return ArtifactText(localized_response("security_attention", findings=", ".join(bits)), artifact)


@client.tool(
    name="get_entity_state",
    description="Get the current state of a Home Assistant entity (on/off, sensor reading, etc.)",
    aliases=["ha_state", "check_state", "is_on"],
)
def get_entity_state(entity: str) -> str:
    """Get the current state of a Home Assistant entity.

    Args:
        entity: Entity name or ID
    """
    entity_id = client._resolve_entity(entity)
    state = client._get_state(entity_id)

    if state is None:
        # Reactive sentinel (not a plain string) so a multi-guess batch —
        # the SLM hedging with several candidate names for the same entity
        # in one turn — replans instead of having every failed guess joined
        # verbatim into the spoken reply alongside a successful one.
        return "Reactive question: " + localized_response("ha_entity_not_found", entity=repr(client._friendly_for(entity_id))) + " Try a different name or be more specific."

    artifact = (
        _media_artifact(entity_id, state)
        if client._domain_of(entity_id) == "media_player"
        else _entity_status_artifact(entity_id, state)
    )
    return ArtifactText(_format_entity_state(entity_id, state), artifact)


@client.tool(
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
    """List the entities Home Assistant has registered in an area or floor. 5 max, then "amongst others" if more exist.

    Args:
        area: Room/area/zone name, e.g. 'downstairs', 'office', 'kitchen'.
        domain: Optional domain filter, e.g. 'light', 'switch', 'sensor', 'climate'.
    """
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")

    area_id = client._resolve_area(area)
    floor_id = client._resolve_floor(area) if area_id is None else None
    if area_id is None and floor_id is None:
        return "Reactive question: " + localized_response("ha_area_not_found", area=repr(area)) + " Try a different room name or be more specific."

    entity_ids = (
        client._area_entities(area_id, domain)
        if area_id
        else client._floor_entities(floor_id, domain)
    )
    entity_ids = [eid for eid in entity_ids if eid not in client._DENIED_ENTITIES]

    area_name = (
        client._AREA_MAP.get(area_id, area) if area_id else client._FLOOR_MAP.get(floor_id, area)
    )
    if not entity_ids:
        scoped = f"{_spoken_domain(domain)} " if domain else ""
        return localized_response("area_no_entities", scope=scoped, area=area_name)

    names = sorted({client._friendly_for(eid) for eid in entity_ids}, key=str.lower)
    return localized_response("area_has_entities", area=area_name, names=", ".join(names[:5]),
                              more=localized_response("area_among_others") if len(names) > 5 else "")


@client.tool(
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
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")

    area_id = client._resolve_area(area)
    floor_id = client._resolve_floor(area) if area_id is None else None
    if area_id is None and floor_id is None:
        return "Reactive question: " + localized_response("ha_area_not_found", area=repr(area)) + " Try a different room name or be more specific."

    entity_ids = [
        entity_id
        for entity_id in (
            client._area_entities(area_id, domain)
            if area_id
            else client._floor_entities(floor_id, domain)
        )
        if entity_id not in client._DENIED_ENTITIES
    ]
    area_name = (
        client._AREA_MAP.get(area_id, area) if area_id else client._FLOOR_MAP.get(floor_id, area)
    )
    if not entity_ids:
        scoped = f"{_spoken_domain(domain)} " if domain else ""
        return localized_response("area_no_entities", scope=scoped, area=area_name)

    state_filter = (only_state or "").lower().strip()
    if state_filter not in ("", "on", "off"):
        return localized_response("area_bad_filter")

    details = []
    artifacts = []
    for entity_id in entity_ids:
        state = client._get_state(entity_id)
        if state is None:
            continue
        if state_filter and state.get("state") != state_filter:
            continue
        details.append(_format_entity_state(entity_id, state))
        if len(artifacts) < 12:
            artifacts.append(_entity_status_artifact(entity_id, state))

    if not details and state_filter:
        language = current_response_language.get()
        if language in {"fr", "de"}:
            scoped = f"{_spoken_domain(domain)} " if domain else ""
        else:
            scoped = f"{domain}s" if domain else "entities"
        state = localized_response(f"state_{state_filter}")
        return localized_response("area_none_in_state", scope=scoped, state=state, area=area_name)
    if not details:
        return localized_response("area_state_unreadable", area=area_name)
    text = "; ".join(details)
    return ArtifactText(
        text,
        {"type": "entity_status", "title": area_name, "entities": artifacts},
    )


def _temperature_unit(attrs: dict) -> str:
    """Map HA's temperature_unit attribute to a spoken phrase."""
    raw = (attrs.get("temperature_unit") or "").upper()
    return "degrees Fahrenheit" if "F" in raw else "degrees Celsius"


def _spoken_temperature_unit(attrs: dict) -> str:
    return localized_response("weather_fahrenheit" if "F" in (attrs.get("temperature_unit") or "").upper()
                              else "weather_celsius")


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

_WEATHER_RESPONSE_KEYS = {
    "clear": "weather_clear", "cloudy": "weather_cloudy", "severe weather": "weather_severe",
    "foggy": "weather_fog", "hail": "weather_hail", "thunderstorms": "weather_thunder",
    "partly cloudy": "weather_partly_cloudy", "heavy rain": "weather_heavy_rain",
    "rainy": "weather_rain", "snowy": "weather_snow", "sleet": "weather_sleet",
    "sunny": "weather_sunny", "windy": "weather_windy", "unknown": "weather_unknown",
}


def _spoken_weather_condition(condition: str) -> str:
    key = _WEATHER_RESPONSE_KEYS.get(condition)
    return localized_response(key) if key else condition


def _humanize_condition(state: Optional[str]) -> str:
    """HA weather condition slug → spoken phrase (closed set; hyphen-swap fallback)."""
    slug = (state or "").strip().lower()
    if not slug:
        return ""
    return _WEATHER_CONDITIONS.get(slug, slug.replace("-", " "))


def _format_day(label: str, day: dict, unit: str) -> str:
    cond = _spoken_weather_condition(_humanize_condition(day.get("condition")))
    hi = day.get("temperature")
    lo = day.get("templow")
    precip = day.get("precipitation_probability")

    bits = [localized_response("weather_day_condition", day=label, condition=cond)] if cond else [label]
    if lo is not None and hi is not None:
        bits.append(localized_response("weather_range", low=round(lo), high=round(hi), unit=unit))
    elif hi is not None:
        bits.append(localized_response("weather_high", high=round(hi), unit=unit))
    if precip is not None:
        bits.append(localized_response("weather_rain_chance", value=int(precip)))
    return " ".join(bits)


def _weather_artifact(
    entity_id: str,
    current_cond: str,
    current_temp,
    unit: str,
    forecast: list,
    start: _dt.date,
    days: int,
    today: _dt.date,
) -> dict:
    """Build a compact, dashboard-safe forecast payload from HA data."""
    entries = []
    for day in forecast:
        try:
            day_date = client._normalise_ha_timestamp(day.get("datetime") or "").date()
        except (TypeError, ValueError):
            continue
        if day_date < start or len(entries) >= days:
            continue
        label = (
            "Today"
            if day_date == today
            else "Tomorrow"
            if day_date == today + _dt.timedelta(days=1)
            else day_date.strftime("%A")
        )
        entry = {
            "date": day_date.isoformat(),
            "label": label,
            "condition": _humanize_condition(day.get("condition")),
        }
        for source, target in (
            ("templow", "low"),
            ("temperature", "high"),
            ("precipitation_probability", "precipitation_probability"),
        ):
            value = day.get(source)
            if isinstance(value, (int, float)):
                entry[target] = round(value)
        entries.append(entry)
    current = {"condition": current_cond}
    if isinstance(current_temp, (int, float)):
        current["temperature"] = round(current_temp)
    return {
        "type": "weather",
        "title": client._friendly_for(entity_id).replace("forecast", "").strip() or "Weather",
        "unit": unit,
        "current": current,
        "forecast": entries,
    }


def _weather_history_summary(entity_id: str, start: _dt.date, days: int, unit: str) -> str:
    """Query HA state history for a weather entity and summarise past conditions."""
    start_dt = _dt.datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=_local_tz.get_tz())
    end_date = start + _dt.timedelta(days=days)
    end_dt = _dt.datetime(
        end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=_local_tz.get_tz()
    )
    now_dt = _local_tz.now()
    if end_dt > now_dt:
        end_dt = now_dt
    try:
        resp = client._get(
            f"/api/history/period/{start_dt.isoformat()}",
            params={"end_time": end_dt.isoformat(), "filter_entity_id": entity_id},
        )
        resp.raise_for_status()
        hist = (resp.json() or [[]])[0]
    except Exception as e:
        logger.warning(f"HA weather history for {entity_id} failed: {e}")
        return "Reactive question: " + localized_response("history_weather_failed", target=repr(entity_id)) + " Tell the user there was an error."

    if not hist:
        return localized_response("weather_history_missing", target=client._friendly_for(entity_id), date=start)

    today = _local_tz.today()
    yesterday = today - _dt.timedelta(days=1)
    daily: dict = {}
    for s in hist:
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = client._normalise_ha_timestamp(ts_str)
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
        return localized_response("weather_history_unusable", date=start)

    parts = [localized_response("weather_history_title", target=client._friendly_for(entity_id).replace('forecast', '').strip())]
    for day in sorted(daily):
        data = daily[day]
        language = current_response_language.get()
        if day == today:
            label = localized_response("weather_today")
        elif day == yesterday:
            label = localized_response("weather_yesterday")
        elif language in {"fr", "de"}:
            from babel.dates import format_date

            label = format_date(day, "EEEE", locale=language)
        else:
            label = day.strftime("%A")
        cond = Counter(data["conditions"]).most_common(1)[0][0] if data["conditions"] else "unknown"
        bits = [localized_response("weather_day_condition", day=label, condition=_spoken_weather_condition(cond))]
        if data["temps"]:
            lo, hi = round(min(data["temps"])), round(max(data["temps"]))
            spoken_unit = localized_response("weather_fahrenheit" if "Fahrenheit" in unit else "weather_celsius")
            bits.append(localized_response("weather_range", low=lo, high=hi, unit=spoken_unit)
                        if lo != hi else f"{lo} {spoken_unit}")
        parts.append(" ".join(bits))
    return ". ".join(parts) + "."


@client.tool(
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
    today = _local_tz.today()
    if start_date:
        try:
            start = _dt.date.fromisoformat(start_date)
        except ValueError:
            start = today
    else:
        start = today

    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")

    entity_id = client._DEFAULT_WEATHER_ENTITY
    if location:
        candidate = client._resolve_entity(location, domain="weather")
        if candidate.startswith("weather."):
            entity_id = candidate

    state = client._get_state(entity_id)
    if state is None:
        return "Reactive question: " + localized_response("ha_weather_entity_not_found", entity=repr(entity_id)) + " Tell the user which entity to configure, or suggest renaming a `weather.*` entity so autodetect picks it up."

    attrs = state.get("attributes", {})
    unit = _temperature_unit(attrs)
    spoken_unit = _spoken_temperature_unit(attrs)

    if start < today:
        return _weather_history_summary(entity_id, start, days, unit)

    current_cond = _humanize_condition(state.get("state")) or "unknown"
    current_temp = attrs.get("temperature")

    parts = [localized_response("weather_forecast", target=client._friendly_for(entity_id).replace('forecast', ''))]
    if start == today:
        if current_temp is not None:
            parts.append(localized_response("weather_current_temperature", condition=_spoken_weather_condition(current_cond),
                                            temperature=round(current_temp), unit=spoken_unit))
        else:
            parts.append(localized_response("weather_current", condition=_spoken_weather_condition(current_cond)))

    try:
        response = client._post(
            "/api/services/weather/get_forecasts?return_response=true",
            json={"entity_id": entity_id, "type": "daily"},
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
            day_date = client._normalise_ha_timestamp(raw_dt).date()
        except (TypeError, ValueError):
            continue
        if day_date < start:
            continue
        if count >= days:
            break
        if day_date == today:
            label = localized_response("weather_today")
        elif day_date == today + _dt.timedelta(days=1):
            label = localized_response("weather_tomorrow")
        else:
            language = current_response_language.get()
            if language in {"fr", "de"}:
                from babel.dates import format_date

                label = format_date(day_date, "EEEE", locale=language)
            else:
                label = day_date.strftime("%A")
        parts.append(_format_day(label, day, spoken_unit))
        count += 1

    text = ". ".join(parts) + "."
    return ArtifactText(
        text,
        _weather_artifact(
            entity_id, current_cond, current_temp, unit, forecast, start, days, today
        ),
    )


def _parse_history_start(start: str):
    """Parse a history `start` arg into (start_dt, default_end_dt, date_only).

    Accepts an ISO date ('2026-06-02' → whole-day window) or ISO datetime
    ('2026-06-02T12:00:00' → start+2h default window). Naive values are
    localised to Home Assistant's configured timezone. Raises ValueError on an
    unparseable arg.
    """
    if "T" in start or " " in start:
        start_dt = _dt.datetime.fromisoformat(start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=_local_tz.get_tz())
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
                end_dt = end_dt.replace(tzinfo=_local_tz.get_tz())
        else:
            d = _dt.date.fromisoformat(end)
            end_dt = _dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=_local_tz.get_tz())
    except ValueError:
        return default_end_dt
    return end_dt


def _fetch_history_states(
    entity_id: str, start_dt, end_dt, *, include_attributes: bool = False
) -> list:
    """GET /api/history/period for one entity; return its list of state dicts.

    Returns an empty list when HA has no recorded changes in the window.
    Raises on a transport/HTTP error so the caller can surface a sentinel.
    """
    params = {
        "end_time": end_dt.isoformat(),
        "filter_entity_id": entity_id,
        "minimal_response": "true",
    }
    if not include_attributes:
        params["no_attributes"] = "true"
    response = client._get(f"/api/history/period/{start_dt.isoformat()}", params=params)
    response.raise_for_status()
    history = response.json()
    if not history or not history[0]:
        return []
    return history[0]


def _temperature_history_artifact(
    entity_id: str, friendly: str, states: list, current: object
) -> dict | None:
    """Return bounded chart data only for temperature-capable HA entities."""
    attrs = current.get("attributes", {}) if isinstance(current, dict) else {}
    raw_unit = str(attrs.get("temperature_unit") or attrs.get("unit_of_measurement") or "")
    is_temperature = (
        "C" in raw_unit.upper()
        or "F" in raw_unit.upper()
        or attrs.get("device_class") == "temperature"
        or client._domain_of(entity_id) in {"climate", "weather"}
    )
    if not is_temperature:
        return None

    points = []
    for item in states:
        try:
            value = float(item.get("state"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        timestamp = item.get("last_changed") or item.get("last_updated")
        if isinstance(timestamp, str) and timestamp:
            points.append({"time": timestamp, "value": round(value, 2)})
    if not points:
        return None
    low = min(point["value"] for point in points)
    high = max(point["value"] for point in points)

    # Preserve the full time span while keeping the stream payload small.
    if len(points) > 40:
        stride = (len(points) - 1) / 39
        points = [points[round(index * stride)] for index in range(40)]

    def _number(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return round(parsed, 2) if math.isfinite(parsed) else None

    current_value = _number(attrs.get("current_temperature"))
    if current_value is None:
        current_value = _number(current.get("state") if isinstance(current, dict) else None)
    return {
        "type": "temperature_history",
        "title": friendly,
        "unit": "°F" if "F" in raw_unit.upper() else "°C",
        "current": current_value if current_value is not None else points[-1]["value"],
        "target": _number(attrs.get("temperature")),
        "min": low,
        "max": high,
        "points": points,
    }


def _light_history_artifact(friendly: str, states: list, current: object) -> dict | None:
    """Build a small on/off and brightness timeline from light history records."""
    points = []
    for item in states:
        timestamp = item.get("last_changed") or item.get("last_updated")
        if not isinstance(timestamp, str) or not timestamp:
            continue
        brightness = item.get("attributes", {}).get("brightness")
        try:
            brightness = round(float(brightness) / 255 * 100) if brightness is not None else None
        except (TypeError, ValueError):
            brightness = None
        points.append(
            {
                "time": timestamp,
                "on": item.get("state") == "on",
                "brightness": brightness if brightness is None else max(0, min(100, brightness)),
            }
        )
    if not points:
        return None
    if len(points) > 40:
        stride = (len(points) - 1) / 39
        points = [points[round(index * stride)] for index in range(40)]
    current_attrs = current.get("attributes", {}) if isinstance(current, dict) else {}
    current_brightness = current_attrs.get("brightness")
    try:
        current_brightness = round(float(current_brightness) / 255 * 100)
    except (TypeError, ValueError):
        current_brightness = None
    return {
        "type": "light_history",
        "title": friendly,
        "state": str(current.get("state", "unknown")) if isinstance(current, dict) else "unknown",
        "brightness": current_brightness,
        "points": points,
    }


@client.tool(
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
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")

    entity_id = client._resolve_entity(entity)
    # If resolution failed to produce a valid entity_id, try domain-specific fallbacks
    if "." not in entity_id:
        weather_terms = {"weather", "forecast", "temperature", "climate", "outdoor", "outside"}
        if any(t in entity.lower() for t in weather_terms):
            entity_id = client._DEFAULT_WEATHER_ENTITY
        else:
            return "Reactive question: " + localized_response("ha_entity_unresolved", entity=entity) + " Ask the user for the exact entity ID (e.g. 'light.kitchen', 'sensor.outdoor_temperature')."

    if not start or not start.strip():
        # No date given — default to a recent window so "when was X last on"
        # answers in one agent call instead of a replan just to pick a start.
        start_dt = _local_tz.now() - _dt.timedelta(days=HISTORY_DEFAULT_LOOKBACK_DAYS)
        default_end_dt = _local_tz.now()
    else:
        try:
            start_dt, default_end_dt, _ = _parse_history_start(start)
        except ValueError:
            return "Reactive question: " + localized_response("history_parse_failed", start=start) + " Ask the user to clarify the date."

    end_dt = _parse_history_end(end, default_end_dt)

    now = _local_tz.now()
    if end_dt > now:
        end_dt = now

    try:
        states = _fetch_history_states(
            entity_id, start_dt, end_dt, include_attributes=client._domain_of(entity_id) == "light"
        )
    except Exception as e:
        logger.warning(f"HA history query for {entity_id} failed: {e}")
        return "Reactive question: " + localized_response("history_fetch_failed", target=entity) + " Tell the user there was an error."

    if not states:
        if start and start.strip():
            window = localized_response("history_window_range", start=start, end=end) if end else start
            window = localized_response("history_window_on", window=window)
        else:
            window = localized_response("history_window_last_days", days=HISTORY_DEFAULT_LOOKBACK_DAYS)
        return localized_response("history_no_states", target=client._friendly_for(entity_id), window=window)

    artifact_states = states
    friendly = client._friendly_for(entity_id)
    today = _local_tz.today()
    yesterday = today - _dt.timedelta(days=1)

    def _when(s) -> str:
        """A spoken 'today/yesterday/<weekday> at <time>' label for one state."""
        ts_str = s.get("last_changed") or s.get("last_updated") or ""
        try:
            ts = client._normalise_ha_timestamp(ts_str)
            ts_date = ts.date()
            if ts_date == today:
                day_label = localized_response("calendar_today")
            elif ts_date == yesterday:
                day_label = localized_response("calendar_yesterday")
            else:
                language = current_response_language.get()
                if language in {"fr", "de"}:
                    from babel.dates import format_date

                    day_label = format_date(ts.date(), "EEEE", locale=language)
                else:
                    day_label = ts.strftime("%A")
            language = current_response_language.get()
            if language in {"fr", "de"}:
                from babel.dates import format_time

                time_label = format_time(ts, format="short", locale=language)
            else:
                time_label = ts.strftime('%I:%M %p').lstrip('0')
            return localized_response("history_when", day=day_label, time=time_label)
        except Exception:
            return ts_str

    MAX_RESULTS = 15
    truncated = len(states) > MAX_RESULTS
    if truncated:
        states = states[-MAX_RESULTS:]

    lines = [localized_response("history_title", target=friendly)]
    for s in states:
        state = s.get("state", "unknown")
        if isinstance(state, str) and state in {"on", "off", "open", "closed", "unlocked", "locked", "unknown", "unavailable"}:
            state = localized_response(f"state_{state}")
        lines.append(f"{_when(s)}: {state}")

    if truncated:
        lines.append(localized_response("history_truncated", count=MAX_RESULTS))

    text = ". ".join(lines) + "."
    current = client._get_state(entity_id)
    artifact = _temperature_history_artifact(entity_id, friendly, artifact_states, current)
    if artifact is None and client._domain_of(entity_id) == "light":
        artifact = _light_history_artifact(friendly, artifact_states, current)
    return ArtifactText(text, artifact)


@client.tool(
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
    if not client.HA_TOKEN:
        return localized_response("ha_not_configured", "error")

    try:
        start_dt, default_end_dt, _ = _parse_history_start(start)
    except ValueError:
        return "Reactive question: " + localized_response("history_parse_failed", start=start) + " Ask the user to clarify the date."

    end_dt = _parse_history_end(end, default_end_dt)
    now = _local_tz.now()
    if end_dt > now:
        end_dt = now

    try:
        user_states = _fetch_history_states("sensor.fulloch_last_utterance", start_dt, end_dt)
        bot_states = _fetch_history_states("sensor.fulloch_last_response", start_dt, end_dt)
    except Exception as e:
        logger.warning(f"HA conversation history query failed: {e}")
        return "Reactive question: " + localized_response("history_conversation_failed") + " Tell the user there was an error."

    # Merge both sensors into a single timeline tagged by speaker, then sort
    # by timestamp so each user question sits next to the reply it drew.
    events = [(s, localized_response("history_user")) for s in user_states] + [(s, "Fulloch") for s in bot_states]
    events.sort(
        key=lambda it: client._normalise_ha_timestamp(
            it[0].get("last_changed") or it[0].get("last_updated") or ""
        )
    )

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
            ts = client._normalise_ha_timestamp(ts_str)
            ts_date = ts.date()
            if ts_date == today:
                day_label = localized_response("calendar_today")
            elif ts_date == yesterday:
                day_label = localized_response("calendar_yesterday")
            else:
                language = current_response_language.get()
                if language in {"fr", "de"}:
                    from babel.dates import format_date

                    day_label = format_date(ts.date(), "EEEE", locale=language)
                else:
                    day_label = ts.strftime("%A")
            language = current_response_language.get()
            if language in {"fr", "de"}:
                from babel.dates import format_time

                time_label = format_time(ts, format="short", locale=language)
            else:
                time_label = ts.strftime("%I:%M %p").lstrip("0")
            when = localized_response("history_when", day=day_label, time=time_label)
            lines.append(f"{when} — {speaker}: {state_val}")
        except Exception:
            lines.append(f"{speaker}: {state_val}")

    if not lines:
        window = localized_response("history_window_range", start=start, end=end) if end else start
        return localized_response("history_no_conversation", window=window)

    # Keep the most recent exchanges if the window is busy — the tail is what
    # "what did we talk about" usually means, and caps the replan payload.
    MAX_RESULTS = 30
    if len(lines) > MAX_RESULTS:
        lines = lines[-MAX_RESULTS:]

    transcript = "\n".join(lines)
    return (
        "Reactive question: " + localized_response("history_transcript_intro") + " Summarise for the user what "
        "was discussed, grouped by topic, in a sentence or two — do NOT read it "
        "back line by line, and do NOT re-research those topics with another "
        "tool.\n" + transcript
    )
