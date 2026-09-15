"""Home Assistant devices implementation.

Uses ha_client as the single owner of configuration and resolution state.
"""

import json
import logging
from typing import Optional

from core.satellite_context import current_satellite_id, get_current_assistant

from . import ha_client as client

logger = logging.getLogger(__name__)

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
    for filler in client._LEADING_FILLERS:
        if key.startswith(filler):
            key = key[len(filler) :].strip()
            break
    if not _is_bare_light_phrase(key):
        return None

    ha_area = _current_satellite_ha_area()
    if not ha_area:
        return None
    area_id = client._resolve_area(ha_area)
    if area_id is None:
        return None

    raw = client._render_template(f"{{{{ area_entities({area_id!r}) | list | tojson }}}}")
    if raw is None:
        return None
    try:
        entity_ids = json.loads(raw)
    except (ValueError, TypeError):
        return None

    lights = [
        eid
        for eid in entity_ids
        if client._domain_of(eid) == "light" and eid not in client._DENIED_ENTITIES
    ]
    if not lights:
        return None
    area_name = client._AREA_MAP.get(area_id, ha_area)
    return lights, area_name


@client.tool(
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
        return client._call_service("light", "turn_on", light_ids, data if data else None, success)

    entity_id = client._resolve_entity(entity, domain="light")
    domain = client._domain_of(entity_id)
    friendly = client._friendly_for(entity_id)

    data = {}
    success = f"{friendly} on"
    if brightness is not None and domain == "light":
        data["brightness"] = int((brightness / 100) * 255)
        success = f"{friendly} on at {brightness} percent"

    return client._call_service(domain, "turn_on", entity_id, data if data else None, success)


@client.tool(
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
        return client._call_service(
            "light", "turn_off", light_ids, success_message=f"Lights off in {area_name}"
        )

    entity_id = client._resolve_entity(entity)
    domain = client._domain_of(entity_id)
    friendly = client._friendly_for(entity_id)

    return client._call_service(domain, "turn_off", entity_id, success_message=f"{friendly} off")


@client.tool(
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
        return client._call_service(
            "light", "toggle", light_ids, success_message=f"Toggled the lights in {area_name}"
        )

    entity_id = client._resolve_entity(entity)
    domain = client._domain_of(entity_id)
    friendly = client._friendly_for(entity_id)

    return client._call_service(domain, "toggle", entity_id, success_message=f"Toggled {friendly}")


@client.tool(
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
        return client._call_service(
            "light",
            "turn_on",
            light_ids,
            {"brightness": brightness_255},
            success_message=f"Lights in {area_name} at {brightness} percent",
        )

    entity_id = client._resolve_entity(entity, domain="light")
    friendly = client._friendly_for(entity_id)

    return client._call_service(
        "light",
        "turn_on",
        entity_id,
        {"brightness": brightness_255},
        success_message=f"{friendly} at {brightness} percent",
    )


@client.tool(
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
    entity_id = client._resolve_entity(entity, domain="light")
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

    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "light",
        "turn_on",
        entity_id,
        {"rgb_color": rgb},
        success_message=f"{friendly} set to {color_lower}",
    )


@client.tool(
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
    entity_id = client._resolve_entity(entity)

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

    friendly = client._friendly_for(entity_id)
    return client._call_service(
        domain,
        service,
        entity_id,
        extra_data,
        success_message=f"{service.replace('_', ' ').capitalize()} {friendly}",
    )


@client.tool(
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
    entity_id = client._resolve_entity(entity, domain="climate")
    friendly = client._friendly_for(entity_id)

    data = {"temperature": temperature}
    if hvac_mode:
        data["hvac_mode"] = hvac_mode.lower()

    return client._call_service(
        "climate",
        "set_temperature",
        entity_id,
        data,
        success_message=f"{friendly} set to {temperature} degrees",
    )


@client.tool(
    name="ha_lock", description="Lock a lock entity in Home Assistant", aliases=["lock_door"]
)
def lock(entity: str) -> str:
    """Lock a lock entity.

    Args:
        entity: Lock entity name or ID
    """
    entity_id = client._resolve_entity(entity, domain="lock")
    friendly = client._friendly_for(entity_id)
    return client._call_service("lock", "lock", entity_id, success_message=f"Locked {friendly}")


@client.tool(
    name="ha_unlock", description="Unlock a lock entity in Home Assistant", aliases=["unlock_door"]
)
def unlock(entity: str) -> str:
    """Unlock a lock entity.

    Args:
        entity: Lock entity name or ID
    """
    entity_id = client._resolve_entity(entity, domain="lock")
    friendly = client._friendly_for(entity_id)
    return client._call_service("lock", "unlock", entity_id, success_message=f"Unlocked {friendly}")


# `valve.*` entities (smart water/gas shutoffs) use their own domain and
# service names but the same open/close/stop/position shape as `cover.*`
# (blinds, garages) — one set of tools handles both rather than duplicating
# every verb under a second "valve" name.
_COVER_LIKE_SERVICES = {
    "cover": {
        "open": "open_cover",
        "close": "close_cover",
        "stop": "stop_cover",
        "set_position": "set_cover_position",
    },
    "valve": {
        "open": "open_valve",
        "close": "close_valve",
        "stop": "stop_valve",
        "set_position": "set_valve_position",
    },
}


def _cover_like_domain(entity_id: str) -> str:
    return "valve" if entity_id.startswith("valve.") else "cover"


@client.tool(
    name="ha_open_cover",
    description="Open a cover, blind, garage door, or valve in Home Assistant",
    aliases=["ha_open", "open_blind", "open_garage", "open_valve"],
)
def open_cover(entity: str) -> str:
    """Open a cover or valve entity (blinds, garage door, water/gas valve, etc.).

    Args:
        entity: Cover or valve entity name or ID
    """
    entity_id = client._resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        domain,
        _COVER_LIKE_SERVICES[domain]["open"],
        entity_id,
        success_message=f"Opened {friendly}",
    )


@client.tool(
    name="ha_close_cover",
    description="Close a cover, blind, garage door, or valve in Home Assistant",
    aliases=["ha_close", "close_blind", "close_garage", "close_valve"],
)
def close_cover(entity: str) -> str:
    """Close a cover or valve entity (blinds, garage door, water/gas valve, etc.).

    Args:
        entity: Cover or valve entity name or ID
    """
    entity_id = client._resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        domain,
        _COVER_LIKE_SERVICES[domain]["close"],
        entity_id,
        success_message=f"Closed {friendly}",
    )


@client.tool(
    name="ha_stop_cover",
    description="Stop a moving cover, blind, garage door, or valve in Home Assistant",
    aliases=["stop_cover", "halt_cover", "stop_blind", "stop_garage", "stop_valve"],
)
def stop_cover(entity: str) -> str:
    """Halt a cover or valve entity mid-travel.

    Args:
        entity: Cover or valve entity name or ID
    """
    entity_id = client._resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        domain,
        _COVER_LIKE_SERVICES[domain]["stop"],
        entity_id,
        success_message=f"Stopped {friendly}",
    )


@client.tool(
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
    entity_id = client._resolve_entity(entity, domain="cover")
    domain = _cover_like_domain(entity_id)
    friendly = client._friendly_for(entity_id)
    pct = max(0, min(100, int(position)))
    return client._call_service(
        domain,
        _COVER_LIKE_SERVICES[domain]["set_position"],
        entity_id,
        {"position": pct},
        success_message=f"{friendly} set to {pct} percent open",
    )


@client.tool(
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
    entity_id = client._resolve_entity(entity, domain="fan")
    friendly = client._friendly_for(entity_id)
    pct = max(0, min(100, int(speed)))
    return client._call_service(
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


@client.tool(
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
    entity_id = client._resolve_entity(entity, domain="vacuum")
    friendly = client._friendly_for(entity_id)
    key = action.lower().strip().replace(" ", "_")
    if key not in _VACUUM_ACTIONS:
        return f"I don't know how to '{action}' a vacuum — try start, pause, stop, dock, or locate."
    service, template = _VACUUM_ACTIONS[key]
    return client._call_service(
        "vacuum", service, entity_id, success_message=template.format(friendly=friendly)
    )


@client.tool(
    name="ha_run_script",
    description="Run a Home Assistant script or automation",
    aliases=["ha_script", "run_automation"],
)
def run_script(script_name: str) -> str:
    """Run a Home Assistant script.

    Args:
        script_name: Script entity ID or name (e.g., 'script.bedtime' or 'bedtime')
    """
    entity_id = client._resolve_entity(script_name, domain="script")
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "script", "turn_on", entity_id, success_message=f"Running {friendly}"
    )


@client.tool(
    name="ha_activate_scene",
    description="Activate a Home Assistant scene",
    aliases=["ha_scene", "set_scene"],
)
def activate_scene(scene_name: str) -> str:
    """Activate a Home Assistant scene.

    Args:
        scene_name: Scene entity ID or name (e.g., 'scene.movie_time' or 'movie time')
    """
    entity_id = client._resolve_entity(scene_name, domain="scene")
    friendly = client._friendly_for(entity_id)
    return client._call_service(
        "scene", "turn_on", entity_id, success_message=f"Scene {friendly} activated"
    )
