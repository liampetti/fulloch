"""Home Assistant integration for Fulloch voice assistant.

Connects to a Home Assistant instance via REST API for home automation control.
Requires a long-lived access token configured in data/config.yml.

This module is only loaded when 'home_assistant' is present in config.yml.
Note: This registers generic tool names like "turn_on" and "turn_off" which
may conflict with other integrations (e.g., Philips Hue lighting).
"""

import requests
import yaml
from typing import Optional

from .tool_registry import tool

# Load configuration
with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)
HA_CONFIG = config.get('home_assistant', {})

# Home Assistant connection settings
HA_URL = HA_CONFIG.get('url', 'http://localhost:8123')
HA_TOKEN = HA_CONFIG.get('token', '')
TIMEOUT = HA_CONFIG.get('timeout', 10)


def _get_headers() -> dict:
    """Return authorization headers for Home Assistant API."""
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _call_service(domain: str, service: str, entity_id: str, data: Optional[dict] = None) -> str:
    """Call a Home Assistant service."""
    if not HA_TOKEN:
        return "Error: Home Assistant token not configured. Add 'token' to home_assistant config."

    url = f"{HA_URL}/api/services/{domain}/{service}"
    payload = {"entity_id": entity_id}
    if data:
        payload.update(data)

    try:
        response = requests.post(url, headers=_get_headers(), json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return f"Successfully called {domain}.{service} on {entity_id}"
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to Home Assistant at {HA_URL}"
    except requests.exceptions.Timeout:
        return f"Error: Home Assistant request timed out"
    except requests.exceptions.HTTPError as e:
        return f"Error: Home Assistant returned {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


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
    """Resolve a friendly name or alias to an entity_id.

    Checks the entity_aliases config mapping first, then assumes
    the name is already a valid entity_id.
    """
    aliases = HA_CONFIG.get('entity_aliases', {})

    # Check if it's a configured alias
    if name.lower() in aliases:
        return aliases[name.lower()]

    # If it looks like an entity_id already, return as-is
    if '.' in name:
        return name

    # Try to construct entity_id from name and domain
    if domain:
        # Convert "living room lights" -> "light.living_room_lights"
        entity_name = name.lower().replace(' ', '_')
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

    # Determine the domain from entity_id
    domain = entity_id.split('.')[0] if '.' in entity_id else 'homeassistant'

    data = {}
    if brightness is not None and domain == 'light':
        # Convert percentage to 0-255 range
        data['brightness'] = int((brightness / 100) * 255)

    return _call_service(domain, "turn_on", entity_id, data if data else None)


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
    domain = entity_id.split('.')[0] if '.' in entity_id else 'homeassistant'

    return _call_service(domain, "turn_off", entity_id)


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
    domain = entity_id.split('.')[0] if '.' in entity_id else 'homeassistant'

    return _call_service(domain, "toggle", entity_id)


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

    # Clamp brightness to valid range
    brightness = max(0, min(100, brightness))
    brightness_255 = int((brightness / 100) * 255)

    return _call_service("light", "turn_on", entity_id, {"brightness": brightness_255})


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

    # Common color name mappings
    color_map = {
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

    color_lower = color.lower().strip()

    if color_lower in color_map:
        rgb = color_map[color_lower]
    elif ',' in color:
        # Parse RGB string like "255,128,0"
        try:
            rgb = [int(c.strip()) for c in color.split(',')]
            if len(rgb) != 3:
                return "Error: RGB color must have 3 values (e.g., '255,128,0')"
        except ValueError:
            return f"Error: Invalid RGB color format '{color}'"
    else:
        return f"Error: Unknown color '{color}'. Use a color name or RGB format."

    return _call_service("light", "turn_on", entity_id, {"rgb_color": rgb})


@tool(
    name="get_entity_state",
    description="Get the current state of a Home Assistant entity",
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
        return f"Error: Could not get state for {entity_id}"

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
    import json

    entity_id = _resolve_entity(entity)

    extra_data = None
    if data:
        try:
            extra_data = json.loads(data)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data: {data}"

    return _call_service(domain, service, entity_id, extra_data)


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

    data = {"temperature": temperature}
    if hvac_mode:
        data["hvac_mode"] = hvac_mode.lower()

    return _call_service("climate", "set_temperature", entity_id, data)


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
    return _call_service("lock", "lock", entity_id)


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
    return _call_service("lock", "unlock", entity_id)


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
    return _call_service("cover", "open_cover", entity_id)


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
    return _call_service("cover", "close_cover", entity_id)


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
    return _call_service("script", "turn_on", entity_id)


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
    return _call_service("scene", "turn_on", entity_id)
