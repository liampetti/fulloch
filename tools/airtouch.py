"""
Airtouch HVAC control tool
"""
import yaml

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

import asyncio
import pyairtouch
from .tool_registry import tool, tool_registry

# Load light names and groups from config file
zone_ids = config['airtouch']


async def get_ac():
    """Get the air conditioner device."""
    devices = await pyairtouch.discover()
    if len(devices) > 0:
        # Connect to the first discovered device
        airtouch = devices[0]
        success = await airtouch.init()
        if success:
            return airtouch
        else:
            return None
    else:
        return None


async def _get_temperature(location):
    """Get temperature for a specific location."""
    ac = await get_ac()
    if ac is not None:
        ac = ac.air_conditioners[0]
        if location.lower() in zone_ids.keys():
            zone_index = zone_ids[location.lower()]
            return f"The {location} temperature is {ac.zones[zone_index].current_temperature} degrees Celcius"
        else:
            return f"{location} not found"
    else:
        return f"No AC device found"


def get_temperature(location: str):
    """
    Get temperature for a specific location.
    
    Args:
        location: The location/zone name
        
    Returns:
        Current temperature information
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, it's safe to use asyncio.run
        return asyncio.run(_get_temperature(location))
    else:
        # There is a running loop (common in Jupyter/servers).
        # Safe solution: Run as a task and get result.
        return loop.create_task(_get_temperature(location))


async def _set_temperature(new_temp, location):
    """Set temperature for a specific location."""
    if new_temp > 25:
        new_temp = 25 # Max temperature limits
    if new_temp < 17:
        new_temp = 17 # Min temperature limits
        
    ac = await get_ac()
    if ac is not None:
        ac = ac.air_conditioners[0]
        if location.lower() in zone_ids.keys():
            zone_index = zone_ids[location.lower()]
            zone = ac.zones[zone_index]
            await ac.set_power(True)
            await zone.set_target_temperature(new_temp)
            return f"The {location} target temperature is now {zone.target_temperature} degrees Celcius"
        else:
            return f"{location} not found"
    else:
        return f"No AC device found"


@tool(
    name="set_temperature",
    description="Set the target temperature for a specific location",
    aliases=["temperature", "set_ac_temperature"]
)
def set_temperature(new_temp: int, location: str):
    """
    Set the target temperature for a specific location.
    
    Args:
        new_temp: Target temperature in degrees Celsius
        location: The location/zone name
        
    Returns:
        Status message about the temperature setting
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_set_temperature(new_temp, location))
    else:
        return loop.create_task(_set_temperature(new_temp, location))


async def _turn_on_ac():
    """Turn on the air conditioner."""
    ac = await get_ac()
    if ac is not None:
        ac = ac.air_conditioners[0]
        success = await ac.set_power(True)
        if success:
            return "Air conditioner turned on"
        else:
            return "Unable to turn on air conditioner"
    else:
        return f"No air conditioner found"


@tool(
    name="turn_on_ac",
    description="Turn on the air conditioner",
    aliases=["ac_on", "start_ac", "turn_on_air_conditioner"]
)
def turn_on_ac():
    """Turn on the air conditioner."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_turn_on_ac())
    else:
        return loop.create_task(_turn_on_ac())


async def _turn_off_ac():
    """Turn off the air conditioner."""
    ac = await get_ac()
    if ac is not None:
        ac = ac.air_conditioners[0]
        success = await ac.set_power(False)
        if success:
            return "Air conditioner turned off"
        else:
            return "Unable to turn off air conditioner"
    else:
        return f"No air conditioner found"


@tool(
    name="turn_off_ac",
    description="Turn off the air conditioner",
    aliases=["ac_off", "stop_ac", "turn_off_air_conditioner"]
)
def turn_off_ac():
    """Turn off the air conditioner."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_turn_off_ac())
    else:
        return loop.create_task(_turn_off_ac())


@tool(
    name="get_temperature",
    description="Get the current temperature for a specific location",
    aliases=["temperature", "current_temperature"]
)
def get_temperature_tool(location: str) -> str:
    """
    Get the current temperature for a specific location.
    
    Args:
        location: The location/zone name
        
    Returns:
        Current temperature information
    """
    return get_temperature(location)


if __name__ == "__main__":
    print("Airtouch HVAC Controller")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calls:")
    result = tool_registry.execute_tool("get_temperature", kwargs={"location": "office"})
    print(f"Temperature: {result}")