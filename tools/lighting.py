"""
Philips Hue lighting control tool
"""
import yaml

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

from phue import Bridge

from .tool_registry import tool, tool_registry

b = Bridge(config['philips']['hue_hub_ip'], config_file_path="./data/.python_hue")


@tool(
    name="turn_on_lights",
    description="Turn on lights in a specific location",
    aliases=["lights_on", "switch_on_lights", "turn_on"]
)
def turn_on_lights(location: str = "Downlights Office") -> str:
    """
    Turn on lights in the specified location.
    
    Args:
        location: The location/room name for the lights
        
    Returns:
        Status message about the action
    """
    location = location.title()  # First Letters Capitalized
    if location in b.get_light_objects('name').keys():
        b.set_light(location, 'on', True)
        return f"{location} lights on"
    
    groups = b.get_group()
    group_names = []
    for i in groups:
        group_names.append(groups[i]['name'])
    if location in group_names:
        b.set_group(location, 'on', True)
        return f"{location} on"
    else:
        return f"No lights or rooms with name {location}"


@tool(
    name="turn_off_lights",
    description="Turn off lights in a specific location",
    aliases=["lights_off", "switch_off_lights", "turn_off"]
)
def turn_off_lights(location: str = "Downlights Office") -> str:
    """
    Turn off lights in the specified location.
    
    Args:
        location: The location/room name for the lights
        
    Returns:
        Status message about the action
    """
    try:
        location = location.title()  # First Letters Capitalized
        if location in b.get_light_objects('name').keys():
            b.set_light(location, 'on', False)
            return f"{location} lights off"
        
        groups = b.get_group()
        group_names = []
        for i in groups:
            group_names.append(groups[i]['name'])
        if location in group_names:
            b.set_group(location, 'on', False)
            return f"{location} off"
        else:
            return f"No lights or rooms with name {location}"
    except Exception as e:
            return f"Unable to connect to lights for {location}"


@tool(
    name="set_brightness",
    description="Set brightness level for lights in a specific location",
    aliases=["brightness", "dim_lights", "brighten_lights"]
)
def set_brightness(percent: int = 100, location: str = "Downlights Office") -> str:
    """
    Set brightness level for lights in the specified location.
    
    Args:
        percent: Brightness percentage (0-100)
        location: The location/room name for the lights
        
    Returns:
        Status message about the action
    """   
    try:
        location = location.title()  # First Letters Capitalized
        if location in b.get_light_objects('name').keys():
            b.set_light(location, 'on', True)
            level = int((int(percent) / 100) * 254)
            b.set_light(location, 'bri', level)
            return f"{location} lights set to {percent} percent."
        
        groups = b.get_group()
        group_names = []
        for i in groups:
            group_names.append(groups[i]['name'])
        if location in group_names:
            b.set_group(location, 'on', True)
            level = int((int(percent) / 100) * 254)
            b.set_group(location, 'bri', level)
            return f"{location} set to {percent} percent."
        else:
            return f"No lights or rooms with name {location}"
    except Exception as e:
            return f"Unable to connect to lights for {location}"


if __name__ == "__main__":
    print("Philips Hue Lighting Controller")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    result = tool_registry.execute_tool("turn_on_lights", kwargs={"location": "kitchen"})
    print(f"Result: {result}")



