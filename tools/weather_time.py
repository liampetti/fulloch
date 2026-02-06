"""
Weather and time information tool using the centralized tool registry.
"""
import yaml

try:
    with open("./data/config.yml", "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}

from datetime import datetime
import re
from ftplib import FTP
import xmltodict
import io
from typing import Optional, Dict
from word2number import w2n
import threading
import json
import os
import time
import sys

from .tool_registry import tool, tool_registry

# Get the absolute path to the parent directory for importing of audio manager
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from audio.beep_manager import BeepManager
# Dictionary to store active timers
active_timers: Dict[str, threading.Timer] = {}

beep_manager = BeepManager()


def summarize_today_tomorrow(forecast_data, location):
    """Summarize weather forecast for today and tomorrow."""
    days = forecast_data['forecast-period'][:2]  # Only first two days
    summary_lines = [f"Forecast for {location}"]

    for day in days:
        date_str = day['@start-time-local'][:10]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        label = "Today" if date_obj.date() == datetime.today().date() else "Tomorrow"

        # Normalize to list
        elements = day.get('element', [])
        if isinstance(elements, dict):
            elements = [elements]
        texts = day.get('text', [])
        if isinstance(texts, dict):
            texts = [texts]

        element_dict = {e['@type']: e['#text'] for e in elements}
        text_dict = {t['@type']: t['#text'] for t in texts}

        min_temp = element_dict.get('air_temperature_minimum')
        max_temp = element_dict.get('air_temperature_maximum')
        precip_range = element_dict.get('precipitation_range')
        chance_of_rain = text_dict.get('probability_of_precipitation')
        precis = text_dict.get('precis', 'No forecast.')

        parts = [f"{label} expect {precis.replace('.', '')}"]
        if min_temp and max_temp:
            parts.append(f"{min_temp} to {max_temp} degrees Celcius")
        elif max_temp:
            parts.append(f"Maximum temperature {max_temp} degrees Celcius")
        if chance_of_rain:
            parts.append(f"with a {chance_of_rain.replace('%', ' percent')} chance of rain")
        if precip_range:
            parts.append(f"from {precip_range.replace('mm', 'millimetres')}")

        summary_lines.append(" ".join(parts))

    return ". ".join(summary_lines)


def load_weather_config():
    return config.get('bom', {
        "host": "ftp.bom.gov.au",
        "path": "/anon/gen/fwo/IDN11060.xml"
    })


@tool(
    name="get_weather_forecast",
    description="Get weather forecast for a location",
    aliases=["weather", "forecast", "get_weather"]
)
def get_weather_forecast(location: str = None) -> str:
    """
    Get weather forecast for the specified location.

    Args:
        location: The location for weather forecast

    Returns:
        Weather forecast summary
    """
    # Load FTP configuration
    ftp_config = load_weather_config()
    if location is None:
        location = ftp_config.get('default', 'Sydney')
    
    # Connect to FTP and retrieve file into memory
    ftp = FTP(ftp_config['host'])
    ftp.login()  # Anonymous login

    xml_data = io.BytesIO()
    ftp.retrbinary(f"RETR {ftp_config['path']}", xml_data.write)
    ftp.quit()

    xml_data.seek(0)
    data = xmltodict.parse(xml_data.read())

    # Find forecast
    areas = data['product']['forecast']['area']
    for area in areas:
        if area['@description'].lower() == location.lower():
            return summarize_today_tomorrow(area, location)

    # Return nothing if no forecast found
    return ""


@tool(
    name="get_current_time",
    description="Get the current date and time",
    aliases=["time", "what_time_is_it", "get_time"]
)
def get_current_time(location: Optional[str] = None) -> str:
    """
    Get the current date and time.
    
    Args:
        location: Optional location (not currently used)
        
    Returns:
        Formatted current date and time
    """
    # TODO: get location time
    # Format into natural spoken English
    text = datetime.now().strftime("%A %B %d %Y at %I %M %p")  # No punctuation

    # Optional: Remove leading zero in hour (e.g., "08" → "8")
    text = re.sub(r'\b0(\d)\b', r'\1', text)
    
    # Convert AM/PM to lowercase if preferred for TTS
    text = text.replace("AM", "A M").replace("PM", "P M")

    return text


@tool(
    name="start_countdown",
    description="Start a countdown timer for specified duration",
    aliases=["timer", "countdown", "set_timer", "start_timer"]
)
def start_countdown(duration: str) -> str:
    """
    Start a countdown timer for the specified duration.
    
    Args:
        duration: Duration string like "ten minutes" or "two hours"
        
    Returns:
        Confirmation message
    """
    def parse_duration(duration_str: str) -> int:
        duration_str = duration_str.lower()
        
        # Extract number words and convert to digits
        number_str = ""
        unit = ""
        for word in duration_str.split():
            try:
                val = w2n.word_to_num(word)
                number_str = str(val)
            except ValueError:
                unit += word + " "
        
        if not number_str:
            # Try direct digit extraction
            numbers = re.findall(r'\d+', duration_str)
            if not numbers:
                raise ValueError("No valid duration value found")
            number_str = numbers[0]
            
        value = int(number_str)
        
        if "hour" in unit:
            seconds = value * 3600
        elif "minute" in unit:
            seconds = value * 60
        elif "second" in unit:
            seconds = value
        else:
            raise ValueError("Unknown duration unit")
            
        return seconds

    def on_timer_complete(timer_id: str):
        if timer_id in active_timers:
            del active_timers[timer_id]
        # Play alarm sound three times with pause when timer completes
        for i in [1,1,1]:
            beep_manager.play_beep(filename="alarm.wav")
            time.sleep(i)

    try:
        seconds = parse_duration(duration)
        timer_id = f"timer_{len(active_timers) + 1}"
        
        # Create and start timer
        timer = threading.Timer(seconds, on_timer_complete, args=[timer_id])
        timer.daemon = True
        timer.start_time = time.time()  # Add this line to track start time
        timer.start()
        
        # Store timer reference
        active_timers[timer_id] = timer
        
        # Format response message
        if seconds >= 3600:
            hours = seconds // 3600
            return f"Timer started for {hours} {'hour' if hours == 1 else 'hours'}"
        elif seconds >= 60:
            minutes = seconds // 60
            return f"Timer started for {minutes} {'minute' if minutes == 1 else 'minutes'}"
        else:
            return f"Timer started for {seconds} {'second' if seconds == 1 else 'seconds'}"
            
    except ValueError as e:
        return f"Error: {str(e)}"


@tool(
    name="cancel_timer",
    description="Cancel an active timer",
    aliases=["stop_timer", "end_timer"]
)
def cancel_timer(timer_id: str) -> str:
    """
    Cancel an active timer.
    
    Args:
        timer_id: ID of timer to cancel
        
    Returns:
        Confirmation message
    """
    if timer_id in active_timers:
        timer = active_timers[timer_id]
        timer.cancel()
        del active_timers[timer_id]
        return f"Timer {timer_id} cancelled"
    return f"Timer {timer_id} not found"

@tool(
    name="get_timer_status",
    description="Get the status of a timer or all timers including time remaining",
    aliases=["timer_status", "check_timer", "show_timers", "get_timers", "list_timers"]
)
def get_timer_status(timer_id: Optional[str] = None) -> str:
    """
    Get status of a specific timer or all timers.
    
    Args:
        timer_id: Optional ID of timer to check. If None, shows all timers.
        
    Returns:
        Timer status information
    """
    def format_time_remaining(seconds: float) -> str:
        """Format remaining time into hours, minutes and seconds."""
        remaining = int(seconds)
        if remaining >= 3600:
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            seconds = remaining % 60
            return f"{hours} hours {minutes} minutes {seconds} seconds"
        elif remaining >= 60:
            minutes = remaining // 60
            seconds = remaining % 60
            return f"{minutes} minutes {seconds} seconds"
        else:
            return f"{remaining} seconds"

    if not active_timers:
        return "No active timers"
        
    if timer_id:
        if timer_id not in active_timers:
            return f"Timer {timer_id} not found"
            
        timer = active_timers[timer_id]
        remaining = max(0, timer.interval - (time.time() - timer.start_time))
        time_str = format_time_remaining(remaining)
        return f"Timer {timer_id} has {time_str} remaining"
    
    # Show status of all timers
    statuses = []
    for tid, timer in active_timers.items():
        remaining = max(0, timer.interval - (time.time() - timer.start_time))
        time_str = format_time_remaining(remaining)
        statuses.append(f"{tid}: {time_str}")
            
    return "Timer status:\n" + "\n".join(statuses)


if __name__ == "__main__":
    print("Weather and Time Information Tool")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calling:")
    result = tool_registry.execute_tool("get_current_time")
    print(f"Current time: {result}")
    
    result = tool_registry.execute_tool("get_weather_forecast", kwargs={"location": "Sydney"})
    print(f"Weather forecast: {result}")
    
    print("\nTesting timer functions:")
    result = tool_registry.execute_tool("start_countdown", kwargs={"duration": "ten minutes"})
    print(result)
    
    result = tool_registry.execute_tool("list_timers")
    print(result)
    
    result = tool_registry.execute_tool("cancel_timer", kwargs={"timer_id": "timer_1"})
    print(result)
    
    print("\nTesting timer status:")
    result = tool_registry.execute_tool("start_countdown", kwargs={"duration": "5 minutes"})
    print(result)

    time.sleep(5)  # Wait for a bit to let the timer start
    
    result = tool_registry.execute_tool("get_timer_status")
    print(result)
    
    result = tool_registry.execute_tool("get_timer_status", kwargs={"timer_id": "timer_1"})
    print(result)
    
    print("\nTesting timer completion:")
    result = tool_registry.execute_tool("start_countdown", kwargs={"duration": "3 seconds"})
    print(result)
    
    # Wait for timer to complete
    print("Waiting for timer to finish...")
    time.sleep(4)  # Wait slightly longer than timer duration
    
    result = tool_registry.execute_tool("list_timers")
    print(f"After completion: {result}")

