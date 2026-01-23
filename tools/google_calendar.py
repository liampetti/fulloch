"""
Google Calendar integration tool using the centralized tool registry.

This module provides calendar functionality with proper
schema definitions and function calling support.
"""
import os
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env vars

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

import datetime
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from .tool_registry import tool, tool_registry

# If modifying SCOPES, delete the token.json file
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

TOKEN = config['google']['token_file']
CREDS = config['google']['cred_file']


def authenticate_google_calendar():
    """Authenticate with Google Calendar API, refreshing or acquiring a new token if needed."""
    creds = None
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    # If credentials are invalid or do not exist, refresh or run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Refresh failed, must re-authenticate
                flow = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN, 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)


def get_events(service, time_min, time_max):
    """Get events from Google Calendar."""
    events_result = service.events().list(
        calendarId='primary', timeMin=time_min.isoformat() + 'Z',
        timeMax=time_max.isoformat() + 'Z', singleEvents=True,
        orderBy='startTime').execute()
    return events_result.get('items', [])


def summarize_events(events):
    """Summarize events in a readable format."""
    if not events:
        return "No events found."
    summary_lines = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        start_time = datetime.datetime.fromisoformat(start).strftime("%a %I:%M %p") if 'T' in start else start
        summary_lines.append(f"- {start_time}: {event.get('summary', 'No title')}")
    return "\n".join(summary_lines)


def tts_friendly_summary(events):
    """Create a TTS-friendly summary of events."""
    if not events:
        return "You have no events scheduled."

    spoken = []
    for event in events:
        raw_start = event['start'].get('dateTime', event['start'].get('date'))
        if 'T' in raw_start:
            start_dt = datetime.datetime.fromisoformat(raw_start)
            day = start_dt.strftime("%A")  # e.g., Monday
            time = start_dt.strftime("%-I %M %p").lower().replace("am", "a m").replace("pm", "p m")
            time = re.sub(r'\b00\b', "o'clock", time)  # 10 00 → 10 o'clock
            spoken.append(f"At {time} on {day}, {event.get('summary', 'an event')}.")
        else:
            day = datetime.datetime.fromisoformat(raw_start).strftime("%A")
            spoken.append(f"All day on {day}, {event.get('summary', 'an event')}.")

    return " ".join(spoken)


@tool(
    name="whats_on",
    description="Get calendar events for a specific time period",
    aliases=["calendar", "events", "schedule"]
)
def whats_on(day: str = "today") -> str:
    """
    Get calendar events for a specific time period.
    
    Args:
        day: Time period - 'today', 'tomorrow', or 'week'
        
    Returns:
        TTS-friendly summary of events
    """
    try:
        service = authenticate_google_calendar()
        if day == 'today':
            start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
            end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
        elif day == 'tomorrow':
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            start = datetime.datetime.combine(tomorrow, datetime.time.min)
            end = datetime.datetime.combine(tomorrow, datetime.time.max)
        elif day == 'week':
            start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
            end = start + datetime.timedelta(days=7)
        else:
            raise ValueError("Use 'today', 'tomorrow', or 'week'.")

        events = get_events(service, start, end)
        return tts_friendly_summary(events)
    except Exception as e:
        return "Unable to get calendar events"


@tool(
    name="whats_on_today",
    description="Get calendar events for today"
)
def whats_on_today() -> str:
    """Get calendar events for today."""
    return whats_on("today")


@tool(
    name="whats_on_tomorrow",
    description="Get calendar events for tomorrow"
)
def whats_on_tomorrow() -> str:
    """Get calendar events for tomorrow."""
    return whats_on("tomorrow")


@tool(
    name="whats_on_this_week",
    description="Get calendar events for this week"
)
def whats_on_this_week() -> str:
    """Get calendar events for this week."""
    return whats_on("week")


if __name__ == '__main__':
    print("Google Calendar Integration Tool")
    
    # Print available tools
    print("\nAvailable tools:")
    for schema in tool_registry.get_all_schemas():
        print(f"  {schema.name}: {schema.description}")
        for param in schema.parameters:
            print(f"    - {param.name} ({param.type.value}): {param.description}")
    
    # Test function calling
    print("\nTesting function calls:")
    for period in ['today', 'tomorrow', 'week']:
        result = tool_registry.execute_tool("whats_on", kwargs={"day": period})
        print(f"Events for {period}: {result}")
