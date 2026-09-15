"""Home Assistant calendar implementation.

Uses ha_client as the single owner of configuration and resolution state.
"""

import datetime as _dt
import difflib
import logging
import re
from typing import Optional

import utils.local_time as _local_tz
from core.datetime_utils import tts_friendly_event_summary

from . import ha_client as client
from .tool_registry import ArtifactText

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calendar — wraps the HA `calendar.get_events` service.
# ---------------------------------------------------------------------------


def _calendar_window(day: str, now: Optional[_dt.datetime] = None) -> tuple[str, str]:
    """Return (start_iso, end_iso) for a spoken day phrase or ISO date.

    Args:
        day: "today", "tomorrow", "this_week" (also "week"), or a specific
            ISO date "YYYY-MM-DD" for a single-day window.
        now: Override for testing; defaults to Home Assistant's local time.
    """
    now = now or _local_tz.now()
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
            start = _dt.datetime.combine(d, _dt.time(), tzinfo=midnight.tzinfo)
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
    all_day = "T" not in start
    if not all_day:
        try:
            start = client._normalise_ha_timestamp(start).isoformat()
        except (AttributeError, TypeError, ValueError):
            pass
    return {
        "start": start,
        "summary": event.get("summary"),
        "all_day": all_day,
    }


def _calendar_artifact(events: list[dict], title: str) -> dict | None:
    """Turn normalised HA events into a bounded, dashboard-only agenda."""
    if not events:
        return None
    now = _local_tz.now()
    entries = []
    for event in events[:12]:
        try:
            start = client._normalise_ha_timestamp(event["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if event.get("all_day"):
            when = "All day"
        elif start.date() == now.date():
            when = start.strftime("%-I:%M %p")
        else:
            when = start.strftime("%a %-d, %-I:%M %p")
        entries.append(
            {
                "title": str(event.get("summary") or "Untitled event"),
                "when": when,
                "all_day": bool(event.get("all_day")),
            }
        )
    return (
        {"type": "calendar", "title": title.replace("_", " ").title(), "events": entries}
        if entries
        else None
    )


def _read_calendars() -> list[str]:
    """Calendars the calendar read tools query.

    The autodetected primary calendar PLUS the configured reminder calendar
    Fulloch writes to (`create_calendar_event`). Without the reminder
    calendar here, events Fulloch creates are invisible to its own
    either lookup tool whenever the write target differs from the autodetected read
    target (e.g. config `calendar: "Fulloch"` vs an auto-picked
    `calendar.primary`). Deduped, order-preserving.
    """
    cals: list[str] = []
    for c in (client.CALENDAR_ENTITY, _reminder_calendar_entity()):
        if c and c not in cals:
            cals.append(c)
    return cals


def _ha_get_events(day: str) -> str:
    """Common body for whats_on and its day-specific aliases."""
    calendars = _read_calendars()
    if not calendars:
        return "No calendar is configured in Home Assistant."

    start_iso, end_iso = _calendar_window(day)
    response = client._call_service_with_response(
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
    # Merged calendars arrive grouped by source; normalised timestamps make
    # ordering correct even when calendars return different UTC offsets.
    events.sort(key=lambda e: client._normalise_ha_timestamp(e["start"]))
    summary = tts_friendly_event_summary(events)
    # Multi-event days benefit from agent summarisation/filtering; route
    # through the replan loop. The "no events" case stays as a direct
    # spoken result.
    text = f"Reactive question: {summary}" if len(events) >= 2 else summary
    return ArtifactText(text, _calendar_artifact(events, day))


@client.tool(
    name="whats_on",
    description=(
        "List every calendar event in a fixed window: pass 'today' (default), "
        "'tomorrow', 'this_week', or a specific date as 'YYYY-MM-DD'. For a "
        "specific named event, use find_calendar_event instead."
    ),
    aliases=[
        "calendar",
        "events",
        "schedule",
    ],
)
def whats_on(day: str = "today") -> str:
    """List every calendar event in a fixed day or week window.

    Args:
        day: "today" (default), "tomorrow", "this_week", or an ISO date
            "YYYY-MM-DD" for a single day.
    """
    return _ha_get_events(day)


@client.tool(
    name="find_calendar_event",
    description=(
        "Find a named calendar event. Pass event_name first. Optionally pass "
        "day as 'today', 'tomorrow', 'this_week', or 'YYYY-MM-DD' to search "
        "only that window; omit day to search one year before and after today."
    ),
    aliases=["find_event", "search_calendar", "when_was", "when_is_it_on"],
)
def find_calendar_event(event_name: str, day: Optional[str] = None, limit: str = "1y") -> str:
    """Find one named event in an explicit window or the broad default range.

    Args:
        event_name: The event title or natural-language description to match.
        day: Optional fixed window: "today", "tomorrow", "this_week", or an
            ISO date. Omit it when the user gave no time scope.
        limit: Broad-search range either side of today, e.g. "30d", "2w", or
            "6m". Ignored when day is set. Default "1y".
    """
    return _ha_get_events_name(event_name, day=day, limit=limit)


def _parse_lookback_days(limit: str, default: int = 365) -> int:
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


def _calendar_match_observation(matches: list[dict]) -> str:
    """Return named-event matches with absolute dates for the next agent step.

    A named event is often only an intermediate fact, such as when the user
    asks how long until a trip.  Keep the timestamp lossless so the agent can
    choose a follow-up action rather than infer a date from relative wording.
    """
    details = []
    for event in matches:
        start = client._normalise_ha_timestamp(event["start"])
        summary = event.get("summary") or "an event"
        if event.get("all_day"):
            details.append(f"{summary}: start date {start.date().isoformat()} (all day)")
        else:
            details.append(f"{summary}: starts {start.isoformat()}")
    return "Reactive question: Calendar search found: " + "; ".join(details) + "."


_CALENDAR_MATCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "for",
        "from",
        "in",
        "is",
        "my",
        "of",
        "on",
        "our",
        "the",
        "to",
        "with",
    }
)
_CALENDAR_MATCH_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise_calendar_match_text(value: object) -> str:
    """Lowercase and remove punctuation so calendar-title matching is stable."""
    return " ".join(_CALENDAR_MATCH_TOKEN_RE.findall(str(value or "").lower()))


def _calendar_match_tokens(value: object) -> frozenset[str]:
    """Return meaningful, order-independent tokens for conservative matching."""
    return frozenset(
        token
        for token in _CALENDAR_MATCH_TOKEN_RE.findall(str(value or "").lower())
        if token not in _CALENDAR_MATCH_STOPWORDS
    )


def _ha_get_events_name(
    event_description: str, *, day: Optional[str] = None, limit: str = "1y"
) -> str:
    """Search calendars for events whose summary matches `event_description`,
    in `day`'s fixed window, or within `limit` days either side of now."""
    calendars = _read_calendars()
    if not calendars:
        return "No calendar is configured in Home Assistant."

    if day:
        start_iso, end_iso = _calendar_window(day)
        range_description = day.replace("_", " ")
    else:
        days = _parse_lookback_days(limit)
        now = _local_tz.now()
        start_iso = (now - _dt.timedelta(days=days)).isoformat()
        end_iso = (now + _dt.timedelta(days=days)).isoformat()
        range_description = f"{days} days before or after today"

    response = client._call_service_with_response(
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

    query = _normalise_calendar_match_text(event_description)
    matches = [
        event
        for event in events
        if query and query in _normalise_calendar_match_text(event.get("summary"))
    ]
    if not matches:
        query_tokens = _calendar_match_tokens(event_description)
        # A one-word query is too broad for token matching; exact-substring
        # matching above already covers the reliable single-word cases.
        if len(query_tokens) >= 2:
            matches = [
                event
                for event in events
                if query_tokens.issubset(_calendar_match_tokens(event.get("summary")))
            ]
    if not matches:
        names = [_normalise_calendar_match_text(event.get("summary")) for event in events]
        close = set(difflib.get_close_matches(query, names, n=5, cutoff=0.5))
        matches = [
            event
            for event in events
            if _normalise_calendar_match_text(event.get("summary")) in close
        ]

    if not matches:
        return f"I couldn't find any events matching '{event_description}' in {range_description}."

    matches.sort(key=lambda e: client._normalise_ha_timestamp(e["start"]))
    return ArtifactText(
        _calendar_match_observation(matches),
        _calendar_artifact(matches, "Calendar match"),
    )


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
    response = client._call_service_with_response(
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
        # Date-only starts are all-day events, not time-specific reminders.
        if not summary or "T" not in start:
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
    client._ensure_loaded()  # reminder-poll path — not a @tool, so load the map explicitly
    name = client.HA_CONFIG.get("calendar")
    if not name:
        return None
    if "." in str(name):
        return name
    entity_id = client._ENTITY_ALIASES.get(name.lower())
    if entity_id and entity_id.startswith("calendar."):
        return entity_id
    logger.warning(f"Reminder calendar '{name}' not found in HA entity aliases")
    return None


@client.tool(
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
    return client._call_service(
        "calendar", "create_event", calendar, extra, success_message=success
    )


# ---------------------------------------------------------------------------
# Todo / task lists — wraps HA `todo.add_item` and `todo.get_items`.
# ---------------------------------------------------------------------------


@client.tool(
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
    if not client.TODO_ENTITY:
        return (
            "User question: No todo list is configured in Home Assistant. "
            "Would you like me to add this to a note instead?"
        )
    return client._call_service(
        "todo",
        "add_item",
        client.TODO_ENTITY,
        {"item": item},
        success_message=f"Added '{item}' to your list",
    )


@client.tool(
    name="get_todo_items",
    description="Read pending items from the Home Assistant todo or shopping list.",
    aliases=["read_todo", "show_todo", "shopping_list", "get_shopping_list", "read_shopping_list"],
)
def get_todo_items() -> str:
    """Return pending (incomplete) items from the configured HA todo list."""
    if not client.TODO_ENTITY:
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
        text = f"You have one item: {names[0]}."
    else:
        text = f"You have {len(names)} items: {', '.join(names[:-1])}, and {names[-1]}."
    return ArtifactText(text, {"type": "todos", "items": names[:20]})


def _fetch_pending_todo_items() -> Optional[list]:
    """Fetch not-yet-completed items from the configured HA todo list, or None on error."""
    response = client._call_service_with_response(
        "todo",
        "get_items",
        {"entity_id": client.TODO_ENTITY, "status": "needs_action"},
    )
    if not response:
        return None
    return (response.get(client.TODO_ENTITY) or {}).get("items") or []


@client.tool(
    name="complete_todo_item",
    description="Mark an item as done on the Home Assistant todo or shopping list.",
    aliases=["check_off_todo", "mark_todo_done", "todo_complete", "complete_shopping_item"],
)
def complete_todo_item(item: str) -> str:
    """Check off a pending todo/shopping-list item by name.

    Args:
        item: The item text to mark done (fuzzy-matched against pending items).
    """
    if not client.TODO_ENTITY:
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
    return client._call_service(
        "todo",
        "update_item",
        client.TODO_ENTITY,
        {"item": summary, "status": "completed"},
        success_message=f"Checked off '{summary}'",
    )
