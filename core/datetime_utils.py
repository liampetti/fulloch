"""Date / time formatting helpers shared between calendar tools.

`tts_friendly_event_summary` renders events as spoken summaries
("ten o'clock" / "ten a m") from a normalised list of dicts:
    - "start"   (str)  ISO 8601 datetime OR ISO date
    - "summary" (str | None) event title
    - "all_day" (bool)  True if the event has no time component
"""

import datetime
import re

from core.satellite_context import current_response_language
from utils.response_language import response


def tts_friendly_event_summary(events: list[dict]) -> str:
    """Render an event list as a TTS-friendly sentence.

    Mirrors the rendering of the original Google-shaped helper:
    timed events emit "At {time} on {day}, {summary}.", all-day events
    emit "All day on {day}, {summary}.".
    """
    if not events:
        return response("calendar_no_events")

    spoken = []
    for event in events:
        raw_start = event["start"]
        summary = event.get("summary") or response("calendar_untitled")
        language = current_response_language.get()
        if event.get("all_day"):
            start_dt = datetime.datetime.fromisoformat(raw_start)
            if language in {"fr", "de"}:
                from babel.dates import format_date

                day = format_date(start_dt.date(), "EEEE", locale=language)
            else:
                day = start_dt.strftime("%A")
            spoken.append(response("calendar_event_all_day", day=day, summary=summary))
        else:
            start_dt = datetime.datetime.fromisoformat(raw_start)
            if language in {"fr", "de"}:
                from babel.dates import format_date, format_time

                day = format_date(start_dt.date(), "EEEE", locale=language)
                time = format_time(start_dt, format="short", locale=language)
            else:
                day = start_dt.strftime("%A")
                time = start_dt.strftime("%-I %M %p").lower().replace("am", "a m").replace("pm", "p m")
                time = re.sub(r"\b00\b", "o'clock", time)
            spoken.append(response("calendar_event_timed", time=time, day=day, summary=summary))

    return " ".join(spoken)
