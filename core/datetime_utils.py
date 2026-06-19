"""Date / time formatting helpers shared between calendar tools.

`tts_friendly_event_summary` renders events as spoken summaries
("ten o'clock" / "ten a m") from a normalised list of dicts:
    - "start"   (str)  ISO 8601 datetime OR ISO date
    - "summary" (str | None) event title
    - "all_day" (bool)  True if the event has no time component
"""

import datetime
import re


def tts_friendly_event_summary(events: list[dict]) -> str:
    """Render an event list as a TTS-friendly sentence.

    Mirrors the rendering of the original Google-shaped helper:
    timed events emit "At {time} on {day}, {summary}.", all-day events
    emit "All day on {day}, {summary}.".
    """
    if not events:
        return "You have no events scheduled."

    spoken = []
    for event in events:
        raw_start = event["start"]
        summary = event.get("summary") or "an event"
        if event.get("all_day"):
            day = datetime.datetime.fromisoformat(raw_start).strftime("%A")
            spoken.append(f"All day on {day}, {summary}.")
        else:
            start_dt = datetime.datetime.fromisoformat(raw_start)
            day = start_dt.strftime("%A")
            time = (
                start_dt.strftime("%-I %M %p")
                .lower()
                .replace("am", "a m")
                .replace("pm", "p m")
            )
            time = re.sub(r"\b00\b", "o'clock", time)
            spoken.append(f"At {time} on {day}, {summary}.")

    return " ".join(spoken)
