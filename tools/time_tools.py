"""Time and timer/alarm tools. (Weather lives in the Home Assistant tool.)"""

import re
import threading
import time
from typing import Dict, Optional

from word2number import w2n

import utils.local_time as _local_time

from .tool_registry import tool

active_timers: Dict[str, threading.Timer] = {}

# Injected by Assistant._load_models so a completed timer can speak its
# reminder through the WebSocket satellite. The local-soundcard beep path is
# gone with sounddevice; Assistant.speak_proactive(alarm=True) plays a
# pre-rendered alert tone (data/wav/alarm.wav) over the same satellite sink
# before the spoken reminder.
_speak_proactive = None


def set_speak_callback(fn) -> None:
    global _speak_proactive
    _speak_proactive = fn


# Word forms used by `get_current_time` below. The output emits dates and
# times entirely in words so ASR round-trips it back to (near-)identical
# text — the self-echo check in core/assistant.py uses substring matching
# against the spoken text, and digits in the spoken side don't match
# their word-form transcription (e.g., "21" vs "twenty-first").
_ONES = (
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty")
_DAY_ORDINALS = (
    "",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "twenty-first",
    "twenty-second",
    "twenty-third",
    "twenty-fourth",
    "twenty-fifth",
    "twenty-sixth",
    "twenty-seventh",
    "twenty-eighth",
    "twenty-ninth",
    "thirtieth",
    "thirty-first",
)


def _num_words(n: int) -> str:
    """Cardinal English words for 0..59 (enough for minutes / years-of-century)."""
    if n < 20:
        return _ONES[n] or "zero"
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_ONES[ones]}"


def _year_words(y: int) -> str:
    """Pronounceable year, e.g. 2026 → 'twenty twenty-six', 2005 → 'twenty oh five'."""
    if y < 1000 or y > 2999:
        return str(y)
    century, rest = divmod(y, 100)
    if rest == 0:
        return f"{_num_words(century)} hundred"
    if rest < 10:
        return f"{_num_words(century)} oh {_num_words(rest)}"
    return f"{_num_words(century)} {_num_words(rest)}"


@tool(
    name="get_current_time",
    description="Get the current date and time",
    aliases=["time", "what_time_is_it", "get_time"],
)
def get_current_time(location: Optional[str] = None) -> str:
    """Current date and time, fully spelled out in words.

    Emitting words rather than digits ("twenty-first" not "21", "twenty
    twenty-six" not "2026") lets ASR re-transcribe the assistant's own
    voice to text that matches `_last_spoken_text` — so the self-echo
    suppressor in core/assistant.py catches the AEC residue cleanly and
    the time response can't loop back as a fake follow-up turn.
    """
    # TODO: get location time
    now = _local_time.now()
    day_name = now.strftime("%A")
    month_name = now.strftime("%B")
    day_word = _DAY_ORDINALS[now.day]
    year_word = _year_words(now.year)
    hour_12 = now.hour % 12 or 12
    hour_word = _ONES[hour_12]
    ampm = "a m" if now.hour < 12 else "p m"
    minute = now.minute
    if minute == 0:
        time_part = f"at {hour_word} {ampm}"
    elif minute < 10:
        time_part = f"at {hour_word} oh {_num_words(minute)} {ampm}"
    else:
        time_part = f"at {hour_word} {_num_words(minute)} {ampm}"
    return f"{day_name} {month_name} {day_word} {year_word} {time_part}"


@tool(
    name="start_countdown",
    description=(
        "Start a countdown for a relative DURATION from now ('in 10 minutes'); "
        "beeps and speaks the reminder message when it fires. ONLY for relative "
        "durations — for specific clock times ('at noon', 'at 3pm') use "
        "create_calendar_event instead."
    ),
    aliases=["timer", "countdown", "set_timer", "start_timer"],
)
def start_countdown(duration: str, message: Optional[str] = None) -> str:
    """Start a countdown timer for the specified duration.

    Args:
        duration: Duration string e.g. "ten minutes", "90 seconds", "2 hours".
                  Bare numbers (e.g. "60") are treated as seconds.
        message:  Optional reminder text spoken after the beeps fire.
    """

    def parse_duration(duration_str: str) -> int:
        duration_str = duration_str.lower()

        number_str = ""
        unit = ""
        for word in duration_str.split():
            try:
                val = w2n.word_to_num(word)
                number_str = str(val)
            except ValueError:
                unit += word + " "

        if not number_str:
            numbers = re.findall(r"\d+", duration_str)
            if not numbers:
                raise ValueError("No valid duration value found")
            number_str = numbers[0]

        value = int(number_str)

        if "hour" in unit:
            return value * 3600
        elif "minute" in unit:
            return value * 60
        elif "second" in unit or not unit.strip():
            # Bare number (no unit) → seconds.
            return value
        else:
            raise ValueError(f"Unknown duration unit: {unit.strip()!r}")

    def on_timer_complete(timer_id: str, reminder: Optional[str]):
        active_timers.pop(timer_id, None)
        text = reminder or "Your timer is up."
        if _speak_proactive:
            try:
                _speak_proactive(text, alarm=True)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Timer speak failed: {e}")

    try:
        seconds = parse_duration(duration)
        timer_id = f"timer_{len(active_timers) + 1}"

        timer = threading.Timer(seconds, on_timer_complete, args=[timer_id, message])
        timer.daemon = True
        timer.start_time = time.time()
        timer.start()

        active_timers[timer_id] = timer

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
    name="cancel_timer", description="Cancel an active timer", aliases=["stop_timer", "end_timer"]
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
    aliases=["timer_status", "check_timer", "show_timers", "get_timers", "list_timers"],
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
