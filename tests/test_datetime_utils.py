"""Tests for the shared spoken-summary helper."""

from core.datetime_utils import tts_friendly_event_summary


def test_no_events_returns_polite_empty_message():
    assert tts_friendly_event_summary([]) == "You have no events scheduled."


def test_single_timed_event_uses_day_and_time():
    events = [
        {"start": "2026-05-25T10:00:00+10:00", "summary": "Dentist", "all_day": False},
    ]
    result = tts_friendly_event_summary(events)
    assert "Monday" in result
    assert "Dentist" in result
    # 10:00 → "10 00 a m" → "10 o'clock a m" (existing behaviour from google_calendar.py)
    assert "o'clock" in result


def test_all_day_event_uses_all_day_prefix():
    events = [
        {"start": "2026-05-26", "summary": "Public holiday", "all_day": True},
    ]
    result = tts_friendly_event_summary(events)
    assert "All day" in result
    assert "Public holiday" in result


def test_multiple_events_joined_with_spaces():
    events = [
        {"start": "2026-05-25T10:00:00+10:00", "summary": "Dentist", "all_day": False},
        {"start": "2026-05-25T14:30:00+10:00", "summary": "Lunch with Alex", "all_day": False},
    ]
    result = tts_friendly_event_summary(events)
    assert "Dentist" in result
    assert "Lunch with Alex" in result


def test_missing_summary_falls_back_to_generic_label():
    events = [{"start": "2026-05-25T10:00:00+10:00", "summary": None, "all_day": False}]
    result = tts_friendly_event_summary(events)
    assert "an event" in result
