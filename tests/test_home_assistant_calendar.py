"""HA calendar behavior against production implementations."""

import datetime
from unittest.mock import MagicMock, patch

from tests.ha_fixtures import loaded_ha  # noqa: F401
from tools import ha_calendar as calendar
from tools import ha_client as client


def test_calendar_window_today_is_midnight_to_midnight():
    from tools.ha_calendar import _calendar_window

    start, end = _calendar_window("today", now=datetime.datetime(2026, 5, 21, 14, 30))
    assert start == "2026-05-21T00:00:00"
    assert end == "2026-05-22T00:00:00"


def test_calendar_window_tomorrow_is_one_day_after():
    from tools.ha_calendar import _calendar_window

    start, end = _calendar_window("tomorrow", now=datetime.datetime(2026, 5, 21, 14, 30))
    assert start == "2026-05-22T00:00:00"
    assert end == "2026-05-23T00:00:00"


def test_calendar_window_week_is_seven_days_from_today():
    from tools.ha_calendar import _calendar_window

    start, end = _calendar_window("this_week", now=datetime.datetime(2026, 5, 21, 14, 30))
    assert start == "2026-05-21T00:00:00"
    assert end == "2026-05-28T00:00:00"


def test_calendar_window_specific_iso_date_is_single_day():
    from tools.ha_calendar import _calendar_window

    start, end = _calendar_window("2026-06-26", now=datetime.datetime(2026, 6, 18, 14, 30))
    assert start == "2026-06-26T00:00:00"
    assert end == "2026-06-27T00:00:00"


def test_calendar_window_unrecognised_string_defaults_to_today():
    from tools.ha_calendar import _calendar_window

    start, end = _calendar_window("sometime", now=datetime.datetime(2026, 6, 18, 14, 30))
    assert start == "2026-06-18T00:00:00"
    assert end == "2026-06-19T00:00:00"


def test_whats_on_lists_only_tomorrows_window():

    now = datetime.datetime(2026, 8, 26, 14, 30, tzinfo=datetime.timezone.utc)
    response = {"calendar.primary": {"events": []}}
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response) as call,
        patch.object(client._local_tz, "now", return_value=now),
    ):
        calendar.whats_on("tomorrow")

    request = call.call_args.args[2]
    assert request["start_date_time"] == "2026-08-27T00:00:00+00:00"
    assert request["end_date_time"] == "2026-08-28T00:00:00+00:00"


def test_whats_on_reads_both_primary_and_reminder_calendars():
    """Events Fulloch writes to its reminder calendar must surface in whats_on
    even when the autodetected read calendar differs (#2 regression)."""

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-06-26T09:00:00", "summary": "Standup"},
            ]
        },
        "calendar.fulloch": {
            "events": [
                {"start": "2026-06-26T12:00:00", "summary": "Australia vs Paraguay"},
            ]
        },
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value="calendar.fulloch"),
        patch.object(client, "_call_service_with_response", return_value=response) as call,
    ):
        out = calendar._ha_get_events("2026-06-26")

    # Both calendars were queried in one call.
    assert call.call_args.args[2]["entity_id"] == ["calendar.primary", "calendar.fulloch"]
    # The reminder-calendar event is present in the spoken summary.
    assert "Australia vs Paraguay" in out
    assert "Standup" in out
    assert out.artifact == {
        "type": "calendar",
        "title": "2026-06-26",
        "events": [
            {"title": "Standup", "when": "Fri 26, 9:00 AM", "all_day": False},
            {"title": "Australia vs Paraguay", "when": "Fri 26, 12:00 PM", "all_day": False},
        ],
    }


def test_whats_on_dedupes_when_read_and_reminder_calendars_match():

    response = {"calendar.primary": {"events": []}}
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value="calendar.primary"),
        patch.object(client, "_call_service_with_response", return_value=response) as call,
    ):
        calendar._ha_get_events("today")
    assert call.call_args.args[2]["entity_id"] == ["calendar.primary"]


def test_parse_lookback_days_units():
    from tools.ha_calendar import _parse_lookback_days

    assert _parse_lookback_days("30d") == 30
    assert _parse_lookback_days("2w") == 14
    assert _parse_lookback_days("6m") == 180
    assert _parse_lookback_days("1y") == 365
    assert _parse_lookback_days("10") == 10  # bare number -> days
    assert _parse_lookback_days("garbage") == 365  # falls back to default


def test_relative_day_phrase():
    from tools.ha_calendar import _relative_day_phrase

    now = datetime.datetime(2026, 6, 18, 9, 0)  # Thursday
    assert _relative_day_phrase(now, now) == "today"
    assert _relative_day_phrase(now + datetime.timedelta(days=1), now) == "tomorrow"
    assert _relative_day_phrase(now - datetime.timedelta(days=1), now) == "yesterday"
    assert _relative_day_phrase(now + datetime.timedelta(days=3), now) == "this Sunday"
    assert _relative_day_phrase(now + datetime.timedelta(days=10), now) == "next Sunday"
    assert _relative_day_phrase(now - datetime.timedelta(days=3), now) == "last Monday"
    # Far outside the near-date window: full calendar date, not a bare weekday
    # (a bare weekday is ambiguous across a multi-week search window).
    far = _relative_day_phrase(now - datetime.timedelta(days=25), now)
    assert far == "Sunday, May 24"


def test_when_is_it_on_matches_by_substring():

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-05-20T09:00:00", "summary": "Dentist appointment"},
                {"start": "2026-06-26T09:00:00", "summary": "Standup"},
            ]
        },
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
    ):
        out = calendar._ha_get_events_name("dentist", limit="30d")

    assert out.startswith("Reactive question:")
    assert "Dentist appointment" in out
    assert "2026-05-20" in out
    assert "Standup" not in out


def test_when_is_it_on_fuzzy_matches_event_word_variants():
    """Natural query wording need not exactly match the calendar title."""

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-10-03T18:25:00", "summary": "Flight to Perth, VA 567"},
            ]
        }
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
    ):
        out = calendar._ha_get_events_name("fly to perth")

    assert out.startswith("Reactive question:")
    assert "Flight to Perth, VA 567" in out
    assert "2026-10-03" in out


def test_when_is_it_on_matches_meaningful_tokens_regardless_of_order():

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-10-03T18:25:00", "summary": "Flight to Perth, VA 567"},
            ]
        }
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
        patch.object(calendar.difflib, "get_close_matches") as fuzzy,
    ):
        out = calendar._ha_get_events_name("Perth flight")

    assert "Flight to Perth, VA 567" in out
    fuzzy.assert_not_called()


def test_when_is_it_on_token_matching_requires_all_query_tokens():

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-10-03T18:25:00", "summary": "Flight to Darwin"},
            ]
        }
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
        patch.object(calendar.difflib, "get_close_matches", return_value=[]),
    ):
        out = calendar._ha_get_events_name("Perth flight")

    assert "couldn't find" in out


def test_find_calendar_event_limits_named_search_to_today():

    now = datetime.datetime(2026, 8, 26, 14, 30, tzinfo=datetime.timezone.utc)
    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-08-26T18:25:00+00:00", "summary": "Doctor appointment"},
            ]
        }
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response) as call,
        patch.object(client._local_tz, "now", return_value=now),
    ):
        out = calendar.find_calendar_event("appointment", "today")

    request = call.call_args.args[2]
    assert request["start_date_time"] == "2026-08-26T00:00:00+00:00"
    assert request["end_date_time"] == "2026-08-27T00:00:00+00:00"
    assert "Doctor appointment" in out


def test_find_calendar_event_limits_named_search_to_this_week():

    now = datetime.datetime(2026, 8, 26, 14, 30, tzinfo=datetime.timezone.utc)
    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-08-28T16:00:00+00:00", "summary": "Dance class"},
            ]
        }
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response) as call,
        patch.object(client._local_tz, "now", return_value=now),
    ):
        out = calendar.find_calendar_event("dance class", "this_week")

    request = call.call_args.args[2]
    assert request["start_date_time"] == "2026-08-26T00:00:00+00:00"
    assert request["end_date_time"] == "2026-09-02T00:00:00+00:00"
    assert "Dance class" in out


def test_when_is_it_on_no_match_is_spoken_directly():

    response = {"calendar.primary": {"events": []}}
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
    ):
        out = calendar._ha_get_events_name("dentist", limit="30d")

    assert "couldn't find" in out
    assert not out.startswith("Reactive question:")


def test_when_is_it_on_no_calendar_configured():

    with (
        patch.object(client, "CALENDAR_ENTITY", None),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
    ):
        out = calendar._ha_get_events_name("dentist", limit="30d")

    assert out == "No calendar is configured in Home Assistant."


def test_calendar_normalises_timed_event():

    ha_event = {"start": "2026-05-25T10:00:00+10:00", "end": "...", "summary": "Dentist"}
    with patch.object(client._local_tz, "get_tz", return_value=datetime.timezone.utc):
        norm = calendar._normalise_ha_event(ha_event)
    assert norm == {"start": "2026-05-25T00:00:00+00:00", "summary": "Dentist", "all_day": False}


def test_calendar_normalises_all_day_event():
    from tools.ha_calendar import _normalise_ha_event

    ha_event = {"start": "2026-05-26", "end": "2026-05-27", "summary": "Public holiday"}
    norm = _normalise_ha_event(ha_event)
    assert norm == {"start": "2026-05-26", "summary": "Public holiday", "all_day": True}


def test_upcoming_events_excludes_all_day_events():

    response = {
        "calendar.fulloch": {
            "events": [{"start": "2026-05-26", "end": "2026-05-27", "summary": "Public holiday"}]
        }
    }
    with (
        patch.object(calendar, "_reminder_calendar_entity", return_value="calendar.fulloch"),
        patch.object(client, "_call_service_with_response", return_value=response),
    ):
        assert calendar.get_upcoming_events() == []


def test_named_calendar_event_normalises_timezone_before_relative_date():

    response = {
        "calendar.primary": {
            "events": [{"start": "2026-05-25T10:00:00+10:00", "summary": "Dentist"}]
        }
    }
    now = datetime.datetime(2026, 5, 24, 12, tzinfo=datetime.timezone.utc)
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
        patch.object(client._local_tz, "now", return_value=now),
        patch.object(client._local_tz, "get_tz", return_value=datetime.timezone.utc),
    ):
        out = calendar._ha_get_events_name("dentist")

    assert (
        out
        == "Reactive question: Calendar search found: Dentist: starts 2026-05-25T00:00:00+00:00."
    )


def test_named_calendar_search_defaults_to_one_year_each_side():

    response = {"calendar.primary": {"events": []}}
    now = datetime.datetime(2026, 8, 25, 12, tzinfo=datetime.timezone.utc)
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response) as call,
        patch.object(client._local_tz, "now", return_value=now),
    ):
        calendar.find_calendar_event("Perth flight")

    request = call.call_args.args[2]
    assert request["start_date_time"] == "2025-08-25T12:00:00+00:00"
    assert request["end_date_time"] == "2027-08-25T12:00:00+00:00"


def test_calendar_events_sort_by_normalised_timestamp():

    response = {
        "calendar.primary": {
            "events": [
                {"start": "2026-06-26T01:00:00+00:00", "summary": "Second"},
                {"start": "2026-06-26T10:00:00+10:00", "summary": "First"},
            ]
        }
    }
    with (
        patch.object(client, "CALENDAR_ENTITY", "calendar.primary"),
        patch.object(calendar, "_reminder_calendar_entity", return_value=None),
        patch.object(client, "_call_service_with_response", return_value=response),
        patch.object(client._local_tz, "get_tz", return_value=datetime.timezone.utc),
    ):
        out = calendar._ha_get_events("2026-06-26")

    assert out.index("First") < out.index("Second")


def test_complete_todo_item_matches_by_substring():

    items = [
        {"summary": "Buy milk", "uid": "1"},
        {"summary": "Call the plumber", "uid": "2"},
    ]
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "TODO_ENTITY", "todo.shopping_list"),
        patch.object(client, "_DENIED_ENTITIES", frozenset()),
        patch(
            "tools.ha_client._call_service_with_response",
            return_value={"todo.shopping_list": {"items": items}},
        ),
        patch("tools.ha_client.requests.post", return_value=resp) as post,
    ):
        result = calendar.complete_todo_item("milk")

    payload = post.call_args.kwargs["json"]
    assert payload["item"] == "Buy milk"
    assert payload["status"] == "completed"
    assert "Buy milk" in result


def test_get_todo_items_attaches_checklist_artifact(monkeypatch):

    monkeypatch.setattr(client, "TODO_ENTITY", "todo.shopping")
    monkeypatch.setattr(client, "_loaded", True)
    monkeypatch.setattr(
        calendar,
        "_fetch_pending_todo_items",
        lambda: [{"summary": "Milk"}, {"summary": "Call the plumber"}],
    )

    result = calendar.get_todo_items()

    assert result == "You have 2 items: Milk, and Call the plumber."
    assert result.artifact == {"type": "todos", "items": ["Milk", "Call the plumber"]}


def test_complete_todo_item_no_match_is_spoken_directly():

    items = [{"summary": "Buy milk", "uid": "1"}]
    with (
        patch.object(client, "HA_TOKEN", "tok"),
        patch.object(client, "TODO_ENTITY", "todo.shopping_list"),
        patch(
            "tools.ha_client._call_service_with_response",
            return_value={"todo.shopping_list": {"items": items}},
        ),
    ):
        result = calendar.complete_todo_item("dentist appointment")

    assert "couldn't find" in result.lower()
