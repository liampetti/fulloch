"""Mocked SerpApi travel shopping and capability-policy tests."""

import importlib
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.capabilities import native_access_class, native_requires_deep_think
from utils.locale import HouseholdLocale


def _module(monkeypatch):
    sys.modules.pop("tools.travel", None)
    module = importlib.import_module("tools.travel")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(module, "local_today", lambda: date(2026, 4, 1))
    return module


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_flights_follow_round_trip_token_and_redact_key(monkeypatch):
    travel = _module(monkeypatch)
    calls = []

    def get(_url, params, timeout):
        calls.append((params, timeout))
        if "departure_token" not in params:
            return _Response({"best_flights": [{"departure_token": "outbound-token"}]})
        return _Response(
            {
                "best_flights": [
                    {
                        "price": 1200,
                        "total_duration": 780,
                        "flights": [
                            {
                                "airline": "Example Air",
                                "departure_airport": {"id": "ZZZ", "time": "2026-04-07T10:00:00+10:00"},
                                "arrival_airport": {"id": "ZZY", "time": "2026-04-08T08:00:00+04:00"},
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.search_flights("ZZZ", "ZZY", "2026-04-07", "2026-04-14", max_stops=0)

    assert len(calls) == 2
    assert calls[0][0]["stops"] == 1
    assert calls[1][0]["departure_token"] == "outbound-token"
    assert calls[1][0]["departure_id"] == "ZZZ"
    assert calls[1][0]["arrival_id"] == "ZZY"
    assert calls[1][0]["outbound_date"] == "2026-04-07"
    assert calls[1][0]["return_date"] == "2026-04-14"
    assert "Example Air" in result and "quoted price 1200" in result
    assert "test-key" not in result
    assert result.artifact == {
        "type": "flight_search",
        "route": {"origin": "ZZZ", "destination": "ZZY"},
        "departure_date": "2026-04-07",
        "return_date": "2026-04-14",
        "travellers": 1,
        "cabin": "Economy",
        "currency": "USD",
        "retrieved_at": result.artifact["retrieved_at"],
        "offer": {
            "price": 1200,
            "duration_minutes": 780,
            "stops": 0,
            "airlines": ["Example Air"],
            "departure": {"id": "ZZZ", "name": "ZZZ", "time": "2026-04-07T10:00:00+10:00", "timezone": None, "timezone_offset": "+10:00", "utc_time": "2026-04-07T00:00:00+00:00"},
            "arrival": {"id": "ZZY", "name": "ZZY", "time": "2026-04-08T08:00:00+04:00", "timezone": None, "timezone_offset": "+04:00", "utc_time": "2026-04-08T04:00:00+00:00"},
            "segments": [
                {
                    "airline": "Example Air",
                    "flight_number": "",
                    "departure": {"id": "ZZZ", "name": "ZZZ", "time": "2026-04-07T10:00:00+10:00", "timezone": None, "timezone_offset": "+10:00", "utc_time": "2026-04-07T00:00:00+00:00"},
                    "arrival": {"id": "ZZY", "name": "ZZY", "time": "2026-04-08T08:00:00+04:00", "timezone": None, "timezone_offset": "+04:00", "utc_time": "2026-04-08T04:00:00+00:00"},
                    "duration_minutes": None,
                }
            ],
        },
    }


def test_one_way_flights_use_provider_type_two_without_return_token(monkeypatch):
    travel = _module(monkeypatch)
    calls = []

    def get(_url, params, timeout):
        calls.append((params, timeout))
        return _Response(
            {
                "best_flights": [
                    {
                        "price": 123,
                        "total_duration": 90,
                        "flights": [
                            {
                                "airline": "Example Air",
                                "departure_airport": {"time": "2026-04-07 10:00"},
                                "arrival_airport": {"time": "2026-04-07 11:30"},
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.search_flights("AAA", "BBB", "2026-04-07")

    assert len(calls) == 1
    assert calls[0][0]["type"] == 2
    assert "return_date" not in calls[0][0]
    assert "quoted price 123" in result


def test_flights_default_one_missing_endpoint_and_currency_from_timezone(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.setattr(
        travel, "household_locale", lambda: HouseholdLocale("America/Chicago", "US", "USD", "miles", "fahrenheit", "us")
    )
    monkeypatch.setattr(travel, "_home_city", lambda: "Chicago")
    monkeypatch.setattr(travel, "_home_airport", lambda: "ORD")
    calls = []

    def get(_url, params, timeout):
        calls.append(params)
        if params["departure_id"] == "Chicago":
            return _Response({"best_flights": []})
        return _Response({"best_flights": [{"price": 123, "flights": []}]})

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.search_flights("", "Paris", "2026-04-07")

    assert [call["departure_id"] for call in calls] == ["Chicago", "ORD"]
    assert calls[1]["arrival_id"] == "CDG"
    assert calls[1]["currency"] == "USD"
    assert result.artifact["currency"] == "USD"


def test_home_airport_uses_matching_timezone_city_metadata(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.setattr(travel.local_time, "get_tz", lambda: SimpleNamespace(key="Test/Metro"))
    monkeypatch.setattr(
        travel,
        "_airports",
        lambda: {
            "AAA": {"iata": "AAA", "tz": "Test/Metro", "city": "Metro", "name": "Metro International"},
            "BBB": {"iata": "BBB", "tz": "Test/Metro", "city": "Elsewhere", "name": "Elsewhere International"},
        },
    )

    assert travel._home_airport() == "AAA"


def test_plan_travel_uses_home_airport_for_a_lone_destination(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.setattr(
        travel, "household_locale", lambda: HouseholdLocale("Australia/Sydney", "AU", "AUD", "kilometres", "celsius", "metric")
    )
    monkeypatch.setattr(travel, "_home_city", lambda: "Sydney")
    monkeypatch.setattr(travel, "_home_airport", lambda: "SYD")
    calls = []

    def get(_url, params, timeout):
        calls.append(params)
        return _Response({"best_flights": [{"price": 123, "flights": []}]})

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.plan_travel("Plan a trip to Paris")

    assert calls[0]["departure_id"] == "Sydney"
    assert calls[0]["arrival_id"] == "CDG"
    assert result.thinking_status == "evidence"


def test_flights_recover_labeled_positional_worker_arguments(monkeypatch):
    travel = _module(monkeypatch)
    calls = []

    def get(_url, params, timeout):
        calls.append(params)
        return _Response({"best_flights": [{"price": 123, "flights": []}]})

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.search_flights(
        "origin", "HND", "destination", "DXB", "departure_date", "2026-09-11"
    )

    assert calls[0]["departure_id"] == "HND"
    assert calls[0]["arrival_id"] == "DXB"
    assert calls[0]["outbound_date"] == "2026-09-11"
    assert "quoted price 123" in result


def test_flights_resolve_city_names_and_replace_a_past_date(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.setattr(travel, "local_today", lambda: date(2026, 8, 31))
    calls = []

    def get(_url, params, timeout):
        calls.append(params)
        return _Response(
            {
                "best_flights": [
                    {
                        "price": 123,
                        "total_duration": 775,
                        "flights": [
                            {
                                "airline": "Example Air",
                                "departure_airport": {"id": "CDG", "time": "2026-09-01 20:40"},
                                "arrival_airport": {"id": "FCO", "time": "2026-09-02 04:35"},
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.search_flights("Paris", "Rome", "2026-08-27")

    assert calls[0]["departure_id"] == "CDG"
    assert calls[0]["arrival_id"] == "FCO"
    assert calls[0]["outbound_date"] == "2026-09-01"
    assert "Resolved route: Paris (CDG) to Rome (FCO)." in result


def test_flights_reject_placeholder_arguments_without_provider_request(monkeypatch):
    travel = _module(monkeypatch)
    called = False

    def get(*_args, **_kwargs):
        nonlocal called
        called = True
        return _Response({})

    monkeypatch.setattr(travel.requests, "get", get)

    assert travel.search_flights("origin", "destination", "departure_date").thinking_status == "needs_input"
    assert called is False


def test_plan_travel_retrieves_every_adjacent_leg(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.setattr(travel, "local_today", lambda: date(2026, 8, 31))
    calls = []

    def get(_url, params, timeout):
        calls.append(params)
        outbound = date.fromisoformat(params["outbound_date"])
        if params["departure_id"] == "CDG":
            flights = [
                {
                    "airline": "Example Air",
                    "departure_airport": {"id": "CDG", "time": f"{outbound} 08:00"},
                    "arrival_airport": {"id": "FCO", "time": f"{outbound} 10:00"},
                }
            ]
        else:
            flights = [
                {
                    "airline": "Example Air",
                    "departure_airport": {"id": "FCO", "time": f"{outbound} 12:00"},
                    "arrival_airport": {"id": "MAD", "time": f"{outbound} 14:00"},
                }
            ]
        return _Response({"best_flights": [{"price": 123, "total_duration": 775, "flights": flights}]})

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.plan_travel(
        "I am travelling from Paris to Rome, then Madrid around Easter next year."
    )

    assert [(call["departure_id"], call["arrival_id"]) for call in calls] == [
        ("CDG", "FCO"),
        ("FCO", "MAD"),
    ]
    assert "Schedule evidence retrieved." in result
    assert "assess_itinerary" in result


def test_itinerary_evaluation_rejects_three_city_time_window_schedule(monkeypatch):
    travel = _module(monkeypatch)
    result = travel.evaluate_itinerary(
        '[{"departure":"2027-04-04T20:40:00+02:00","arrival":"2027-04-05T04:35:00+02:00"},{"departure":"2027-04-05T05:50:00+02:00","arrival":"2027-04-05T19:26:00+02:00"}]',
        '[{"location":"Paris conference","start":"2027-04-04T07:00:00+02:00","end":"2027-04-04T09:00:00+02:00"},{"location":"Rome meeting","start":"2027-04-04T12:00:00+02:00","end":"2027-04-04T14:00:00+02:00"},{"location":"Madrid event","start":"2027-04-04T18:00:00+02:00","end":"2027-04-04T20:00:00+02:00"}]',
    )

    assert result.startswith("Itinerary is not feasible:")
    assert "Rome meeting" in result
    assert "Madrid event" in result


def test_airport_artifact_resolves_naive_provider_time_to_utc(monkeypatch):
    travel = _module(monkeypatch)

    airport = travel._airport({"id": "HND", "time": "2026-08-31 08:00"}, "HND")

    assert airport["timezone"] == "Asia/Tokyo"
    assert airport["timezone_offset"] == "+09:00"
    assert airport["utc_time"] == "2026-08-30T23:00:00+00:00"


def test_hotels_are_bounded_and_include_only_normalized_fields(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.setattr(
        travel.requests,
        "get",
        lambda *_args, **_kwargs: _Response(
            {
                "properties": [
                    {
                        "name": "Station Hotel",
                        "total_rate": {"lowest": "$900"},
                        "overall_rating": 4.5,
                        "address": "1 Main Street",
                        "free_cancellation": True,
                    }
                ]
            }
        ),
    )

    result = travel.search_hotels("Tokyo", "2026-04-07", "2026-04-14", adults=2)

    assert "Station Hotel" in result
    assert "$900" in result
    assert "free cancellation" in result
    assert result.thinking_status == "evidence"
    assert result.artifact["type"] == "hotel_search"


def test_invalid_constraints_skip_provider_calls(monkeypatch):
    travel = _module(monkeypatch)
    called = False

    def get(*_args, **_kwargs):
        nonlocal called
        called = True
        return _Response({})

    monkeypatch.setattr(travel.requests, "get", get)
    assert "after departure_date" in travel.search_flights("AAA", "BBB", "2026-04-14", "2026-04-07")
    assert "YYYY-MM-DD" in travel.search_hotels("Tokyo", "tomorrow", "2026-04-14")
    assert called is False


def test_provider_failures_are_safe_and_keyless(monkeypatch):
    travel = _module(monkeypatch)

    def get(*_args, **_kwargs):
        raise travel.requests.Timeout("test-key")

    monkeypatch.setattr(travel.requests, "get", get)
    result = travel.search_hotels("Tokyo", "2026-04-07", "2026-04-14")
    assert result == "Travel search is temporarily unavailable."
    assert "test-key" not in result


def test_provider_authentication_failure_explains_how_to_reconfigure(monkeypatch):
    travel = _module(monkeypatch)

    class _UnauthorizedResponse(_Response):
        status_code = 401

    monkeypatch.setattr(travel.requests, "get", lambda *_args, **_kwargs: _UnauthorizedResponse({}))

    result = travel.search_flights("AAA", "BBB", "2026-04-07")

    assert result == "Travel search authentication failed; update serpapi_api_key in credentials.json."


def test_provider_rejection_preserves_the_safe_diagnostic(monkeypatch):
    travel = _module(monkeypatch)

    class _RejectedResponse(_Response):
        status_code = 400

    monkeypatch.setattr(
        travel.requests,
        "get",
        lambda *_args, **_kwargs: _RejectedResponse({"error": "departure_id is invalid"}),
    )

    assert travel.search_flights("BAD", "DXB", "2026-04-07") == (
        "Travel search provider rejected the request: departure_id is invalid"
    )


def test_travel_capabilities_are_read_only(monkeypatch):
    travel = _module(monkeypatch)

    assert travel._available() is True
    assert native_access_class("search_flights") == "read"
    assert native_access_class("search_hotels") == "read"
    assert native_access_class("assess_itinerary") == "read"
    assert native_access_class("plan_travel") == "read"
    assert native_requires_deep_think("plan_travel") is True


def test_assess_itinerary_reads_the_plan_artifact_without_schedule_json(monkeypatch):
    travel = _module(monkeypatch)
    from tools.thinking_context import reset_artifacts, set_artifacts

    token = set_artifacts(
        {
            "artifact-001": {
                "tool": "plan_travel",
                "data": {
                    "type": "travel_plan",
                    "legs": [
                        {
                            "departure": {"utc_time": "2026-04-07T00:00:00+00:00"},
                            "arrival": {"utc_time": "2026-04-07T04:00:00+00:00"},
                        },
                        {
                            "departure": {"utc_time": "2026-04-07T06:00:00+00:00"},
                            "arrival": {"utc_time": "2026-04-07T10:00:00+00:00"},
                        },
                    ],
                },
            }
        }
    )
    try:
        result = travel.assess_itinerary("artifact-001")
    finally:
        reset_artifacts(token)

    assert result.thinking_status == "evidence"
    assert "chronologically feasible" in result


def test_assess_itinerary_checks_later_retrieved_offer_combinations(monkeypatch):
    travel = _module(monkeypatch)
    from tools.thinking_context import reset_artifacts, set_artifacts

    token = set_artifacts(
        {
            "artifact-001": {
                "data": {
                    "type": "travel_plan",
                    "legs": [],
                    "leg_offers": [
                        [{"departure": {"utc_time": "2026-04-07T00:00:00+00:00"}, "arrival": {"utc_time": "2026-04-07T04:00:00+00:00"}}],
                        [
                            {"departure": {"utc_time": "2026-04-07T03:00:00+00:00"}, "arrival": {"utc_time": "2026-04-07T08:00:00+00:00"}},
                            {"departure": {"utc_time": "2026-04-07T06:00:00+00:00"}, "arrival": {"utc_time": "2026-04-07T10:00:00+00:00"}},
                        ],
                    ],
                }
            }
        }
    )
    try:
        result = travel.assess_itinerary("artifact-001")
    finally:
        reset_artifacts(token)

    assert result.thinking_status == "evidence"
    assert result.evidence["combination_checked"] is True


def test_assess_itinerary_reports_activity_timing_as_preliminary(monkeypatch):
    travel = _module(monkeypatch)
    from tools.thinking_context import reset_artifacts, set_artifacts

    token = set_artifacts(
        {
            "artifact-001": {
                "data": {
                    "type": "travel_plan",
                    "request": "Have breakfast in Tokyo and lunch in Dubai.",
                    "legs": [],
                }
            }
        }
    )
    try:
        result = travel.assess_itinerary("artifact-001")
    finally:
        reset_artifacts(token)

    assert result.thinking_status == "evidence"
    assert "Preliminary assessment" in result
    assert "chronologically feasible" in result
    assert result.evidence["activity_windows_checked"] is False
    assert native_requires_deep_think("search_flights") is True


def test_travel_uses_credentials_store_when_environment_is_unset(monkeypatch):
    travel = _module(monkeypatch)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setattr(travel, "get_credential", lambda key: "stored-key" if key == "serpapi_api_key" else "")

    assert travel._api_key() == "stored-key"
    assert travel._available() is True


def test_credentials_map_serpapi_key():
    from server.credentials_store import _ENV_MAP

    assert _ENV_MAP["serpapi_api_key"] == "SERPAPI_API_KEY"
