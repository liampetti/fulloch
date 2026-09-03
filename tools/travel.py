"""Bounded, read-only trip shopping through SerpApi's Google travel engines."""

import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from itertools import islice, product
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import airportsdata
import requests
from dateutil.easter import easter

import utils.local_time as local_time
from server.credentials_store import get_credential
from utils.local_time import today as local_today
from utils.locale import household_locale

from .thinking_context import get_artifact
from .thinking_playbooks import thinking_playbook
from .tool_registry import ThinkingResult, tool

API_URL = "https://serpapi.com/search.json"
TIMEOUT_S = 10
MAX_OFFERS = 5

logger = logging.getLogger(__name__)

def _api_key() -> str:
    """Prefer explicit runtime configuration, falling back to credentials.json."""
    return os.environ.get("SERPAPI_API_KEY") or get_credential("serpapi_api_key")


def _available() -> bool:
    return bool(_api_key())


def _date(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must use YYYY-MM-DD.")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError(f"{name} must use YYYY-MM-DD.") from None


_PLACEHOLDER_VALUES = frozenset(
    {"origin", "destination", "departure_date", "return_date", "date", "city", "airport"}
)


@lru_cache(maxsize=1)
def _airports() -> dict:
    return airportsdata.load("IATA")


def _resolve_airport(value: object, field: str) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    query = " ".join(value.split()).strip()
    if query.lower() in _PLACEHOLDER_VALUES:
        return None
    airports = _airports()
    direct = airports.get(query.upper())
    if isinstance(direct, dict):
        return query.upper(), str(direct.get("city") or direct.get("name") or query)
    if re.fullmatch(r"[A-Za-z]{3}", query):
        return query.upper(), query.upper()
    query_lower = query.lower()
    candidates = [
        airport
        for airport in airports.values()
        if isinstance(airport, dict)
        and (
            str(airport.get("city") or "").lower() == query_lower
            or str(airport.get("subd") or "").lower() == query_lower
        )
        and airport.get("iata")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda airport: _airport_rank(airport, query_lower))
    airport = candidates[0]
    return str(airport["iata"]), str(airport.get("city") or airport.get("name") or query)


def _airport_rank(airport: dict, query_lower: str) -> tuple[bool, bool, str]:
    return (
        "international" not in str(airport.get("name") or "").lower(),
        not str(airport.get("name") or "").lower().startswith(query_lower),
        str(airport.get("iata")),
    )


def _home_airport() -> str | None:
    """Resolve an airport from the timezone's own city and airport metadata."""
    timezone_name = str(getattr(local_time.get_tz(), "key", "") or "")
    if "/" not in timezone_name:
        return None
    city = timezone_name.rsplit("/", 1)[1].replace("_", " ").lower()
    candidates = [
        airport
        for airport in _airports().values()
        if isinstance(airport, dict)
        and airport.get("iata")
        and str(airport.get("tz") or "") == timezone_name
        and str(airport.get("city") or "").lower() == city
    ]
    if not candidates:
        return None
    return str(min(candidates, key=lambda airport: _airport_rank(airport, city))["iata"])


def _home_city() -> str | None:
    """Return the city segment of a configured IANA timezone when available."""
    timezone_name = str(getattr(local_time.get_tz(), "key", "") or "")
    if "/" not in timezone_name:
        return None
    return timezone_name.rsplit("/", 1)[1].replace("_", " ")


def _future_departure_date(value: object) -> str | None:
    if not isinstance(value, str) or value.strip().lower() in _PLACEHOLDER_VALUES:
        return None
    try:
        requested = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return max(requested, local_today() + timedelta(days=1)).isoformat()


@lru_cache(maxsize=1)
def _named_places() -> dict[str, tuple[str, str]]:
    candidates: dict[str, list[dict]] = {}
    for airport in _airports().values():
        if not isinstance(airport, dict) or not airport.get("iata"):
            continue
        for field in ("city", "subd"):
            name = str(airport.get(field) or "").strip()
            if len(name) < 3:
                continue
            candidates.setdefault(name.lower(), []).append(airport)
    places = {}
    for name, airports in candidates.items():
        airport = min(airports, key=lambda item: _airport_rank(item, name))
        places[name] = (str(airport["iata"]), str(airport.get("city") or airport.get("name") or name))
    return places


def _locations_in_request(request: str) -> list[tuple[str, str]]:
    matches = []
    for name, resolved in _named_places().items():
        found = re.search(rf"\b{re.escape(name)}\b", request, re.IGNORECASE)
        if found is not None:
            matches.append((found.start(), resolved))
    matches.sort(key=lambda item: item[0])
    locations = []
    seen = set()
    for _position, location in matches:
        if location[0] not in seen:
            locations.append(location)
            seen.add(location[0])
    return locations


def _representative_departure_date(request: str) -> str:
    explicit_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", request)
    if explicit_dates:
        resolved = _future_departure_date(explicit_dates[0])
        if resolved is not None:
            return resolved
    today = local_today()
    if re.search(r"\bnext year\b", request, re.IGNORECASE):
        year = today.year + 1
        if re.search(r"\beaster\b", request, re.IGNORECASE):
            return easter(year).isoformat()
        return date(year, today.month, today.day).isoformat()
    if re.search(r"\btomorrow\b", request, re.IGNORECASE):
        return (today + timedelta(days=1)).isoformat()
    return (today + timedelta(days=14)).isoformat()


def _is_travel_request(request: str) -> bool:
    if re.search(
        r"\b(flight|plane|travel|trip|itinerary|airport|layover|stopover|hotel|holiday|vacation)\b",
        request,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(go|leave|arrive|get)\b.*\b(to|from|somewhere|away|there)\b", request, re.IGNORECASE):
        return True
    return len(_locations_in_request(request)) >= 2 and bool(
        re.search(r"\b(then|after|before|possible|doable|same day)\b", request, re.IGNORECASE)
    )


thinking_playbook(
    name="travel planning",
    triggers=(),
    capabilities=("plan_travel", "search_flights", "assess_itinerary", "search_hotels"),
    solve_path=(
        "For a natural-language trip or route-planning request, start with plan_travel.",
        "For a one-way specific flight search, call search_flights with exactly origin, destination, and a future ISO departure date. Do not pass return_date unless the user asked for a return journey.",
        "After plan_travel returns an Artifact reference, call assess_itinerary with that reference. It evaluates the retrieved schedules directly; never serialize schedule JSON into an action.",
        "A failed itinerary assessment rejects only that candidate. Check other returned schedule combinations, then use a materially different date, route, or flight search when it could change feasibility before concluding no option works.",
        "Use hotel search only after transport feasibility is established or when accommodation is independently requested.",
    ),
    completion_rule="A feasible candidate is deterministically evaluated, or every retrieved candidate combination and materially different available search path is accounted for with its scope stated.",
    prohibited_shortcuts=("Do not infer transport feasibility from timezone or calendar arithmetic alone.",),
    matcher=_is_travel_request,
    fallback_capability="plan_travel",
)


def _positive(value: Any, name: str, maximum: int = 9) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number.") from None
    if not 0 <= number <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}.")
    return number


def _normalise_labeled_flight_args(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None,
    adults: int,
    children: int,
    cabin_class: str,
    max_stops: int,
) -> tuple[str, str, str, str | None, int, int, str, int]:
    """Recover the model's occasional ["origin", "HND", ...] positional form."""
    raw = (origin, destination, departure_date, return_date, adults, children, cabin_class, max_stops)
    values = {
        str(key).strip().lower(): value
        for key, value in zip(raw[::2], raw[1::2], strict=False)
        if isinstance(key, str)
    }
    required = {"origin", "destination", "departure_date"}
    if not required <= values.keys():
        return origin, destination, departure_date, return_date, adults, children, cabin_class, max_stops
    return (
        values["origin"],
        values["destination"],
        values["departure_date"],
        values.get("return_date"),
        values.get("adults", 1),
        values.get("children", 0),
        values.get("cabin_class", "economy"),
        values.get("max_stops", 2),
    )


def _search(params: dict[str, Any]) -> dict:
    key = _api_key()
    if not key:
        return {
            "error": "Travel planning is not configured; add serpapi_api_key to credentials.json."
        }
    try:
        response = requests.get(API_URL, params={**params, "api_key": key}, timeout=TIMEOUT_S)
        status_code = getattr(response, "status_code", 200)
        if status_code in {401, 403}:
            return {"error": "Travel search authentication failed; update serpapi_api_key in credentials.json."}
        if status_code >= 400:
            try:
                provider_error = str(response.json().get("error") or "").strip()
            except (AttributeError, ValueError):
                provider_error = ""
            safe_params = {name: value for name, value in params.items() if name != "api_key"}
            logger.warning(
                "Travel search provider rejected request (HTTP %d, params=%r, error=%s)",
                status_code,
                safe_params,
                provider_error or "unavailable",
            )
            detail = f": {provider_error}" if provider_error else "."
            return {"error": "Travel search provider rejected the request" + detail}
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return {"error": "Travel search is temporarily unavailable."}
    if not isinstance(payload, dict):
        return {"error": "Travel search returned an invalid response."}
    if payload.get("error"):
        if "api key" in str(payload["error"]).lower():
            return {"error": "Travel search authentication failed; update serpapi_api_key in credentials.json."}
        return {"error": "Travel search could not be completed."}
    return payload


def _flight_summary(item: dict) -> str:
    legs = item.get("flights") or []
    first = legs[0] if legs else {}
    last = legs[-1] if legs else {}
    departure = _airport(first.get("departure_airport"), "origin")
    arrival = _airport(last.get("arrival_airport"), "destination")
    airline = first.get("airline") or "carrier unavailable"
    duration = item.get("total_duration")
    duration_text = f"{duration} minutes" if isinstance(duration, int) else "duration unavailable"
    stops = max(len(legs) - 1, 0)
    price = item.get("price")
    price_text = str(price) if price is not None else "price unavailable"
    return (
        f"{airline}; depart {_schedule_text(departure)}; arrive {_schedule_text(arrival)}; {stops} stops; "
        f"{duration_text}; quoted price {price_text}."
    )


def _offset_text(value: datetime) -> str | None:
    offset = value.utcoffset()
    if offset is None:
        return None
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    hours, remainder = divmod(abs(total_seconds), 3600)
    return f"{sign}{hours:02d}:{remainder // 60:02d}"


@lru_cache(maxsize=1024)
def _airport_timezone(airport_id: str) -> str | None:
    airport = _airports().get(airport_id.upper())
    timezone_name = airport.get("tz") if isinstance(airport, dict) else None
    return str(timezone_name) if timezone_name else None


def _airport(value: object, fallback: str) -> dict:
    value = value if isinstance(value, dict) else {}
    airport_id = str(value.get("id") or fallback)
    local_time = str(value.get("time") or "")
    utc_time = None
    timezone_offset = None
    timezone_name = _airport_timezone(airport_id)
    try:
        parsed = datetime.fromisoformat(local_time.replace("Z", "+00:00"))
        if parsed.tzinfo is None and timezone_name is not None:
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
            except ZoneInfoNotFoundError:
                timezone_name = None
        timezone_offset = _offset_text(parsed)
        if timezone_offset is not None:
            utc_time = parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    return {
        "id": airport_id,
        "name": str(value.get("name") or value.get("id") or fallback),
        "time": local_time,
        "timezone": timezone_name,
        "timezone_offset": timezone_offset,
        "utc_time": utc_time,
    }


def _schedule_text(airport: dict) -> str:
    local_time = airport.get("time") or "time unavailable"
    timezone_offset = airport.get("timezone_offset")
    return f"{airport.get('id', 'airport')} local {local_time}{f' ({timezone_offset})' if timezone_offset else ''}"


def _schedule_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp with a UTC offset.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} must be an ISO timestamp with a UTC offset.") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset.")
    return parsed


def _schedule_value(record: dict, field: str) -> object:
    value = record.get(field)
    if isinstance(value, dict):
        return value.get("utc_time") or value.get("local_time") or value.get("time")
    return value


def _evaluate_itinerary(legs: list, windows: list) -> str:
    if len(windows) != len(legs) + 1:
        return "Provide one stop window for the journey start and each leg destination."
    parsed_windows = []
    for index, window in enumerate(windows, 1):
        if not isinstance(window, dict):
            return f"Stop window {index} must be an object."
        try:
            start = _schedule_datetime(window.get("start"), f"stop window {index} start")
            end = _schedule_datetime(window.get("end"), f"stop window {index} end")
        except ValueError as exc:
            return str(exc)
        if end < start:
            return f"Stop window {index} ends before it starts."
        parsed_windows.append((start, end, str(window.get("location") or f"stop {index}")))

    parsed_legs = []
    for index, leg in enumerate(legs, 1):
        if not isinstance(leg, dict):
            return f"Leg {index} must be an object."
        try:
            departure = _schedule_datetime(_schedule_value(leg, "departure"), f"leg {index} departure")
            arrival = _schedule_datetime(_schedule_value(leg, "arrival"), f"leg {index} arrival")
        except ValueError as exc:
            return str(exc)
        if arrival < departure:
            return f"Leg {index} arrives before it departs."
        parsed_legs.append((departure, arrival))

    failures = []
    for index, (departure, arrival) in enumerate(parsed_legs):
        source_start, _, source_name = parsed_windows[index]
        _, destination_end, destination_name = parsed_windows[index + 1]
        if departure < source_start:
            failures.append(
                f"Leg {index + 1} leaves {source_name} at {departure.isoformat()} before its requested window begins at {source_start.isoformat()}."
            )
        if arrival > destination_end:
            failures.append(
                f"Leg {index + 1} reaches {destination_name} at {arrival.isoformat()}, after its requested window ends at {destination_end.isoformat()}."
            )
        if index and departure < parsed_legs[index - 1][1]:
            failures.append(f"Leg {index + 1} departs before leg {index} arrives.")
    if failures:
        return "Itinerary is not feasible:\n- " + "\n- ".join(failures)
    return "Itinerary is feasible within the supplied local-time windows."


def _flight_artifact(item: dict, params: dict, departure_date: str, return_date: str | None) -> dict:
    """Return the small, token-free record used by the completed plan card."""
    legs = [leg for leg in (item.get("flights") or []) if isinstance(leg, dict)]
    first, last = (legs[0], legs[-1]) if legs else ({}, {})
    segments = []
    airlines = []
    for leg in legs:
        airline = str(leg.get("airline") or "")
        if airline and airline not in airlines:
            airlines.append(airline)
        segments.append(
            {
                "airline": airline or "Carrier unavailable",
                "flight_number": str(leg.get("flight_number") or ""),
                "departure": _airport(leg.get("departure_airport"), params["departure_id"]),
                "arrival": _airport(leg.get("arrival_airport"), params["arrival_id"]),
                "duration_minutes": leg.get("duration") if isinstance(leg.get("duration"), int) else None,
            }
        )
    return {
        "type": "flight_search",
        "route": {"origin": params["departure_id"], "destination": params["arrival_id"]},
        "departure_date": departure_date,
        "return_date": return_date,
        "travellers": params["adults"] + params["children"],
        "cabin": {1: "Economy", 2: "Premium economy", 3: "Business", 4: "First"}[params["travel_class"]],
        "currency": params["currency"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "offer": {
            "price": item.get("price") if isinstance(item.get("price"), (int, float, str)) else None,
            "duration_minutes": item.get("total_duration") if isinstance(item.get("total_duration"), int) else None,
            "stops": max(len(legs) - 1, 0),
            "airlines": airlines,
            "departure": _airport(first.get("departure_airport"), params["departure_id"]),
            "arrival": _airport(last.get("arrival_airport"), params["arrival_id"]),
            "segments": segments,
        },
    }


@tool(
    name="search_flights",
    description=(
        "Planning-worker-only flight search. Requires a destination and departure_date; when exactly one "
        "endpoint is omitted, the household's timezone-derived home airport is used. "
        "return_date is optional for a one-way search. Use IATA airport/city codes and ISO dates. "
        "Results are read-only, not bookings."
    ),
    available=_available,
    deep_think_only=True,
    thinking_outcome=True,
)
def search_flights(
    origin: str = "",
    destination: str = "",
    departure_date: str = "",
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    cabin_class: str = "economy",
    max_stops: int = 2,
) -> str:
    """Return bounded, normalized one-way or round-trip offers without booking links."""
    (
        origin,
        destination,
        departure_date,
        return_date,
        adults,
        children,
        cabin_class,
        max_stops,
    ) = _normalise_labeled_flight_args(
        origin, destination, departure_date, return_date, adults, children, cabin_class, max_stops
    )
    locale = household_locale()
    home_city = _home_city()
    home_airport = _home_airport()
    inferred_home_endpoint = False
    if home_city or home_airport:
        if not str(origin or "").strip() and str(destination or "").strip():
            origin = home_city or home_airport
            inferred_home_endpoint = True
        elif str(origin or "").strip() and not str(destination or "").strip():
            destination = home_city or home_airport
            inferred_home_endpoint = True
    if (
        not isinstance(origin, str)
        or not origin.strip()
        or not isinstance(destination, str)
        or not destination.strip()
    ):
        return ThinkingResult("Origin and destination airport or city names are required.", status="needs_input", scope="Both flight endpoints are required.")
    resolved_origin = _resolve_airport(origin, "origin")
    resolved_destination = _resolve_airport(destination, "destination")
    if resolved_origin is None or resolved_destination is None:
        return ThinkingResult("Use a real origin and destination city or IATA airport code.", status="needs_input", scope="At least one flight endpoint could not be resolved.")
    depart = _future_departure_date(departure_date)
    if depart is None:
        return ThinkingResult("departure_date must use YYYY-MM-DD; use a future date for flight schedules.", status="needs_input", scope="A future ISO departure date is required.")
    try:
        returning = _date(return_date, "return_date") if return_date is not None else None
        adults, children = _positive(adults, "adults"), _positive(children, "children")
    except ValueError as exc:
        return ThinkingResult(str(exc), status="needs_input", scope="The requested flight dates or passenger counts are invalid.")
    if returning is not None and returning < local_today().isoformat():
        returning = (local_today() + timedelta(days=2)).isoformat()
    if returning is not None and returning <= depart:
        return ThinkingResult("return_date must be after departure_date.", status="needs_input", scope="The requested return date is invalid.")
    if adults < 1:
        return ThinkingResult("At least one adult is required.", status="needs_input", scope="At least one adult is required.")
    try:
        travel_class = {"economy": 1, "premium_economy": 2, "business": 3, "first": 4}[cabin_class]
        stops = min(max(int(max_stops), 0), 2) + 1
    except (KeyError, TypeError, ValueError):
        return ThinkingResult("cabin_class must be economy, premium_economy, business, or first; max_stops must be 0, 1, or 2.", status="needs_input", scope="The requested cabin class or stop limit is invalid.")
    params = {
        "engine": "google_flights",
        "departure_id": origin if inferred_home_endpoint and origin == home_city else resolved_origin[0],
        "arrival_id": destination if inferred_home_endpoint and destination == home_city else resolved_destination[0],
        "outbound_date": depart,
        "type": 1 if returning is not None else 2,
        "adults": adults,
        "children": children,
        "travel_class": travel_class,
        "stops": stops,
        "currency": locale.currency,
        "hl": "en",
    }
    if returning is not None:
        params["return_date"] = returning
    first = _search(params)
    departures = first.get("best_flights") or first.get("other_flights") or []
    if inferred_home_endpoint and home_city and home_airport and (first.get("error") or not departures):
        params = {
            **params,
            **({"departure_id": home_airport} if origin == home_city else {"arrival_id": home_airport}),
        }
        first = _search(params)
    if first.get("error"):
        return ThinkingResult(str(first["error"]), status="unavailable", scope="The flight provider did not return schedule evidence.")
    departures = first.get("best_flights") or first.get("other_flights") or []
    if not departures:
        return ThinkingResult("No flight offers matched those constraints.", status="rejected", scope="No retrieved offer matched the requested flight constraints.", next_actions=("search_flights",))
    token = departures[0].get("departure_token") if isinstance(departures[0], dict) else None
    result = (
        _search(
            {
                **params,
                "departure_token": token,
            }
        )
        if token and returning is not None
        else first
    )
    if result.get("error"):
        return ThinkingResult(str(result["error"]), status="unavailable", scope="The flight provider did not return schedule evidence.")
    offers = (result.get("best_flights") or result.get("other_flights") or [])[:MAX_OFFERS]
    if not offers:
        return ThinkingResult("No flight offers matched those constraints.", status="rejected", scope="No retrieved offer matched the requested flight constraints.", next_actions=("search_flights",))
    text = "Google Flights results (prices can change):\n" + "\n".join(
        f"{index}. {_flight_summary(item)}"
        for index, item in enumerate(offers, 1)
        if isinstance(item, dict)
    )
    first_offer = next((item for item in offers if isinstance(item, dict)), None)
    artifact = _flight_artifact(first_offer, params, depart, returning) if first_offer is not None else None
    evidence_offers = [
        _flight_artifact(item, params, depart, returning)["offer"]
        for item in offers
        if isinstance(item, dict)
    ]
    return ThinkingResult(
        (
            f"Representative future date: {depart}. "
            f"Resolved route: {resolved_origin[1]} ({resolved_origin[0]}) to "
            f"{resolved_destination[1]} ({resolved_destination[0]}).\n{text}"
        ),
        evidence={"route": {"origin": resolved_origin[0], "destination": resolved_destination[0]}, "date": depart, "offers": evidence_offers},
        scope=f"Up to {len(evidence_offers)} retrieved offers for {resolved_origin[0]} to {resolved_destination[0]} on {depart}.",
        next_actions=("search_flights", "plan_travel"),
        artifact=artifact,
    )


@tool(
    name="plan_travel",
    description=(
        "Plan a natural-language multi-stop trip. Resolves named places, selects a representative future date, "
        "and retrieves each adjacent flight leg with timezone-aware schedule data."
    ),
    available=_available,
    deep_think_only=True,
    thinking_outcome=True,
)
def plan_travel(request: str) -> str:
    """Run the travel tool's complete evidence path for an ordered itinerary."""
    if not isinstance(request, str) or not request.strip():
        return ThinkingResult("Describe at least two places to plan a trip.", status="needs_input", scope="An ordered multi-stop travel request is required.")
    stops = _locations_in_request(request)
    home_city = _home_city()
    home = _resolve_airport(home_city, "home") if home_city else None
    if len(stops) == 1 and home is not None:
        # A lone destination ordinarily means a trip from home; retain the
        # reverse direction when the request explicitly starts from that city.
        city = re.escape(stops[0][1])
        stops = [stops[0], home] if re.search(rf"\bfrom\s+{city}\b", request, re.IGNORECASE) else [home, stops[0]]
    if len(stops) < 2:
        return ThinkingResult("Name a destination; a home airport could not be inferred from the household timezone.", status="needs_input", scope="At least one travel endpoint is required and no home airport default is available.")
    departure_date = _representative_departure_date(request)

    results = []
    legs = []
    leg_offers = []
    artifact = None
    for index, ((origin, _location), (destination, _next_location)) in enumerate(
        zip(stops, stops[1:], strict=False), 1
    ):
        origin_arg = "" if home is not None and origin == home[0] else origin
        destination_arg = "" if home is not None and destination == home[0] else destination
        result = search_flights(origin_arg, destination_arg, departure_date)
        if isinstance(result, ThinkingResult) and result.thinking_status != "evidence":
            return result
        flight_artifact = getattr(result, "artifact", None)
        if not isinstance(flight_artifact, dict):
            return ThinkingResult("The travel provider did not return schedule evidence for every itinerary leg.", status="unavailable", scope="At least one itinerary leg had no schedule artifact.")
        offer = flight_artifact.get("offer")
        if not isinstance(offer, dict):
            return ThinkingResult("The travel provider returned an incomplete flight schedule.", status="unavailable", scope="At least one itinerary leg had an incomplete schedule artifact.")
        legs.append({"departure": offer.get("departure"), "arrival": offer.get("arrival")})
        offers = result.evidence.get("offers") if isinstance(result.evidence, dict) else None
        if not isinstance(offers, list) or not offers:
            return ThinkingResult("The travel provider returned no usable flight offers.", status="unavailable", scope="At least one itinerary leg had no bounded offer list.")
        leg_offers.append(offers)
        results.append(f"Leg {index}: {result}")
        artifact = artifact or flight_artifact

    text = (
        f"Representative future date: {departure_date}.\n"
        + "\n".join(results)
        + "\n\nSchedule evidence retrieved. Use assess_itinerary with the returned Artifact reference to check the retrieved flight chronology."
    )
    plan_artifact = {"type": "travel_plan", "request": request, "legs": legs, "leg_offers": leg_offers, "representative": artifact}
    return ThinkingResult(
        text,
        evidence={"departure_date": departure_date, "legs": legs, "offers_per_leg": [len(items) for items in leg_offers]},
        scope=f"Retrieved one representative offer for each of {len(legs)} itinerary legs.",
        next_actions=("assess_itinerary", "search_flights", "search_hotels"),
        artifact=plan_artifact,
    )


def _assess_itinerary_legs(legs: list) -> str:
    """Check retrieved leg chronology without asking the worker to reconstruct schedules."""
    parsed = []
    for index, leg in enumerate(legs, 1):
        try:
            departure = _schedule_datetime(_schedule_value(leg, "departure"), f"leg {index} departure")
            arrival = _schedule_datetime(_schedule_value(leg, "arrival"), f"leg {index} arrival")
        except ValueError as exc:
            return str(exc)
        if arrival < departure:
            return f"Leg {index} arrives before it departs."
        if parsed and departure < parsed[-1][1]:
            return f"Leg {index} departs before leg {index - 1} arrives."
        parsed.append((departure, arrival))
    return "Retrieved itinerary is chronologically feasible between its flight legs."


def _best_retrieved_itinerary(leg_offers: list) -> list | None:
    """Return the first chronologically compatible bounded offer combination."""
    if not isinstance(leg_offers, list) or not leg_offers or any(
        not isinstance(offers, list) or not offers for offers in leg_offers
    ):
        return None
    for offers in islice(product(*leg_offers), 125):
        legs = [
            {"departure": offer.get("departure"), "arrival": offer.get("arrival")}
            for offer in offers
            if isinstance(offer, dict)
        ]
        if len(legs) == len(leg_offers) and "feasible" in _assess_itinerary_legs(legs).lower():
            return legs
    return None


@tool(
    name="assess_itinerary",
    description="Assess a plan_travel Artifact reference using its retrieved schedules. Do not pass schedule JSON.",
    available=_available,
    deep_think_only=True,
    thinking_outcome=True,
)
def assess_itinerary(artifact_id: str) -> str:
    record = get_artifact(artifact_id)
    data = record.get("data") if isinstance(record, dict) else None
    legs = data.get("legs") if isinstance(data, dict) else None
    if not isinstance(legs, list):
        return ThinkingResult(
            "That itinerary artifact is unavailable.", status="needs_input", scope="A valid travel-plan artifact reference is required."
        )
    leg_offers = data.get("leg_offers") if isinstance(data, dict) else None
    best_legs = _best_retrieved_itinerary(leg_offers) if leg_offers is not None else legs
    assessment = _assess_itinerary_legs(best_legs) if isinstance(best_legs, list) else "No chronologically compatible itinerary exists among the retrieved offer combinations."
    request = data.get("request") if isinstance(data, dict) else ""
    if isinstance(request, str) and re.search(
        r"\b(breakfast|lunch|dinner|morning|afternoon|evening|appointment|meeting|event)\b",
        request,
        re.IGNORECASE,
    ):
        return ThinkingResult(
            f"Preliminary assessment: {assessment} The requested activity timing cannot be confirmed without exact local windows.",
            evidence={"artifact_id": artifact_id, "legs_checked": len(legs), "activity_windows_checked": False, "schedule_assessment": assessment},
            scope="This preliminary result covers bounded retrieved flight combinations, not the requested activity windows.",
            next_actions=("search_flights",),
        )
    status = "evidence" if "feasible" in assessment.lower() else "rejected"
    return ThinkingResult(
        assessment,
        status=status,
        evidence={"artifact_id": artifact_id, "legs_checked": len(legs), "combination_checked": best_legs is not None},
        scope="Only bounded retrieved flight combinations were checked; requested activity windows need additional evidence.",
        next_actions=("search_flights",),
    )


def evaluate_itinerary(legs_json: str, windows_json: str) -> str:
    """Legacy deterministic helper retained for direct callers and tests only."""
    try:
        legs = json.loads(legs_json)
        windows = json.loads(windows_json)
    except (TypeError, json.JSONDecodeError):
        return "legs_json and windows_json must each be valid JSON arrays."
    if not isinstance(legs, list) or not isinstance(windows, list):
        return "legs_json and windows_json must each be JSON arrays."
    return _evaluate_itinerary(legs, windows)


@tool(
    name="search_hotels",
    description="Search up to five read-only Google Hotels offers via SerpApi for fixed dates and guests. Results are not bookings.",
    available=_available,
    deep_think_only=True,
    thinking_outcome=True,
)
def search_hotels(
    location: str, check_in: str, check_out: str, adults: int = 1, children: int = 0
) -> str:
    """Return bounded, normalized accommodation options without booking actions."""
    if not isinstance(location, str) or not location.strip():
        return ThinkingResult("A city or location is required for hotel search.", status="needs_input", scope="A hotel-search location is required.")
    try:
        check_in, check_out = _date(check_in, "check_in"), _date(check_out, "check_out")
        adults, children = _positive(adults, "adults"), _positive(children, "children")
    except ValueError as exc:
        return ThinkingResult(str(exc), status="needs_input", scope="The requested hotel dates or guest counts are invalid.")
    if check_out <= check_in:
        return ThinkingResult("check_out must be after check_in.", status="needs_input", scope="The requested hotel dates are invalid.")
    if adults < 1:
        return ThinkingResult("At least one adult is required.", status="needs_input", scope="At least one adult is required.")
    result = _search(
        {
            "engine": "google_hotels",
            "q": location.strip(),
            "check_in_date": check_in,
            "check_out_date": check_out,
            "adults": adults,
            "children": children,
            "currency": household_locale().currency,
            "hl": "en",
        }
    )
    if result.get("error"):
        return ThinkingResult(str(result["error"]), status="unavailable", scope="The hotel provider did not return offers.")
    properties = (result.get("properties") or [])[:MAX_OFFERS]
    if not properties:
        return ThinkingResult("No hotel offers matched those constraints.", status="rejected", scope="No retrieved hotel offer matched the requested location and dates.", next_actions=("search_hotels",))
    lines = []
    offers = []
    for index, item in enumerate(properties, 1):
        if not isinstance(item, dict):
            continue
        total = (item.get("total_rate") or {}).get("lowest") or "price unavailable"
        rating = item.get("overall_rating", "rating unavailable")
        address = item.get("address") or "address unavailable"
        cancellation = (
            "free cancellation"
            if item.get("free_cancellation")
            else "cancellation terms unavailable"
        )
        lines.append(
            f"{index}. {item.get('name', 'Unnamed property')}; total {total}; rating {rating}; {address}; {cancellation}."
        )
        offers.append({"name": str(item.get("name") or "Unnamed property"), "total": total, "rating": rating, "address": str(address), "free_cancellation": bool(item.get("free_cancellation"))})
    return ThinkingResult(
        "Google Hotels results (prices can change):\n" + "\n".join(lines),
        evidence={"location": location.strip(), "check_in": check_in, "check_out": check_out, "offers": offers},
        scope=f"Up to {len(offers)} retrieved hotel offers in {location.strip()} for {check_in} to {check_out}.",
        next_actions=("search_hotels",),
        artifact={"type": "hotel_search", "location": location.strip(), "check_in": check_in, "check_out": check_out, "offers": offers},
    )
