"""Restricted, separately loaded Fulloch response resources.

Missing or invalid files produce no templates; callers retain the original
truthful observation.
"""

import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from string import Formatter

from babel.core import UnknownLocaleError
from babel.dates import format_date, format_time
from babel.lists import format_list
from babel.numbers import format_decimal
from babel.units import format_unit
from ruamel.yaml import YAML

from tools.tool_registry import OperationOutcome
from utils.hassil_fastpath import MatchMetadata

logger = logging.getLogger(__name__)

# A key's allowed fields are fixed by code, not by editable resources.
RESPONSE_FIELDS = {
    "time": frozenset({"time"}),
    "date": frozenset({"date"}),
    "timer_started": frozenset({"duration"}),
    "timer_cancelled": frozenset(),
    "timer_completed": frozenset(),
    "timer_status": frozenset({"duration"}),
    "timer_status_multiple": frozenset({"durations"}),
    "timer_none": frozenset(),
    "timer_ambiguous": frozenset(),
    "timer_limit": frozenset(),
    "timer_too_long": frozenset(),
    "timer_invalid_duration": frozenset(),
    "light_brightness": frozenset({"value"}),
    "climate_temperature": frozenset({"value"}),
    "cover_position": frozenset({"value"}),
    "media_volume": frozenset({"value"}),
    "ha_unreachable": frozenset(),
    "ha_timeout": frozenset(),
    "ha_not_configured": frozenset(),
    "voice_denied": frozenset(),
    "light_on": frozenset({"target"}),
    "light_off": frozenset({"target"}),
    "operation_failed": frozenset(),
}
_FORMATTER = Formatter()
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")


def _valid_template(text: object, fields: frozenset[str]) -> bool:
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        return False
    # Reject Jinja and escaped braces rather than treating them as executable
    # syntax or permitting confusing user edits.
    if "{{" in text or "}}" in text or "{%" in text or "{#" in text:
        return False
    try:
        for _, field, spec, conversion in _FORMATTER.parse(text):
            if field is not None and (field not in fields or spec or conversion):
                return False
    except ValueError:
        return False
    return True


def load_responses(intents_dir: str | Path, language: str) -> dict:
    """Load and validate one file; never execute templates or read other packs."""
    if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
        return {}
    path = Path(intents_dir) / language / "fulloch-responses.yaml"
    try:
        if path.stat().st_size > 65536:
            raise ValueError("response file exceeds 64 KiB")
        yaml = YAML(typ="safe")
        with path.open(encoding="utf-8") as stream:
            data = yaml.load(stream)
        if not isinstance(data, dict) or data.get("language") != language or data.get("schema_version") != 1:
            raise ValueError("invalid response file header")
        responses = data.get("responses")
        if not isinstance(responses, dict):
            raise ValueError("responses must be a mapping")
        validated = {}
        for key, outcomes in responses.items():
            if key not in RESPONSE_FIELDS or not isinstance(outcomes, dict) or not outcomes:
                raise ValueError("invalid response key")
            validated[key] = {}
            for outcome, choices in outcomes.items():
                if outcome not in {"success", "error"} or not isinstance(choices, list) or not choices or len(choices) > 20:
                    raise ValueError("invalid outcome choices")
                if not all(_valid_template(choice, RESPONSE_FIELDS[key]) for choice in choices):
                    raise ValueError("invalid response template")
                validated[key][outcome] = tuple(choices)
        return validated
    except FileNotFoundError:
        return {}
    except Exception as exc:  # A user's malformed file must not affect actions.
        logger.warning("Ignoring invalid response file %s: %s", path, exc)
        return {}


def render_response(responses: dict, key: str, outcome: str, values: dict) -> str | None:
    """Render the first validated choice, or return None for a safe fallback."""
    choices = responses.get(key, {}).get(outcome, ())
    if not choices or key not in RESPONSE_FIELDS or outcome not in {"success", "error"}:
        return None
    template = choices[0]
    if not _valid_template(template, RESPONSE_FIELDS[key]):
        return None
    try:
        return template.format_map(values)
    except (KeyError, ValueError, TypeError):
        return None


def format_local_datetime(value: datetime, language: str) -> dict[str, str]:
    """Format an already-zoned instant without using process-local timezone."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("a timezone-aware datetime is required")
    return {
        "date": format_date(value.date(), format="full", locale=language),
        "time": format_time(value, format="short", tzinfo=value.tzinfo, locale=language),
    }


def format_local_duration(seconds: int, language: str) -> str:
    """Format every nonzero duration component using CLDR plurals and lists."""
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
        raise ValueError("duration must be a non-negative integer number of seconds")
    parts = []
    for unit, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        value, seconds = divmod(seconds, size)
        if value:
            parts.append(format_unit(value, unit, length="long", locale=language))
    return format_list(parts, locale=language) if parts else format_unit(0, "second", length="long", locale=language)


def _render_with_bundled_fallback(responses, language, key, outcome, values):
    """Use new bundled keys without overwriting existing customized resources."""
    result = render_response(responses, key, outcome, values)
    if result or not responses or key in responses:
        return result
    seed = Path(os.environ.get("FULLOCH_SEED_DIR", "/app/seed")) / "intents"
    if not seed.is_dir():
        seed = Path(__file__).resolve().parent.parent / "data" / "intents"
    return render_response(load_responses(seed, language), key, outcome, values)


def _verified_numeric_service(metadata, outcome):
    """Return response key/value only for an exact, successful HA service."""
    specs = {
        "HassLightSet": ("brightness", "light", "turn_on", "brightness", "light_brightness"),
        "HassClimateSetTemperature": ("temperature", "climate", "set_temperature", "temperature", "climate_temperature"),
        "HassSetPosition": ("position", None, "set_cover_position", "position", "cover_position"),
        "HassSetVolume": ("volume_level", "media_player", "volume_set", "volume_level", "media_volume"),
    }
    spec = specs.get(metadata.intent)
    if not spec or outcome.code != "service_called":
        return None
    slot, domain, service, field, key = spec
    value = metadata.slots.get(slot)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    actual = outcome.values
    if actual.get("domain") != domain and not (domain is None and actual.get("domain") in {"cover", "valve"}):
        return None
    if actual.get("service") != ("set_valve_position" if metadata.intent == "HassSetPosition" and actual.get("domain") == "valve" else service):
        return None
    targets = actual.get("entity_id")
    targets = [targets] if isinstance(targets, str) else targets
    if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t.startswith(actual["domain"] + ".") for t in targets):
        return None
    named = metadata.slots.get("name")
    if named and targets != [named]:
        return None
    data = actual.get("data")
    if not isinstance(data, dict) or field not in data:
        return None
    sent = data[field]
    expected = int(value * 255 / 100) if slot == "brightness" else value / 100 if slot == "volume_level" else value
    if isinstance(sent, bool) or not isinstance(sent, (int, float)) or not math.isclose(sent, expected, abs_tol=1e-9):
        return None
    return key, value


def select_fastpath_response(
    metadata: MatchMetadata | None,
    outcome: OperationOutcome | None,
    observation: str,
    intents_dir: str | Path,
    *,
    language_confirmed: bool = False,
) -> str:
    """Select a verified local success or retain the original observation.

    Callers must first establish an unambiguous response language. This
    supports verified time/date, timers, light, climate, position and volume
    outcomes; unknown actions and failures retain their original observations.
    No choice is inferred from an action name, a nonempty string, or a normal
    agent step. Resources are re-read on each call, so edits are visible on the
    next eligible turn without replacing a customized file.
    """
    if not language_confirmed or not isinstance(metadata, MatchMetadata) or not isinstance(outcome, OperationOutcome):
        return observation
    if metadata.language not in {"en", "fr", "de"}:
        return observation
    if not isinstance(outcome.values, dict):
        return observation
    key = None
    values = {}
    try:
        if outcome.success is False:
            errors = {"HassStartTimer": {"timer_limit": "timer_limit", "duration_too_long": "timer_too_long", "invalid_duration": "timer_invalid_duration"},
                      "HassCancelTimer": {"no_active_timers": "timer_none", "ambiguous_timer": "timer_ambiguous"}}
            key = errors.get(metadata.intent, {}).get(outcome.code)
            if metadata.intent in {"HassTurnOn", "HassTurnOff", "HassLightSet", "HassClimateSetTemperature", "HassSetPosition", "HassSetVolume"} and outcome.code in {"ha_unreachable", "ha_timeout", "ha_not_configured", "voice_denied"}:
                key = outcome.code
            if not key:
                return observation
            responses = load_responses(intents_dir, metadata.language)
            return _render_with_bundled_fallback(responses, metadata.language, key, "error", {}) or observation
        if outcome.success is not True:
            return observation
        if metadata.intent in {"HassGetCurrentTime", "HassGetCurrentDate"} and outcome.code == "current_time":
            key = "time" if metadata.intent == "HassGetCurrentTime" else "date"
            values = format_local_datetime(outcome.values.get("datetime"), metadata.language)
        elif metadata.intent == "HassStartTimer" and outcome.code == "timer_started":
            seconds = outcome.values.get("duration_seconds")
            if not isinstance(outcome.values.get("timer_id"), str) or not outcome.values["timer_id"]:
                return observation
            if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
                return observation
            key = "timer_started"
            values = {"duration": format_local_duration(seconds, metadata.language)}
        elif metadata.intent == "HassCancelTimer" and outcome.code == "timer_cancelled":
            if not isinstance(outcome.values.get("timer_id"), str) or not outcome.values["timer_id"]:
                return observation
            key = "timer_cancelled"
        elif metadata.intent == "HassTimerStatus" and outcome.code == "no_active_timers" and outcome.values.get("timers") == []:
            key = "timer_none"
        elif metadata.intent == "HassTimerStatus" and outcome.code == "timer_status":
            timers = outcome.values.get("timers")
            if not isinstance(timers, list) or not timers or len(timers) > 32:
                return observation
            durations = []
            for timer in timers:
                if not isinstance(timer, dict) or not isinstance(timer.get("timer_id"), str):
                    return observation
                seconds = timer.get("remaining_seconds")
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    return observation
                durations.append(format_local_duration(int(seconds), metadata.language))
            key = "timer_status" if len(durations) == 1 else "timer_status_multiple"
            values = {"duration": durations[0], "durations": format_list(durations, locale=metadata.language)}
        elif metadata.intent in {"HassTurnOn", "HassTurnOff"} and outcome.code == "service_called":
            service = "turn_on" if metadata.intent == "HassTurnOn" else "turn_off"
            targets = outcome.values.get("entity_id")
            targets = [targets] if isinstance(targets, str) else targets
            if (outcome.values.get("domain") != "light" or outcome.values.get("service") != service
                    or not isinstance(targets, list) or not targets
                    or any(not isinstance(target, str) or not target.startswith("light.") for target in targets)):
                return observation
            named_target = metadata.slots.get("name")
            if named_target and (not isinstance(named_target, str) or targets != [named_target]):
                return observation
            if not named_target and not (metadata.slots.get("area") or metadata.slots.get("floor")):
                return observation
            key = "light_on" if service == "turn_on" else "light_off"
        else:
            numeric = _verified_numeric_service(metadata, outcome)
            if not numeric:
                return observation
            key, number = numeric
            values = {"value": format_decimal(number, locale=metadata.language)}
        responses = load_responses(intents_dir, metadata.language)
        return _render_with_bundled_fallback(responses, metadata.language, key, "success", values) or observation
    except (ValueError, TypeError, UnknownLocaleError, OverflowError):
        return observation
