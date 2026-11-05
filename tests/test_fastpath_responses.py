"""Editable response files are data, never executable templates."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tools.tool_registry import OperationOutcome
from utils.fastpath_responses import (
    format_local_datetime,
    format_local_duration,
    load_responses,
    render_response,
    select_fastpath_response,
)
from utils.hassil_fastpath import MatchMetadata

ROOT = Path(__file__).resolve().parents[1] / "data" / "intents"


@pytest.mark.parametrize("language", ["en", "fr", "de"])
def test_tracked_resources_load_and_keep_outcomes_separate(language):
    responses = load_responses(ROOT, language)
    assert render_response(responses, "time", "success", {"time": "10:00"})
    assert render_response(responses, "time", "error", {"time": "10:00"}) is None
    assert render_response(responses, "operation_failed", "success", {}) is None
    assert render_response(responses, "operation_failed", "error", {})


@pytest.mark.parametrize("template", [
    "{time.__class__}", "{time!r}", "{time:>10}", "{other}",
    "{{ value }}", "{% if x %}hi{% endif %}", "{time",
])
def test_untrusted_templates_are_rejected(tmp_path, template):
    path = tmp_path / "fr" / "fulloch-responses.yaml"
    path.parent.mkdir()
    import json

    path.write_text(f'language: fr\nschema_version: 1\nresponses:\n  time:\n    success: [{json.dumps(template)}]\n')
    assert load_responses(tmp_path, "fr") == {}


def test_custom_file_and_revision_do_not_require_matching_pack(tmp_path):
    path = tmp_path / "de" / "fulloch-responses.yaml"
    path.parent.mkdir()
    path.write_text('language: de\nschema_version: 1\nsource_revision: old\nresponses:\n  time:\n    success: ["Zeit: {time}"]\n')
    responses = load_responses(tmp_path, "de")
    assert render_response(responses, "time", "success", {"time": "12:30"}) == "Zeit: 12:30"
    assert render_response(responses, "time", "success", {}) is None


def test_missing_malformed_and_wrong_language_are_safe(tmp_path):
    assert load_responses(tmp_path, "../en") == {}
    assert load_responses(tmp_path, "en") == {}
    path = tmp_path / "en" / "fulloch-responses.yaml"
    path.parent.mkdir()
    path.write_text("language: [invalid")
    assert load_responses(tmp_path, "en") == {}
    path.write_text("language: de\nschema_version: 1\nresponses: {}\n")
    assert load_responses(tmp_path, "en") == {}


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.cpu"])
def test_images_explicitly_package_only_response_files(dockerfile):
    root = ROOT.parent.parent
    contents = (root / dockerfile).read_text()
    for language in ("en", "fr", "de"):
        assert f"data/intents/{language}/fulloch-responses.yaml /app/seed/intents/{language}/fulloch-responses.yaml" in contents
    assert "data/intents/*/*" in (root / ".dockerignore").read_text()


@pytest.mark.parametrize("language, expected", [
    ("en", "1 minute and 2 seconds"),
    ("fr", "1 minute et 2\u00a0secondes"),
    ("de", "1 Minute und 2 Sekunden"),
])
def test_duration_formatting_uses_locale_plural_and_retains_remainder(language, expected):
    assert format_local_duration(62, language) == expected


@pytest.mark.parametrize("seconds", [-1, 0.5, True, "60"])
def test_duration_formatting_rejects_invalid_values(seconds):
    with pytest.raises(ValueError):
        format_local_duration(seconds, "en")


def test_datetime_uses_the_instant_timezone_not_host_timezone():
    instant = datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Paris"))
    before = format_local_datetime(instant, "fr")
    after = format_local_datetime(datetime(2026, 3, 29, 1, 30, tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Paris")), "fr")
    assert before["time"] == "01:30"
    assert after["time"] == "03:30"
    assert "mars" in before["date"]
    with pytest.raises(ValueError):
        format_local_datetime(datetime(2026, 3, 29), "fr")


def _match(language, intent):
    return MatchMetadata(language, intent, intent, {}, None)


@pytest.mark.parametrize("language, prefix", [("en", "The time is"), ("fr", "Il est"), ("de", "Es ist")])
def test_select_response_uses_verified_time_outcome(language, prefix):
    outcome = OperationOutcome(True, "current_time", {"datetime": datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc)})
    result = select_fastpath_response(_match(language, "HassGetCurrentTime"), outcome, "original", ROOT, language_confirmed=True)
    assert result.startswith(prefix)


def test_select_response_uses_verified_timer_values():
    outcome = OperationOutcome(True, "timer_started", {"timer_id": "timer_1", "duration_seconds": 62})
    result = select_fastpath_response(_match("fr", "HassStartTimer"), outcome, "original", ROOT, language_confirmed=True)
    assert "1 minute et 2\u00a0secondes" in result
    cancelled = OperationOutcome(True, "timer_cancelled", {"timer_id": "timer_1"})
    assert select_fastpath_response(_match("de", "HassCancelTimer"), cancelled, "original", ROOT, language_confirmed=True) == "Timer abgebrochen."


@pytest.mark.parametrize("metadata, outcome, confirmed", [
    (_match("de", "HassStartTimer"), OperationOutcome(False, "timer_started", {"timer_id": "timer_1", "duration_seconds": 60}), True),
    (_match("de", "HassStartTimer"), OperationOutcome(True, "timer_cancelled", {"timer_id": "timer_1"}), True),
    (_match("de", "HassStartTimer"), OperationOutcome(True, "timer_started", {"duration_seconds": 60}), True),
    (_match("de", "HassStartTimer"), OperationOutcome(True, "timer_started", {"timer_id": "timer_1", "duration_seconds": 0}), True),
    (_match("de", "HassStartTimer"), OperationOutcome(True, "timer_started", {"timer_id": "timer_1", "duration_seconds": 60}), False),
    (_match("es", "HassStartTimer"), OperationOutcome(True, "timer_started", {"timer_id": "timer_1", "duration_seconds": 60}), True),
    (_match("de", "HassStartTimer"), None, True),
    (None, OperationOutcome(True, "timer_started", {"timer_id": "timer_1", "duration_seconds": 60}), True),
])
def test_select_response_never_invents_success(metadata, outcome, confirmed):
    assert select_fastpath_response(metadata, outcome, "original failure", ROOT, language_confirmed=confirmed) == "original failure"


def test_select_response_reloads_edits_and_falls_back_on_malformed_file(tmp_path):
    path = tmp_path / "fr" / "fulloch-responses.yaml"
    path.parent.mkdir()
    path.write_text('language: fr\nschema_version: 1\nresponses:\n  timer_cancelled:\n    success: ["Annulé."]\n')
    outcome = OperationOutcome(True, "timer_cancelled", {"timer_id": "timer_1"})
    args = (_match("fr", "HassCancelTimer"), outcome, "Timer cancelled", tmp_path)
    assert select_fastpath_response(*args, language_confirmed=True) == "Annulé."
    path.write_text('language: fr\nschema_version: 1\nresponses:\n  timer_cancelled:\n    success: ["Autre."]\n')
    assert select_fastpath_response(*args, language_confirmed=True) == "Autre."
    path.write_text("invalid: [")
    assert select_fastpath_response(*args, language_confirmed=True) == "Timer cancelled"


def test_light_reply_requires_confirmed_light_service_and_exact_named_target():
    metadata = MatchMetadata("fr", "HassTurnOn", "HassTurnOn", {"name": "light.kitchen"}, "light.kitchen")
    values = {"domain": "light", "service": "turn_on", "entity_id": "light.kitchen", "data": None}
    outcome = OperationOutcome(True, "service_called", values)
    assert select_fastpath_response(metadata, outcome, "original", ROOT, language_confirmed=True) == "Commande d'allumage des lumières envoyée."
    for changed in ({"domain": "switch"}, {"service": "turn_off"}, {"entity_id": "light.other"}, {"entity_id": []}):
        invalid = OperationOutcome(True, "service_called", {**values, **changed})
        assert select_fastpath_response(metadata, invalid, "original", ROOT, language_confirmed=True) == "original"
    failed = OperationOutcome(False, "voice_denied", values)
    assert "Aucune commande" in select_fastpath_response(metadata, failed, "denied", ROOT, language_confirmed=True)


def test_light_area_reply_requires_nonempty_light_targets():
    metadata = MatchMetadata("de", "HassTurnOff", "HassTurnOff", {"area": "kitchen"}, "kitchen")
    values = {"domain": "light", "service": "turn_off", "entity_id": ["light.one", "light.two"]}
    result = select_fastpath_response(metadata, OperationOutcome(True, "service_called", values), "original", ROOT, language_confirmed=True)
    assert result == "Befehl zum Ausschalten der Beleuchtung gesendet."


@pytest.mark.parametrize("language, expected", [("en", "25 percent"), ("fr", "25 pour cent"), ("de", "25 Prozent")])
def test_numeric_confirmation_uses_matched_language_and_verified_payload(language, expected):
    metadata = MatchMetadata(language, "HassLightSet", "brightness", {"name": "light.kitchen", "brightness": 25}, "light.kitchen")
    values = {"domain": "light", "service": "turn_on", "entity_id": "light.kitchen", "data": {"brightness": 63}}
    outcome = OperationOutcome(True, "service_called", values)
    assert expected in select_fastpath_response(metadata, outcome, "original", ROOT, language_confirmed=True)
    for change in ({"data": {"brightness": 100}}, {"entity_id": "light.other"}, {"service": "turn_off"}):
        assert select_fastpath_response(metadata, OperationOutcome(True, "service_called", {**values, **change}), "original", ROOT, language_confirmed=True) == "original"


@pytest.mark.parametrize("intent, slots, domain, service, data, expected", [
    ("HassSetVolume", {"volume_level": 25}, "media_player", "volume_set", {"volume_level": 0.25}, "25 pour cent"),
    ("HassClimateSetTemperature", {"temperature": 21.5}, "climate", "set_temperature", {"temperature": 21.5}, "21,5 degrés"),
    ("HassSetPosition", {"position": 50}, "cover", "set_cover_position", {"position": 50}, "50 pour cent"),
    ("HassSetPosition", {"position": 50}, "valve", "set_valve_position", {"position": 50}, "50 pour cent"),
])
def test_numeric_actions_format_values_from_verified_service(intent, slots, domain, service, data, expected):
    metadata = MatchMetadata("fr", intent, intent, slots, None)
    outcome = OperationOutcome(True, "service_called", {"domain": domain, "service": service, "entity_id": f"{domain}.test", "data": data})
    assert expected in select_fastpath_response(metadata, outcome, "original", ROOT, language_confirmed=True)


@pytest.mark.parametrize("language, expected", [("en", "no active timers"), ("fr", "Aucun minuteur"), ("de", "keine Timer")])
def test_timer_status_and_cancel_errors_are_localized(language, expected):
    status = OperationOutcome(True, "no_active_timers", {"timers": []})
    cancel = OperationOutcome(False, "no_active_timers", {})
    assert expected in select_fastpath_response(_match(language, "HassTimerStatus"), status, "original", ROOT, language_confirmed=True)
    assert expected in select_fastpath_response(_match(language, "HassCancelTimer"), cancel, "original", ROOT, language_confirmed=True)


def test_timer_status_formats_structured_remaining_time():
    outcome = OperationOutcome(True, "timer_status", {"timers": [{"timer_id": "timer_1", "remaining_seconds": 62.9}]})
    assert "1 minute et 2\u00a0secondes" in select_fastpath_response(_match("fr", "HassTimerStatus"), outcome, "original", ROOT, language_confirmed=True)


def test_existing_custom_response_file_gets_missing_bundled_keys(tmp_path):
    path = tmp_path / "fr" / "fulloch-responses.yaml"
    path.parent.mkdir()
    path.write_text('language: fr\nschema_version: 1\nresponses:\n  time:\n    success: ["Personnalisé : {time}"]\n')
    outcome = OperationOutcome(True, "no_active_timers", {"timers": []})
    result = select_fastpath_response(_match("fr", "HassTimerStatus"), outcome, "original", tmp_path, language_confirmed=True)
    assert result == "Aucun minuteur n'est actif."
    assert "Personnalisé" in path.read_text()
