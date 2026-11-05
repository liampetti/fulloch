"""Multilingual matching, safety, dispatch, and live language replacement."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hassil import Intents

from tools import ha_client, ha_devices
from utils.completeness import should_commit_provisional
from utils.hassil_fastpath import HassilFastPath
from utils.intent_catch import catch_with_hassil

FIXTURES = Path(__file__).parent / "fixtures" / "hassil"
ENGLISH = Path(__file__).parent.parent / "third_party/home-assistant-intents/en/2025.6.23.yaml"


@pytest.fixture
def matcher(tmp_path, monkeypatch):
    for code, source in (("en", ENGLISH), ("fr", FIXTURES / "fr/2025.6.23.yaml"), ("de", FIXTURES / "de/2025.6.23.yaml")):
        target = tmp_path / code / "2025.6.23.yaml"
        target.parent.mkdir()
        shutil.copyfile(source, target)
    monkeypatch.setattr(ha_client, "_loaded", True)
    aliases = {
        "kitchen lights": ["light.kitchen_lights"],
        "lumière de la cuisine": ["light.kitchen_lights"],
        "küchenlicht": ["light.kitchen_lights"],
    }
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", aliases)
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {name: ids[0] for name, ids in aliases.items()})
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"kitchen": "kitchen"})
    monkeypatch.setattr(ha_client, "_FLOOR_MAP", {"upstairs": "upstairs"})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    return HassilFastPath(["en", "fr", "de"], str(tmp_path))


@pytest.mark.parametrize("text", [
    "turn on the kitchen lights",
    "allume la lumière de la cuisine",
    "schalte das küchenlicht ein",
    "um schalte das küchenlicht ein",
])
def test_multilingual_named_lights(matcher, text):
    assert matcher.match(text) == {"actions": [{"intent": "turn_on", "args": ["light.kitchen_lights"]}]}


@pytest.mark.parametrize("text, language", [
    ("turn on the kitchen lights", "en"),
    ("allume la lumière de la cuisine", "fr"),
    ("schalte das küchenlicht ein", "de"),
])
def test_match_metadata_is_per_emission(matcher, text, language):
    emission = matcher.match(text)
    assert emission.metadata.language == language
    assert emission.metadata.intent == "HassTurnOn"
    assert emission.metadata.response_key == "HassTurnOn"
    assert emission.metadata.slots["name"] == "light.kitchen_lights"
    assert emission.metadata.target == "light.kitchen_lights"
    assert list(emission) == ["actions"]
    other = matcher.match("schalte das küchenlicht ein")
    assert emission.metadata.language == language
    assert other.metadata.language == "de"


@pytest.mark.parametrize("text, language, expected", [
    ("set the kitchen lights brightness to 25 percent", "en", "25 percent"),
    ("règle la luminosité de la lumière de la cuisine à 25 pour cent", "fr", "25 pour cent"),
    ("dimme küchenlicht auf 25 prozent", "de", "25 Prozent"),
])
def test_hassil_numeric_slots_feed_localized_verified_response(matcher, text, language, expected):
    from tools.tool_registry import OperationOutcome
    from utils.fastpath_responses import select_fastpath_response

    emission = matcher.match(text)
    assert emission.metadata.language == language
    assert emission.metadata.slots["brightness"] == 25
    assert matcher.response_language_confirmed(text, emission)
    outcome = OperationOutcome(True, "service_called", {"domain": "light", "service": "turn_on",
        "entity_id": "light.kitchen_lights", "data": {"brightness": 63}})
    responses = Path(__file__).parent.parent / "data/intents"
    assert expected in select_fastpath_response(emission.metadata, outcome, "original", responses, language_confirmed=True)


def test_regex_priority_does_not_inherit_hassil_metadata(matcher):
    matcher.match("schalte das küchenlicht ein")
    assert not hasattr(catch_with_hassil("stop", matcher), "metadata")


def test_local_match_retains_metadata_without_synthetic_target(matcher):
    emission = catch_with_hassil("wie spät ist es", matcher)
    assert emission == {"actions": [{"intent": "get_current_time", "args": []}]}
    assert emission.metadata.language == "de"
    assert emission.metadata.intent == "HassGetCurrentTime"
    assert emission.metadata.slots == {}
    assert emission.metadata.target is None


def test_response_language_confirms_only_unchanged_unambiguous_match(matcher):
    text = "wie spät ist es"
    emission = matcher.match(text)
    assert matcher.response_language_confirmed(text, emission)
    assert not matcher.response_language_confirmed("quelle heure est il", emission)
    assert not matcher.response_language_confirmed(text, {"actions": emission["actions"]})
    assert not matcher.response_language_confirmed("stop", catch_with_hassil("stop", matcher))


def test_competing_language_does_not_change_action_but_blocks_confirmation(matcher):
    text = "wie spät ist es"
    emission = matcher.match(text)
    competing = Intents.from_dict({
        "language": "fr",
        "intents": {"HassGetCurrentTime": {"data": [{"sentences": [text]}]}},
    })
    matcher.local_tables = (*matcher.local_tables, ("fr", competing))
    assert matcher.match(text) == emission
    assert not matcher.response_language_confirmed(text, emission)


def test_language_uncertainty_and_matcher_errors_never_confirm(matcher, monkeypatch):
    text = "wie spät ist es"
    emission = matcher.match(text)
    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    assert not matcher.response_language_confirmed(text, emission)
    monkeypatch.setattr(ha_client, "_loaded", True)
    with patch("utils.hassil_fastpath.recognize", side_effect=RuntimeError("bad pack")):
        assert not matcher.response_language_confirmed(text, emission)


def test_single_language_confirms_local_match_without_discovery(tmp_path, monkeypatch):
    target = tmp_path / "de" / "2025.6.23.yaml"
    target.parent.mkdir()
    shutil.copyfile(FIXTURES / "de/2025.6.23.yaml", target)
    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    matcher = HassilFastPath(["de"], str(tmp_path))
    text = "wie spät ist es"
    assert matcher.response_language_confirmed(text, matcher.match(text))


def test_local_stop_precedes_matcher(matcher):
    assert catch_with_hassil("stop", matcher) == {"actions": [{"intent": "pause", "args": []}]}


def test_brightness_and_unsafe_slot_combinations(matcher):
    assert matcher.match("set the kitchen lights to 50 percent") == {
        "actions": [{"intent": "ha_set_brightness", "args": ["light.kitchen_lights", 50]}],
    }
    assert matcher._action("HassTurnOn", {"area": "kitchen", "domain": "light", "device_class": "garage"}) is None
    assert matcher._action("HassTurnOn", {"name": "light.kitchen_lights", "area": "kitchen"}) is None


@pytest.mark.parametrize("text, alias, entity, intent, args", [
    ("set thermostat to 21 degrees", "thermostat", "climate.thermostat", "ha_set_climate", ["climate.thermostat", 21]),
    ("set bedroom blinds to 50 percent", "bedroom blinds", "cover.blinds", "ha_set_cover_position", ["cover.blinds", 50]),
    ("unlock the front door", "front door", "lock.front", "ha_unlock", ["lock.front"]),
])
def test_device_intent_mappings(matcher, monkeypatch, text, alias, entity, intent, args):
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {alias: [entity]})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {alias: entity})
    assert matcher.match(text) == {"actions": [{"intent": intent, "args": args}]}


def test_names_and_denylist_are_live(matcher, monkeypatch):
    assert matcher.match("turn on the kitchen lights")
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.kitchen_lights"}))
    assert matcher.match("turn on the kitchen lights") is None
    assert matcher.match("allume la lumière de la cuisine") is None
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"kitchen lights": ["light.kitchen_lights", "switch.kitchen"]})
    assert matcher.match("turn on the kitchen lights") is None


def test_area_dispatch_filters_denied_and_service_rechecks(matcher, monkeypatch):
    action = matcher.match("turn on the lights in the kitchen")
    assert action == {"actions": [{"intent": "ha_hassil_area", "args": ["area", "kitchen", "light", "on", None]}]}
    monkeypatch.setattr(ha_client, "_area_entities", lambda *_: ["light.kitchen_lights", "light.blocked"])
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.blocked"}))
    calls = []
    monkeypatch.setattr(ha_client, "_call_service", lambda *args: calls.append(args) or "Done.")
    assert ha_devices.hassil_area("area", "kitchen", "light", "on") == "Done."
    assert calls[0][2] == ["light.kitchen_lights"]


def test_safe_provisional_and_security_policy(matcher):
    text = "turn on the kitchen lights"
    assert should_commit_provisional(text, catch_with_hassil(text, matcher))
    assert not should_commit_provisional("unlock the door", {"actions": [{"intent": "ha_unlock", "args": ["lock.door"]}]})
    assert not should_commit_provisional("turn on movie night", {"actions": [{"intent": "turn_on", "args": ["scene.movie_night"]}]})


def test_list_service_refuses_newly_denied_entity(monkeypatch):
    monkeypatch.setattr(ha_client, "HA_TOKEN", "token")
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.blocked"}))
    with patch.object(ha_client, "_post") as post:
        result = ha_client._call_service("light", "turn_on", ["light.allowed", "light.blocked"])
    assert "isn't available for voice control" in result
    post.assert_not_called()


def test_hot_reload_is_atomic(tmp_path, matcher):
    from core.assistant import Assistant

    with patch("core.assistant.AudioCapture", return_value=MagicMock()):
        assistant = Assistant(wakeword="hey atticus")
    assistant._tts_backend = "qwen"
    assistant._hassil_fastpath = matcher
    assistant._hassil_intents_dir = str(tmp_path)
    assert assistant.apply_hot_config([{"path": "general.intent_languages", "value": ["de"]}]) == {"general.intent_languages"}
    assert [language for language, _ in assistant._hassil_fastpath.tables] == ["de"]
    previous = assistant._hassil_fastpath
    with patch("server.downloader.ensure_hassil_intents", side_effect=OSError("offline")):
        assert assistant.apply_hot_config([{"path": "general.intent_languages", "value": ["fr"]}]) == set()
    assert assistant._hassil_fastpath is previous
