import time
from pathlib import Path

import pytest

from core.agent_loop import _record_fastpath_light_context
from core.satellite import SatelliteSession
from tools import ha_client
from tools.tool_registry import OperationOutcome
from utils.intents import StepKind, StepResult
from utils.voice_fastpath import VoiceFastPath, fastpath_pattern, load_hassil_supplements

ROOT = Path(__file__).resolve().parents[1] / "data" / "intents"


@pytest.fixture
def rules():
    return VoiceFastPath(["en", "de", "fr"], ROOT)


@pytest.mark.parametrize("text,operation,value", [
    ("brighten them", "brightness_step", 20),
    ("brighton then", "brightness_step", 20),
    ("dim them", "brightness_step", -20),
    ("mach sie heller", "brightness_step", 20),
    ("tamise les", "brightness_step", -20),
    ("45 pour cent", "brightness", 45),
    ("100 prozent", "brightness", 100),
    ("hundred per cent", "brightness", 100),
    ("turn them off", "off", None),
])
def test_followups_match_pack_and_preserve_scope(rules, text, operation, value):
    context = {"scope": "area", "target": "office", "satellite_area": "kitchen", "time": time.monotonic()}
    assert rules.followup(text, context, "kitchen") == {
        "actions": [{"intent": "ha_hassil_area", "args": ["area", "office", "light", operation, value]}],
    }
    assert rules.followup(text, context, "another room") is None
    context["time"] -= 1801
    assert rules.followup(text, context, "kitchen") is None


def test_correction_does_not_override_a_real_named_target(rules, monkeypatch):
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {"slides": "cover.slides"})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {})
    class Matcher:
        def match(self, text):
            raise AssertionError("must not reinterpret a named device")
    assert rules.corrected_match("turn on downstairs office slides", Matcher()) is None


def test_only_verified_successful_light_service_becomes_context():
    sat = SatelliteSession(id="one", ha_area="kitchen")
    action = {"intent": "ha_hassil_area", "args": ["area", "office", "light", "on", None]}
    outcome = OperationOutcome(True, "service_called", {"domain": "light", "entity_id": ["light.one"]})
    _record_fastpath_light_context(sat, "kitchen", action, StepResult(StepKind.NORMAL, "on", False, operation_outcome=outcome))
    assert sat.fastpath_light_context["target"] == "office"
    sat.fastpath_light_context = None
    failed = OperationOutcome(False, "voice_denied", {"domain": "light", "entity_id": ["light.one"]})
    _record_fastpath_light_context(sat, "kitchen", action, StepResult(StepKind.NORMAL, "denied", False, operation_outcome=failed))
    assert sat.fastpath_light_context is None


def test_light_clarification_cross_language_and_negation():
    from core.light_clarification import clarify_light

    sat = SatelliteSession(id="one", ha_area="kitchen")
    sat.fastpath_light_context = {"scope": "entity", "target": "light.table", "satellite_area": "kitchen", "time": time.monotonic()}
    args = (sat, "lumière", "kitchen", ["en", "de", "fr"], str(ROOT))
    assert clarify_light(*args)[0] == "Allumer ou éteindre la lumière ?"
    assert clarify_light(sat, "off", *args[2:])[1] == {"actions": [{"intent": "turn_off", "args": ["light.table"]}]}
    assert clarify_light(sat, "nicht ausmachen", *args[2:]) == (None, None)
    assert clarify_light(sat, "étendre le lumière", *args[2:])[0]
    assert clarify_light(sat, "nicht ausmachen", *args[2:]) == (None, None)


def test_clarification_retains_action_and_explicit_room(monkeypatch):
    from core.light_clarification import clarify_light

    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"kitchen": "Kitchen"})
    sat = SatelliteSession(id="one")
    def ask(text):
        return clarify_light(sat, text, None, ["en", "fr"], str(ROOT))
    assert ask("lumière")[0]
    assert ask("off")[0] == "Dans quelle pièce ?"
    assert ask("kitchen light")[1] == {"actions": [{"intent": "ha_hassil_area", "args": ["area", "kitchen", "light", "off", None]}]}
    assert ask("light")[0]
    assert ask("kitchen")[0] == "Turn the light on or off?"
    assert ask("éteins")[1]["actions"][0]["args"][3] == "off"


def test_relative_brightness_uses_home_assistant_step(monkeypatch):
    from tools.ha_devices import adjust_light_brightness, hassil_area

    calls = []
    monkeypatch.setattr(ha_client, "_resolve_entity", lambda entity, domain=None: "light.table")
    monkeypatch.setattr(ha_client, "_bare_light_area_entities", lambda entity: None, raising=False)
    monkeypatch.setattr("tools.ha_devices._bare_light_area_entities", lambda entity: None)
    monkeypatch.setattr(ha_client, "_call_service", lambda *args, **kwargs: calls.append((args, kwargs)))
    adjust_light_brightness("table", 20)
    assert calls[-1][0][3] == {"brightness_step_pct": 20}
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"office": "Office"})
    monkeypatch.setattr(ha_client, "_area_entities", lambda *args: ["light.table"])
    monkeypatch.setattr(ha_client, "get_denylist", lambda: set())
    hassil_area("area", "office", "light", "brightness_step", -20)
    assert calls[-1][0][3] == {"brightness_step_pct": -20}


def test_light_trace_correlates_text_and_verified_tool_result(monkeypatch):
    from types import SimpleNamespace

    from core.agent_loop import AgentLoop

    events = []
    monkeypatch.setattr("core.agent_loop.telemetry_event", lambda kind, **fields: events.append((kind, fields)))
    loop = AgentLoop(SimpleNamespace(), source="voice", satellite_id="sat", stats=SimpleNamespace(turn_id="turn", route="regex"))
    outcome = OperationOutcome(True, "service_called", {"domain": "light"})
    def run(*args):
        loop._trace_light_action({"intent": "turn_off", "args": ["light.table"]},
                                 StepResult(StepKind.NORMAL, "Done.", False, operation_outcome=outcome))
        return "Done."
    monkeypatch.setattr(loop, "_run", run)
    assert loop.run("turn off table") == "Done."
    kind, trace = events[0]
    assert kind == "light_turn"
    assert (trace["turn_id"], trace["satellite_id"], trace["transcript"], trace["response"]) == (
        "turn", "sat", "turn off table", "Done.")
    assert trace["actions"][0]["success"] is True


def test_hassil_supplements_use_language_specific_yaml(tmp_path):
    target = tmp_path / "de" / "fulloch-fastpath.yaml"
    target.parent.mkdir()
    target.write_text("hassil:\n  local:\n    HassGetCurrentTime: ['test phrase']\n", encoding="utf-8")
    assert load_hassil_supplements("de", tmp_path) == {
        "local": {"HassGetCurrentTime": ["test phrase"]},
    }


def test_all_existing_hassil_supplement_languages_are_bundled():
    for language in ("en", "de", "fr"):
        assert load_hassil_supplements(language, ROOT)


def test_pattern_can_be_customized_in_fastpath_yaml(tmp_path, monkeypatch):
    directory = tmp_path / "data" / "intents" / "en"
    directory.mkdir(parents=True)
    (directory / "fulloch-fastpath.yaml").write_text(
        "patterns:\n  play: '^my-custom-play$'\n", encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    fastpath_pattern.cache_clear()
    try:
        assert fastpath_pattern("play").fullmatch("MY-CUSTOM-PLAY")
        assert not fastpath_pattern("play").fullmatch("play music")
    finally:
        fastpath_pattern.cache_clear()


def test_custom_language_commands_and_pronunciation_repairs(tmp_path, monkeypatch):
    directory = tmp_path / "es"
    directory.mkdir()
    (directory / "fulloch-fastpath.yaml").write_text(
        "patterns:\n  room_lights: '(?P<area>oficina) luces'\n"
        "commands:\n"
        "  - operation: light_on\n"
        "    pattern_key: room_lights\n"
        "corrections:\n"
        "  - pattern: '(?P<area>oficina) (?P<word>lucesita)'\n"
        "    replacement: '\\g<area> luces'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"office": "oficina"})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {})
    rules = VoiceFastPath(["es"], tmp_path)
    assert fastpath_pattern("room_lights", language="es", directory=str(tmp_path)).fullmatch("oficina luces")
    expected = {"actions": [{"intent": "ha_hassil_area", "args": ["area", "office", "light", "on", None]}]}
    assert rules.command("oficina luces", None) == expected

    class Matcher:
        def match(self, text):
            raise AssertionError("repair should use user-defined fast-path command")

    assert rules.corrected_match("oficina lucesita", Matcher()) == expected


@pytest.mark.parametrize("text,operation,area", [
    ("allumer les lumières", "on", "downstairs_office"),
    ("éteindre les lumières", "off", "downstairs_office"),
    ("allume la lumière", "on", "downstairs_office"),
    ("éteins la lumière", "off", "downstairs_office"),
    ("allumer les lumières dans la cuisine", "on", "kitchen"),
    ("éteindre les lumières dans la cuisine", "off", "kitchen"),
])
def test_french_light_commands_route_before_hassil(monkeypatch, text, operation, area):
    from utils.intent_catch import catch_with_hassil

    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"downstairs_office": "Downstairs Office", "kitchen": "cuisine"})

    class Matcher:
        voice_fastpath = VoiceFastPath(["fr"], ROOT)

        def match(self, text):
            raise AssertionError("French light command should not reach HassIL")

    expected = ({"intent": "ha_hassil_area", "args": ["area", area, "light", operation, None]}
                if area == "kitchen" else {"intent": f"turn_{operation}", "args": ["lights"]})
    assert catch_with_hassil(text, Matcher(), voice=True, ha_area="downstairs_office") == {"actions": [expected]}


def test_custom_command_does_not_guess_ambiguous_or_denied_names(tmp_path, monkeypatch):
    directory = tmp_path / "es"
    directory.mkdir()
    (directory / "fulloch-fastpath.yaml").write_text(
        "commands:\n  - {operation: light_on, pattern: '(?P<name>.+) luces'}\n", encoding="utf-8",
    )
    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"oficina": ["light.office", "light.other"]})
    rules = VoiceFastPath(["es"], tmp_path)
    assert rules.command("oficina luces", None) is None
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"oficina": ["light.office"]})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.office"}))
    assert rules.command("oficina luces", None) is None


def test_yaml_routes_are_ordered_and_reject_unknown_tool_mappings(monkeypatch):
    from utils.intent_catch import _configured_routes, extract_stop

    monkeypatch.setattr("utils.voice_fastpath.fastpath_section", lambda _: [
        {"extractor": "stop", "operation": "pause"},
        {"extractor": "stop", "operation": "unlock"},
        {"extractor": "unknown", "operation": "pause"},
    ])
    routes = _configured_routes()
    assert len(routes) == 1
    assert routes[0][0] is extract_stop
    assert routes[0][1](True) == {"intent": "pause", "args": []}


def test_ambiguous_date_and_time_yaml_does_not_guess(tmp_path):
    directory = tmp_path / "es"
    directory.mkdir()
    (directory / "fulloch-fastpath.yaml").write_text(
        "commands:\n  - {operation: time, pattern: consulta}\n"
        "  - {operation: date, pattern: consulta}\n", encoding="utf-8",
    )
    assert VoiceFastPath(["es"], tmp_path).command("consulta", None) is None


@pytest.mark.parametrize("language,text,base,quote", [
    ("de", "15 euro in aud", "EUR", "AUD"),
    ("fr", "15 euros en usd", "EUR", "USD"),
])
def test_currency_names_are_normalized_before_dispatch(language, text, base, quote):
    assert VoiceFastPath([language], ROOT).command(text, None) == {
        "actions": [{"intent": "get_exchange_rate", "args": [base, quote, 15.0]}],
    }
