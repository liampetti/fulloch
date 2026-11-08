"""A2: AgentLoop sets `TurnStats.route` ("regex" / "agent" / "no_llm") so
`GET /status` and the structured per-turn log line can show which path
resolved a turn. Host stubs mirror the pattern in test_llm_openai.py's
`test_agent_loop_degrades_to_regex_on_remote_unreachable` and test_intents.py's
`TestNoLlmReactive` — the minimal set of host attributes `AgentLoop._run`
touches before returning, not a full Assistant.
"""

import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_follow_up import startup_greeting_follow_up  # noqa: E402
from core.agent_loop import _llm_unavailable_label  # noqa: E402
from core.slm import RemoteUnreachable  # noqa: E402
from core.turn_stats import TurnStats  # noqa: E402
from utils.intents import StepKind, StepResult  # noqa: E402


def _host(**overrides):
    base = {
        "llm_enabled": True,
        "_history": [],
        "_history_for": lambda session: [],
        "_turn_local": types.SimpleNamespace(sink=None, tts_active_event=None),
        "_trim_history": lambda: None,
        "_emit_agent_event": lambda *a, **k: None,
        "_compact_completed_turns": lambda: None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_nested_days_between_is_repaired_before_plan_and_dispatch():
    import json

    import core.agent_loop as al

    expression = 'days_between("2026-11-08", "2026-12-25")'
    history, events = [], []
    generate = Mock(return_value=json.dumps({"actions": [{"intent": "calculate", "args": [expression]}]}))
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None,
        replan_stall_cache=[], web_search_stall_cache=[], note_write_stall_cache=[],
        play_chunks=lambda *a, **k: None, _note_llm_remote_status=lambda *a: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
    )
    assert al.AgentLoop(host, source="text").run("how many days till christmas") == (
        "There are 47 days between those two dates."
    )
    assert generate.call_count == 1
    action = {"intent": "days_between", "args": ["2026-11-08", "2026-12-25"]}
    assert next(payload for kind, payload in events if kind == "plan")["actions"] == [action]
    assert next(payload for kind, payload in events if kind == "step") == action
    assert any(entry.get("name") == "days_between" for entry in history)


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("language,prompt,expected,excluded", [
    ("en", "what time is it?", "2:30", "November"),
    ("en", "what day is it?", "November", "2:30"),
    ("de", "wie spät ist es", "14:30", "November"),
    ("de", "welcher tag ist heute", "November", "14:30"),
    ("fr", "quelle heure est il", "14:30", "novembre"),
    ("fr", "quelle est la date", "novembre", "14:30"),
])
def test_clock_fastpaths_answer_only_requested_part_without_model(
    monkeypatch, source, language, prompt, expected, excluded,
):
    from datetime import datetime, timezone

    import core.agent_loop as al
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    root = Path(__file__).parent.parent
    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools._local_time, "now", lambda: datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc))
    generate = Mock(side_effect=AssertionError("fast path must not call model"))
    host = _host(
        _hassil_fastpath=(HassilFastPath([language], str(root / "tests/fixtures/hassil"))
                          if language != "en" else None),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
        _generate_with_context_recovery=generate,
        web_search_stall_cache=[], note_write_stall_cache=[],
    )
    result = al.AgentLoop(host, source=source).run(prompt)
    assert expected in result
    assert excluded not in result
    generate.assert_not_called()


def test_explicit_translation_uses_agent_after_clock_tool(monkeypatch):
    from datetime import datetime, timezone

    import core.agent_loop as al
    from tools import time_tools

    monkeypatch.setattr(time_tools._local_time, "now", lambda: datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc))
    history = []
    generate = Mock(side_effect=[
        '{"actions": [{"intent": "get_current_time", "args": []}]}',
        '{"reply": "Il est quatorze heures trente."}',
    ])
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None,
        replan_stall_cache=[], web_search_stall_cache=[], note_write_stall_cache=[],
        play_chunks=lambda *a, **k: None, _note_llm_remote_status=lambda *a: None,
        _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
    )
    result = al.AgentLoop(host, source="text").run("what time is it? in french")
    assert result == "Il est 14:30."
    assert generate.call_count == 2
    assert any(item.get("name") == "get_current_time" for item in history)


@pytest.mark.parametrize("source", ["text", "voice"])
def test_guessed_translated_time_must_use_clock_and_cannot_change_its_value(monkeypatch, source):
    from datetime import datetime, timedelta, timezone

    import core.agent_loop as al
    from tools import time_tools

    now = datetime(2026, 11, 8, 18, 43, tzinfo=timezone(timedelta(hours=11)))
    monkeypatch.setattr(time_tools._local_time, "now", lambda: now)
    dispatch = Mock(wraps=al.intents.handle_action)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    history, events = [], []
    generate = Mock(side_effect=[
        '{"reply": "Il est 14:41. Dimanche 8 novembre 2026."}',
        '{"actions": [{"intent": "get_current_time", "args": []}]}',
        '{"reply": "Il est 14:41. Dimanche 8 novembre 2026."}',
    ])
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
    )

    result = al.AgentLoop(host, source=source).run("what time is it in french")

    assert result == "Il est 18:43."
    assert generate.call_count == 3
    dispatch.assert_called_once_with({"intent": "get_current_time", "args": []})
    assert any(item.get("name") == "live_fact_guard" for item in history)
    assert all("14:41" not in str(item.get("content", "")) for item in history)
    assert any(kind == "step" and payload["intent"] == "get_current_time" for kind, payload in events)
    if source == "voice":
        assert host._turn_local.response_language == "fr"


def test_guessed_time_is_not_returned_when_model_ignores_tool_guard():
    import core.agent_loop as al

    history = []
    generate = Mock(return_value='{"reply": "Il est 14:41."}')
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        play_chunks=lambda *a, **k: None, _note_llm_remote_status=lambda *a: None,
        _play_random_ack=lambda *a, **k: None, _generate_with_context_recovery=generate,
        _record_spoken=lambda _: None, _history_for=lambda _: history,
    )

    result = al.AgentLoop(host, source="text").run("what time is it in french")

    assert "couldn't verify" in result
    assert "14:41" not in result
    assert generate.call_count == 2


def test_guessed_calculation_gets_a_tool_replan_before_translation(monkeypatch):
    import core.agent_loop as al

    dispatch = Mock(wraps=al.intents.handle_action)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    history = []
    generate = Mock(side_effect=[
        '{"reply": "Cela fait cinq."}',
        '{"actions": [{"intent": "calculate", "args": ["2+2"]}]}',
        '{"reply": "Cela fait quatre."}',
    ])
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
    )

    assert al.AgentLoop(host, source="text").run("what is 2 plus 2 in French?") == "Cela fait quatre."
    dispatch.assert_called_once_with({"intent": "calculate", "args": ["2+2"]})
    assert any(item.get("name") == "live_fact_guard" for item in history)
    assert not any("Cela fait cinq" in str(item.get("content", "")) for item in history)


def test_pre_execution_bundled_math_guess_does_not_override_tool_result():
    import core.agent_loop as al

    history = []
    generate = Mock(return_value=(
        '{"actions": [{"intent": "calculate", "args": ["2+2"]}], "reply": "It is five."}'
    ))
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _generate_with_context_recovery=generate,
        _record_spoken=lambda _: None, _history_for=lambda _: history,
    )

    result = al.AgentLoop(host, source="text").run("what is 2 plus 2?")

    assert result == "That comes to 4."
    assert generate.call_count == 1


def test_unavailable_clock_tool_returns_truthful_fallback(monkeypatch):
    import core.agent_loop as al

    availability = al.tool_registry.is_available
    monkeypatch.setattr(al.tool_registry, "is_available", lambda name: name != "get_current_time" and availability(name))
    generate = Mock(return_value='{"reply": "Il est 14:41."}')
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        play_chunks=lambda *a, **k: None, _note_llm_remote_status=lambda *a: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
    )

    result = al.AgentLoop(host, source="text").run("what time is it in french")

    assert "couldn't verify" in result
    assert "14:41" not in result
    assert generate.call_count == 1


def test_translation_without_llm_does_not_use_laya():
    import core.agent_loop as al

    router = Mock()
    host = _host(
        llm_enabled=False, laya_enabled=True, laya_router=types.SimpleNamespace(route=router),
        _speak_no_ai_fallback=lambda *args, **kwargs: "AI unavailable",
    )
    assert al.AgentLoop(host, source="text").run("what time is it? in french") == "AI unavailable"
    router.assert_not_called()


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("prompt,action,observation,reply,target", [
    ("what is 2 plus 2 in French?", {"intent": "calculate", "args": ["2+2"]},
     "That comes to 4.", "Cela fait 4.", "fr"),
    ("what is the weather in German?", {"intent": "get_weather_forecast", "args": []},
     "It is sunny and 20 degrees.", "Es ist sonnig und 20 Grad.", "de"),
    ("what date is next Tuesday in French?", {"intent": "date_of", "args": ["Tuesday", "next"]},
     "Tuesday 10 November 2026.", "Cela fait 4.", "fr"),
])
def test_translation_composes_read_only_tool_results_once(
    monkeypatch, source, prompt, action, observation, reply, target,
):
    import json

    import core.agent_loop as al

    history = []
    generate = Mock(side_effect=[json.dumps({"actions": [action]}), json.dumps({"reply": reply})])
    dispatch = Mock(return_value=observation)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda _: True)
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
    )
    assert al.AgentLoop(host, source=source).run(prompt) == reply
    assert generate.call_count == 2
    dispatch.assert_called_once_with(action)
    assert any(item.get("content") == observation for item in history)
    if source == "voice":
        from core.text_utils import spoken_for_tts

        assert host._turn_local.response_language == target
        assert spoken_for_tts(reply, target) == (
            "Cela fait quatre." if target == "fr" else "Es ist sonnig und zwanzig Grad."
        )


@pytest.mark.parametrize("observation", ["Kitchen lights turned on.", "Error: cannot reach Home Assistant."])
def test_translation_of_action_preserves_actual_outcome_without_repeat(monkeypatch, observation):
    import json

    import core.agent_loop as al

    action = {"intent": "turn_on", "args": ["kitchen"]}
    dispatch = Mock(return_value=observation)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda _: True)
    generate = Mock(return_value=json.dumps({"actions": [action]}))
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _generate_with_context_recovery=generate,
        _record_spoken=lambda _: None,
    )
    result = al.AgentLoop(host, source="text").run("turn on the lights and answer in French")
    assert observation in result
    assert "couldn't provide" in result
    dispatch.assert_called_once_with(action)
    assert generate.call_count == 1


def test_translation_does_not_replace_reactive_action_failure_with_model_claim(monkeypatch):
    import json

    import core.agent_loop as al

    action = {"intent": "turn_on", "args": ["kitchen"]}
    dispatch = Mock(return_value="Reactive question: The light was not found. Ask for the correct light.")
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda _: True)
    generate = Mock(side_effect=[
        json.dumps({"actions": [action]}), json.dumps({"reply": "La lumière est allumée."}),
    ])
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
    )
    result = al.AgentLoop(host, source="text").run("turn on the lights and answer in French")
    assert "The light was not found" in result
    assert "La lumière est allumée" not in result
    assert "couldn't provide" in result
    dispatch.assert_called_once_with(action)


def test_translation_fallback_after_read_does_not_return_untranslated_result(monkeypatch):
    import core.agent_loop as al

    history = []
    generate = Mock(side_effect=[
        '{"actions": [{"intent": "calculate", "args": ["2+2"]}]}',
        RemoteUnreachable("offline"),
    ])
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
        _speak_llm_error_fallback=lambda *a, **k: "Model unavailable",
    )
    assert al.AgentLoop(host, source="text").run("what is 2 plus 2 in French?") == "Model unavailable"
    assert any(item.get("name") == "calculate" for item in history)


def test_empty_translation_does_not_leak_the_untranslated_result():
    import core.agent_loop as al

    history = []
    generate = Mock(side_effect=[
        '{"actions": [{"intent": "calculate", "args": ["2+2"]}]}',
        '{"reply": ""}',
    ])
    host = _host(
        grammar=object(), wakeword_name="Fulloch", tts_session=None, replan_stall_cache=[],
        web_search_stall_cache=[], note_write_stall_cache=[], play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda *a: None, _play_random_ack=lambda *a, **k: None,
        _generate_with_context_recovery=generate, _record_spoken=lambda _: None,
        _history_for=lambda _: history,
    )
    result = al.AgentLoop(host, source="text").run("what is 2 plus 2 in French?")
    assert "couldn't provide" in result
    assert "That comes to" not in result


def test_dashboard_localized_time_uses_verified_outcome_and_preserves_raw(monkeypatch):
    from datetime import datetime, timezone

    import core.agent_loop as al
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools._local_time, "now", lambda: datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc))
    root = Path(__file__).parent.parent
    matcher = HassilFastPath(["de"], str(root / "tests/fixtures/hassil"))
    history = []
    events = []
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=matcher,
        _hassil_intents_dir=str(root / "data/intents"),
        _history_for=lambda _: history,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
        _record_spoken=lambda _: None,
    )
    result = al.AgentLoop(host, source="text").run("wie spät ist es")
    assert result == "Es ist 14:30 Uhr."
    observation = next(payload for kind, payload in events if kind == "observation")
    assert observation["result"] == result
    assert "zweitausendsechsundzwanzig" in observation["raw_result"]
    assert any(entry.get("content") == observation["raw_result"] for entry in history)


def test_llm_enabled_dashboard_fastpath_localizes_without_model_or_second_dispatch(monkeypatch):
    from datetime import datetime, timezone

    import core.agent_loop as al
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools._local_time, "now", lambda: datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc))
    dispatch = Mock(wraps=al.intents.handle_action)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    root = Path(__file__).parent.parent
    events, history = [], []
    generate = Mock(side_effect=AssertionError("fast-path must not call the model"))
    host = _host(
        llm_enabled=True,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _history_for=lambda _: history,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
        _record_spoken=lambda _: None,
        _generate_with_context_recovery=generate,
        web_search_stall_cache=[], note_write_stall_cache=[],
    )
    result = al.AgentLoop(host, source="text").run("wie spät ist es")
    assert result == "Es ist 14:30 Uhr."
    dispatch.assert_called_once()
    generate.assert_not_called()
    observation = next(payload for kind, payload in events if kind == "observation")
    assert observation["result"] == result
    assert "zweitausendsechsundzwanzig" in observation["raw_result"]
    assert any(entry.get("content") == observation["raw_result"] for entry in history)


def test_voice_turn_allows_localized_text_for_selected_tts(monkeypatch):
    from datetime import datetime, timezone

    import core.agent_loop as al
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools._local_time, "now", lambda: datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc))
    root = Path(__file__).parent.parent
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
    )
    for backend in ("pocket-tts-pytorch", "pocket-tts-onnx", "custom-model"):
        host._tts_backend = backend
        result = al.AgentLoop(host, source="voice").run("wie spät ist es")
        assert result == "Es ist 14:30 Uhr."


def test_dashboard_does_not_localize_unverified_failure(monkeypatch):
    import core.agent_loop as al
    from tools import ha_client
    from tools.tool_registry import OutcomeText
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(al.intents, "handle_action", lambda _: OutcomeText("Clock failed.", success=False, code="clock_failed"))
    root = Path(__file__).parent.parent
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
    )
    assert al.AgentLoop(host, source="text").run("wie spät ist es") == "Clock failed."


def test_dashboard_light_localization_requires_actual_success(monkeypatch):
    import requests

    import core.agent_loop as al
    from tools import ha_client, ha_devices
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "token")
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {"küchenlicht": "light.kitchen"})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"küchenlicht": ["light.kitchen"]})
    monkeypatch.setattr(ha_client, "_AREA_MAP", {})
    monkeypatch.setattr(ha_client, "_FLOOR_MAP", {})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    post = Mock()
    monkeypatch.setattr(ha_client, "_post", post)
    monkeypatch.setattr(al.intents, "handle_action", lambda action: ha_devices.turn_on(action["args"][0]))
    root = Path(__file__).parent.parent
    events = []
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
    )
    text = "schalte das küchenlicht ein"
    result = al.AgentLoop(host, source="text").run(text)
    assert result == "Befehl zum Einschalten der Beleuchtung gesendet."
    post.assert_called_once()
    observation = next(payload for kind, payload in events if kind == "observation")
    assert observation["result"] == result
    assert observation["raw_result"] != result

    post.side_effect = requests.exceptions.ConnectionError()
    result = al.AgentLoop(host, source="text").run(text)
    assert result == "Ich konnte Home Assistant nicht erreichen."


def test_dashboard_light_denied_after_match_never_localizes_success(monkeypatch):
    import core.agent_loop as al
    from tools import ha_client, ha_devices
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "token")
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {"küchenlicht": "light.kitchen"})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"küchenlicht": ["light.kitchen"]})
    monkeypatch.setattr(ha_client, "_AREA_MAP", {})
    monkeypatch.setattr(ha_client, "_FLOOR_MAP", {})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    post = Mock()
    monkeypatch.setattr(ha_client, "_post", post)

    def dispatch(action):
        monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"light.kitchen"}))
        return ha_devices.turn_on(action["args"][0])

    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    root = Path(__file__).parent.parent
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
    )
    result = al.AgentLoop(host, source="text").run("schalte das küchenlicht ein")
    assert "nicht für die Sprachsteuerung verfügbar" in result
    assert "kein Befehl gesendet" in result
    post.assert_not_called()


def test_dashboard_timer_localizes_and_retains_language_for_completion(monkeypatch):
    import core.agent_loop as al
    from core.satellite_context import current_timer_response_language
    from tests.test_time_tools import _Timer
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools.threading, "Timer", _Timer)
    monkeypatch.setattr(time_tools, "active_timers", {})
    spoken = []
    monkeypatch.setattr(time_tools, "_speak_proactive", lambda text, **_: spoken.append(text))
    root = Path(__file__).parent.parent
    events = []
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
    )
    result = al.AgentLoop(host, source="text").run("stelle einen timer auf fünf minuten")
    assert result == "Timer für 5 Minuten gestartet."
    timer = next(iter(time_tools.active_timers.values()))
    assert timer.response_language == "de"
    assert current_timer_response_language.get() == "en"
    observation = next(payload for kind, payload in events if kind == "observation")
    assert observation["result"] == result
    assert observation["raw_result"] == "Timer started for 5 minutes"
    assert observation["artifact"]["type"] == "timers"
    timer.function(*timer.args)
    assert spoken == ["Dein Timer ist abgelaufen."]


def test_voice_timer_uses_selected_language_with_any_tts_backend(monkeypatch):
    import core.agent_loop as al
    from tests.test_time_tools import _Timer
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools.threading, "Timer", _Timer)
    monkeypatch.setattr(time_tools, "active_timers", {})
    spoken = []
    monkeypatch.setattr(time_tools, "_speak_proactive", lambda text, **_: spoken.append(text))
    root = Path(__file__).parent.parent
    host = _host(
        llm_enabled=False,
        _tts_backend="custom-model",
        _hassil_fastpath=HassilFastPath(["fr"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
    )
    result = al.AgentLoop(host, source="voice").run("minuteur de cinq minutes")
    assert result == "Minuteur lancé pour 5 minutes."
    timer = next(iter(time_tools.active_timers.values()))
    assert timer.response_language == "fr"
    timer.function(*timer.args)
    assert spoken == ["Votre minuteur est terminé."]


def test_dashboard_timer_cancellation_localizes_only_after_actual_cancel(monkeypatch):
    import core.agent_loop as al
    from tests.test_time_tools import _Timer
    from tools import ha_client, time_tools
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    monkeypatch.setattr(time_tools.threading, "Timer", _Timer)
    monkeypatch.setattr(time_tools, "active_timers", {})
    time_tools.start_countdown("60")
    timer = next(iter(time_tools.active_timers.values()))
    root = Path(__file__).parent.parent
    events = []
    host = _host(
        llm_enabled=False,
        _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
        _hassil_intents_dir=str(root / "data/intents"),
        _record_spoken=lambda _: None,
        _emit_agent_event=lambda kind, payload, **_: events.append((kind, payload)),
    )
    text = "stoppe den timer"
    assert al.AgentLoop(host, source="text").run(text) == "Timer abgebrochen."
    assert timer.cancelled
    assert not time_tools.active_timers
    observation = next(payload for kind, payload in events if kind == "observation")
    assert "cancelled" in observation["raw_result"]
    assert observation["artifact"] == {"type": "timers", "timers": []}
    assert al.AgentLoop(host, source="text").run(text) == "Es sind keine Timer aktiv."


def test_german_brightness_fastpath_localizes_actual_service_value(monkeypatch):
    import core.agent_loop as al
    from tools import ha_client, ha_devices
    from utils.hassil_fastpath import HassilFastPath

    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "token")
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {"küchenlicht": "light.kitchen"})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"küchenlicht": ["light.kitchen"]})
    monkeypatch.setattr(ha_client, "_AREA_MAP", {})
    monkeypatch.setattr(ha_client, "_FLOOR_MAP", {})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    post = Mock()
    monkeypatch.setattr(ha_client, "_post", post)
    monkeypatch.setattr(al.intents, "handle_action", lambda action: ha_devices.set_ha_brightness(*action["args"]))
    root = Path(__file__).parent.parent
    host = _host(llm_enabled=False,
                 _hassil_fastpath=HassilFastPath(["de"], str(root / "tests/fixtures/hassil")),
                 _hassil_intents_dir=str(root / "data/intents"), _record_spoken=lambda _: None)
    result = al.AgentLoop(host, source="text").run("dimme küchenlicht auf 25 prozent")
    assert result == "Die Helligkeit wurde auf 25 Prozent eingestellt."
    assert post.call_args.kwargs["json"]["brightness"] == 63


def test_local_llm_failure_label_names_llama_server():
    assert _llm_unavailable_label(_host(llm_backend="local")) == "Local llama-server unavailable"


def test_startup_greeting_response_is_selected_for_an_explicit_follow_up():
    response = "Storms can make frogs rain from the sky."
    assert startup_greeting_follow_up(response, "Could you tell me more about that topic?") == response


def test_startup_greeting_response_is_discarded_from_history_for_unrelated_first_turn(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda _prompt: {"reply": "The kitchen lights are on."})
    history = []
    host = _host(
        _history_for=lambda _session: history,
        _record_spoken=lambda _text: None,
        _startup_greeting_response="Storms can make frogs rain from the sky.",
    )

    assert al.AgentLoop(host, source="text").run("Turn on the kitchen lights.") == "The kitchen lights are on."
    assert history == [
        {"role": "user", "content": "Turn on the kitchen lights."},
        {"role": "assistant", "content": '{"reply": "The kitchen lights are on."}'},
    ]
    assert host._startup_greeting_response is None


def test_startup_greeting_response_enters_history_for_explicit_first_follow_up(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return '{"reply": "A weather explanation."}'

    history = []
    host = _host(
        grammar=object(),
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[],
        play_chunks=lambda *args, **kwargs: None,
        _note_llm_remote_status=lambda *args: None,
        _generate_with_context_recovery=generate,
        _record_spoken=lambda _text: None,
        _history_for=lambda _session: history,
        _startup_greeting_response="Storms can make frogs rain from the sky.",
    )

    assert al.AgentLoop(host, source="text").run("Tell me more about that.") == "A weather explanation."
    assert captured["history"] == [
        {"role": "assistant", "content": "Storms can make frogs rain from the sky."},
        {"role": "user", "content": "Tell me more about that."},
    ]
    assert host._startup_greeting_response is None


def test_route_is_regex_when_catchall_reply_resolves_without_slm(monkeypatch):
    import core.agent_loop as al

    # catchAll's own {"reply": ...} shape (e.g. the note-delete refusal) is a
    # complete turn resolved on iteration 0 — the SLM is never called.
    monkeypatch.setattr(al, "catchAll", lambda prompt: {"reply": "no can do"})
    stats = TurnStats()
    on_slm_start = Mock()
    loop = al.AgentLoop(_host(), source="text", stats=stats, on_slm_start=on_slm_start)
    out = loop.run("delete my note")
    assert out == "no can do"
    assert stats.route == "regex"
    on_slm_start.assert_not_called()


def test_route_is_no_llm_when_llm_disabled(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    spoken = {}
    host = _host(
        llm_enabled=False,
        _generate_with_context_recovery=Mock(),
        _record_spoken=lambda s: spoken.__setitem__("said", s),
        _speak_no_ai_fallback=lambda session, source, satellite_id=None: (
            spoken.__setitem__("said", "NO_AI") or "NO_AI"
        ),
    )
    stats = TurnStats()
    loop = al.AgentLoop(host, session=None, source="text", stats=stats)
    out = loop.run("tell me a joke")
    assert out == "NO_AI"
    assert stats.route == "no_llm"
    host._generate_with_context_recovery.assert_not_called()


def test_laya_routes_only_after_a_regex_miss(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: prompt)
    captured = {}
    monkeypatch.setattr(
        al.AgentLoop,
        "_run_without_llm",
        lambda self, prompt, emission: captured.__setitem__("emission", emission) or "paused",
    )
    router = Mock(return_value={"actions": [{"intent": "pause", "args": []}]})
    host = _host(
        llm_enabled=False,
        laya_enabled=True,
        laya_router=types.SimpleNamespace(route=router),
    )
    stats = TurnStats()

    assert al.AgentLoop(host, source="text", stats=stats).run("please stop playback") == "paused"
    router.assert_called_once_with("please stop playback")
    assert captured["emission"] == {"actions": [{"intent": "pause", "args": []}]}
    assert stats.route == "laya"
    assert stats.laya_seconds is not None


def test_laya_enabled_keeps_regex_telemetry_as_regex(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: {"reply": "done"})
    router = Mock()
    stats = TurnStats()
    host = _host(
        llm_enabled=False,
        laya_enabled=True,
        laya_router=types.SimpleNamespace(route=router),
        _record_spoken=lambda _text: None,
    )

    assert al.AgentLoop(host, source="text", stats=stats).run("a regex command") == "done"
    router.assert_not_called()
    assert stats.route == "regex"
    assert stats.laya_seconds is None


def test_follow_up_is_routed_to_agent_when_no_report_is_available(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    history = []
    generate = Mock(return_value='{"reply": "What would you like me to investigate?"}')
    host = _host(
        grammar=None,
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[],
        play_chunks=Mock(),
        _note_llm_remote_status=Mock(),
        _generate_with_context_recovery=generate,
        _history_for=lambda satellite: history,
        consume_completed_thinking_report=Mock(return_value=None),
    )

    assert al.AgentLoop(host, source="text").run("yes") == "What would you like me to investigate?"
    host.consume_completed_thinking_report.assert_called_once_with(None)
    generate.assert_called_once()
    assert generate.call_args.kwargs["history"] == [{"role": "user", "content": "yes"}]


def test_turn_them_back_on_uses_last_successful_off_target():
    import json

    from core.agent_loop import _contextual_turn_onoff
    from core.turn_history import compact

    history = [
        {"role": "user", "content": "turn off downstairs office lights"},
        {"role": "assistant", "content": json.dumps({
            "actions": [{"intent": "turn_off", "args": ["downstairs office lights"]}],
        })},
        {"role": "tool", "name": "turn_off", "content": "downstairs office off"},
        {"role": "user", "content": "turn them back on"},
    ]
    compact(history, 160)

    assert _contextual_turn_onoff("turn them back on", history) == {
        "actions": [{"intent": "turn_on", "args": ["downstairs office lights"]}],
    }


def test_turn_them_back_on_does_not_reuse_failed_or_wrong_direction_action():
    import json

    from core.agent_loop import _contextual_turn_onoff

    def history(intent, result):
        return [
            {"role": "user", "content": "previous command"},
            {"role": "assistant", "content": json.dumps({
                "actions": [{"intent": intent, "args": ["office lights"]}],
            })},
            {"role": "tool", "name": intent, "content": result},
            {"role": "user", "content": "turn them back on"},
        ]

    assert _contextual_turn_onoff(
        "turn them back on", history("turn_off", "Reactive question: couldn't find it"),
    ) is None
    assert _contextual_turn_onoff(
        "turn them back on", history("turn_on", "office lights on"),
    ) is None
    assert _contextual_turn_onoff(
        "turn them back on", history("turn_off", "office lights off"),
    ) == {"actions": [{"intent": "turn_on", "args": ["office lights"]}]}


def test_pronoun_fast_path_rejects_ambiguous_empty_or_denied_observations():
    import json

    from core.agent_loop import _contextual_turn_onoff

    history = [
        {"role": "user", "content": "turn off lights"},
        {"role": "assistant", "content": json.dumps({
            "actions": [{"intent": "turn_off", "args": {"entity": "lights"}}],
        })},
        {"role": "tool", "name": "turn_off", "content": "Lights off in Office", "ha_area": "office"},
        {"role": "user", "content": "turn them back on"},
    ]
    assert _contextual_turn_onoff("turn them back on", history, ha_area="office") is not None
    assert _contextual_turn_onoff("turn them back on", history, ha_area="kitchen") is None
    for result in ("", "Sorry, those lights aren't available for voice control."):
        history[2]["content"] = result
        assert _contextual_turn_onoff("turn them back on", history, ha_area="office") is None
    history[2]["content"] = "Lights off in Office"
    history.insert(3, {"role": "tool", "name": "turn_off", "content": "Kitchen off"})
    assert _contextual_turn_onoff("turn them back on", history, ha_area="office") is None


def test_topicless_web_search_reuses_prior_question(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: prompt)
    captured = {}
    monkeypatch.setattr(
        al.AgentLoop,
        "_run_without_llm",
        lambda self, prompt, emission: captured.setdefault("emission", emission) or "searched",
    )
    history = [{"role": "user", "content": "did we ever find out what that supermarket staple is"}]
    host = _host(
        llm_enabled=False,
        _history_for=lambda session: history,
    )
    loop = al.AgentLoop(host, session=None, source="text")
    loop.run("can you search the internet")

    assert captured["emission"] == {
        "actions": [
            {
                "intent": "external_information",
                "args": ["did we ever find out what that supermarket staple is"],
            }
        ]
    }


def test_challenged_assistant_claim_is_explicitly_disputed_in_next_prompt(monkeypatch):
    import json

    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda _prompt: None)
    history = [{
        "role": "assistant",
        "content": "The green C means the current leader.",
    }]
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return json.dumps({"reply": "I'll verify that against current sources."})

    host = _host(
        grammar=None, wakeword_name="Fulloch", tts_session=None,
        replan_stall_cache=[], play_chunks=Mock(), _note_llm_remote_status=Mock(),
        _generate_with_context_recovery=generate, _record_spoken=Mock(),
        _history_for=lambda _session: history,
    )

    assert al.AgentLoop(host, source="text").run(
        "check web for answer rather than making things up",
    ) == "I'll verify that against current sources."
    prompt_history = captured["history"]
    marker = next(item for item in prompt_history if item.get("role") == "system")
    assert "DISPUTED PRIOR ASSISTANT CLAIM" in marker["content"]
    assert "do not use it as evidence" in marker["content"]
    assert "green C means the current leader" in marker["content"]


def test_route_is_agent_once_slm_call_starts(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)

    def _raise(**k):
        raise RemoteUnreachable("down")

    host = _host(
        grammar=object(),
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[],  # empty -> progress watchdog is a no-op
        play_chunks=lambda *a, **k: None,
        _note_llm_remote_status=lambda ok, error="": None,
        _speak_llm_error_fallback=lambda session, source, satellite_id=None: "LLM SERVER UNREACHABLE",
        _generate_with_context_recovery=_raise,
    )
    stats = TurnStats()
    loop = al.AgentLoop(host, session=None, source="text", stats=stats)
    out = loop.run("tell me a story about the sea")
    # Route is set the moment the SLM call is attempted, before its outcome
    # is known — even though this particular call fails over to a fallback.
    assert out == "LLM SERVER UNREACHABLE"
    assert stats.route == "agent"


def test_replan_ack_finishes_before_the_final_reply_returns(monkeypatch):
    """A late ack `tts_end` must not deactivate the final browser TTS stream."""
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda _intent: True)
    monkeypatch.setattr(
        al.intents,
        "classify_step",
        lambda _result: StepResult(StepKind.REACTIVE, "try again", False),
    )
    monkeypatch.setattr(al.intents, "handle_action", lambda _action: "ignored")
    emissions = iter([
        '{"actions": [{"intent": "get_entity_state", "args": ["lamp"]}]}',
        '{"reply": "The final answer."}',
    ])
    ack_finished = []

    def play_ack(*_args, **_kwargs):
        time.sleep(0.03)
        ack_finished.append(True)

    host = _host(
        grammar=object(),
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[([], 24000)],
        _play_random_ack=play_ack,
        play_chunks=lambda *args, **kwargs: None,
        _note_llm_remote_status=lambda *args: None,
        _generate_with_context_recovery=lambda **kwargs: next(emissions),
        _record_spoken=lambda _text: None,
    )

    assert al.AgentLoop(host, source="voice").run("look it up") == "The final answer."
    assert ack_finished == [True]


def test_replan_can_use_a_calendar_date_in_a_follow_up_calculation(monkeypatch):
    """An intermediate lookup observation must allow a dependent tool action."""
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    actions = []
    emissions = iter(
        [
            '{"actions": [{"intent": "find_calendar_event", "args": ["Perth flight"]}]}',
            '{"actions": [{"intent": "days_between", "args": ["2026-08-25", "2026-10-03"]}]}',
        ]
    )

    def handle_action(action):
        actions.append(action)
        if action["intent"] == "find_calendar_event":
            return "Reactive question: Calendar search found: Perth flight: start date 2026-10-03 (all day)."
        return "There are 39 days between those two dates."

    history = []
    host = _host(
        grammar=object(),
        wakeword_name="Fulloch",
        tts_session=None,
        replan_stall_cache=[],
        _play_random_ack=lambda *args, **kwargs: None,
        play_chunks=lambda *args, **kwargs: None,
        _note_llm_remote_status=lambda *args: None,
        _generate_with_context_recovery=lambda **kwargs: next(emissions),
        _record_spoken=lambda _text: None,
        _history_for=lambda _session: history,
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda _intent: True)
    monkeypatch.setattr(al.intents, "handle_action", handle_action)

    out = al.AgentLoop(host, source="text").run("how many days until we fly to Perth?")

    assert out == "There are 39 days between those two dates."
    assert actions == [
        {"intent": "find_calendar_event", "args": ["Perth flight"]},
        {"intent": "days_between", "args": ["2026-08-25", "2026-10-03"]},
    ]


def test_route_stays_none_without_a_stats_object():
    import core.agent_loop as al

    # stats=None (e.g. a caller that doesn't track turn stats) must not raise.
    host = _host(llm_enabled=False, _speak_no_ai_fallback=lambda *a, **k: "NO_AI")
    loop = al.AgentLoop(host, session=None, source="text", stats=None)
    assert loop.run("tell me a joke") == "NO_AI"


def test_remote_cooldown_skips_generation_and_uses_outage_fallback(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    calls = {"generate": 0, "fallback": 0}
    host = _host(
        _remote_llm_retry_blocked=lambda: True,
        _remote_llm_unavailable_fallback=lambda *args, **kwargs: (
            calls.__setitem__("fallback", calls["fallback"] + 1) or "REMOTE UNAVAILABLE"
        ),
        _generate_with_context_recovery=lambda **kwargs: (
            calls.__setitem__("generate", calls["generate"] + 1)
            or (_ for _ in ()).throw(AssertionError("cooldown must skip generation"))
        ),
    )

    out = al.AgentLoop(host, session=None, source="voice").run("tell me a story")

    assert out == "REMOTE UNAVAILABLE"
    assert calls == {"generate": 0, "fallback": 1}


def test_remote_cooldown_keeps_regex_reply_available(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(al, "catchAll", lambda prompt: {"reply": "The time is noon."})
    host = _host(
        _remote_llm_retry_blocked=lambda: True,
        _remote_llm_unavailable_fallback=lambda *args, **kwargs: "REMOTE UNAVAILABLE",
    )

    out = al.AgentLoop(host, session=None, source="voice").run("what time is it")

    assert out == "The time is noon."


def test_non_balanced_satellite_message_uses_agent_but_keeps_regex_fallback(monkeypatch):
    import core.agent_loop as al

    emission = {"actions": [{"intent": "send_satellite_message", "args": ["kitchen", "Dinner is ready"]}]}
    monkeypatch.setattr(al, "catchAll", lambda prompt: emission)
    captured = {}
    host = _host(
        personality="wry",
        _remote_llm_retry_blocked=lambda: True,
        _remote_llm_unavailable_fallback=lambda *args, **kwargs: "REMOTE UNAVAILABLE",
    )

    def fallback(self, prompt, fallback_emission, unavailable_fallback=None):
        captured["emission"] = fallback_emission
        return "queued"

    monkeypatch.setattr(
        al.AgentLoop,
        "_run_without_llm",
        fallback,
    )

    out = al.AgentLoop(host, session=None, source="text").run("tell kitchen dinner is ready")

    assert out == "queued"
    assert captured["emission"] == emission


def test_music_search_ack_plays_before_regex_spotify_dispatch(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(
        al,
        "catchAll",
        lambda prompt: {"actions": [{"intent": "play_song", "args": ["Take Five"]}]},
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: name == "play_song")
    monkeypatch.setattr(al.intents, "handle_action", lambda action: "Playing Take Five.")
    calls = []
    host = _host(
        music_search_stall_cache=[("pcm", 16000)],
        _play_random_ack=lambda session, cache: calls.append((session, cache)),
        _record_spoken=lambda spoken: None,
    )

    out = al.AgentLoop(host, session=None, source="voice").run("play Take Five")

    assert out == "Playing Take Five."
    assert calls == [(None, host.music_search_stall_cache)]


def test_wry_announcement_fallback_changes_an_echoed_llm_message():
    from core.agent_emission import apply_announcement_fallback

    raw = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "dinner is ready"]}]}
    emission = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "dinner is ready"]}]}

    apply_announcement_fallback("wry", "tell kitchen dinner is ready", raw, emission)

    assert emission["actions"][0]["args"][1] == "dinner is ready. The kitchen's patience has been noted."


def test_announcement_fallback_keeps_verbatim_and_safety_messages_literal():
    from core.agent_emission import apply_announcement_fallback

    raw = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "take your medication"]}]}
    emission = {"actions": [{"intent": "send_satellite_message", "args": ["Kitchen", "take your medication"]}]}

    apply_announcement_fallback("wry", "tell kitchen verbatim: take your medication", raw, emission)

    assert emission == raw


def test_delivery_is_not_spoken_for_lock_actions():
    from core.agent_emission import can_speak_delivery
    from tools.capabilities import native_access_class

    assert not can_speak_delivery("wry", [{"intent": "ha_lock", "args": ["front door"]}], access_class=native_access_class)
    assert can_speak_delivery("wry", [{"intent": "turn_on", "args": ["lamp"]}], access_class=native_access_class)


def test_delivery_is_not_spoken_for_read_actions():
    from core.agent_emission import can_speak_delivery
    from tools.capabilities import native_access_class

    assert not can_speak_delivery(
        "wry", [{"intent": "days_between", "args": ["today", "2026-10-03"]}],
        access_class=native_access_class,
    )


def test_execution_results_replace_model_delivery(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(
        al,
        "catchAll",
        lambda prompt: {
            "actions": [
                {"intent": "turn_on", "args": ["lights"]},
                {"intent": "play_song", "args": ["music"]},
            ],
            "delivery": "The lights are on and music is playing.",
        },
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: name in {"turn_on", "play_song"})
    monkeypatch.setattr(
        al.intents,
        "handle_action",
        lambda action: "Raw result for " + action["intent"],
    )
    spoken = []
    host = _host(personality="wry", _record_spoken=spoken.append)

    out = al.AgentLoop(host, session=None, source="text").run("turn on lights and play music")

    assert out == "Raw result for turn_on Raw result for play_song"
    assert spoken == [out]


def test_read_action_delivery_cannot_hide_the_tool_result(monkeypatch):
    import core.agent_loop as al

    monkeypatch.setattr(
        al,
        "catchAll",
        lambda _prompt: {
            "actions": [{"intent": "days_between", "args": ["2026-08-26", "2026-10-03"]}],
            "delivery": "Calculating the days until departure.",
        },
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: name == "days_between")
    monkeypatch.setattr(al.intents, "handle_action", lambda _action: "There are 38 days between those two dates.")
    spoken = []
    host = _host(personality="wry", _record_spoken=spoken.append)

    out = al.AgentLoop(host, session=None, source="text").run("how many days until departure")

    assert out == "There are 38 days between those two dates."
    assert spoken == [out]
