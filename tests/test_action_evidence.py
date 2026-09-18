"""Reply-boundary regressions: no model judge and no extra call on normal answers."""

import json
from unittest.mock import Mock

import pytest

from core.action_evidence import FALLBACK, ActionEvidence, claims_execution
from tests.test_agent_loop_route import _host
from utils.intents import classify_step


@pytest.mark.parametrize("text", [
    "I turned the Living Room Lamp on.", "I've opened the blind.",
    "The lamp has been switched on.", "Done!", "Opened the blind.",
    "I saved that to your notes.", "We have already set the timer.",
    "Done, the living room lamp is on.", "The Living Room Lamp is now on.",
])
def test_detects_completion_claims(text):
    assert claims_execution(text)


@pytest.mark.parametrize("text", [
    "A ripe banana is usually yellow.", "I couldn't turn the lamp on.",
    "I haven't opened the blind.", "Would you like me to turn it on?",
    "You can turn it on in Home Assistant.", "I'll turn the lamp on.",
])
def test_ordinary_answers_and_failure_reports_pass(text):
    assert not claims_execution(text)


def test_normal_routing_kind_is_not_proof_of_success():
    evidence = ActionEvidence()
    evidence.record({"intent": "get_current_time"}, classify_step("12:00"))
    assert not evidence.records
    evidence.record({"intent": "turn_on", "args": ["lamp"]},
                    classify_step("I couldn't reach Home Assistant."))
    assert evidence.response() == "I couldn't reach Home Assistant."


def make_loop(monkeypatch, emissions, results=None):
    import core.agent_loop as al

    history = []
    generate = Mock(side_effect=[json.dumps(item) for item in emissions])
    dispatch = Mock(side_effect=results or [])
    monkeypatch.setattr(al, "catchAll", lambda prompt: None)
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: True)
    host = _host(
        grammar=None, wakeword_name="Fulloch", tts_session=None,
        replan_stall_cache=[], web_search_stall_cache=[], note_write_stall_cache=[],
        play_chunks=Mock(), _play_random_ack=Mock(), _record_spoken=Mock(),
        _note_llm_remote_status=Mock(), _generate_with_context_recovery=generate,
        _history_for=lambda satellite: history,
    )
    return al.AgentLoop(host, source="text"), generate, dispatch, history


def test_banana_answer_needs_one_call_and_no_tool(monkeypatch):
    loop, generate, dispatch, _ = make_loop(monkeypatch, [{"reply": "Bananas are yellow."}])
    assert loop.run("What colour is a banana?") == "Bananas are yellow."
    assert generate.call_count == 1
    dispatch.assert_not_called()


def test_unsupported_claim_repairs_into_real_action(monkeypatch):
    claim = "I turned the living room lamp on."
    loop, generate, dispatch, history = make_loop(monkeypatch, [
        {"reply": claim},
        {"actions": [{"intent": "turn_on", "args": ["living room lamp"]}]},
    ], ["Living room lamp on"])
    assert loop.run("Could we get some light from the living room lamp?") == "Living room lamp on"
    assert generate.call_count == 2
    assert dispatch.call_count == 1
    assert not any(claim in entry["content"] for entry in history)


def test_repeated_fabrication_has_one_repair_then_fallback(monkeypatch):
    loop, generate, dispatch, history = make_loop(monkeypatch, [
        {"reply": "I've opened the blind."}, {"reply": "Done!"},
    ])
    assert loop.run("Open the blind") == FALLBACK
    assert generate.call_count == 2
    dispatch.assert_not_called()
    assert json.loads(history[-1]["content"])["reply"] == FALLBACK


def test_failed_tool_cannot_back_success_claim(monkeypatch):
    loop, _, dispatch, _ = make_loop(monkeypatch, [
        {"actions": [{"intent": "turn_on", "args": ["lamp"]}]},
        {"reply": "I turned the lamp on."},
    ], ["Reactive question: I couldn't find that lamp."])
    assert loop.run("Turn on the lamp") == "I couldn't find that lamp."
    assert dispatch.call_count == 1


def test_failure_can_replan_and_retry(monkeypatch):
    loop, _, dispatch, _ = make_loop(monkeypatch, [
        {"actions": [{"intent": "turn_on", "args": ["lamp"]}]},
        {"actions": [{"intent": "turn_on", "args": ["living room lamp"]}]},
    ], ["Reactive question: I couldn't find that lamp.", "Living room lamp on"])
    answer = loop.run("Turn on the lamp")
    assert "Living room lamp on" in answer
    assert dispatch.call_count == 2


def test_bundled_claim_cannot_replace_plain_failure(monkeypatch):
    loop, _, _, history = make_loop(monkeypatch, [{
        "actions": [{"intent": "turn_on", "args": ["lamp"]}],
        "reply": "I opened the bedroom blind.",
    }], ["I couldn't reach Home Assistant."])
    assert loop.run("Turn on the lamp") == "I couldn't reach Home Assistant."
    assert not any("opened the bedroom" in entry["content"] for entry in history)


def test_old_history_does_not_count_as_execution(monkeypatch):
    loop, generate, dispatch, history = make_loop(monkeypatch, [
        {"reply": "I turned the lamp on."}, {"reply": "I haven't changed the lamp yet."},
    ])
    history.append({"role": "assistant", "content": "I turned the lamp on."})
    assert loop.run("Turn it on again") == "I haven't changed the lamp yet."
    assert generate.call_count == 2
    dispatch.assert_not_called()


def test_unrelated_lookup_cannot_authorize_bundled_action_claim(monkeypatch):
    loop, _, dispatch, _ = make_loop(monkeypatch, [{
        "actions": [{"intent": "get_current_time", "args": []}],
        "reply": "I turned the lamp on.",
    }], ["It is noon."])
    assert loop.run("What time is it?") == "It is noon."
    assert dispatch.call_count == 1


def test_mixed_action_and_lookup_keeps_both_results(monkeypatch):
    loop, _, _, _ = make_loop(monkeypatch, [{"actions": [
        {"intent": "turn_on", "args": ["lamp"]},
        {"intent": "get_current_time", "args": []},
    ]}], ["Lamp on.", "It is noon."])
    assert loop.run("Turn on the lamp and tell me the time") == "Lamp on. It is noon."


def test_success_for_one_target_cannot_authorize_another(monkeypatch):
    loop, generate, dispatch, _ = make_loop(monkeypatch, [{
        "actions": [{"intent": "turn_on", "args": ["kitchen lamp"]}],
        "reply": "I opened the bedroom blind.",
    }], ["Kitchen lamp on."])
    assert loop.run("Turn on the kitchen lamp") == "Kitchen lamp on."
    assert dispatch.call_count == generate.call_count == 1


def test_empty_lookup_output_does_not_default_to_done(monkeypatch):
    loop, _, _, _ = make_loop(monkeypatch, [{
        "actions": [{"intent": "get_current_time", "args": []}],
        "reply": "Done!",
    }], [""])
    assert loop.run("What time is it?") == FALLBACK
