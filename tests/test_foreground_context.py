"""Narrow prompt profiles never replace contextual or mixed-action routing."""

import json

import pytest

from core import agent_loop as al
from core.foreground_context import is_standalone_research_request, select_prompt_profile
from tests.test_agent_plan import research_tool
from tests.test_assistant_web_search import _emissions
from tests.test_assistant_web_search import turn as turn


def profile(prompt, prior=(), **kwargs):
    return select_prompt_profile(prompt, [*prior, {"role": "user", "content": prompt}], **kwargs)


@pytest.mark.parametrize("prompt", ["Hello!", "How are you?", "Good morning", "Thanks", "Tell me a joke"])
def test_standalone_social_turns_are_lean(prompt):
    assert profile(prompt) == "conversation"


@pytest.mark.parametrize("prompt", [
    "Research the latest local AI trends for running capable models at home",
    "Please investigate home AI privacy and performance",
])
def test_explicit_standalone_research_is_a_scoped_handoff(prompt):
    assert profile(prompt) == "research"


@pytest.mark.parametrize("prompt", [
    "can you find latest info on trending local models with 9B parameters? Keep sources recent",
    "Please look up current information on home AI models",
])
def test_current_discovery_request_is_a_scoped_handoff(prompt):
    assert is_standalone_research_request(prompt)
    assert profile(prompt) == "research"


@pytest.mark.parametrize("prompt", [
    "Yes", "Go ahead", "Do it", "Turn those lights off", "Look it up",
    "Research that", "Research AI again", "Research AI and save the report",
    "Research GPUs then play music", "Hello and turn on the lights",
    "Investigate our previous plan", "What's the weather in London?",
    "Can you find my keys?", "Search for recent information and save it to my notes",
])
def test_ambiguous_contextual_and_mixed_turns_keep_full_routing(prompt):
    assert profile(prompt) == "full"


@pytest.mark.parametrize("prior", [
    {"role": "assistant", "content": '{"reply":"Shall I dim the lights?"}'},
    {"role": "assistant", "content": '{"actions":[{"intent":"turn_on"}]}'},
    {"role": "tool", "content": "Reactive question: Which city?"},
    {"role": "system", "content": "Open task: confirm the route"},
])
def test_pending_work_and_clarifications_keep_context_even_for_social_turns(prior):
    assert profile("Hello", [prior]) == "full"


def test_replans_never_discard_tool_results_or_plan_state():
    assert profile("Hello", plan_active=True) == "full"


def test_social_agent_call_omits_old_history_without_mutating_canonical_turns(turn):
    turn.history.extend([
        {"role": "user", "content": "old household request"},
        {"role": "tool", "name": "turn_on", "content": "old household outcome"},
        {"role": "assistant", "content": '{"reply":"The old task is complete."}'},
    ])
    _emissions(turn, {"reply": "Hello!"})
    assert turn.loop.run("Hello") == "Hello!"
    request = turn.host._generate_with_context_recovery.call_args.kwargs
    assert request["history"] == [{"role": "user", "content": "Hello"}]
    assert "Available tools:" not in request["system_prompt"]
    assert turn.history[0]["content"] == "old household request"
    assert turn.stats.prompt_profile == "conversation"


def test_confirmation_agent_call_keeps_previous_proposal(turn):
    proposal = {"role": "assistant", "content": '{"reply":"Shall I save the findings?"}'}
    turn.history.append(proposal)
    _emissions(turn, {"reply": "Please specify which note."})
    turn.loop.run("Yes")
    request = turn.host._generate_with_context_recovery.call_args.kwargs
    assert proposal in request["history"]
    assert "Available tools:" in request["system_prompt"]
    assert turn.stats.prompt_profile == "full"


def test_reply_guard_replan_restores_full_schema_and_current_outcomes(turn):
    _emissions(turn, {"reply": "I've saved the note."}, {"reply": "Hello!"})
    assert turn.loop.run("Hello") == "Hello!"
    first, second = turn.host._generate_with_context_recovery.call_args_list
    assert "Available tools:" not in first.kwargs["system_prompt"]
    assert "Available tools:" in second.kwargs["system_prompt"]
    assert any(item.get("name") == "reply_guard" for item in second.kwargs["history"])


def test_current_request_and_inflight_tool_results_are_not_character_clipped():
    from utils.prompts import assemble_foreground_history

    request = "Research AI. " + "constraint " * 100
    observation = "verified outcome " * 100
    history = [
        {"role": "user", "content": "old " * 1000},
        {"role": "assistant", "content": "old reply " * 1000},
        {"role": "user", "content": request},
        {"role": "tool", "content": observation},
    ]
    compact = assemble_foreground_history(history)
    assert compact[-2]["content"] == request
    assert compact[-1]["content"] == observation
    assert len(compact[0]["content"]) < len(history[0]["content"])


def test_scoped_research_agent_still_dispatches_original_request_to_worker(turn, monkeypatch):
    assistant = research_tool(monkeypatch)
    question = "Research the latest local AI trends for running capable models at home"
    _emissions(turn, {"actions": [{"intent": "deep_think", "args": [question]}]})
    assert "I'll look into that" in turn.loop.run(question)
    assert assistant.run_thinking_task.call_args.args == (question,)
    assistant.run_thinking_task.assert_called_once()
    request = turn.host._generate_with_context_recovery.call_args.kwargs
    assert "ha_set_climate" not in request["system_prompt"]
    assert "emit a deep_think action now" in request["system_prompt"]
    assert turn.stats.prompt_profile == "research"
    assert al.tool_registry.get_tool("deep_think") is not None


RESEARCH_PROMISE = (
    "Happy to dig into that. This is a substantial research task, so I'll queue "
    "a bounded background investigation and report back when it's ready."
)


@pytest.mark.parametrize("emission", [
    {"reply": RESEARCH_PROMISE},
    RESEARCH_PROMISE,
    {"actions": [{"intent": "reply", "args": [RESEARCH_PROMISE]}]},
    {"reply": "I've queued the investigation."},
])
def test_research_reply_without_dispatch_is_replaced_with_real_handoff(turn, monkeypatch, emission):
    assistant = research_tool(monkeypatch)
    question = "Research the latest local AI trends for running capable models at home"
    _emissions(turn, emission)

    answer = turn.loop.run(question)

    assistant.run_thinking_task.assert_called_once()
    queued = assistant.run_thinking_task.call_args
    assert queued.args == (question,)
    assert queued.kwargs["origin_satellite_id"] is None
    assert queued.kwargs["origin_source"] == "conversation"
    assert queued.kwargs["handoff"]["original_request"] == question
    assert answer == "I'll look into that and let you know when I'm done."
    assert turn.stats.tool_dispatches == 1
    turn.host._generate_with_context_recovery.assert_called_once()
    plan_events = [call.args[1] for call in turn.host._emit_agent_event.call_args_list if call.args[0] == "plan"]
    assert plan_events == [{"actions": [{"intent": "deep_think", "args": [question]}]}]
    assert not any(RESEARCH_PROMISE in entry["content"] for entry in turn.history)
    assert turn.history[-1] == {"role": "tool", "name": "deep_think", "content": answer}
    turn.host._record_spoken.assert_called_once_with(answer)


def test_research_guard_reports_queue_failure_instead_of_model_promise(turn, monkeypatch):
    assistant = research_tool(monkeypatch)
    assistant.run_thinking_task.side_effect = RuntimeError("The research queue is full.")
    _emissions(turn, {"reply": RESEARCH_PROMISE})

    answer = turn.loop.run("Research capable home AI models")

    assert "The research queue is full." in answer
    assert "I'll queue" not in answer and "I'll look into" not in answer
    assistant.run_thinking_task.assert_called_once()


def test_research_guard_also_applies_when_prior_question_requires_full_profile(turn, monkeypatch):
    assistant = research_tool(monkeypatch)
    turn.history.append({"role": "assistant", "content": '{"reply":"How can I help?"}'})
    question = "Research the latest local AI trends for running capable models at home"
    _emissions(turn, {"reply": RESEARCH_PROMISE})

    assert turn.loop.run(question) == "I'll look into that and let you know when I'm done."
    assert turn.stats.prompt_profile == "full"
    assistant.run_thinking_task.assert_called_once()
    assert assistant.run_thinking_task.call_args.args == (question,)


@pytest.mark.parametrize("registered", [False, True])
def test_research_guard_cannot_claim_queued_work_when_tool_unavailable(turn, monkeypatch, registered):
    assistant = research_tool(monkeypatch) if registered else None
    if registered:
        al.tool_registry._availability["deep_think"] = lambda: False
    _emissions(turn, {"reply": RESEARCH_PROMISE})

    answer = turn.loop.run("Research capable home AI models")

    assert "unavailable" in answer.lower()
    assert "haven't queued" in answer
    assert turn.stats.tool_dispatches == 0
    if assistant:
        assistant.run_thinking_task.assert_not_called()


@pytest.mark.parametrize("question", ["Research that", "Research AI and save the report", "Hello"])
def test_research_guard_does_not_force_contextual_mixed_or_social_requests(turn, monkeypatch, question):
    assistant = research_tool(monkeypatch)
    _emissions(turn, {"reply": "Which topic would you like?"})
    assert turn.loop.run(question) == "Which topic would you like?"
    assistant.run_thinking_task.assert_not_called()


def test_cancelled_research_generation_never_queues_work(turn, monkeypatch):
    assistant = research_tool(monkeypatch)

    def generate(**_kwargs):
        turn.session.stop()
        return json.dumps({"reply": RESEARCH_PROMISE})

    turn.host._generate_with_context_recovery.side_effect = generate
    assert turn.loop.run("Research capable home AI models") == ""
    assistant.run_thinking_task.assert_not_called()
