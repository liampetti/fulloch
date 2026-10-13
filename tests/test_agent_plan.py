"""Plan continuation and tool-owned evidence handoff regression tests."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from core import agent_loop as al
from core.agent_emission import route_deep_think_only_tools
from core.agent_plan import TurnPlan, action_key
from core.agent_search import TurnSearch
from core.slm import RemoteUnreachable
from tests.test_assistant_web_search import _emissions, _search
from tests.test_assistant_web_search import turn as turn
from tools import thinking
from tools.thinking_context import get_handoff
from tools.tool_registry import ThinkingResult, ToolRegistry
from utils.intents import classify_step


def research_tool(monkeypatch):
    job = {"id": "job-1", "status": "QUEUED"}
    assistant = SimpleNamespace(run_thinking_task=Mock(return_value=job),
                                active_thinking_task=lambda: job)
    monkeypatch.setattr(thinking, "get_current_assistant", lambda: assistant)
    al.tool_registry.register_tool(thinking.deep_think, name="deep_think", aliases=["investigate"])
    return assistant


@pytest.mark.parametrize("name,result,args", [
    ("external_information", "User question: raw results", ["topic"]),
    ("read_note", "October rainfall averaged 80 mm in the retained station records.", ["weather"]),
    ("get_weather_forecast", "Reactive question: Which city?", []),
    ("get_weather_forecast", None, []),
    ("get_weather_forecast", ThinkingResult("Provider unavailable.", status="unavailable",
                                          scope="Weather provider failed."), []),
])
def test_every_replan_kind_preserves_independent_work_and_research(turn, monkeypatch, name, result, args):
    assistant = research_tool(monkeypatch)
    lookup = Mock(return_value=result)
    if name == "external_information":
        turn.search.return_value = result
        lookup = turn.search
    else:
        al.tool_registry.register_tool(lookup, name=name)
    pause = Mock(return_value="Music paused.")
    al.tool_registry.register_tool(pause, name="pause")
    turn.catch.return_value = {"actions": [
        {"intent": name, "args": args},
        {"intent": "pause", "args": [], "depends_on": []},
        {"intent": "investigate", "args": ["seasonal weather trends"]},
    ]}
    prompt = "Give me the forecast, pause music, and investigate seasonal weather trends."
    answer = turn.loop.run(prompt)
    assert "Music paused." in answer
    assert "I'll look into that" in answer
    lookup.assert_called_once()
    pause.assert_called_once()
    turn.host._generate_with_context_recovery.assert_not_called()
    assert assistant.run_thinking_task.call_args.args == (prompt,)
    handoff = assistant.run_thinking_task.call_args.kwargs["handoff"]
    assert handoff["original_request"] == prompt
    assert handoff["research_task"] == "seasonal weather trends"
    assert handoff["observations"][0]["tool"] == name
    if name == "external_information":
        assert handoff["observations"][0]["evidence"]["text"] == result
    assert get_handoff() is None


@pytest.mark.parametrize("search_status", ["evidence", "rejected", "unavailable"])
def test_quote_and_original_web_evidence_reach_worker_not_summary(turn, monkeypatch, search_status):
    assistant = research_tool(monkeypatch)
    quote = {"symbol": "NVDA", "price": "USD229.28", "numeric_price": 229.28,
             "currency": "USD", "exchange": "NASDAQ", "quoted_at": "2026-10-09T20:30:00-04:00",
             "retrieved_at": "2026-10-10T02:33:41+00:00", "cache_mode": "cache allowed",
             "source_url": "https://example.com/quote"}
    al.tool_registry.register_tool(lambda symbol: ThinkingResult(
        "NVIDIA's latest quoted price is USD229.28.", evidence={"quote": quote}, scope="One market snapshot.",
        artifact={"type": "finance_quote", "quote": quote}), name="get_finance_quote")
    source = {"url": "https://example.com/earnings", "evidence": "Revenue increased 10 percent in the reported quarter.",
              "published_at": "2026-09-01", "eligible": True}
    sources = [source] if search_status == "evidence" else []
    turn.search.return_value = ThinkingResult(
        "User question: NVIDIA results\nOriginal earnings source text.", status=search_status,
        evidence={"query": "NVIDIA results", "sources": sources}, scope="Bounded earnings search.",
        artifact={"type": "web_research", "sources": sources})
    turn.host._summarise_search_result.return_value = "A model-written summary, not source evidence."
    turn.catch.return_value = {"actions": [
        {"intent": "get_finance_quote", "args": ["NVDA"]},
        _search("NVIDIA results"),
        {"intent": "deep_think", "args": ["risks"]},
    ]}
    prompt = "Give me a deep NVIDIA briefing: quote, results, recent news, and uncertainty."
    answer = turn.loop.run(prompt)
    assert "USD229.28" in answer and "I'll look into that" in answer
    observations = assistant.run_thinking_task.call_args.kwargs["handoff"]["observations"]
    assert assistant.run_thinking_task.call_args.kwargs["handoff"]["research_task"] == "risks"
    assert observations[0]["evidence"]["quote"] == quote
    assert observations[1]["evidence"]["sources"] == sources
    assert observations[1]["status"] == search_status
    assert observations[1]["artifact"]["sources"] == sources
    assert observations[1]["retrieved_at"]
    turn.host._generate_with_context_recovery.assert_not_called()


def test_premature_reply_cannot_drop_a_dependent_save(turn):
    turn.catch.return_value = {"actions": [_search(),
        {"intent": "append_to_today", "args": ["Unresearched guess"], "depends_on": [0]}]}
    _emissions(turn, {"reply": "Here is the news."}, {"actions": [
        {"intent": "append_to_today", "args": ["Grounded findings."], "step_id": 2}]})
    answer = turn.loop.run("Find news and save the findings.")
    assert "Grounded findings." in answer and "Saved the note." in answer
    turn.save.assert_called_once_with("Grounded findings.")
    assert turn.host._generate_with_context_recovery.call_count == 2
    progress = json.loads(turn.host._generate_with_context_recovery.call_args.kwargs["history"][-1]["content"])
    assert progress["steps"][1]["status"] == "pending"
    assert progress["steps"][1]["needs_revision"]


def test_completed_mutation_is_not_repeated_on_replan(turn):
    light = Mock(return_value="Lamp on.")
    al.tool_registry.register_tool(lambda target: light(target), name="turn_on", aliases=["light_on"])
    turn.catch.return_value = {"actions": [{"intent": "turn_on", "args": ["lamp"]}, _search()]}
    _emissions(turn, {"actions": [{"intent": "light_on", "args": {"target": "lamp"}}]})
    answer = turn.loop.run("Turn on the lamp and find the news.")
    assert "Lamp on." in answer and "Grounded findings." in answer
    light.assert_called_once_with("lamp")
    assert turn.stats.tool_dispatches == 2


def test_distinct_identical_actions_in_original_plan_are_not_suppressed(turn):
    timer = Mock(side_effect=["First timer started.", "Second timer started."])
    al.tool_registry.register_tool(lambda seconds: timer(seconds), name="start_countdown")
    turn.catch.return_value = {"actions": [
        {"intent": "start_countdown", "args": [60]}, {"intent": "start_countdown", "args": [60]}]}
    answer = turn.loop.run("Start two one-minute timers.")
    assert "First timer started." in answer and "Second timer started." in answer
    assert timer.call_count == 2


def test_uncertain_mutation_is_not_retried_blindly(turn):
    light = Mock(side_effect=RuntimeError("lost confirmation after dispatch"))
    al.tool_registry.register_tool(lambda target: light(target), name="toggle")
    turn.catch.return_value = {"actions": [{"intent": "toggle", "args": ["lamp"]}]}
    _emissions(turn, {"actions": [{"intent": "toggle", "args": ["lamp"]}]},
               {"reply": "I couldn't confirm the lamp state."})
    answer = turn.loop.run("Toggle the lamp.")
    assert "couldn't confirm" in answer and "Not completed: toggle." in answer
    light.assert_called_once_with("lamp")


def test_explicit_independent_tasks_after_research_are_not_dropped(turn, monkeypatch):
    assistant = research_tool(monkeypatch)
    pause = Mock(return_value="Music paused.")
    al.tool_registry.register_tool(pause, name="pause")
    turn.catch.return_value = {"actions": [
        {"intent": "deep_think", "args": ["weather"]},
        {"intent": "append_to_today", "args": ["Guessed report"]},
        {"intent": "pause", "args": [], "depends_on": []}]}
    answer = turn.loop.run("Research weather, save its report when ready, and pause music.")
    assert "I'll look into that" in answer and "Music paused." in answer
    assert "Not completed: append to today." in answer
    turn.save.assert_not_called()
    pause.assert_called_once()
    assistant.run_thinking_task.assert_called_once()
    turn.host._generate_with_context_recovery.assert_not_called()


@pytest.mark.parametrize("last_lookup,args", [
    ("get_conversation_history", ["2026-10-12"]),
    ("read_note", ["race context"]),
])
def test_research_acknowledgement_omits_unexecuted_context_lookups(turn, monkeypatch, last_lookup, args):
    assistant = research_tool(monkeypatch)
    history = Mock(return_value="Earlier conversation.")
    note = Mock(return_value="Saved context.")
    al.tool_registry.register_tool(history, name="get_conversation_history")
    al.tool_registry.register_tool(note, name="read_note")
    turn.catch.return_value = None
    _emissions(turn, {"actions": [
        {"intent": "deep_think", "args": ["next Formula 1 race"]},
        {"intent": "get_conversation_history", "args": ["2026-10-11"]},
        {"intent": last_lookup, "args": args},
    ]})

    prompt = "Give me a deep analysis on the next Formula 1 race coming up."
    answer = turn.loop.run(prompt)

    assert answer == "I'll look into that and let you know when I'm done."
    history.assert_not_called()
    note.assert_not_called()
    assistant.run_thinking_task.assert_called_once()
    handoff = assistant.run_thinking_task.call_args.kwargs["handoff"]
    assert handoff["original_request"] == prompt
    assert len(handoff["plan"]["steps"]) == 3
    assert all(step["status"] == "pending" for step in handoff["plan"]["steps"][1:])


@pytest.mark.parametrize("queued", [False, True])
def test_suppressing_context_lookup_notice_does_not_change_plan_state(queued):
    registry = ToolRegistry()
    registry.register_tool(lambda query: "", name="deep_think")
    registry.register_tool(lambda date: "", name="get_conversation_history")
    registry.register_tool(lambda content: "", name="append_to_today")
    plan = TurnPlan("Research and save the report", registry, lambda name: "read")
    research, history, save = plan.stage([
        {"intent": "deep_think", "args": ["topic"]},
        {"intent": "get_conversation_history", "args": ["2026-10-12"]},
        {"intent": "append_to_today", "args": ["unverified placeholder"]},
    ])
    if queued:
        plan.record(research, classify_step("I'll look into that and let you know when I'm done."))
    history.status = save.status = "blocked"
    history.needs_revision = save.needs_revision = True
    before = plan.progress()

    answer = plan.finish(include_results=True)

    assert ("get conversation history" not in answer) is queued
    assert "append to today" in answer
    assert plan.progress() == before


def test_executed_lookup_failure_is_not_hidden_by_research_acknowledgement():
    registry = ToolRegistry()
    registry.register_tool(lambda query: "", name="deep_think")
    registry.register_tool(lambda date: "", name="get_conversation_history")
    plan = TurnPlan("Research and check past conversations", registry, lambda name: "read")
    research, history = plan.stage([
        {"intent": "deep_think", "args": ["topic"]},
        {"intent": "get_conversation_history", "args": ["2026-10-12"], "depends_on": []},
    ])
    plan.record(research, classify_step("I'll look into that and let you know when I'm done."))
    plan.record(history, classify_step(None))

    assert "get conversation history" in plan.finish(include_results=True)


def test_generation_failure_keeps_results_and_reports_remaining_work(turn):
    turn.catch.return_value = {"actions": [_search(),
        {"intent": "append_to_today", "args": ["Unresearched guess"], "depends_on": [0]}]}
    turn.host._generate_with_context_recovery.side_effect = RuntimeError("backend error")
    answer = turn.loop.run("Find news and save it.")
    assert "Grounded findings." in answer and "Not completed: append to today." in answer
    assert "language model failed" in answer
    turn.save.assert_not_called()


def test_remote_outage_after_a_model_plan_preserves_findings_and_pending_goals(turn):
    turn.catch.return_value = None
    turn.host._speak_llm_error_fallback = Mock(return_value="Language model unavailable.")
    turn.host._generate_with_context_recovery.side_effect = [
        json.dumps({"actions": [_search(), {"intent": "append_to_today", "args": ["Guess"], "depends_on": [0]}]}),
        RemoteUnreachable("backend down"),
    ]
    answer = turn.loop.run("Find news and save the actual findings.")
    assert "Grounded findings." in answer and "Not completed: append to today." in answer
    turn.search.assert_called_once()
    turn.save.assert_not_called()


def test_replan_can_replace_a_pending_goal_with_an_alternative_tool(turn):
    lookup = Mock(return_value="Note results from the alternative source.")
    al.tool_registry.register_tool(lambda name: lookup(name), name="read_note")
    turn.catch.return_value = {"actions": [_search(), _search("more detail")]}
    _emissions(turn, {"actions": [{"intent": "read_note", "args": ["news"], "step_id": 2}]},
               {"reply": "The web and saved note agree on the news."})
    assert turn.loop.run("Look up the news and its details.") == "The web and saved note agree on the news."
    lookup.assert_called_once_with("news")
    turn.search.assert_called_once()


def test_each_distinct_raw_lookup_gets_a_composing_replan(turn):
    lookup = Mock(side_effect=["First raw note record.", "Second raw note record."])
    al.tool_registry.register_tool(lambda name: lookup(name), name="read_note")
    turn.catch.return_value = {"actions": [
        {"intent": "read_note", "args": ["first"]}, {"intent": "read_note", "args": ["second"]}]}
    _emissions(turn, {"actions": [{"intent": "read_note", "args": ["second"], "step_id": 2}]},
               {"reply": "The two notes agree on the result."})
    assert turn.loop.run("Compare both notes.") == "The two notes agree on the result."
    assert lookup.call_args_list == [call("first"), call("second")]
    assert turn.host._generate_with_context_recovery.call_count == 2


def test_operation_confirmation_does_not_erase_a_composed_lookup_answer(turn):
    al.tool_registry.register_tool(lambda: "Music paused.", name="pause")
    al.tool_registry.register_tool(lambda name: "Raw note: October rainfall was 80 millimetres.", name="read_note")
    turn.catch.return_value = {"actions": [
        {"intent": "pause", "args": []}, {"intent": "read_note", "args": ["weather"]}]}
    _emissions(turn, {"reply": "I've paused the music. The note records October rainfall of 80 millimetres."})
    answer = turn.loop.run("Pause music and tell me what my weather note says.")
    assert answer == "Music paused. The note records October rainfall of 80 millimetres."


def test_clarification_cannot_claim_an_unexecuted_dependent_action(turn):
    al.tool_registry.register_tool(lambda: ThinkingResult("Which city?", status="needs_input",
        scope="Forecast city is missing."), name="get_weather_forecast")
    turn.catch.return_value = {"actions": [
        {"intent": "get_weather_forecast", "args": []},
        {"intent": "append_to_today", "args": ["Guessed forecast"], "depends_on": [0]}]}
    _emissions(turn, {"reply": "I've saved the forecast. Which city should I use?"})
    answer = turn.loop.run("Get the forecast and save it.")
    assert "I've saved" not in answer and "Which city" in answer
    assert "Not completed: append to today." in answer
    turn.save.assert_not_called()


def test_mixed_final_reply_is_not_replaced_by_search_failure(turn):
    al.tool_registry.register_tool(lambda symbol: "NVDA is quoted at USD229.28.", name="get_finance_quote")
    turn.host._summarise_search_result.return_value = "No usable news results."
    turn.catch.return_value = {"actions": [{"intent": "get_finance_quote", "args": ["NVDA"]}, _search()]}
    _emissions(turn, {"reply": "NVDA is quoted at USD229.28; I couldn't verify the news."})
    assert turn.loop.run("Get NVIDIA's quote and news.") == "NVDA is quoted at USD229.28; I couldn't verify the news."


@pytest.mark.parametrize("error", [RemoteUnreachable("offline"), RuntimeError("summary failed")])
def test_summary_failure_preserves_research_and_never_repeats_mutation(turn, monkeypatch, error):
    assistant = research_tool(monkeypatch)
    light = Mock(return_value="Lamp off.")
    al.tool_registry.register_tool(light, name="turn_off")
    turn.host._summarise_search_result.side_effect = error
    turn.catch.return_value = {"actions": [
        {"intent": "turn_off", "args": []}, _search(), {"intent": "deep_think", "args": ["research"]}]}
    answer = turn.loop.run("Turn off the lamp and research the news.")
    assert "Lamp off." in answer and "I'll look into that" in answer
    light.assert_called_once()
    turn.search.assert_called_once()
    assistant.run_thinking_task.assert_called_once()


def test_cancellation_after_result_prevents_pending_research(turn, monkeypatch):
    assistant = research_tool(monkeypatch)
    def lookup():
        turn.session.stop()
        return "Forecast retrieved."
    al.tool_registry.register_tool(lookup, name="get_weather_forecast")
    turn.catch.return_value = {"actions": [
        {"intent": "get_weather_forecast", "args": []}, {"intent": "deep_think", "args": ["weather"]}]}
    assert turn.loop.run("Forecast and research.") == ""
    assistant.run_thinking_task.assert_not_called()
    assert get_handoff() is None


def test_cancellation_during_a_failed_generation_stays_silent(turn):
    turn.catch.return_value = None
    turn.host._speak_llm_error_fallback = Mock(return_value="Language model unavailable.")
    def generate(**kwargs):
        turn.session.stop()
        raise RemoteUnreachable("generation cancelled")
    turn.host._generate_with_context_recovery.side_effect = generate
    assert turn.loop.run("Research the news.") == ""
    turn.search.assert_not_called()
    turn.host._speak_llm_error_fallback.assert_not_called()


def test_invalid_tool_metadata_does_not_stop_independent_work(turn):
    al.tool_registry.register_tool(lambda: ThinkingResult("Invalid outcome.", status={},
        scope="Invalid provider status."), name="get_weather_forecast")
    pause = Mock(return_value="Music paused.")
    al.tool_registry.register_tool(pause, name="pause")
    turn.catch.return_value = {"actions": [
        {"intent": "get_weather_forecast", "args": []}, {"intent": "pause", "args": [], "depends_on": []}]}
    _emissions(turn, {"reply": "The forecast provider returned an invalid outcome."})
    answer = turn.loop.run("Get the forecast and pause music.")
    assert "Not completed: get weather forecast." in answer
    pause.assert_called_once()


def test_real_clarification_completes_independent_work_and_reports_pending_dependency(turn):
    al.tool_registry.register_tool(lambda: ThinkingResult("Which city?", status="needs_input",
        scope="Forecast city is missing."), name="get_weather_forecast")
    pause = Mock(return_value="Music paused.")
    al.tool_registry.register_tool(pause, name="pause")
    turn.catch.return_value = {"actions": [
        {"intent": "get_weather_forecast", "args": []},
        {"intent": "pause", "args": [], "depends_on": []},
        {"intent": "append_to_today", "args": ["Unknown forecast"], "depends_on": [0]}]}
    _emissions(turn, {"reply": "Which city should I use before saving the forecast?"})
    answer = turn.loop.run("Pause music, get the forecast, and save it.")
    assert "Music paused." in answer and "Which city" in answer
    assert "Not completed: append to today." in answer
    turn.save.assert_not_called()
    pause.assert_called_once()


def test_no_model_still_completes_independent_steps_after_a_search(turn):
    pause = Mock(return_value="Music paused.")
    al.tool_registry.register_tool(pause, name="pause")
    turn.host._speak_no_ai_fallback = Mock(return_value="No language model.")
    answer = turn.loop._run_without_llm("Find news and pause music.", {"actions": [
        _search(), {"intent": "pause", "args": [], "depends_on": []}]})
    assert "Music paused." in answer
    assert "Not completed: external information." in answer
    pause.assert_called_once()
    turn.host._generate_with_context_recovery.assert_not_called()


def test_no_model_reads_notes_but_never_saves_a_dependent_placeholder(turn):
    al.tool_registry.register_tool(lambda name: "A real note body.", name="read_note")
    turn.host._speak_no_ai_fallback = Mock(return_value="No language model.")
    answer = turn.loop._run_without_llm("Read the note and save a summary.", {"actions": [
        {"intent": "read_note", "args": ["note"]},
        {"intent": "append_to_today", "args": ["Guessed summary"], "depends_on": [0]}]})
    assert "A real note body." in answer and "Not completed: append to today." in answer
    turn.save.assert_not_called()
    turn.host._generate_with_context_recovery.assert_not_called()


def test_progress_handoff_and_cached_sources_are_deep_copied():
    registry = ToolRegistry()
    registry.register_tool(lambda query, fresh=False: "", name="lookup", aliases=["alias"])
    registry.register_tool(lambda query: "", name="deep_think")
    plan = TurnPlan("Research", registry, lambda name: "read")
    step = plan.stage([{"intent": "lookup", "args": ["topic"]}])[0]
    assert action_key(step.action, registry) == action_key(
        {"intent": "alias", "args": {"query": "topic", "fresh": False}}, registry)
    original = classify_step(ThinkingResult("User question: topic", evidence={"sources": [{"evidence": "original"}]},
                                            scope="One source.", artifact={"sources": [{"evidence": "original"}]}))
    search = TurnSearch()
    summarised = search.accept("topic", "Summary", original)
    plan.record(step, summarised)
    progress = plan.progress()
    assert progress["task"] == "Research"
    assert progress["steps"][0]["status"] == "completed"
    assert progress["steps"][0]["outcome_ref"] == 1
    assert progress["outcome_refs"] == [{
        "id": 1, "tool": "lookup", "status": "evidence", "scope": "One source.",
    }]
    handoff = plan.handoff()
    handoff["observations"][0]["evidence"]["sources"][0]["evidence"] = "mutated"
    assert plan.observations[0]["evidence"]["sources"][0]["evidence"] == "original"
    cached = search.cached("topic")
    assert cached.outcome == original.outcome and cached.artifact == original.artifact
    assert classify_step(ThinkingResult("No results.", status="rejected", scope="Empty search.")).should_replan


def test_explicit_step_ids_distinguish_two_pending_calls_to_same_tool():
    registry = ToolRegistry()
    registry.register_tool(lambda name: "", name="read_note")
    plan = TurnPlan("Compare notes", registry, lambda name: "read")
    first, second = plan.stage([{"intent": "read_note", "args": ["one"]},
                                {"intent": "read_note", "args": ["two"]}])
    revised = plan.stage([{"intent": "read_note", "args": ["corrected two"], "step_id": second.id}])[0]
    assert revised is second and first.action["args"] == ["one"]
    saved = deepcopy(plan.handoff())
    plan.stage([{"intent": "read_note", "args": ["new"], "step_id": 99}])
    assert plan.steps[-1].needs_revision
    assert len(saved["plan"]["steps"]) == 2


def test_specialist_routing_keeps_prefixes_and_never_moves_a_save_before_research():
    registry = ToolRegistry()
    for name in ("pause", "search_papers", "get_paper_detail", "append_to_today", "deep_think", "get_finance_quote"):
        registry.register_tool(lambda query="": "", name=name)
    def route(actions, task):
        return route_deep_think_only_tools({"actions": actions}, task,
            registry=registry, requires_deep_think=lambda name: name in {"search_papers", "get_paper_detail"})
    assert route([
        {"intent": "pause", "args": []}, {"intent": "get_finance_quote", "args": ["NVDA"]}],
        "Pause music and investigate whether I should buy NVIDIA.")["actions"] == [
            {"intent": "pause", "args": []},
            {"intent": "deep_think", "args": ["Pause music and investigate whether I should buy NVIDIA."]}]
    routed = route([
        {"intent": "search_papers", "args": ["weather"]},
        {"intent": "get_paper_detail", "args": ["guessed paper"]},
        {"intent": "append_to_today", "args": ["Guessed results"], "depends_on": [1]}], "Research and save results.")
    assert routed["actions"][0]["intent"] == "deep_think"
    assert routed["actions"][1]["intent"] == "append_to_today"
    assert routed["actions"][1]["depends_on"] == [0]
