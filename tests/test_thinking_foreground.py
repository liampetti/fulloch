"""Thinking submission, foreground routing and Assistant callback integration."""

import importlib
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core import assistant as assistant_module
from core.assistant import Assistant
from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus
from tools import thinking


def test_queues_conversational_job(monkeypatch):
    assistant = SimpleNamespace(
        run_thinking_task=MagicMock(return_value={"id": "job-1", "status": "QUEUED"}),
        active_thinking_task=lambda: {"id": "job-1", "status": "QUEUED"},
    )
    monkeypatch.setattr(thinking, "get_current_assistant", lambda: assistant)
    result = thinking.deep_think("research new models")
    assert "look into" in result
    assert assistant.run_thinking_task.call_args.args == ("research new models",)
    assert assistant.run_thinking_task.call_args.kwargs["origin_source"] == "conversation"


def test_is_exposed_without_a_thinking_config_block():
    from tools.tool_registry import tool_registry

    assert tool_registry.is_available("deep_think") is True


def test_oversized_slot_count_clamps_to_two():
    with patch("core.assistant.AudioCapture"):
        assistant = Assistant(wakeword="hey atticus", thinking={"server_slots": 3})
    assert assistant.thinking_enabled is True
    assert assistant.thinking_server_slots == 2


def test_planning_worker_receives_saved_facts_as_job_context(monkeypatch):
    monkeypatch.setattr(assistant_module.notes, "recall_facts", lambda: "I live in Sydney.")
    assistant = Assistant.__new__(Assistant)
    assistant._history = [{"role": "user", "content": "Compare heating options"}]
    assistant.thinking_jobs = MagicMock()
    assistant.thinking_jobs.submit.return_value = "job-1"
    assistant.thinking_jobs.status.return_value = {"id": "job-1", "status": "QUEUED"}
    assert assistant.run_thinking_task("Investigate") == {"id": "job-1", "status": "QUEUED"}
    assistant.thinking_jobs.submit.assert_called_once_with(
        "Investigate",
        conversation=assistant._history,
        notes="I live in Sydney.",
        origin_satellite_id=None,
        origin_source="integration",
    )


def test_deep_think_action_stops_bundled_foreground_actions(monkeypatch):
    from core import agent_loop as al

    prompt = "Investigate heating costs for my three-bedroom home in Sydney"
    monkeypatch.setattr(
        al,
        "catchAll",
        lambda text: {
            "actions": [
                {"intent": "deep_think", "args": ["Heating costs"]},
                {"intent": "append_to_today", "args": ["An unresearched guess"]},
            ]
        },
    )
    monkeypatch.setattr(al.intents, "is_registered_tool", lambda name: True)
    dispatch = MagicMock(return_value="I'll investigate that.")
    monkeypatch.setattr(al.intents, "handle_action", dispatch)
    history = []
    host = SimpleNamespace(
        llm_enabled=True,
        _history_for=lambda sat: history,
        _compact_completed_turns=MagicMock(),
        _trim_history=MagicMock(),
        _emit_agent_event=MagicMock(),
        _record_spoken=MagicMock(),
        _generate_with_context_recovery=MagicMock(),
    )
    assert al.AgentLoop(host, source="text").run(prompt) == "I'll investigate that."
    dispatch.assert_called_once_with({"intent": "deep_think", "args": [prompt]})
    host._generate_with_context_recovery.assert_not_called()
    host._record_spoken.assert_called_once_with("I'll investigate that.")


@pytest.mark.parametrize(
    "intent,args,prompt",
    [
        ("plan_travel", ["Tokyo, Dubai, and Hawaii in one day"], "Can this itinerary work?"),
        ("search_papers", ["battery chemistry"], "Find papers on battery chemistry"),
        ("get_finance_quote", ["TSLA:NASDAQ"], "Should I buy TSLA?"),
    ],
)
def test_foreground_deep_think_tools_and_finance_advice_are_routed_to_worker(
    monkeypatch, intent, args, prompt
):
    monkeypatch.setitem(sys.modules, "arxiv", SimpleNamespace())
    importlib.import_module("tools.research")
    importlib.import_module("tools.travel")
    from core.agent_emission import route_deep_think_only_tools
    from tools.capabilities import native_requires_deep_think
    from tools.tool_registry import tool_registry

    monkeypatch.setattr(
        tool_registry, "is_available", lambda name: name == "deep_think"
    )
    assert route_deep_think_only_tools(
        {"actions": [{"intent": intent, "args": args}]}, prompt,
        registry=tool_registry, requires_deep_think=native_requires_deep_think,
    ) == {"actions": [{"intent": "deep_think", "args": [prompt]}]}


@pytest.mark.parametrize("prompt", ["Yes, give me a short summary.", "yes", "go ahead"])
def test_explicit_summary_request_consumes_the_completed_report(prompt):
    from core.agent_loop import AgentLoop

    host = SimpleNamespace(
        consume_completed_thinking_report=MagicMock(return_value="Grounded report conclusion.")
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.satellite_id = "satellite"
    assert loop._run(host, None, "voice", None, None, None, prompt) == "Grounded report conclusion."
    host.consume_completed_thinking_report.assert_called_once_with("satellite")


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "What did the report say about installation costs?",
            "The report lists installation costs of $4,000.",
        ),
        (
            "Does it say that the route is feasible?",
            "The report found no feasible option among the retrieved itineraries.",
        ),
    ],
)
def test_report_follow_up_uses_grounded_reader(question, answer):
    from core.agent_loop import AgentLoop

    host = SimpleNamespace(answer_completed_thinking_report=MagicMock(return_value=answer))
    loop = AgentLoop.__new__(AgentLoop)
    loop.satellite_id = "satellite"
    assert loop._run(host, None, "voice", None, None, None, question) == answer
    host.answer_completed_thinking_report.assert_called_once_with("satellite", question, None, None)


def test_report_routing_preserves_command_priority_and_bypass():
    from core.agent_follow_up import route_report_follow_up

    answer = MagicMock(return_value="Report answer")
    command = {"actions": [{"intent": "read_note", "args": ["report"]}]}
    catch = MagicMock(return_value=command)
    consume = MagicMock()
    services = {
        "satellite_id": "satellite", "cancel_check": None, "stats": None,
        "consume_report": consume, "answer_report": answer, "active_task": lambda: None,
        "catch_intent": catch,
    }
    route = route_report_follow_up("Read the report note", **services)
    assert route.caught == command
    assert route.reply is None
    catch.return_value = None
    assert route_report_follow_up("Search again for the report", **services).reply is None
    answer.assert_not_called()
    consume.assert_not_called()


@pytest.mark.parametrize("status", ["QUEUED", "RUNNING", "PAUSED"])
def test_report_routing_waits_for_active_report_after_completed_report_miss(status):
    from core.agent_follow_up import route_report_follow_up

    calls = []
    route = route_report_follow_up(
        "yes", satellite_id="satellite", cancel_check=None, stats=None,
        consume_report=lambda sid: calls.append(("consume", sid)),
        answer_report=None,
        catch_intent=lambda prompt: calls.append(("catch", prompt)),
        active_task=lambda: {"status": status},
    )
    assert calls == [("consume", "satellite"), ("catch", "yes")]
    assert route.reply == "I'm still working on that. I'll let you know as soon as the report is ready."


def test_worker_callback_binds_current_model_and_services(monkeypatch):
    """The stable bound callback must see model replacements made after registration."""
    assistant = Assistant.__new__(Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant._turn_lock = threading.Lock()
    assistant.thinking_jobs = MagicMock()
    assistant.thinking_server_slots = 2
    callback = assistant._run_background_thinking_job
    replacement = object()
    assistant.slm_model = replacement
    generate = MagicMock(side_effect=['{"reply":"sufficient findings collected"}', "Report."])
    monkeypatch.setattr(assistant_module, "generate_slm", generate)
    monkeypatch.setattr(assistant_module, "native_capabilities", lambda: {})
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    assert callback(job, lambda: False)[0] == "Report."
    assert all(call.args == (replacement,) for call in generate.call_args_list)
    assert assistant.thinking_jobs.update_stage.call_args.args == (job, "Synthesising report")


def test_terminal_job_callback_serializes_history_and_forwards_report_card():
    assistant = Assistant.__new__(Assistant)
    assistant._turn_lock = threading.Lock()
    assistant._history = []
    assistant._pending_thinking_tasks = {}
    assistant._completed_thinking_reports = {}
    assistant._dispatch_event = MagicMock()
    assistant._emit_turn_event = MagicMock()

    def trim_under_lock():
        assert assistant._turn_lock.locked()
        Assistant._trim_history(assistant)

    assistant._trim_history = trim_under_lock
    job = BackgroundJob(
        "12345678",
        JobSnapshot(
            "Investigate", origin_source="conversation", origin_satellite_id="dashboard-text"
        ),
        status=JobStatus.READY,
        note_id="fulloch-reports/2026-08-27-12345678",
        summary="## Summary\n\nResult.",
    )
    assistant._on_thinking_job_status(job)
    assert job.note_id in assistant._history[0]["content"]
    assert not assistant._turn_lock.locked()
    assert (
        assistant._emit_turn_event.call_args.kwargs["artifact"]["report_url"]
        == f"/reports/{job.note_id}"
    )


def test_report_followup_entrypoint_passes_live_model_and_request_services(monkeypatch):
    from core import thinking_reports

    assistant = Assistant.__new__(Assistant)
    assistant.slm_model = object()
    assistant.satellites = {"satellite": object()}
    assistant._completed_thinking_reports = {
        "satellite": {"note_id": "fulloch-reports/2026-08-27-12345678"}
    }
    monkeypatch.setattr(thinking_reports, "read_report", lambda _entry: "## Summary\n\nResult.")
    monkeypatch.setattr(thinking_reports, "read_evidence", lambda _entry: {})
    generate = MagicMock(return_value="Grounded answer.")
    monkeypatch.setattr(assistant_module, "generate_slm", generate)
    cancel, stats = MagicMock(return_value=False), object()
    assert (
        assistant.answer_completed_thinking_report("satellite", "What did it say?", cancel, stats)
        == "Grounded answer."
    )
    assert generate.call_args.args == (assistant.slm_model,)
    assert generate.call_args.kwargs["cancel_check"] is cancel
    assert generate.call_args.kwargs["stats"] is stats
