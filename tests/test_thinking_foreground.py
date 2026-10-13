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
from core.thinking_worker import ReportSynthesisError
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


def test_started_job_is_not_mistaken_for_an_earlier_job(monkeypatch):
    assistant = SimpleNamespace(
        run_thinking_task=lambda *args, **kwargs: {"id": "job-1", "status": "QUEUED"},
        active_thinking_task=lambda: {"id": "job-1", "status": "RUNNING"},
    )
    monkeypatch.setattr(thinking, "get_current_assistant", lambda: assistant)
    assert thinking.deep_think("Investigate") == "I'll look into that and let you know when I'm done."


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
        origin_turn_id=None,
    )


def test_thinking_clarification_resumes_existing_job_instead_of_submitting(monkeypatch):
    assistant = Assistant.__new__(Assistant)
    assistant._pending_thinking_tasks = {
        "satellite": {"job_id": "job-1", "task": "Find flights", "question": "Did you mean 2027?"}
    }
    assistant.thinking_jobs = MagicMock()
    assistant.thinking_jobs.resume_with_input.return_value = True
    assistant.thinking_jobs.status.return_value = {"id": "job-1", "status": "QUEUED"}

    assert assistant.run_thinking_task("yes", origin_satellite_id="satellite", origin_source="conversation") == {
        "id": "job-1", "status": "QUEUED"
    }
    assistant.thinking_jobs.resume_with_input.assert_called_once_with("job-1", "yes")
    assistant.thinking_jobs.submit.assert_not_called()
    assert assistant._pending_thinking_tasks == {}


def test_conversation_retry_keeps_task_identity_and_links_failed_predecessor(monkeypatch):
    from core.background_jobs import BackgroundJobManager, JobStatus

    monkeypatch.setattr(assistant_module.notes, "recall_facts", lambda: "")
    assistant = Assistant.__new__(Assistant)
    assistant._history = []
    assistant._satellite_histories = {
        "thread-a": [{"role": "user", "content": "Thread A only"}],
        "thread-b": [{"role": "user", "content": "Thread B secret context"}],
    }
    assistant._pending_thinking_tasks = {}
    assistant.thinking_jobs = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)

    initial = assistant.run_thinking_task(
        "Research latest local models with 9B parameters; keep sources recent",
        origin_satellite_id="thread-a", origin_source="conversation",
    )
    first_job = assistant.thinking_jobs._jobs[initial["id"]]
    assert list(first_job.snapshot.conversation) == [
        {"role": "user", "content": "Thread A only"},
    ]
    first_job.record_outcome(
        "external_information", "evidence", "One original model-card source.",
        {"sources": [{"url": "https://models.example.test/model-9b", "text": "Model X has 9B parameters."}]},
        (), {"type": "web_research", "sources": [
            {"url": "https://models.example.test/model-9b", "source_text": "Model X has 9B parameters."},
        ]},
    )
    first_job.status = JobStatus.FAILED

    retry = assistant.run_thinking_task(
        "just do it", origin_satellite_id="thread-a", origin_source="conversation",
    )
    retry_job = assistant.thinking_jobs._jobs[retry["id"]]

    assert retry["id"] != initial["id"]
    assert retry_job.snapshot.task == first_job.snapshot.task
    assert retry_job.snapshot.task_record["task_id"] == first_job.snapshot.task_record["task_id"]
    assert retry_job.snapshot.task_record["parent_job_id"] == initial["id"]
    assert retry_job.snapshot.task_record["attempt"] == 2
    [retained] = retry_job.snapshot.handoff["observations"]
    from core.thinking_source_store import load_document

    artifact_source = retained["artifact"]["sources"][0]
    evidence_source = retained["evidence"]["sources"][0]
    assert load_document(artifact_source["source_ref"])["text"] == "Model X has 9B parameters."
    assert load_document(evidence_source["source_ref"])["text"] == "Model X has 9B parameters."
    assert retry_job.snapshot.handoff["source_job_id"] == initial["id"]
    assert "9B parameters" in retry_job.snapshot.task
    assert "sources recent" in retry_job.snapshot.task
    assistant.thinking_jobs.cancel(retry["id"])


def test_conversation_retry_reuses_same_thread_active_job(monkeypatch):
    from core.background_jobs import BackgroundJobManager

    monkeypatch.setattr(assistant_module.notes, "recall_facts", lambda: "")
    assistant = Assistant.__new__(Assistant)
    assistant._history = []
    assistant._pending_thinking_tasks = {}
    assistant.thinking_jobs = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)
    initial = assistant.run_thinking_task(
        "Research current local models", origin_satellite_id="thread-a", origin_source="conversation",
    )

    repeated = assistant.run_thinking_task(
        "just do it", origin_satellite_id="thread-a", origin_source="conversation",
    )

    assert repeated["id"] == initial["id"]
    assert len(assistant.thinking_jobs._jobs) == 1
    assistant.thinking_jobs.cancel(initial["id"])


def test_topicless_followup_without_thread_task_is_rejected(monkeypatch):
    from core.background_jobs import BackgroundJobManager

    monkeypatch.setattr(assistant_module.notes, "recall_facts", lambda: "")
    assistant = Assistant.__new__(Assistant)
    assistant._history = []
    assistant._pending_thinking_tasks = {}
    assistant.thinking_jobs = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)

    with pytest.raises(RuntimeError, match="research topic"):
        assistant.run_thinking_task(
            "just do it", origin_satellite_id="thread-empty", origin_source="conversation",
        )
    assert not assistant.thinking_jobs._jobs


def test_complaint_and_why_not_are_answered_from_thread_job_without_new_submission():
    from core.background_jobs import BackgroundJobManager, JobStatus

    assistant = Assistant.__new__(Assistant)
    assistant.thinking_jobs = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)
    job_id = assistant.thinking_jobs.submit(
        "Research recent 9B local models", origin_source="conversation",
        origin_satellite_id="thread-status",
    )
    job = assistant.thinking_jobs._jobs[job_id]
    job.status = JobStatus.FAILED
    job.stage = "primary-source verification"
    job.error = "Lookup provider returned no eligible sources."
    job.snapshot.task_record.update({
        "failure_stage": "primary-source verification",
        "failure_reason": job.error,
        "partial_artifact_link": "/reports/partial-job",
    })

    complaint = assistant.research_task_follow_up("thread-status", "you didn't do anything")
    explanation = assistant.research_task_follow_up("thread-status", "why not?")

    assert "last attempt" in complaint
    assert "failed" in complaint
    assert "Lookup provider returned no eligible sources" in explanation
    assert "primary-source verification" in explanation
    assert "/reports/partial-job" in explanation
    assert len(assistant.thinking_jobs._jobs) == 1
    assistant.thinking_jobs.cancel(job_id)


def test_status_question_without_prior_thread_job_requests_a_topic():
    from core.background_jobs import BackgroundJobManager

    assistant = Assistant.__new__(Assistant)
    assistant.thinking_jobs = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)

    reply = assistant.research_task_follow_up("empty-thread", "you didn't do anything")

    assert "What topic should I investigate?" in reply
    assert not assistant.thinking_jobs._jobs


def test_refinement_records_source_turn_on_versioned_amendment(monkeypatch):
    from types import SimpleNamespace

    from core.background_jobs import BackgroundJobManager, JobStatus

    monkeypatch.setattr(assistant_module.notes, "recall_facts", lambda: "")
    assistant = Assistant.__new__(Assistant)
    assistant._history = []
    assistant._pending_thinking_tasks = {}
    assistant._turn_local = SimpleNamespace(turn_id="turn-refinement")
    assistant.thinking_jobs = BackgroundJobManager(lambda *_: ("", ""), admission_check=lambda: False)
    original = assistant.thinking_jobs.submit(
        "Research the latest local AI trends for running capable models at home",
        origin_source="conversation", origin_satellite_id="thread-refinement",
        task_record={
            "task_id": "task-refinement", "objective": "Research the latest local AI trends for running capable models at home",
            "constraints": [], "amendments": [],
        },
    )
    assistant.thinking_jobs._jobs[original].status = JobStatus.FAILED

    refined = assistant.run_thinking_task(
        "can you find latest info on trending local models with 9B parameters? Keep sources recent",
        origin_satellite_id="thread-refinement", origin_source="conversation",
    )
    record = assistant.thinking_jobs._jobs[refined["id"]].snapshot.task_record

    assert record["revision"] == 2
    assert "turn-refinement" in record["source_turn_ids"]
    assert record["amendments"][0]["source_turn_id"] == "turn-refinement"
    assert "9B parameters" in assistant.thinking_jobs._jobs[refined["id"]].snapshot.task
    assistant.thinking_jobs.cancel(refined["id"])


def test_bare_confirmation_resumes_pending_thinking_before_report_routing():
    from core.agent_loop import AgentLoop

    host = SimpleNamespace(resume_pending_thinking_task=MagicMock(return_value=True))
    loop = AgentLoop.__new__(AgentLoop)
    loop.satellite_id = "satellite"

    assert loop._run(host, None, "voice", None, None, None, "yes") == (
        "Thanks, I'll continue that investigation with the proposed correction."
    )
    host.resume_pending_thinking_task.assert_called_once_with("satellite", "yes")


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
    expected = "I'll investigate that. Not completed: append to today."
    assert al.AgentLoop(host, source="text").run(prompt) == expected
    dispatch.assert_called_once_with({"intent": "deep_think", "args": [prompt]})
    host._generate_with_context_recovery.assert_not_called()
    host._record_spoken.assert_called_once_with(expected)


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
    generate = MagicMock(side_effect=['{"reply":"sufficient findings collected"}'])
    monkeypatch.setattr(assistant_module, "generate_slm", generate)
    monkeypatch.setattr(assistant_module, "native_capabilities", lambda: {})
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    with pytest.raises(ReportSynthesisError, match="No verified evidence"):
        callback(job, lambda: False)
    assert all(call.args == (replacement,) for call in generate.call_args_list)
    assistant.thinking_jobs.update_stage.assert_called_once_with(job, "Planning research")
    assert job.planning_complete is True
    assert job.synthesis_state["chapter_planning"] == "deterministic_fallback"


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
