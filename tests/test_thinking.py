"""Tests for queued, tool-capable deliberate work."""

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import thinking  # noqa: E402


class TestDeepThinkTool:
    def test_queues_conversational_job(self, monkeypatch):
        class Assistant:
            def run_thinking_task(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return {"id": "job-1", "status": "QUEUED"}

            def active_thinking_task(self):
                return {"id": "job-1", "status": "QUEUED"}

        assistant = Assistant()
        monkeypatch.setattr(thinking, "get_current_assistant", lambda: assistant)
        result = thinking.deep_think("research new models")
        assert "look into" in result
        assert assistant.args == ("research new models",)
        assert assistant.kwargs["origin_source"] == "conversation"

    def test_is_exposed_without_a_thinking_config_block(self):
        from tools.tool_registry import tool_registry

        assert tool_registry.is_available("deep_think") is True

    def test_oversized_slot_count_clamps_to_two(self):
        with patch("core.assistant.AudioCapture") as capture:
            capture.return_value = MagicMock()
            from core.assistant import Assistant

            assistant = Assistant(wakeword="hey atticus", thinking={"server_slots": 3})

        assert assistant.thinking_enabled is True
        assert assistant.thinking_server_slots == 2


class TestThinkingPromptStripping:
    """The <think>...</think> block should not reach the TTS path."""

    def test_clean_for_tts_strips_think_blocks(self):
        from core.text_utils import clean_for_tts

        raw = (
            "<think>OK so they're asking about cars. "
            "Let me weigh the options...</think>"
            "Electric cars are probably worth it if you have home charging."
        )
        cleaned = clean_for_tts(raw)
        assert "<think>" not in cleaned
        assert "weigh the options" not in cleaned
        assert "Electric cars" in cleaned


def test_foreground_history_is_copied_and_bounded():
    from utils.prompts import FOREGROUND_HISTORY_MESSAGES, assemble_foreground_history

    history = [
        {"role": "user", "content": "x" * 1000} for _ in range(FOREGROUND_HISTORY_MESSAGES + 1)
    ]
    compact = assemble_foreground_history(history)
    assert len(compact) == FOREGROUND_HISTORY_MESSAGES
    assert compact[0] is not history[1]
    assert len(compact[-1]["content"]) < 1000


def test_thinking_worker_prompt_excludes_foreground_personality_and_includes_inputs():
    from utils.prompts import get_thinking_worker_prompt

    prompt = get_thinking_worker_prompt(
        "Compare options",
        [{"role": "user", "content": "context"}],
        notes="saved note",
        job_state="one lead remains",
        capabilities="search_notes (read)",
    )
    assert "Compare options" in prompt
    assert "saved note" in prompt and "one lead remains" in prompt
    assert "search_notes (read)" in prompt
    assert "personality" not in prompt.lower()


def test_thinking_worker_prompt_uses_generic_progress_guardrails():
    from utils.prompts import get_thinking_worker_prompt

    prompt = get_thinking_worker_prompt("Compare options", [], capabilities="search_notes (read)")

    assert "highest information gain" in prompt
    assert "materially different" in prompt
    assert "runtime rejects duplicates" in prompt
    assert "sufficient findings collected" in prompt
    assert "useful preliminary report" in prompt
    assert "single source" in prompt
    assert "materially different available option" in prompt
    assert "book" in prompt.lower()
    assert "plan" in prompt
    assert "Reactive question:" in prompt
    assert "Do not write the final report yourself" in prompt
    assert "Today is " in prompt


def test_thinking_worker_prompt_has_no_domain_specific_playbooks():
    from utils.prompts import get_thinking_worker_prompt

    prompt = get_thinking_worker_prompt("Compare options", [], capabilities="search_notes(query)")

    assert "flight" not in prompt.lower()
    assert "hotel" not in prompt.lower()
    assert "iata" not in prompt.lower()
    assert "highest information gain" in prompt
    assert "untrusted data" in prompt
    assert "<tool_observations>" in prompt


def test_thinking_worker_prompt_includes_matching_tool_owned_playbook():
    import tools.travel  # noqa: F401
    from tools.thinking_playbooks import matching_playbooks
    from utils.prompts import get_thinking_worker_prompt

    playbooks = matching_playbooks(
        "I am travelling from Paris to Rome, then Madrid around Easter next year.",
        {"plan_travel", "search_flights", "assess_itinerary"},
    )
    prompt = get_thinking_worker_prompt(
        "Check this itinerary",
        [],
        capabilities="plan_travel(request)",
        capability_playbooks="\n\n".join(playbook.render() for playbook in playbooks),
    )

    assert [playbook.name for playbook in playbooks] == ["travel planning"]
    assert "travel planning" in prompt
    assert "start with plan_travel" in prompt
    assert "timezone or calendar arithmetic alone" in prompt
    assert "failed itinerary assessment rejects only that candidate" in prompt
    assert "materially different date, route, or flight search" in prompt
    assert "exactly origin, destination, and a future ISO departure date" in prompt
    assert "Artifact reference" in prompt
    assert "never serialize schedule JSON" in prompt
    assert playbooks[0].fallback_capability == "plan_travel"


def test_thinking_playbook_requires_an_enabled_capability():
    import tools.travel  # noqa: F401
    from tools.thinking_playbooks import matching_playbooks

    assert matching_playbooks("Find flights to Tokyo", {"calculate"}) == []


def test_travel_playbook_matches_movement_not_meal_words():
    import tools.travel  # noqa: F401
    from tools.thinking_playbooks import matching_playbooks

    capabilities = {"plan_travel", "search_flights", "assess_itinerary"}
    for request in (
        "I want to go somewhere for lunch.",
        "I want to go somewhere to see the sunset.",
        "I want to go somewhere and fly overnight.",
        "I want to catch an early plane to somewhere so I arrive in time for dinner.",
    ):
        assert [playbook.name for playbook in matching_playbooks(request, capabilities)] == [
            "travel planning"
        ]

    assert matching_playbooks("What should I have for dinner?", capabilities) == []


def test_thinking_transcript_compaction_preserves_each_tool_observation():
    a = _import_assistant_module()
    transcript = ""
    for index in range(8):
        transcript = a._append_thinking_observation(
            transcript, f"tool:search_{index}", f"result-{index} " + "x" * 2_900
        )

    assert len(transcript) <= a.DEEP_THINK_TRANSCRIPT_MAX_CHARS
    for index in range(8):
        assert f"[tool:search_{index}]" in transcript
        assert f"result-{index}" in transcript


def test_thinking_capability_description_includes_callable_signature():
    from tools.tool_registry import Param, ToolSchema

    a = _import_assistant_module()
    description = a._describe_thinking_capability(
        "search_notes",
        ToolSchema(
            "search_notes",
            "Search saved notes.",
            [Param("query", True), Param("limit", False, 5)],
        ),
    )

    assert description == "- search_notes(query, limit=5): Search saved notes."


def test_agentic_worker_exposes_only_read_tools_and_bounds_capability_calls():
    import inspect

    a = _import_assistant_module()
    source = inspect.getsource(a.Assistant._run_background_thinking_job)
    assert a.MAX_THINKING_CAPABILITY_CALLS == 3
    assert "attempted_actions" in source
    assert "Duplicate capability request" in source
    assert "Blocked unavailable capability" in source
    assert "capability_calls[name] -= 1" in source
    assert "Worker supplied a plan without an action" in source
    assert 'if capability.access_class == "read"' in source


def test_agentic_worker_synthesises_after_its_investigation():
    import inspect

    a = _import_assistant_module()
    source = inspect.getsource(a.Assistant._run_background_thinking_job)

    assert 'update_stage(job, "Synthesising report")' in source
    assert "get_thinking_report_prompt" in source
    assert "max_new_tokens=8192" in source
    assert "synthesise current findings" in source


def test_thinking_report_prompt_requires_a_direct_evidence_scoped_summary():
    from utils.prompts import get_thinking_report_prompt

    prompt = get_thinking_report_prompt("Compare heating options", "Retrieved prices")

    assert "## Summary" in prompt
    assert "central question" in prompt
    assert "beyond the retrieved evidence" in prompt


def test_typed_evidence_is_used_without_raw_worker_observations(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot

    a = _import_assistant_module()
    captured = {}
    monkeypatch.setattr(
        a,
        "generate_slm",
        lambda *_args, **kwargs: captured.update(kwargs) or "## Summary\n\nA scoped answer.",
    )
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob(
        "job",
        JobSnapshot("Investigate"),
        state="[worker]\nprivate planning text",
        evidence=[{"tool": "lookup", "status": "evidence", "scope": "One source.", "evidence": {"fact": "retrieved"}}],
    )

    assistant._run_background_thinking_job(job, lambda: False)

    assert "private planning text" not in captured["system_prompt"]
    assert '"fact": "retrieved"' in captured["system_prompt"]


def test_report_answer_prompt_is_grounded_to_one_completed_report():
    from utils.prompts import get_thinking_report_answer_prompt

    prompt = get_thinking_report_answer_prompt(
        "The installed price is $4,000.", {"artifacts": {"artifact-001": {"price": 4000}}}
    )

    assert "only the completed report" in prompt
    assert '"The report does not answer that."' in prompt
    assert "The installed price is $4,000." in prompt
    assert "artifact-001" in prompt


def test_planning_worker_receives_saved_facts_as_job_context():
    import inspect

    a = _import_assistant_module()
    source = inspect.getsource(a.Assistant.run_thinking_task)

    assert "notes=notes.recall_facts()" in source


def test_deep_think_worker_has_its_own_twelve_step_budget():
    import inspect

    a = _import_assistant_module()
    source = inspect.getsource(a.Assistant._run_background_thinking_job)

    assert a.MAX_THINKING_WORKER_CALLS == 12
    assert "for _ in range(MAX_THINKING_WORKER_CALLS):" in source
    assert a.DEEP_THINK_STEP_MAX_TOKENS == 1024
    assert "read_timeout=DEEP_THINK_READ_TIMEOUT_S" in source
    assert "generation_timeout=DEEP_THINK_GENERATION_TIMEOUT_S" in source


def test_deep_think_worker_stops_on_a_duplicate_action(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability

    a = _import_assistant_module()
    calls = []
    stages = []
    capability = ToolCapability(
        name="lookup",
        invoke=lambda args, _kwargs: calls.append(args) or "evidence",
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    responses = iter(
        [
            '{"actions":[{"intent":"lookup","args":["same"]}]}',
            '{"actions":[{"intent":"lookup","args":["same"]}]}',
            "Evidence-based report.",
        ]
    )
    monkeypatch.setattr(a, "generate_slm", lambda *_args, **_kwargs: next(responses))
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type(
        "Jobs", (), {"update_stage": lambda _self, _job, stage: stages.append(stage)}
    )()
    job = BackgroundJob("job", JobSnapshot("Investigate something"))

    report, findings = assistant._run_background_thinking_job(job, lambda: False)

    assert calls == [["same"]]
    assert "Duplicate capability request for lookup" in findings
    assert report == "Evidence-based report."
    assert stages[-1] == "Synthesising report"


def test_deep_think_invalid_worker_json_is_not_saved_as_a_report(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability

    a = _import_assistant_module()
    capability = ToolCapability(
        name="lookup",
        invoke=lambda _args, _kwargs: "source evidence",
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    responses = iter(
        [
            '{"actions":[{"intent":"lookup","args":["query"]}]}',
            '{"actions":[{"intent":"evaluate_itinerary","args":["truncated"]}',
            '{"reply":"sufficient findings collected"}',
            "Evidence-based final report.",
        ]
    )
    monkeypatch.setattr(a, "generate_slm", lambda *_args, **_kwargs: next(responses))
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob("job", JobSnapshot("Investigate something"))

    report, findings = assistant._run_background_thinking_job(job, lambda: False)

    assert report == "Evidence-based final report."
    assert "invalid planning response; select the next capability" in findings
    assert "evaluate_itinerary" not in report


def test_deep_think_empty_worker_response_still_synthesises_collected_evidence(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability

    a = _import_assistant_module()
    capability = ToolCapability(
        name="lookup",
        invoke=lambda _args, _kwargs: "source evidence",
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    responses = iter(
        [
            '{"actions":[{"intent":"lookup","args":["query"]}]}',
            "",
            "Evidence-based final report.",
        ]
    )
    monkeypatch.setattr(a, "generate_slm", lambda *_args, **_kwargs: next(responses))
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob("job", JobSnapshot("Investigate something"))

    report, findings = assistant._run_background_thinking_job(job, lambda: False)

    assert report == "Evidence-based final report."
    assert "Worker stopped without a next action" in findings
    assert "source evidence" in findings


def test_deep_think_rejects_invalid_typed_outcome_before_recording_evidence(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability
    from tools.tool_registry import Param, ThinkingResult, ToolSchema

    a = _import_assistant_module()
    capability = ToolCapability(
        name="lookup",
        invoke=lambda _args, _kwargs: ThinkingResult("Bad result", status="exhausted", scope="Bad."),
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    monkeypatch.setitem(
        a.tool_registry._schemas,
        "lookup",
        ToolSchema("lookup", "", [Param("refinement", False, "")], thinking_outcome=True),
    )
    responses = iter(['{"actions":[{"intent":"lookup","args":[]}]}', '{"reply":"enough"}', "Report."])
    monkeypatch.setattr(a, "generate_slm", lambda *_args, **_kwargs: next(responses))
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob("job", JobSnapshot("Investigate"))

    report, _findings = assistant._run_background_thinking_job(job, lambda: False)

    assert report == "Report."
    assert job.evidence[0]["status"] == "failed"
    assert job.evidence[0]["evidence"] == {}


def test_deep_think_typed_needs_input_stops_without_synthesising(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability
    from tools.tool_registry import ThinkingResult, ToolSchema

    a = _import_assistant_module()
    capability = ToolCapability(
        name="lookup",
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Which city should I use?", status="needs_input", scope="The city was not supplied."
        ),
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    monkeypatch.setitem(a.tool_registry._schemas, "lookup", ToolSchema("lookup", "", [], thinking_outcome=True))
    monkeypatch.setattr(
        a, "generate_slm", lambda *_args, **_kwargs: '{"actions":[{"intent":"lookup","args":[]}]}'
    )
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob("job", JobSnapshot("Investigate"))

    report, _findings = assistant._run_background_thinking_job(job, lambda: False)

    assert report == "Reactive question: Which city should I use?"
    assert job.evidence[0]["status"] == "needs_input"


def test_deep_think_synthesises_preliminary_report_when_input_follows_evidence(monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability
    from tools.tool_registry import Param, ThinkingResult, ToolSchema

    a = _import_assistant_module()
    results = iter(
        [
            ThinkingResult("Retrieved schedule.", evidence={"schedule": True}, scope="One schedule."),
            ThinkingResult("Which exact time?", status="needs_input", scope="Event timing is missing."),
        ]
    )
    capability = ToolCapability(
        name="lookup",
        invoke=lambda _args, _kwargs: next(results),
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    monkeypatch.setitem(
        a.tool_registry._schemas,
        "lookup",
        ToolSchema("lookup", "", [Param("refinement", False, "")], thinking_outcome=True),
    )
    responses = iter(
        [
            '{"actions":[{"intent":"lookup","args":[]}]}',
            '{"actions":[{"intent":"lookup","args":["refine"]}]}',
            "Preliminary report.",
        ]
    )
    monkeypatch.setattr(a, "generate_slm", lambda *_args, **_kwargs: next(responses))
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob("job", JobSnapshot("Investigate"))

    report, findings = assistant._run_background_thinking_job(job, lambda: False)

    assert report == "Preliminary report."
    assert "sufficient evidence exists for a preliminary scoped report" in findings


def test_deep_think_synthesises_collected_evidence_after_invalid_planning_exhausts_budget(
    monkeypatch,
):
    from core.background_jobs import BackgroundJob, JobSnapshot
    from tools.capabilities import ToolCapability

    a = _import_assistant_module()
    capability = ToolCapability(
        name="lookup",
        invoke=lambda _args, _kwargs: "source evidence",
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class="read",
    )
    monkeypatch.setattr(a, "native_capabilities", lambda: {"lookup": capability})
    monkeypatch.setattr(a.tool_registry, "canonical_name", lambda name: name)
    responses = iter(
        ['{"actions":[{"intent":"lookup","args":["query"]}]}']
        + ['{"actions":[{"intent":"lookup","args":["truncated"]}']
        * (a.MAX_THINKING_WORKER_CALLS - 1)
        + ["Evidence-based final report."]
    )
    monkeypatch.setattr(a, "generate_slm", lambda *_args, **_kwargs: next(responses))
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.grammar = "grammar"
    assistant.thinking_server_slots = 2
    assistant.thinking_jobs = type("Jobs", (), {"update_stage": lambda *_args: None})()
    job = BackgroundJob("job", JobSnapshot("Investigate something"))

    report, findings = assistant._run_background_thinking_job(job, lambda: False)

    assert report == "Evidence-based final report."
    assert "source evidence" in findings
    assert findings.count("invalid planning response") == a.MAX_THINKING_WORKER_CALLS - 1


def test_deep_think_has_no_regex_profile_routing():
    a = _import_assistant_module()
    assert not hasattr(a.Assistant, "_select_thinking_profile")


def test_agentic_worker_logs_its_internal_plan_and_uses_long_synthesis():
    import inspect

    a = _import_assistant_module()
    source = inspect.getsource(a.Assistant._run_background_thinking_job)
    assert "fallback_report()" in source
    assert "thinking_mode=False" in source
    assert 'logger.debug("Deep-think job %s plan: %s"' in source
    assert "get_thinking_report_prompt" in source
    assert "thinking_mode=True" in source
    assert "max_new_tokens=8192" in source


def test_deep_think_action_stops_bundled_foreground_actions():
    import inspect

    from core.agent_loop import AgentLoop

    source = inspect.getsource(AgentLoop._run)
    assert 'if intent_name == "deep_think"' in source
    assert "return spoken" in source
    assert 'action = {**action, "args": [user_prompt]}' in source


def test_foreground_deep_think_only_tools_are_routed_to_the_planning_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "arxiv", types.SimpleNamespace())
    importlib.import_module("tools.research")
    importlib.import_module("tools.travel")

    from core.agent_loop import _route_deep_think_only_tools

    monkeypatch.setattr(
        "core.agent_loop.tool_registry.is_available", lambda name: name == "deep_think"
    )
    travel_emission = {
        "actions": [
            {
                "intent": "plan_travel",
                "args": ["Tokyo, Dubai, and Hawaii in one day"],
            }
        ]
    }

    assert _route_deep_think_only_tools(travel_emission, "Can this itinerary work?") == {
        "actions": [
            {
                "intent": "deep_think",
                "args": ["Can this itinerary work?"],
            }
        ]
    }
    research_emission = {"actions": [{"intent": "search_papers", "args": ["battery chemistry"]}]}
    assert _route_deep_think_only_tools(research_emission, "Find papers on battery chemistry") == {
        "actions": [{"intent": "deep_think", "args": ["Find papers on battery chemistry"]}]
    }


def test_affirmative_follow_up_consumes_the_completed_report():
    import inspect

    from core.agent_loop import AgentLoop

    source = inspect.getsource(AgentLoop._run)
    assert "consume_completed_thinking_report" in source
    assert "if completed_report:" in source


def test_spoken_report_summary_is_short_and_sentence_complete():
    a = _import_assistant_module()
    report = "One finding is useful. Two finding is useful. Three finding is useful. Four finding is useful. Five finding is useful."
    assert a.Assistant._spoken_report_summary(report) == "Four finding is useful. Five finding is useful."


def test_completed_report_is_available_after_a_satellite_reconnect(monkeypatch):
    a = _import_assistant_module()
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.satellites = {"new-session": object()}
    assistant._completed_thinking_reports = {
        "old-session": {
            "note_id": "fulloch-reports/2026-08-27-12345678",
            "task": "Compare options",
            "summary_delivered": False,
        }
    }
    monkeypatch.setattr(
        a.Assistant,
        "_read_completed_thinking_report",
        staticmethod(lambda _pending: "Finding. Conclusion."),
    )

    response = assistant.consume_completed_thinking_report("new-session")

    assert response == (
        "Here's the short version. Finding. Conclusion. "
        "The full report is saved in Fulloch Reports. Would you like me to read the full report?"
    )
    assert assistant._completed_thinking_reports == {
        "new-session": {
            "note_id": "fulloch-reports/2026-08-27-12345678",
            "task": "Compare options",
            "summary_delivered": True,
        }
    }


def test_explicit_summary_request_consumes_the_completed_report():
    from core.agent_loop import AgentLoop

    consumed = []

    class Host:
        def consume_completed_thinking_report(self, satellite_id):
            consumed.append(satellite_id)
            return "Grounded report conclusion."

    loop = AgentLoop.__new__(AgentLoop)
    loop.satellite_id = "satellite"

    assert loop._run(
        Host(), None, "voice", None, None, None, "Yes, give me a short summary."
    ) == "Grounded report conclusion."
    assert consumed == ["satellite"]


def test_report_follow_up_uses_the_grounded_report_reader_for_non_travel_question():
    from core.agent_loop import AgentLoop

    class Host:
        def answer_completed_thinking_report(self, satellite_id, question, cancel_check, stats):
            assert satellite_id == "satellite"
            assert question == "What did the report say about installation costs?"
            assert cancel_check is None and stats is None
            return "The report lists installation costs of $4,000."

    loop = AgentLoop.__new__(AgentLoop)
    loop.satellite_id = "satellite"

    assert loop._run(
        Host(),
        None,
        "voice",
        None,
        None,
        None,
        "What did the report say about installation costs?",
    ) == "The report lists installation costs of $4,000."


def test_report_follow_up_uses_the_grounded_report_reader_for_feasibility_question():
    from core.agent_loop import AgentLoop

    class Host:
        def answer_completed_thinking_report(self, _satellite_id, _question, _cancel_check, _stats):
            return "The report found no feasible option among the retrieved itineraries."

    loop = AgentLoop.__new__(AgentLoop)
    loop.satellite_id = "satellite"

    assert loop._run(
        Host(), None, "voice", None, None, None, "Does it say that the route is feasible?"
    ) == "The report found no feasible option among the retrieved itineraries."


def test_report_question_reads_the_saved_report_without_conversation_history(tmp_path, monkeypatch):
    a = _import_assistant_module()
    report_path = tmp_path / "fulloch-reports" / "2026-08-27-12345678.md"
    report_path.parent.mkdir()
    report_path.write_text("# Deep Think Report\n\nThe installation cost is $4,000.", encoding="utf-8")
    report_path.with_suffix(".evidence.json").write_text(
        '{"artifacts": {"artifact-001": {"data": {"price": 4000}}}}', encoding="utf-8"
    )
    monkeypatch.setattr(a.notes_root, "get_notes_root", lambda: tmp_path)
    captured = {}
    monkeypatch.setattr(
        a,
        "generate_slm",
        lambda *_args, **kwargs: captured.update(kwargs) or "The report lists $4,000.",
    )
    assistant = a.Assistant.__new__(a.Assistant)
    assistant.slm_model = object()
    assistant.satellites = {"dashboard-text": object()}
    assistant._completed_thinking_reports = {
        "satellite": {
            "note_id": "fulloch-reports/2026-08-27-12345678",
            "task": "Compare heating systems",
            "summary_delivered": True,
        }
    }

    answer = assistant.answer_completed_thinking_report(
        "satellite", "What did it say about installation costs?", lambda: False
    )

    assert answer == "The report lists $4,000."
    assert captured["user_prompt"] == "What did it say about installation costs?"
    assert "The installation cost is $4,000." in captured["system_prompt"]
    assert "artifact-001" in captured["system_prompt"]
    assert "Compare heating systems" not in captured["system_prompt"]


def test_completed_report_persists_typed_evidence_and_artifacts(tmp_path, monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot

    a = _import_assistant_module()
    monkeypatch.setattr(a.notes_root, "get_notes_root", lambda: tmp_path)
    monkeypatch.setattr(a.notes, "_after_write", lambda _path: None)
    assistant = a.Assistant.__new__(a.Assistant)
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Compare options"),
        summary="## Summary\n\nA scoped result.",
        evidence=[{"tool": "lookup", "status": "evidence", "artifact_id": "artifact-001"}],
        artifacts={"artifact-001": {"tool": "lookup", "data": {"price": 4000}}},
    )

    note_id = assistant._save_thinking_report(job)
    evidence = json.loads((tmp_path / f"{note_id}.evidence.json").read_text(encoding="utf-8"))

    assert evidence["task"] == "Compare options"
    assert evidence["evidence"][0]["artifact_id"] == "artifact-001"
    assert evidence["artifacts"]["artifact-001"]["data"]["price"] == 4000


def test_completed_report_adds_a_scoped_summary_when_worker_omits_one(tmp_path, monkeypatch):
    from core.background_jobs import BackgroundJob, JobSnapshot

    a = _import_assistant_module()
    monkeypatch.setattr(a.notes_root, "get_notes_root", lambda: tmp_path)
    monkeypatch.setattr(a.notes, "_after_write", lambda _path: None)
    assistant = a.Assistant.__new__(a.Assistant)
    job = BackgroundJob("12345678", JobSnapshot("Compare options"), summary="Option A has the lower price.")

    note_id = assistant._save_thinking_report(job)
    report = (tmp_path / f"{note_id}.md").read_text(encoding="utf-8")

    assert "## Summary" in report
    assert "Option A has the lower price." in report
    assert "limited to the retrieved evidence" in report


def test_completed_report_history_keeps_summary_and_durable_filename():
    from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus

    a = _import_assistant_module()
    assistant = a.Assistant.__new__(a.Assistant)
    assistant._history = []
    assistant._completed_thinking_reports = {}
    assistant._pending_thinking_tasks = {}
    assistant._trim_history = lambda: None
    assistant._dispatch_event = lambda _event: None
    assistant._emit_turn_event = lambda *_args, **_kwargs: None
    assistant.satellites = {"dashboard-text": object()}
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Compare heat pumps", origin_source="conversation", origin_satellite_id="dashboard-text"),
        status=JobStatus.READY,
        note_id="fulloch-reports/2026-08-27-12345678",
        summary=(
            "## Summary\n\nHeat pumps are suitable for this insulated home, based on the retrieved quotes. "
            "Installation estimates vary by installer.\n\n## Analysis\n\n" + "detail " * 500
        ),
    )

    assistant._on_thinking_job_status(job)

    trace = assistant._history[-1]["content"]
    assert "Heat pumps are suitable" in trace
    assert "fulloch-reports/2026-08-27-12345678" in trace
    assert "detail detail" not in trace


def test_reports_use_a_user_facing_vault_directory():
    import inspect

    a = _import_assistant_module()
    source = inspect.getsource(a.Assistant._save_thinking_report)
    assert 'f"fulloch-reports/' in source


def test_completed_flight_job_emits_its_flight_plan_card():
    from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus

    a = _import_assistant_module()
    assistant = a.Assistant.__new__(a.Assistant)
    assistant._history = []
    assistant._completed_thinking_reports = {}
    assistant._trim_history = lambda: None
    assistant._dispatch_event = lambda _event: None
    emitted = []
    assistant._emit_turn_event = lambda *args, **kwargs: emitted.append((args, kwargs))
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Find flights", origin_source="integration"),
        status=JobStatus.READY,
        note_id="fulloch-reports/2026-08-27-12345678",
        artifact={
            "type": "flight_search",
            "route": {"origin": "SYD", "destination": "NRT"},
            "offer": {},
        },
    )

    assistant._on_thinking_job_status(job)

    assert emitted == [
        (
            (
                "assistant",
                "I found a recommended flight option and saved the full comparison.",
                "proactive",
            ),
            {
                "artifact": {
                    "type": "flight_plan",
                    "route": {"origin": "SYD", "destination": "NRT"},
                    "offer": {},
                    "note_id": "fulloch-reports/2026-08-27-12345678",
                    "report_url": "/reports/fulloch-reports/2026-08-27-12345678",
                    "prices_can_change": True,
                }
            },
        )
    ]


def test_completed_research_job_emits_generated_report_card():
    from core.background_jobs import BackgroundJob, JobSnapshot, JobStatus

    a = _import_assistant_module()
    assistant = a.Assistant.__new__(a.Assistant)
    assistant._history = []
    assistant._completed_thinking_reports = {}
    assistant._trim_history = lambda: None
    assistant._dispatch_event = lambda _event: None
    emitted = []
    assistant._emit_turn_event = lambda *args, **kwargs: emitted.append((args, kwargs))
    job = BackgroundJob(
        "12345678",
        JobSnapshot("Compare heat pumps", origin_source="integration"),
        status=JobStatus.READY,
        summary="A heat pump is likely suitable with insulation improvements.",
        note_id="fulloch-reports/2026-08-27-12345678",
        created_at=1_788_000_000,
    )

    assistant._on_thinking_job_status(job)

    assert emitted == [
        (
            ("assistant", "I've completed the research report and saved the full version.", "proactive"),
            {
                "artifact": {
                    "type": "generated_report",
                    "title": "Compare heat pumps",
                    "created_at": 1_788_000_000,
                    "summary": "A heat pump is likely suitable with insulation improvements.",
                    "report_url": "/reports/fulloch-reports/2026-08-27-12345678",
                }
            },
        )
    ]


def _import_assistant_module():
    """Import core.assistant with the heavy audio/ASR/SLM/TTS deps stubbed."""
    import types

    fake = {
        "core.audio": ["AudioCapture"],
        "core.asr": ["load_asr_pipeline"],
        "core.tts": [
            "set_voice",
            "warmup_model",
            "synthesize",
            "play_chunks",
            "speak_stream",
            "set_output_device",
            "set_tts_active_event",
            "model",
        ],
        "core.slm": ["load_slm", "generate_slm"],
    }
    for name, attrs in fake.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, lambda *a, **k: None)
        if name == "core.slm":
            # Real exception class — assistant.py does `except
            # ContextExhaustedError`, which a lambda stub can't satisfy.
            mod.ContextExhaustedError = type("ContextExhaustedError", (RuntimeError,), {})
            mod.RemoteUnreachable = type("RemoteUnreachable", (RuntimeError,), {})
        sys.modules[name] = mod
    import core.assistant as assistant  # noqa: E402

    return assistant
