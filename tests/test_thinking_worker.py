"""Investigations tested through explicit model, tool and scheduler services."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core import thinking_worker as worker
from core.background_jobs import BackgroundJob, JobSnapshot
from tools.capabilities import ToolCapability
from tools.tool_registry import Param, ThinkingResult, ToolSchema


def make_worker(responses, *, invoke=None, access_class="read", schema=None, slots=2):
    """A real worker with a private registry and model; no Assistant globals."""
    capability = ToolCapability(
        name="lookup",
        invoke=invoke or MagicMock(return_value="source evidence"),
        source="native",
        timeout_seconds=1,
        format_result=lambda result: result,
        access_class=access_class,
    )
    registry = SimpleNamespace(
        _schemas={"lookup": schema} if schema else {}, canonical_name=lambda name: name
    )
    generate = MagicMock(side_effect=responses)
    return worker.ThinkingWorker(
        model=object(),
        grammar="grammar",
        jobs=MagicMock(),
        generate=generate,
        capabilities=lambda: {"lookup": capability},
        registry=registry,
        summarise_search=MagicMock(return_value="Summarised search evidence."),
        model_lock=threading.Lock(),
        server_slots=slots,
    )


def action(*args):
    return json.dumps({"actions": [{"intent": "lookup", "args": list(args)}]})


@pytest.mark.parametrize("access_class,expected_calls", [("read", 3), ("write", 0)])
def test_agentic_worker_exposes_only_read_tools_and_bounds_capability_calls(
    access_class, expected_calls
):
    invoke = MagicMock(return_value="Retrieved evidence.")
    count = 0

    def generate(_model, **kwargs):
        nonlocal count
        if kwargs.get("thinking_mode"):
            return "Final report."
        count += 1
        return action(count)

    service = make_worker(generate, invoke=invoke, access_class=access_class)
    report, findings = service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)
    assert report == "Final report."
    assert invoke.call_count == expected_calls
    assert count == expected_calls + 1
    assert (
        "Capability budget reached" if expected_calls else "Blocked unavailable capability"
    ) in findings


def test_typed_evidence_is_used_without_raw_worker_observations():
    service = make_worker(['{"reply":"enough"}', "## Summary\n\nA scoped answer."])
    job = BackgroundJob(
        "job",
        JobSnapshot("Investigate"),
        state="[worker]\nprivate planning text",
        evidence=[
            {
                "tool": "lookup",
                "status": "evidence",
                "scope": "One source.",
                "evidence": {"fact": "retrieved"},
            }
        ],
    )
    service.run(job, lambda: False)
    prompt = service.generate.call_args.kwargs["system_prompt"]
    assert "private planning text" not in prompt
    assert '"fact": "retrieved"' in prompt


def test_deep_think_worker_stops_on_a_duplicate_action():
    invoke = MagicMock(return_value="evidence")
    service = make_worker(
        [action("same"), action(" SAME "), "Evidence-based report."], invoke=invoke
    )
    report, findings = service.run(
        BackgroundJob("job", JobSnapshot("Investigate something")), lambda: False
    )
    invoke.assert_called_once_with(["same"], {})
    assert "Duplicate capability request for lookup" in findings
    assert report == "Evidence-based report."
    assert service.jobs.update_stage.call_args.args[1] == "Synthesising report"
    assert service.generate.call_count == 3
    for invocation in service.generate.call_args_list[:2]:
        assert invocation.kwargs["thinking_mode"] is False
        assert invocation.kwargs["grammar"] == "grammar"
        assert invocation.kwargs["max_new_tokens"] == 1024
        assert invocation.kwargs["read_timeout"] == worker.DEEP_THINK_READ_TIMEOUT_S
        assert invocation.kwargs["generation_timeout"] == worker.DEEP_THINK_GENERATION_TIMEOUT_S
        assert invocation.kwargs["recover_on_failure"] is False
    synthesis = service.generate.call_args.kwargs
    assert synthesis["thinking_mode"] is True and synthesis["max_new_tokens"] == 8192
    assert "grammar" not in synthesis
    assert "evidence" in synthesis["system_prompt"]


def test_deep_think_cancel_after_tool_result_skips_report_synthesis():
    cancelled = threading.Event()

    def invoke(_args, _kwargs):
        cancelled.set()
        return "evidence"

    service = make_worker([action()], invoke=invoke)
    report, _ = service.run(BackgroundJob("job", JobSnapshot("Investigate")), cancelled.is_set)
    assert report == ""
    assert service.generate.call_count == 1


@pytest.mark.parametrize(
    "planning,marker",
    [
        (
            [
                '{"actions":[{"intent":"evaluate_itinerary","args":["truncated"]}',
                '{"reply":"enough"}',
            ],
            "invalid planning response; select the next capability",
        ),
        ([""], "Worker stopped without a next action"),
        (
            ['{"actions":[{"intent":"lookup","args":["truncated"]}'] * 11,
            "invalid planning response",
        ),
    ],
)
def test_deep_think_synthesises_collected_evidence_after_planning_failure(planning, marker):
    service = make_worker([action("query"), *planning, "Evidence-based final report."])
    report, findings = service.run(
        BackgroundJob("job", JobSnapshot("Investigate something")), lambda: False
    )
    assert report == "Evidence-based final report."
    assert marker in findings and "source evidence" in findings
    assert "evaluate_itinerary" not in report
    if len(planning) == 11:
        assert findings.count("invalid planning response") == 11
        assert service.generate.call_count == worker.MAX_THINKING_WORKER_CALLS + 1


def test_deep_think_empty_final_report_preserves_collected_evidence():
    service = make_worker(
        [action(), '{"reply":"enough"}', ""],
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Retrieved schedule.",
            evidence={"departure": "Tokyo", "arrival": "Dubai"},
            scope="One retrieved flight schedule.",
        ),
        schema=ToolSchema("lookup", "", [], thinking_outcome=True),
    )
    report, _ = service.run(
        BackgroundJob("job", JobSnapshot("Can this itinerary work?")), lambda: False
    )
    assert "## Collected evidence" in report and '"departure": "Tokyo"' in report


def test_deep_think_rejects_invalid_typed_outcome_before_recording_evidence():
    service = make_worker(
        [action(), '{"reply":"enough"}', "Report."],
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Bad result", status="exhausted", scope="Bad."
        ),
        schema=ToolSchema("lookup", "", [Param("refinement", False, "")], thinking_outcome=True),
    )
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    report, _ = service.run(job, lambda: False)
    assert report == "Report."
    assert job.evidence[0]["status"] == "failed" and job.evidence[0]["evidence"] == {}


def test_deep_think_typed_needs_input_stops_without_synthesising():
    service = make_worker(
        [action()],
        invoke=lambda _args, _kwargs: ThinkingResult(
            "Which city should I use?", status="needs_input", scope="The city was not supplied."
        ),
        schema=ToolSchema("lookup", "", [], thinking_outcome=True),
    )
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    report, _ = service.run(job, lambda: False)
    assert report == "Reactive question: Which city should I use?"
    assert job.evidence[0]["status"] == "needs_input"
    assert service.generate.call_count == 1


def test_deep_think_synthesises_preliminary_report_when_input_follows_evidence():
    results = iter(
        [
            ThinkingResult(
                "Retrieved schedule.", evidence={"schedule": True}, scope="One schedule."
            ),
            ThinkingResult(
                "Which exact time?", status="needs_input", scope="Event timing is missing."
            ),
        ]
    )
    service = make_worker(
        [action(), action("refine"), "Preliminary report."],
        invoke=lambda _args, _kwargs: next(results),
        schema=ToolSchema("lookup", "", [Param("refinement", False, "")], thinking_outcome=True),
    )
    report, findings = service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)
    assert report == "Preliminary report."
    assert "sufficient evidence exists for a preliminary scoped report" in findings


@pytest.mark.parametrize("slots", [1, 2])
def test_worker_owns_model_lock_only_for_single_slot_generation(slots):
    responses = iter([action(), '{"reply":"enough"}', "Report."])

    def generate(_model, **_kwargs):
        assert service.model_lock.locked() is (slots == 1)
        return next(responses)

    def invoke(_args, _kwargs):
        assert not service.model_lock.locked()
        return "Evidence."

    service = make_worker(generate, slots=slots, invoke=invoke)
    assert (
        service.run(BackgroundJob("job", JobSnapshot("Investigate")), lambda: False)[0] == "Report."
    )
    assert not service.model_lock.locked()
