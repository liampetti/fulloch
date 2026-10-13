import threading
import time
from types import SimpleNamespace

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.thinking_budget import ThinkingBudget
from core.thinking_worker import ThinkingWorker, _recover_empty_section


def test_empty_recovery_changes_strategy_once_and_circuit_survives_resume():
    job = BackgroundJob("job", JobSnapshot("Research"))
    calls = []

    def generate(_model, **kwargs):
        calls.append(kwargs)
        return ""

    worker = ThinkingWorker(object(), None, SimpleNamespace(update_stage=lambda *_: None),
                            generate, lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    budget = ThinkingBudget(worker.model, job.model_budget)
    for fingerprint in ("one", "one", "two", "three"):
        assert not _recover_empty_section(worker, job, budget, fingerprint, "short", "write", lambda: False)
    assert len(calls) == 2
    assert all(call["max_new_tokens"] == 4096 and not call["thinking_mode"]
               and call["output_policy"] == "research" for call in calls)
    resumed = BackgroundJob.from_checkpoint(job.checkpoint())
    assert not _recover_empty_section(worker, resumed, ThinkingBudget(worker.model, resumed.model_budget),
                                     "four", "short", "write", lambda: False)
    assert len(calls) == 2


def test_stage_policy_and_actual_usage_are_separate_from_reservations():
    state = {
        "stage_policies": {"section": {"output_tokens": 6144, "thinking_mode": True}},
        "deadline_epoch": time.time() + 300,
        "responses": [{
            "prompt_tokens": 128, "ttft_ms": 100,
            "completion_tokens": 6144, "elapsed_ms": 10_100,
        }],
    }
    budget = ThinkingBudget(object(), state)
    limit, started = budget.begin("section", system_prompt="short")
    budget.finish("section", started, "Answer", diagnostics={"completion_tokens": 300,
                  "reasoning_tokens": 200, "termination": "completed"})
    assert limit == state["reserved_tokens"] == 6144
    assert state["actual_generated_tokens"] == 300
    assert state["responses"][-1]["reasoning_tokens"] == 200


def test_deep_think_generation_always_starts_with_fresh_messages():
    calls = []
    worker = ThinkingWorker(object(), None, None, lambda _model, **kwargs: calls.append(kwargs) or "ok",
                            lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    worker._generate(system_prompt="Only plan research chapters", user_prompt="Plan")
    worker._generate(system_prompt="Only audit these source claims", user_prompt="Audit")
    assert all(call["history"] == [] for call in calls)
    assert calls[0]["history"] is not calls[1]["history"]
    with pytest.raises(ValueError, match="fresh context"):
        worker._generate(system_prompt="Audit", history=[{"role": "user", "content": "old chat"}])
    assert len(calls) == 2


def test_stage_prompts_receive_only_their_bounded_stage_payloads():
    from core.thinking_worker import (
        MAX_THINKING_CAPABILITY_DESCRIPTION_CHARS,
        _describe_thinking_capability,
    )
    from tools.tool_registry import Param, ToolSchema
    from utils.prompts import get_thinking_chapter_plan_prompt, get_thinking_report_audit_prompt

    schema = ToolSchema(
        "lookup", "A selected read-only lookup. " * 100,
        [Param("query", True), Param("profile", False, "research")], thinking_outcome=True,
    )
    capability = _describe_thinking_capability("lookup", schema)
    plan = get_thinking_chapter_plan_prompt(
        "Research this assignment", "bounded scope", "one obligation", capability,
    )
    audit = get_thinking_report_audit_prompt(
        "Research this assignment", '{"claims":[]}', "One cited finding.",
    )

    assert len(capability) <= MAX_THINKING_CAPABILITY_DESCRIPTION_CHARS + 100
    assert "lookup" in plan and "lookup" not in audit
    assert "entire foreground tool registry" not in plan
    assert "background thread transcript" not in audit


def test_one_slot_worker_releases_model_lock_between_calls_for_foreground_turns():
    lock = threading.Lock()
    calls = []

    def generate(_model, **kwargs):
        assert lock.locked()
        calls.append(kwargs["user_prompt"])
        return "bounded stage result"

    worker = ThinkingWorker(
        object(), None, SimpleNamespace(update_stage=lambda *_: None),
        generate, lambda: {}, None, lambda *_: "", lock, 1,
    )

    worker._generate(system_prompt="Plan", user_prompt="first stage")
    assert lock.acquire(blocking=False), "foreground should acquire the slot between background calls"
    lock.release()
    worker._generate(system_prompt="Audit", user_prompt="next stage")

    assert calls == ["first stage", "next stage"]
    assert not lock.locked()
