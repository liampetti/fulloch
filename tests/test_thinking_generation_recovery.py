import threading
import time
from types import SimpleNamespace

import pytest

from core.background_jobs import BackgroundJob, JobSnapshot
from core.slm import RemoteUnreachable
from core.thinking_budget import ThinkingBudget
from core.thinking_worker import ThinkingWorker, _budgeted_generate, _recover_empty_section


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


def test_planning_schema_requires_report_title():
    calls = []

    def generate(_model, **kwargs):
        calls.append(kwargs)
        return ('{"report_title":"XYZQ outlook","chapters":[{"title":"XYZQ",'
                '"priority":"high","questions":["What changed?"],'
                '"tools":[],"queries":["XYZQ news"]}]}')

    worker = ThinkingWorker(object(), None, None, generate, lambda: {}, None,
                            lambda *_: "", threading.Lock(), 1)
    result = _budgeted_generate(worker, ThinkingBudget(worker.model, {}), "chapter_planning",
                                system_prompt="Plan XYZQ", user_prompt="Plan")
    assert '"report_title":"XYZQ outlook"' in result
    assert "report_title" in calls[0]["response_schema"]["required"]


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


def test_backend_recovery_waits_for_ready_then_resumes_with_durable_state(monkeypatch):
    import core.thinking_worker as module

    model = SimpleNamespace(_fulloch_backend_state="RECOVERING", _fulloch_process_generation=1)
    checkpoints = []
    jobs = SimpleNamespace(checkpoint=lambda job: checkpoints.append(job.checkpoint()))
    worker = ThinkingWorker(model, None, jobs, None, lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    job = BackgroundJob("recovery", JobSnapshot("Research"))
    job.attempted_actions.add("previous-action")
    attempts = []

    def run_once(current, _cancelled, **_kwargs):
        attempts.append(model._fulloch_backend_state)
        if len(attempts) == 1:
            raise RemoteUnreachable("backend recovering")
        assert current.attempted_actions == {"previous-action"}
        assert "backend_recovery" not in current.synthesis_state
        return "verified result", "state"

    def recover(_seconds):
        assert len(attempts) == 1
        model._fulloch_backend_state = "READY"
        model._fulloch_process_generation = 2

    monkeypatch.setattr(worker, "_run_once", run_once)
    monkeypatch.setattr(module.time, "sleep", recover)
    assert worker.run(job, lambda: False) == ("verified result", "state")
    assert attempts == ["RECOVERING", "READY"]
    assert any(item["synthesis_state"].get("backend_recovery_limits", {}).get("attempts") == 1
               for item in checkpoints)
    assert job.synthesis_state["backend_recovery_limits"]["attempts"] == 1


def test_backend_recovery_obeys_persisted_retry_limit_and_cancellation(monkeypatch):
    model = SimpleNamespace(_fulloch_backend_state="READY")
    jobs = SimpleNamespace(checkpoint=lambda _job: None)
    worker = ThinkingWorker(model, None, jobs, None, lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    job = BackgroundJob("limited", JobSnapshot("Research"))
    job.synthesis_state["backend_recovery_limits"] = {"attempts": 2, "deadline_epoch": time.time() + 60}
    monkeypatch.setattr(worker, "_run_once", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RemoteUnreachable("still unavailable")
    ))
    with pytest.raises(RemoteUnreachable):
        worker.run(job, lambda: False)
    assert job.synthesis_state["backend_recovery_limits"]["attempts"] == 2

    job.synthesis_state["backend_recovery_limits"]["attempts"] = 0
    model._fulloch_backend_state = "RECOVERING"
    calls = iter((False, False, True))
    assert worker.run(job, lambda: next(calls, True)) == ("", "")


def test_backend_recovery_refuses_resume_when_checkpoint_fails(monkeypatch):
    model = SimpleNamespace(_fulloch_backend_state="READY")

    def fail_checkpoint(job):
        job.checkpoint_error = True

    worker = ThinkingWorker(model, None, SimpleNamespace(checkpoint=fail_checkpoint), None,
                            lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    job = BackgroundJob("not-durable", JobSnapshot("Research"))
    monkeypatch.setattr(worker, "_run_once", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RemoteUnreachable("disconnected")
    ))
    with pytest.raises(RemoteUnreachable):
        worker.run(job, lambda: False)
    assert job.checkpoint_error


def test_backend_recovery_stops_after_three_failures(monkeypatch):
    model = SimpleNamespace(_fulloch_backend_state="READY")
    worker = ThinkingWorker(model, None, SimpleNamespace(checkpoint=lambda _job: None), None,
                            lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    job = BackgroundJob("three-failures", JobSnapshot("Research"))
    attempts = []

    def fail(*_args, **_kwargs):
        attempts.append(1)
        raise RemoteUnreachable("backend crashed")

    monkeypatch.setattr(worker, "_run_once", fail)
    with pytest.raises(RemoteUnreachable, match="backend crashed"):
        worker.run(job, lambda: False)
    assert len(attempts) == 3


def test_backend_recovery_deadline_and_retry_budget_survive_restart(monkeypatch):
    import core.thinking_worker as module

    model = SimpleNamespace(_fulloch_backend_state="RECOVERING")
    checkpoints = []
    jobs = SimpleNamespace(checkpoint=lambda job: checkpoints.append(job.checkpoint()))
    worker = ThinkingWorker(model, None, jobs, None, lambda: {}, None, lambda *_: "", threading.Lock(), 1)
    job = BackgroundJob("deadline", JobSnapshot("Research"))
    monkeypatch.setattr(module, "BACKEND_RECOVERY_SECONDS", 0)
    monkeypatch.setattr(worker, "_run_once", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RemoteUnreachable("backend recovering")
    ))

    with pytest.raises(RemoteUnreachable):
        worker.run(job, lambda: False)
    restored = BackgroundJob.from_checkpoint(checkpoints[-1])
    assert restored.synthesis_state["backend_recovery_limits"]["attempts"] == 1
    assert restored.synthesis_state["backend_recovery"]["checkpointed"] is True
    assert restored.synthesis_state["backend_recovery_limits"]["deadline_epoch"] <= time.time()
    with pytest.raises(RemoteUnreachable):
        worker.run(restored, lambda: False)
    assert restored.synthesis_state["backend_recovery_limits"]["attempts"] == 2
