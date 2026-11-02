"""Single-worker background thinking scheduler."""

import json
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.background_jobs import (  # noqa: E402
    MAX_ARTIFACTS,
    MAX_EVIDENCE_ENTRIES,
    MAX_EVIDENCE_STRING_CHARS,
    BackgroundJob,
    BackgroundJobManager,
    JobSnapshot,
    JobStatus,
)


def test_job_runs_with_immutable_snapshot():
    done = threading.Event()
    seen = {}

    def worker(job, _cancelled):
        seen["conversation"] = job.snapshot.conversation
        done.set()
        return "complete", "saved state"

    manager = BackgroundJobManager(worker)
    history = [{"role": "user", "content": "original"}]
    job_id = manager.submit("research", conversation=history)
    history[0]["content"] = "mutated"
    assert done.wait(1)
    assert seen["conversation"][0]["content"] == "original"
    assert manager.status(job_id)["status"] == JobStatus.READY
    assert manager.status(job_id)["note_id"] == ""


def test_job_status_is_string_serializable_on_supported_python_versions():
    assert isinstance(JobStatus.QUEUED, str)
    assert str(JobStatus.QUEUED) == "QUEUED"
    assert JobStatus.QUEUED.value == "QUEUED"


def test_queued_job_can_be_cancelled():
    blocker = threading.Event()
    manager = BackgroundJobManager(lambda _job, _cancelled: (blocker.wait(), ""))
    first = manager.submit("first")
    second = manager.submit("second")
    assert manager.cancel(second) is True
    blocker.set()
    assert manager.status(second)["status"] == JobStatus.CANCELLED
    assert manager.status(first) is not None


def test_memory_pressure_cancels_running_and_queued_jobs_without_retry():
    started = threading.Event()
    finished = threading.Event()
    calls = []

    def worker(job, cancelled):
        calls.append(job.id)
        started.set()
        while not cancelled():
            finished.wait(0.01)
        return "", ""

    manager = BackgroundJobManager(worker)
    first = manager.submit("first")
    assert started.wait(1)
    second = manager.submit("second")
    manager.cancel_for_memory_pressure()
    for _ in range(100):
        if manager.status(first)["status"] == JobStatus.CANCELLED:
            break
        finished.wait(0.01)
    assert manager.status(first)["status"] == JobStatus.CANCELLED
    assert manager.status(second)["status"] == JobStatus.CANCELLED
    assert calls == [first]


def test_reactive_worker_result_needs_input_without_becoming_a_report():
    done = threading.Event()

    def worker(_job, _cancelled):
        done.set()
        return "Reactive question: Which airport will you leave from?", ""

    manager = BackgroundJobManager(worker)
    job_id = manager.submit("find a flight")

    assert done.wait(1)
    assert manager.status(job_id)["status"] == JobStatus.NEEDS_INPUT
    assert manager.cancel(job_id) is False


def test_needs_input_job_resumes_with_user_clarification_and_keeps_evidence():
    done = threading.Event()
    calls = []

    def worker(job, _cancelled):
        calls.append(job.continuation)
        if not job.continuation:
            job.evidence.append({"tool": "lookup", "status": "evidence"})
            return "Reactive question: Did you mean 2027?", "first attempt"
        done.set()
        return "complete", "resumed attempt"

    manager = BackgroundJobManager(worker)
    job_id = manager.submit("find a flight")
    for _ in range(100):
        if manager.status(job_id)["status"] == JobStatus.NEEDS_INPUT:
            break
        threading.Event().wait(0.01)

    assert manager.resume_with_input(job_id, "yes") is True
    assert done.wait(1)
    assert manager.status(job_id)["status"] == JobStatus.READY
    assert calls == ["", "yes"]


def test_job_waits_for_idle_admission_and_resumes_after_pause():
    admitted = threading.Event()
    started = threading.Event()
    release = threading.Event()
    paused = threading.Event()

    def worker(_job, cancelled):
        started.set()
        while not release.wait(0.01):
            if cancelled():
                paused.set()
                return "", "checkpoint"
        return "complete", "state"

    manager = BackgroundJobManager(worker, admission_check=admitted.is_set)
    job_id = manager.submit("research")
    assert not started.wait(0.05)
    admitted.set()
    manager.wake()
    assert started.wait(1)
    admitted.clear()
    assert manager.pause_active() is True
    assert paused.wait(1)
    assert manager.status(job_id)["status"] == JobStatus.PAUSED

    admitted.clear()
    manager.wake()
    assert manager.status(job_id)["status"] == JobStatus.PAUSED
    admitted.set()
    release.set()
    manager.wake()
    for _ in range(100):
        if manager.status(job_id)["status"] == JobStatus.READY:
            break
        threading.Event().wait(0.01)
    assert manager.status(job_id)["status"] == JobStatus.READY


def test_job_outcomes_have_stable_artifact_references_and_bounded_evidence():
    job = BackgroundJob("job", JobSnapshot("Research something"))

    first_id = job.record_outcome(
        "lookup",
        "evidence",
        "One source.",
        {"body": "x" * (MAX_EVIDENCE_STRING_CHARS + 1)},
        ("lookup",),
        {"type": "source", "body": "y" * (MAX_EVIDENCE_STRING_CHARS + 1)},
    )
    for _ in range(MAX_EVIDENCE_ENTRIES + 1):
        job.record_outcome("lookup", "evidence", "Another source.", {}, (), None)
    for _ in range(MAX_ARTIFACTS):
        job.record_outcome("lookup", "evidence", "Artifact.", {}, (), {"type": "source"})

    assert first_id == "artifact-001"
    assert job.evidence[0]["artifact_id"] == first_id
    assert job.artifacts[first_id]["tool"] == "lookup"
    assert len(job.evidence[0]["evidence"]["body"]) == MAX_EVIDENCE_STRING_CHARS + 1
    assert len(job.artifacts[first_id]["data"]["body"]) == MAX_EVIDENCE_STRING_CHARS + 1
    assert len(job.evidence) == MAX_EVIDENCE_ENTRIES
    assert len(job.artifacts) == MAX_ARTIFACTS


def test_research_work_queue_states_and_references_survive_checkpoint():
    job = BackgroundJob("queue-job", JobSnapshot("Research"))
    unit = job.enqueue_work(
        "section-1", "section_write", references=["card-a", "card-a", "card-b"],
        requirement_ids=["answer"], fingerprint="f" * 64,
    )
    assert unit["references"] == ["card-a", "card-b"]
    assert job.enqueue_work("section-1", "section_write") is unit

    transitions = (
        "running", "packet_full", "queued", "running", "needs_evidence",
        "queued", "running", "complete",
    )
    for state in transitions:
        job.transition_work("section-1", state, reason="bounded transition")
    job.enqueue_work("blocked", "audit")
    job.transition_work("blocked", "blocked", reason="invalid output")
    job.enqueue_work("budget", "merge")
    job.transition_work("budget", "budget_limited", reason="stage allowance")
    for kind in ("source_chunk", "card_reduction", "audit", "merge"):
        job.enqueue_work(f"pending-{kind}", kind, references=["artifact-1"])

    resumed = BackgroundJob.from_checkpoint(job.checkpoint())

    assert resumed.checkpoint_version == 6
    assert [(item["id"], item["state"]) for item in resumed.work_queue[:3]] == [
        ("section-1", "complete"), ("blocked", "blocked"), ("budget", "budget_limited"),
    ]
    assert resumed.work_queue[0]["attempts"] == 3
    assert resumed.work_queue[0]["references"] == ["card-a", "card-b"]
    assert {item["kind"] for item in resumed.work_queue[3:]} == {
        "source_chunk", "card_reduction", "audit", "merge",
    }


def test_changed_work_fingerprint_reopens_only_that_unit_without_resetting_attempts():
    job = BackgroundJob("fingerprints", JobSnapshot("Research"))
    first = job.enqueue_work("section-a", "section_write", references=["card-a"], fingerprint="old-a")
    second = job.enqueue_work("section-b", "section_write", references=["card-b"], fingerprint="stable-b")
    job.transition_work("section-a", "running")
    job.transition_work("section-a", "complete")
    job.transition_work("section-b", "running")
    job.transition_work("section-b", "complete")

    reopened = job.enqueue_work(
        "section-a", "section_write", references=["card-a", "card-new"],
        requirement_ids=["obligation-a"], fingerprint="new-a",
    )

    assert reopened is first
    assert first["state"] == "queued" and first["reason"] == "fingerprint_changed"
    assert first["attempts"] == 1
    assert first["references"] == ["card-a", "card-new"]
    assert first["requirement_ids"] == ["obligation-a"]
    assert second["state"] == "complete" and second["fingerprint"] == "stable-b"


def test_atomic_checkpoint_restores_accepted_batch_but_excludes_unaudited_draft(tmp_path):
    job = BackgroundJob(
        "resume-job", JobSnapshot("Research a bounded assignment"), status=JobStatus.RUNNING,
    )
    job.report_draft = "UNAUDITED DRAFT MUST NOT RESUME AS ACCEPTED WORK"
    job.chapter_plan = [{
        "id": "chapter-01", "title": "Accepted section", "analysis_state": "complete",
        "body": "Accepted section body.", "digest": "Accepted cited digest.",
        "analysis_fingerprint": "fingerprint-01",
    }]
    job.work_queue = [{
        "id": "section-01", "kind": "section_write", "state": "complete",
        "references": ["card-01"], "requirement_ids": ["answer"],
    }]

    writer = BackgroundJobManager(lambda *_: ("unused", ""), checkpoint_dir=tmp_path)
    writer.checkpoint(job)
    saved = json.loads((tmp_path / "resume-job.json").read_text())
    assert saved["report_draft"] == ""

    restored_done = threading.Event()
    seen = {}

    def resume(restored, _cancelled):
        seen["job"] = restored
        restored_done.set()
        return "resumed", "state"

    manager = BackgroundJobManager(resume, checkpoint_dir=tmp_path)
    assert restored_done.wait(1)

    assert manager.status("resume-job")["status"] == JobStatus.READY
    assert seen["job"].report_draft == ""
    assert seen["job"].chapter_plan[0]["body"] == "Accepted section body."
    assert seen["job"].chapter_plan[0]["digest"] == "Accepted cited digest."
    assert seen["job"].work_queue[0]["state"] == "complete"


def test_chapter_digest_checkpoint_roundtrip_preserves_bindings_and_runtime_state(tmp_path):
    job = BackgroundJob("digest-job", JobSnapshot("Research"), status=JobStatus.PAUSED)
    job.chapter_plan = [{"id": "chapter-1", "evidence_digest_ids": {"card-b", "card-a"}}]
    job.attempted_actions = {"action-b", "action-a"}
    job.exhausted_capabilities = {"search"}
    job.model_budget = {"stages": {"chapter_digest": {"calls": 1}}}
    job.fact_cards = [{"id": "card-a", "source_id": "source-1"}]
    manager = BackgroundJobManager(lambda *_: ("unused", ""), checkpoint_dir=tmp_path)

    manager.checkpoint(job)
    saved = json.loads((tmp_path / "digest-job.json").read_text())
    restored = BackgroundJob.from_checkpoint(saved)

    assert saved["chapter_plan"][0]["evidence_digest_ids"] == ["card-a", "card-b"]
    assert job.chapter_plan[0]["evidence_digest_ids"] == {"card-a", "card-b"}
    assert restored.chapter_plan == saved["chapter_plan"]
    assert restored.attempted_actions == job.attempted_actions
    assert restored.exhausted_capabilities == job.exhausted_capabilities
    assert restored.model_budget == job.model_budget
    assert restored.fact_cards == job.fact_cards
    assert not job.checkpoint_error


def test_checkpoint_failure_preserves_last_good_file_and_reports_recovery(tmp_path, caplog):
    job = BackgroundJob("failure-job", JobSnapshot("Research"))
    manager = BackgroundJobManager(lambda *_: ("unused", ""), checkpoint_dir=tmp_path)
    manager.checkpoint(job)
    path = tmp_path / "failure-job.json"
    original = path.read_bytes()
    job.synthesis_state["unexpected_set"] = {"not silently discarded"}

    with caplog.at_level(logging.INFO, logger="core.background_jobs"):
        manager.checkpoint(job)
        manager.checkpoint(job)
        assert job.view()["checkpoint_error"] is True
        assert path.read_bytes() == original
        assert caplog.text.count("Could not persist thinking job checkpoint") == 1
        del job.synthesis_state["unexpected_set"]
        job.status = JobStatus.FAILED
        manager.checkpoint(job)

    assert not job.checkpoint_error
    assert json.loads(path.read_text())["status"] == "FAILED"
    assert "checkpoint persistence recovered" in caplog.text


def test_lifecycle_keeps_queued_delivery_pending_and_reports_partial_terminal_assessment():
    done = threading.Event()
    seen = {}

    def worker(job, _cancelled):
        seen["job"] = job
        job.synthesis_state["delivery_assessment"] = {"status": "partial"}
        job.synthesis_state["partial_report"] = True
        done.set()
        return "A supported partial report.", ""

    manager = BackgroundJobManager(worker, admission_check=lambda: False)
    job_id = manager.submit("Research the requested comparison")
    queued = manager._jobs[job_id]
    assert queued.snapshot.task_record["execution_status"] == "queued"
    assert queued.snapshot.task_record["delivery_status"] == "pending"
    assert "completed" not in queued.stage.casefold()

    manager._admission_check = lambda: True
    manager.wake()
    assert done.wait(1)
    for _ in range(100):
        if manager.status(job_id)["status"] == JobStatus.READY:
            break
        threading.Event().wait(0.01)

    completed = seen["job"]
    assert completed.snapshot.task_record["execution_status"] == "completed"
    assert completed.snapshot.task_record["delivery_status"] == "partial"
    assert completed.stage == "Report ready"


def test_worker_failure_persists_actual_stage_and_exception_reason(tmp_path):
    done = threading.Event()

    def worker(job, _cancelled):
        job.stage = "card extraction"
        raise ValueError("source record failed schema validation")

    manager = BackgroundJobManager(
        worker, task_ledger_path=tmp_path / "tasks.json", checkpoint_dir=tmp_path / "checkpoints",
    )
    job_id = manager.submit("Research documented options")
    for _ in range(100):
        job = manager._jobs[job_id]
        if job.status == JobStatus.FAILED:
            done.set()
            break
        threading.Event().wait(0.01)

    assert done.wait(0)
    assert job.snapshot.task_record["failure_stage"] == "card extraction"
    assert job.snapshot.task_record["failure_reason"] == (
        "ValueError: source record failed schema validation"
    )
    assert job.snapshot.task_record["delivery_status"] == "blocked"


def test_worker_failure_does_not_deliver_unaudited_draft(tmp_path):
    done = threading.Event()

    def worker(job, _cancelled):
        job.report_draft = "Unsupported claim awaiting audit"
        job.report_bindings = [{"card_id": "C1"}]
        job.synthesis_state["partial_report"] = True
        raise RuntimeError("backend lost")

    def notify(job):
        if job.status == JobStatus.FAILED:
            done.set()

    manager = BackgroundJobManager(worker, status_listener=notify, checkpoint_dir=tmp_path)
    job_id = manager.submit("Research")
    assert done.wait(2)
    job = manager._jobs[job_id]
    assert job.summary == ""
    assert "partial_report" not in job.synthesis_state
    assert job.snapshot.task_record["delivery_status"] == "blocked"
    for _ in range(100):
        if not (tmp_path / f"{job_id}.json").exists():
            break
        threading.Event().wait(0.01)
    assert not (tmp_path / f"{job_id}.json").exists()


def test_status_separates_acknowledgement_latency_from_active_job_duration():
    job = BackgroundJob("timed-job", JobSnapshot("Research"))
    job.snapshot.task_record.update({
        "acknowledgement_seconds": 0.0123,
        "started_at": time.time() - 4,
        "finished_at": time.time() - 1,
    })
    job.model_budget.update({
        "context_tokens": 8192,
        "request_reservations": [{
            "stage": "planning", "input_tokens": 500, "output_reservation": 1024,
        }],
    })

    view = job.view()

    assert view["acknowledgement_seconds"] == 0.0123
    assert 2.9 <= view["job_duration_seconds"] <= 3.1
    assert view["context_tokens"] == 8192
    assert view["request_metrics"][0]["output_reservation"] == 1024


def test_resolved_clarification_preserves_nested_evidence_shape():
    job = BackgroundJob("job", JobSnapshot("Investigate"))
    recovery = {"kind": "confirmation", "resume_action": {"intent": "lookup", "args": ["fixed"]}}
    job.record_outcome("lookup", "needs_input", "Need confirmation.", {}, (), None, recovery)
    job.resolve_pending_clarification(recovery, "confirmed")
    job.record_outcome(
        "lookup",
        "evidence",
        "One itinerary.",
        {"offer": {"outbound": {"departure": {"id": "HAM"}}}},
        (),
        {"offer": {"outbound": {"departure": {"id": "HAM"}}}},
    )

    assert job.evidence[0]["resolution"] == {"status": "confirmed"}
    assert job.evidence[1]["evidence"]["offer"]["outbound"]["departure"] == {"id": "HAM"}


def test_stage_diagnostics_checkpoint_safe_codes_without_logging_prompts_or_source(caplog):
    job = BackgroundJob("job", JobSnapshot("Research something"))
    secret = "Authorization: Bearer test-secret; source body: confidential paragraph"

    with caplog.at_level(logging.INFO, logger="core.background_jobs"):
        job.record_stage_event(
            "chapter_research", "action_rejection", "malformed_json", record_ids=["src-safe-card"],
        )
        job.record_stage_event("chapter_research", "action_rejection", secret)

    restored = BackgroundJob.from_checkpoint(job.checkpoint())

    assert restored.stage_log == [
        {
            "stage": "chapter_research", "category": "action_rejection",
            "reason": "malformed_json", "count": 1, "record_ids": ["src-safe-card"],
        },
        {
            "stage": "chapter_research", "category": "action_rejection",
            "reason": "invalid_action_shape", "count": 1, "record_ids": [],
        },
    ]
    assert "category=action_rejection reason=malformed_json" in caplog.text
    assert secret not in caplog.text

    for _ in range(140):
        job.record_stage_event("synthesis", "evidence_omission", "not_synthesized")
    assert len(job.checkpoint()["stage_log"]) == 128


def test_terminal_checkpoints_are_archived_not_rerun(tmp_path):
    """A restart must not re-run jobs that already reached a terminal state.

    Checkpoints for READY/FAILED/CANCELLED jobs are recorded for status
    lookups and archived (file removed) so a fresh process does not spam the
    dashboard with stale "I couldn't complete that report" messages.
    In-flight jobs are re-queued instead.
    """
    def terminal_job(job_id, status, *, error="", execution_status=""):
        job = BackgroundJob(job_id, JobSnapshot("Research something"), status=status)
        job.error = error
        job.stage = "Thinking" if status is JobStatus.FAILED else "Report ready"
        if execution_status:
            job.snapshot.task_record["execution_status"] = execution_status
        return job

    writer = BackgroundJobManager(lambda *_: ("unused", ""), checkpoint_dir=tmp_path)
    writer.checkpoint(terminal_job("done-job", JobStatus.READY))
    writer.checkpoint(terminal_job(
        "failed-job", JobStatus.FAILED,
        error="RuntimeError: local language model is not loaded",
        execution_status="failed",
    ))
    writer.checkpoint(terminal_job("cancelled-job", JobStatus.CANCELLED, execution_status="cancelled"))
    # An in-flight job left RUNNING by a crashed process must resume, not archive.
    writer.checkpoint(terminal_job("queued-job", JobStatus.RUNNING, execution_status="running"))

    done = threading.Event()
    seen = {}

    def worker(restored, _cancelled):
        seen[restored.id] = restored
        done.set()
        return "resumed", "state"

    manager = BackgroundJobManager(worker, checkpoint_dir=tmp_path)
    assert done.wait(1)
    assert list(seen) == ["queued-job"], "terminal jobs must not be re-queued"

    # In-flight job was re-queued and ran.
    assert manager.status("queued-job")["status"] == JobStatus.READY
    # Terminal jobs are visible to status() for lookups...
    assert manager.status("failed-job")["status"] == JobStatus.FAILED
    assert manager.status("failed-job")["error"] == "RuntimeError: local language model is not loaded"
    assert manager.status("done-job")["status"] == JobStatus.READY
    assert manager.status("cancelled-job")["status"] == JobStatus.CANCELLED
    # ...and their checkpoint files are archived away.
    assert not (tmp_path / "done-job.json").exists()
    assert not (tmp_path / "failed-job.json").exists()
    assert not (tmp_path / "cancelled-job.json").exists()


def test_running_checkpoint_is_requeued_on_restart(tmp_path):
    """A job interrupted mid-run (checkpoint left RUNNING) resumes on restart."""
    writer = BackgroundJobManager(lambda *_: ("unused", ""), checkpoint_dir=tmp_path)
    job = BackgroundJob("inflight-job", JobSnapshot("Research something"), status=JobStatus.RUNNING)
    writer.checkpoint(job)

    done = threading.Event()

    def worker(restored, _cancelled):
        done.set()
        return "resumed", "state"

    manager = BackgroundJobManager(worker, checkpoint_dir=tmp_path)
    assert done.wait(1)
    assert manager.status("inflight-job")["status"] == JobStatus.READY
    # Once the resumed job finishes, its working checkpoint is removed.
    for _ in range(100):
        if not (tmp_path / "inflight-job.json").exists():
            break
        threading.Event().wait(0.01)
    assert not (tmp_path / "inflight-job.json").exists()
