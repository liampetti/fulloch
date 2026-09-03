"""Single-worker background thinking scheduler."""

import sys
import threading
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
    assert len(job.evidence[0]["evidence"]["body"]) == MAX_EVIDENCE_STRING_CHARS
    assert len(job.artifacts[first_id]["data"]["body"]) == MAX_EVIDENCE_STRING_CHARS
    assert len(job.evidence) == MAX_EVIDENCE_ENTRIES
    assert len(job.artifacts) == MAX_ARTIFACTS
