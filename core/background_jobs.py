"""Bounded, single-worker background thinking job scheduler."""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

MAX_QUEUED_JOBS = 16
MAX_STATE_CHARS = 12_000
MAX_EVIDENCE_ENTRIES = 36
MAX_ARTIFACTS = 36
MAX_EVIDENCE_MAPPING_ITEMS = 24
MAX_EVIDENCE_LIST_ITEMS = 12
MAX_EVIDENCE_STRING_CHARS = 1_000


def _bounded_evidence_value(value, depth: int = 0):
    """Keep job evidence structured and bounded before retaining it in memory."""
    if depth >= 4:
        return str(value)[:MAX_EVIDENCE_STRING_CHARS]
    if isinstance(value, str):
        return value[:MAX_EVIDENCE_STRING_CHARS]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _bounded_evidence_value(item, depth + 1)
            for key, item in list(value.items())[:MAX_EVIDENCE_MAPPING_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_evidence_value(item, depth + 1)
            for item in value[:MAX_EVIDENCE_LIST_ITEMS]
        ]
    return str(value)[:MAX_EVIDENCE_STRING_CHARS]


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    READY = "READY"
    NEEDS_INPUT = "NEEDS_INPUT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class JobSnapshot:
    task: str
    conversation: tuple[dict, ...] = ()
    notes: str = ""
    origin_satellite_id: str | None = None
    origin_source: str = "integration"


@dataclass
class BackgroundJob:
    id: str
    snapshot: JobSnapshot
    status: JobStatus = JobStatus.QUEUED
    state: str = ""
    summary: str = ""
    error: str = ""
    note_id: str = ""
    artifact: dict | None = None
    evidence: list[dict] = field(default_factory=list)
    artifacts: dict[str, dict] = field(default_factory=dict)
    stage: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    pause_requested: threading.Event = field(default_factory=threading.Event, repr=False)

    def record_outcome(
        self,
        tool: str,
        status: str,
        scope: str,
        evidence: dict,
        next_actions: tuple[str, ...],
        artifact: dict | None,
    ) -> str | None:
        """Record one typed result and return its job-local artifact reference."""
        artifact_id = None
        if artifact is not None and len(self.artifacts) < MAX_ARTIFACTS:
            artifact_id = f"artifact-{len(self.artifacts) + 1:03d}"
            self.artifacts[artifact_id] = {
                "tool": tool,
                "data": _bounded_evidence_value(artifact),
            }
        if len(self.evidence) < MAX_EVIDENCE_ENTRIES:
            self.evidence.append(
                {
                    "tool": tool,
                    "status": status,
                    "scope": scope[:MAX_EVIDENCE_STRING_CHARS],
                    "evidence": _bounded_evidence_value(evidence),
                    "next_actions": list(next_actions),
                    "artifact_id": artifact_id,
                }
            )
        return artifact_id

    def view(self) -> dict:
        return {
            "id": self.id,
            "task": self.snapshot.task,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
            "note_id": self.note_id,
            "stage": self.stage,
            "origin_source": self.snapshot.origin_source,
            "origin_satellite_id": self.snapshot.origin_satellite_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


Worker = Callable[[BackgroundJob, Callable[[], bool]], tuple[str, str]]
StatusListener = Callable[[BackgroundJob], None]
AdmissionCheck = Callable[[], bool]


class BackgroundJobManager:
    """Own exactly one bounded background worker and FIFO queue."""

    def __init__(
        self,
        worker: Worker,
        status_listener: StatusListener | None = None,
        admission_check: AdmissionCheck | None = None,
    ):
        self._worker = worker
        self._status_listener = status_listener
        self._admission_check = admission_check or (lambda: True)
        self._jobs: dict[str, BackgroundJob] = {}
        self._queue: queue.Queue[str] = queue.Queue(MAX_QUEUED_JOBS)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="thinking-background-worker"
        )
        self._thread.start()

    def _notify(self, job: BackgroundJob) -> None:
        if self._status_listener is None:
            return
        try:
            self._status_listener(job)
        except Exception:  # Observability must never stop the worker.
            logger.exception("Background thinking status listener failed")

    def wake(self) -> None:
        """Recheck idle admission after foreground activity finishes."""
        self._wake.set()

    def pause_active(self) -> bool:
        """Yield the active generation to a foreground request, if any."""
        with self._lock:
            job = next(
                (item for item in self._jobs.values() if item.status is JobStatus.RUNNING), None
            )
            if job is None:
                return False
            job.pause_requested.set()
            return True

    def submit(
        self,
        task: str,
        conversation: list[dict] | None = None,
        notes: str = "",
        origin_satellite_id: str | None = None,
        origin_source: str = "integration",
    ) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        snapshot = JobSnapshot(
            task.strip(),
            tuple(dict(item) for item in (conversation or ())),
            notes[:MAX_STATE_CHARS],
            origin_satellite_id,
            origin_source,
        )
        job = BackgroundJob(uuid.uuid4().hex, snapshot)
        with self._lock:
            self._jobs[job.id] = job
        try:
            self._queue.put_nowait(job.id)
        except queue.Full:
            with self._lock:
                self._jobs.pop(job.id, None)
            raise RuntimeError("thinking job queue is full") from None
        self._notify(job)
        self.wake()
        return job.id

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.view() if job else None

    def active(self) -> dict | None:
        with self._lock:
            job = next(
                (
                    item
                    for item in self._jobs.values()
                    if item.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}
                ),
                None,
            )
            return job.view() if job else None

    def update_stage(self, job: BackgroundJob, stage: str) -> None:
        """Publish a concise worker milestone without exposing internal reasoning."""
        with self._lock:
            job.stage = stage[:160]
            job.updated_at = time.time()
        self._notify(job)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {
                JobStatus.READY,
                JobStatus.NEEDS_INPUT,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return False
            job.cancel_requested.set()
            if job.status is JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.updated_at = time.time()
                self._notify(job)
            return True

    def pause(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.RUNNING:
                return False
            job.pause_requested.set()
            return True

    def resume(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.PAUSED:
                return False
            job.pause_requested.clear()
            job.status = JobStatus.QUEUED
            job.updated_at = time.time()
        self._queue.put_nowait(job_id)
        self.wake()
        return True

    def _wait_for_admission(self, job: BackgroundJob) -> bool:
        """Wait without polling hot while a foreground turn owns the one slot."""
        while not job.cancel_requested.is_set() and not self._admission_check():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
        return not job.cancel_requested.is_set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            try:
                job_id = self._queue.get_nowait()
            except queue.Empty:
                self._wake.clear()
                continue
            with self._lock:
                job = self._jobs[job_id]
                if job.status is JobStatus.CANCELLED:
                    continue
            if not self._wait_for_admission(job):
                with self._lock:
                    job.status = JobStatus.CANCELLED
                    job.updated_at = time.time()
                self._notify(job)
                continue
            with self._lock:
                job.status = JobStatus.RUNNING
                job.stage = "Thinking"
                job.updated_at = time.time()
            self._notify(job)
            try:
                summary, state = self._worker(
                    job,
                    lambda job=job: job.cancel_requested.is_set() or job.pause_requested.is_set(),
                )
                with self._lock:
                    if job.cancel_requested.is_set():
                        job.status = JobStatus.CANCELLED
                    elif job.pause_requested.is_set():
                        job.status = JobStatus.PAUSED
                        job.stage = "Paused for foreground conversation"
                    elif summary.startswith("Reactive question:"):
                        job.summary = summary[:MAX_STATE_CHARS]
                        job.state = state[:MAX_STATE_CHARS]
                        job.status = JobStatus.NEEDS_INPUT
                        job.stage = "Input needed"
                    else:
                        job.summary = summary[:MAX_STATE_CHARS]
                        job.state = state[:MAX_STATE_CHARS]
                        job.status = JobStatus.READY
                        job.stage = "Report ready"
                    job.updated_at = time.time()
                self._notify(job)
                if job.status is JobStatus.PAUSED and not job.cancel_requested.is_set():
                    job.pause_requested.clear()
                    if self._wait_for_admission(job):
                        with self._lock:
                            job.status = JobStatus.QUEUED
                            job.stage = "Queued, waiting for a quiet moment"
                            job.updated_at = time.time()
                        self._queue.put_nowait(job.id)
                        self._notify(job)
                        self.wake()
                    else:
                        with self._lock:
                            job.status = JobStatus.CANCELLED
                            job.updated_at = time.time()
                        self._notify(job)
            except Exception as exc:  # Worker failures must not kill the scheduler.
                logger.exception("Background thinking job %s failed", job.id)
                with self._lock:
                    job.error = f"{type(exc).__name__}: {exc}"[:MAX_STATE_CHARS]
                    job.status = JobStatus.FAILED
                    job.updated_at = time.time()
                self._notify(job)
