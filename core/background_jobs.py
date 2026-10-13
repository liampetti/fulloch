"""Bounded, single-worker background thinking job scheduler."""

import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

MAX_QUEUED_JOBS = 16
MAX_STATE_CHARS = 12_000
MAX_EVIDENCE_ENTRIES = 36
MAX_ARTIFACTS = 36
MAX_EVIDENCE_MAPPING_ITEMS = 48
MAX_EVIDENCE_LIST_ITEMS = 24
MAX_EVIDENCE_STRING_CHARS = 8_000
MAX_EVIDENCE_DEPTH = 12
MAX_STAGE_LOG_ENTRIES = 128
WORK_UNIT_STATES = frozenset({
    "queued", "running", "packet_full", "needs_evidence", "complete", "blocked", "budget_limited",
})
MAX_PERSISTED_TASKS = 256

_STAGE_DIAGNOSTIC_REASONS = {
    "action_rejection": frozenset({
        "malformed_json", "invalid_action_shape", "unavailable_tool", "out_of_scope_tool",
        "invalid_arguments", "duplicate_action", "near_duplicate_action",
        "capability_exhausted", "capability_budget", "tool_budget", "decision_budget",
        "generation_failed", "generation_budget",
    }),
    "tool_blocked": frozenset({
        "unavailable_tool", "out_of_scope_tool", "capability_exhausted",
        "capability_budget", "tool_budget", "invalid_arguments",
    }),
    "evidence_omission": frozenset({
        "not_selected_by_requirements", "context_limit", "server_context_limit",
        "time_budget", "model_call_or_token_budget", "generation_failed",
        "empty_generation", "no_verified_section", "not_synthesized", "display_omitted",
        "ineligible_record", "no_question_eligible_cards",
    }),
    "citation_failure": frozenset({
        "missing_citation", "unknown_card_id", "invalid_support_id", "model_authored_marker",
        "invalid_passage_shape", "audit_unavailable", "repair_failed",
    }),
    "output_rejection": frozenset({
        "malformed_json", "invalid_output_shape", "missing_support_ids", "unknown_support_ids",
        "model_authored_marker", "invalid_passage_shape", "digest_contract",
    }),
    "generation_failure": frozenset({"model_error", "model_budget", "context_limit", "empty_output"}),
}
_STAGE_DIAGNOSTIC_STAGES = frozenset({
    "chapter_planning", "chapter_research", "capability_dispatch", "chapter_analysis",
    "report_audit", "final_assembly", "synthesis", "chapter_coverage",
})
_STAGE_DIAGNOSTIC_FALLBACK_REASONS = {
    "action_rejection": "invalid_action_shape",
    "tool_blocked": "unavailable_tool",
    "evidence_omission": "not_synthesized",
    "citation_failure": "invalid_support_id",
    "output_rejection": "invalid_output_shape",
    "generation_failure": "model_error",
}
_SAFE_DIAGNOSTIC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")


def _bounded_evidence_value(value, depth: int = 0):
    """Retain intact tool-owned records; prompt budgets are a separate concern.

    Collection size is bounded by tool contracts and the investigation call cap,
    not by cutting arbitrary strings, list entries, or mapping fields.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _bounded_evidence_value(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_evidence_value(item, depth + 1)
            for item in value
        ]
    return str(value)


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
    handoff: dict = field(default_factory=dict)
    origin_turn_id: str | None = None
    task_record: dict = field(default_factory=dict)


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
    pending_handoffs: list[dict] = field(default_factory=list)
    foreground_steps: list[dict] = field(default_factory=list)
    pending_clarification: dict = field(default_factory=dict)
    continuation: str = ""
    capability_calls: dict[str, int] = field(default_factory=dict)
    exhausted_capabilities: set[str] = field(default_factory=set)
    tool_budget_exhausted: bool = False
    attempted_actions: set[str] = field(default_factory=set)
    duplicate_actions_avoided: int = 0
    evidence_requirements: list[dict] = field(default_factory=list)
    action_selections: list[dict] = field(default_factory=list)
    stage_log: list[dict] = field(default_factory=list)
    fact_cards: list[dict] = field(default_factory=list)
    report_draft: str = ""
    report_audit: dict = field(default_factory=dict)
    report_revision: str = ""
    report_bindings: list[dict] = field(default_factory=list)
    research_context: dict = field(default_factory=dict)
    source_registry: dict[str, dict] = field(default_factory=dict)
    extraction_state: dict[str, dict] = field(default_factory=dict)
    section_plan: list[dict] = field(default_factory=list)
    chapter_plan: list[dict] = field(default_factory=list)
    synthesis_state: dict = field(default_factory=dict)
    work_queue: list[dict] = field(default_factory=list)
    model_budget: dict = field(default_factory=dict)
    planning_complete: bool = False
    checkpoint_version: int = 6
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
        recovery: dict | None = None,
        retrieved_at: str | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """Record one typed result and return its job-local artifact reference."""
        from .thinking_source_store import archive_source

        def archive_sources(value):
            if isinstance(value, dict):
                result = dict(value)
                sources = result.get("sources")
                if isinstance(sources, list):
                    result["sources"] = [archive_source(source) for source in sources]
                for key, item in list(result.items()):
                    if isinstance(item, dict):
                        result[key] = archive_sources(item)
                return result
            if isinstance(value, list):
                return [archive_sources(item) for item in value]
            return value

        evidence = archive_sources(evidence)
        artifact = archive_sources(artifact) if artifact is not None else None
        artifact_id = None
        retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
        if artifact is not None and len(self.artifacts) < MAX_ARTIFACTS:
            artifact = dict(artifact)
            artifact.setdefault("retrieved_at", retrieved_at)
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
                    "scope": scope,
                    "evidence": _bounded_evidence_value(evidence),
                    "next_actions": list(next_actions),
                    "recovery": _bounded_evidence_value(recovery or {}),
                    "artifact_id": artifact_id,
                    "retrieved_at": retrieved_at,
                    "metadata": _bounded_evidence_value(metadata or {}),
                }
            )
        return artifact_id

    def enqueue_work(
        self, unit_id: str, kind: str, *, references=(), requirement_ids=(), fingerprint: str = "",
    ) -> dict:
        """Create or return durable work using references, never copied source bodies."""
        existing = next((item for item in self.work_queue if item.get("id") == unit_id), None)
        if existing is not None:
            if fingerprint and existing.get("fingerprint") != fingerprint:
                existing.update({
                    "state": "queued", "fingerprint": str(fingerprint)[:128],
                    "reason": "fingerprint_changed", "references": list(dict.fromkeys(
                        str(value)[:160] for value in references if isinstance(value, (str, int))
                    ))[:100],
                    "requirement_ids": list(dict.fromkeys(
                        str(value)[:160] for value in requirement_ids if isinstance(value, (str, int))
                    ))[:100],
                })
            return existing
        unit = {
            "id": str(unit_id)[:160], "kind": str(kind)[:64], "state": "queued",
            "references": list(dict.fromkeys(
                str(value)[:160] for value in references if isinstance(value, (str, int))
            ))[:100],
            "requirement_ids": list(dict.fromkeys(
                str(value)[:160] for value in requirement_ids if isinstance(value, (str, int))
            ))[:100],
            "fingerprint": str(fingerprint)[:128], "reason": "", "attempts": 0,
        }
        self.work_queue.append(unit)
        return unit

    def transition_work(self, unit_id: str, state: str, *, reason: str = "") -> dict:
        """Persist a bounded queue transition; unknown states are rejected."""
        if state not in WORK_UNIT_STATES:
            raise ValueError(f"Unknown research work state: {state}")
        unit = next((item for item in self.work_queue if item.get("id") == unit_id), None)
        if unit is None:
            raise KeyError(unit_id)
        if state == "running":
            unit["attempts"] = min(10_000, int(unit.get("attempts", 0)) + 1)
        unit["state"] = state
        unit["reason"] = str(reason)[:200]
        return unit

    def checkpoint(self) -> dict:
        """JSON-compatible durable worker state (without threading primitives)."""
        from dataclasses import fields

        return deepcopy({
            item.name: sorted(getattr(self, item.name)) if isinstance(getattr(self, item.name), set)
            else dict(vars(self.snapshot)) if item.name == "snapshot"
            else str(self.status) if item.name == "status" else getattr(self, item.name)
            for item in fields(self) if item.name not in {"cancel_requested", "pause_requested"}
        })

    def record_stage_event(
        self, stage: str, category: str, reason: str, *, count: int = 1,
        record_ids=(),
    ) -> None:
        """Persist and log bounded diagnostic codes, never prompts or source text."""
        stage_code = stage if isinstance(stage, str) and stage in _STAGE_DIAGNOSTIC_STAGES else "synthesis"
        allowed_reasons = _STAGE_DIAGNOSTIC_REASONS.get(category) if isinstance(category, str) else None
        if allowed_reasons is None:
            category, allowed_reasons = "generation_failure", _STAGE_DIAGNOSTIC_REASONS["generation_failure"]
        reason_code = (
            reason if isinstance(reason, str) and reason in allowed_reasons
            else _STAGE_DIAGNOSTIC_FALLBACK_REASONS[category]
        )
        try:
            event_count = max(1, min(int(count), 10_000))
        except (TypeError, ValueError):
            event_count = 1
        if isinstance(record_ids, str):
            ids = [record_ids]
        else:
            try:
                ids = list(record_ids)
            except TypeError:
                ids = []
        safe_ids = list(dict.fromkeys(
            value for value in ids
            if isinstance(value, str) and _SAFE_DIAGNOSTIC_ID_RE.fullmatch(value)
        ))[:20]
        event = {
            "stage": stage_code, "category": category, "reason": reason_code,
            "count": event_count, "record_ids": safe_ids,
        }
        self.stage_log.append(event)
        if len(self.stage_log) > MAX_STAGE_LOG_ENTRIES:
            del self.stage_log[:-MAX_STAGE_LOG_ENTRIES]
        logger.info(
            "Deep-think stage diagnostic job=%s stage=%s category=%s reason=%s count=%d",
            self.id, stage_code, category, reason_code, event_count,
        )

    @classmethod
    def from_checkpoint(cls, value: dict):
        """Load old checkpoints too; new fields have safe dataclass defaults."""
        from dataclasses import fields

        saved = deepcopy(value)
        if saved.get("checkpoint_version", 1) < 2:
            for requirement in saved.get("evidence_requirements", []):
                requirement["answer_coverage"] = "not_verified"
                requirement["delivered_synthesis"] = "not_started"
                if requirement.get("state") == "resolved":
                    requirement["state"] = "open"
            saved["checkpoint_version"] = 2
        if saved.get("checkpoint_version", 1) < 3:
            # Older section plans did not persist independently resumable
            # chapters. Re-plan once from the frozen task/scope on resume.
            saved["chapter_plan"] = []
            saved["planning_complete"] = False
            saved["checkpoint_version"] = 3
        if saved.get("checkpoint_version", 1) < 4:
            saved.setdefault("stage_log", [])
            saved["checkpoint_version"] = 4
        if saved.get("checkpoint_version", 1) < 5:
            saved.setdefault("work_queue", [])
            saved["checkpoint_version"] = 5
        if saved.get("checkpoint_version", 1) < 6:
            from .research_task import version_task_contract

            snapshot = saved.get("snapshot") or {}
            snapshot["task_record"] = version_task_contract(
                snapshot.get("task_record"), snapshot.get("task", ""),
                thread_id=snapshot.get("origin_satellite_id"),
                turn_id=snapshot.get("origin_turn_id"),
            )
            saved["checkpoint_version"] = 6
        snapshot = saved.pop("snapshot")
        snapshot["conversation"] = tuple(snapshot.get("conversation") or ())
        saved["status"] = JobStatus(saved.get("status", "QUEUED"))
        for key in ("attempted_actions", "exhausted_capabilities"):
            saved[key] = set(saved.get(key) or ())
        allowed = {item.name for item in fields(cls)} - {"snapshot", "cancel_requested", "pause_requested"}
        return cls(snapshot=JobSnapshot(**snapshot), **{key: item for key, item in saved.items() if key in allowed})

    def resolve_pending_clarification(self, recovery: dict, resolution: str) -> None:
        """Keep an answered input request as history, not an active report caveat."""
        for entry in reversed(self.evidence):
            if (
                entry.get("status") == "needs_input"
                and not entry.get("resolution")
                and entry.get("recovery") == _bounded_evidence_value(recovery or {})
            ):
                entry["resolution"] = {"status": resolution}
                return

    def view(self) -> dict:
        from urllib.parse import quote

        from .thinking_evidence import research_progress

        now = time.time()
        task_record = self.snapshot.task_record
        started_at = task_record.get("started_at")
        finished_at = task_record.get("finished_at")
        duration_end = finished_at if isinstance(finished_at, (int, float)) else now
        requests = self.model_budget.get("request_reservations", [])
        return {
            "id": self.id,
            "task": self.snapshot.task,
            "objective": task_record.get("objective") or self.snapshot.task,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
            "note_id": self.note_id,
            "stage": self.stage,
            "origin_source": self.snapshot.origin_source,
            "origin_satellite_id": self.snapshot.origin_satellite_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed_seconds": round(max(0.0, duration_end - self.created_at), 2),
            "progress": research_progress(self),
            "partial_artifact_link": f"/reports/{quote(self.note_id, safe='/')}" if self.note_id else "",
            "acknowledgement_seconds": task_record.get("acknowledgement_seconds"),
            "job_duration_seconds": (
                round(max(0.0, duration_end - started_at), 3)
                if isinstance(started_at, (int, float)) else 0.0
            ),
            "request_metrics": [dict(item) for item in requests[-64:] if isinstance(item, dict)],
            "context_tokens": self.model_budget.get("context_tokens"),
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
        task_ledger_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        self._worker = worker
        self._status_listener = status_listener
        self._admission_check = admission_check or (lambda: True)
        self._jobs: dict[str, BackgroundJob] = {}
        configured_path = task_ledger_path or os.getenv("FULLOCH_TASK_LEDGER")
        self._task_ledger_path = Path(configured_path) if configured_path else None
        self._persisted_tasks = self._load_task_ledger()
        configured_checkpoints = checkpoint_dir or os.getenv(
            "FULLOCH_THINKING_CHECKPOINT_DIR", "./data/thinking-jobs",
        )
        self._checkpoint_dir = Path(configured_checkpoints) if configured_checkpoints else None
        self._queue: queue.Queue[str] = queue.Queue(MAX_QUEUED_JOBS)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._load_job_checkpoints()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="thinking-background-worker"
        )
        self._thread.start()
        if not self._queue.empty():
            self.wake()

    def _checkpoint_path(self, job_id: str) -> Path | None:
        if self._checkpoint_dir is None or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", job_id):
            return None
        return self._checkpoint_dir / f"{job_id}.json"

    def _persist_checkpoint(self, job: BackgroundJob) -> None:
        path = self._checkpoint_path(job.id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            checkpoint = job.checkpoint()
            # Draft text is never an accepted batch artifact. Accepted bodies
            # live on their chapter/section records with explicit audit state.
            checkpoint["report_draft"] = ""
            checkpoint["report_revision"] = ""
            temporary.write_text(
                json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError):
            logger.exception("Could not persist thinking job checkpoint id=%s", job.id)

    def checkpoint(self, job: BackgroundJob) -> None:
        """Atomically persist a worker-owned batch boundary."""
        self._persist_checkpoint(job)

    def _load_job_checkpoints(self) -> None:
        if self._checkpoint_dir is None or not self._checkpoint_dir.is_dir():
            return
        for path in sorted(self._checkpoint_dir.glob("*.json")):
            try:
                job = BackgroundJob.from_checkpoint(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, KeyError):
                logger.warning("Ignoring invalid thinking checkpoint file=%s", path.name)
                continue
            if path.stem != job.id:
                continue
            if job.status in {JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.QUEUED}:
                job.status = JobStatus.QUEUED
                job.stage = "Resuming saved research checkpoint"
                job.snapshot.task_record["execution_status"] = "queued"
                try:
                    self._queue.put_nowait(job.id)
                except queue.Full:
                    logger.warning("Thinking checkpoint queue limit reached id=%s", job.id)
                    break
            self._jobs[job.id] = job

    def _load_task_ledger(self) -> dict[str, dict]:
        if self._task_ledger_path is None:
            return {}
        try:
            payload = json.loads(self._task_ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        tasks = payload.get("threads") if isinstance(payload, dict) else None
        if not isinstance(tasks, dict):
            return {}
        restored = {}
        for thread_id, item in list(tasks.items())[-MAX_PERSISTED_TASKS:]:
            if (not isinstance(thread_id, str) or not isinstance(item, dict)
                    or not isinstance(item.get("task"), str)
                    or not isinstance(item.get("task_record"), dict)):
                continue
            status = item.get("status", JobStatus.FAILED.value)
            record = deepcopy(item["task_record"])
            if status in {"QUEUED", "RUNNING", "PAUSED", "NEEDS_INPUT"}:
                status = JobStatus.FAILED.value
                record["execution_status"] = "interrupted"
            restored[thread_id] = {**item, "status": status, "task_record": record}
        return restored

    def _persist_task(self, job: BackgroundJob) -> None:
        thread_id = job.snapshot.origin_satellite_id
        if self._task_ledger_path is None or job.snapshot.origin_source != "conversation" or not thread_id:
            return
        current = self._persisted_tasks.get(thread_id)
        if current and float(current.get("created_at", 0)) > job.created_at:
            # A slow older task must not replace a newer task's thread pointer
            # when it completes later.
            return
        self._persisted_tasks[thread_id] = {
            "task": job.snapshot.task,
            "task_record": deepcopy(job.snapshot.task_record),
            "status": job.status.value,
            "created_at": job.created_at,
        }
        while len(self._persisted_tasks) > MAX_PERSISTED_TASKS:
            oldest = min(self._persisted_tasks, key=lambda key: self._persisted_tasks[key].get("created_at", 0))
            del self._persisted_tasks[oldest]
        try:
            self._task_ledger_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._task_ledger_path.with_name(f".{self._task_ledger_path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps({"version": 1, "threads": self._persisted_tasks}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, self._task_ledger_path)
        except OSError:
            logger.exception("Could not persist conversation research task ledger")

    def _notify(self, job: BackgroundJob) -> None:
        if self._status_listener is None:
            return
        try:
            self._status_listener(job)
        except Exception:  # Observability must never stop the worker.
            logger.exception("Background thinking status listener failed")
        finally:
            # The listener may attach the durable report link or final delivery
            # classification to task state; persist those fields after it runs.
            with self._lock:
                self._persist_task(job)
                self._persist_checkpoint(job)

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
        handoff: dict | None = None,
        origin_turn_id: str | None = None,
        task_record: dict | None = None,
    ) -> str:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        acknowledged_started = time.monotonic()
        job_id = uuid.uuid4().hex
        from .research_task import version_task_contract

        record = version_task_contract(
            task_record, task.strip(), thread_id=origin_satellite_id, turn_id=origin_turn_id,
        )
        source_turn_ids = list(dict.fromkeys(
            value for value in [*(record.get("source_turn_ids") or []), origin_turn_id]
            if isinstance(value, str) and value
        ))
        snapshot = JobSnapshot(
            task=task.strip(),
            conversation=tuple(deepcopy(item) for item in (conversation or ())),
            notes=notes[:MAX_STATE_CHARS],
            origin_satellite_id=origin_satellite_id,
            origin_source=origin_source,
            handoff=deepcopy(handoff or {}),
            origin_turn_id=origin_turn_id,
            task_record={
                "version": 2,
                "contract_version": 2,
                "task_id": record.get("task_id") or uuid.uuid4().hex,
                "thread_id": origin_satellite_id,
                "objective": str(record.get("objective") or task.strip()),
                "constraints": deepcopy(record.get("constraints") or []),
                "amendments": deepcopy(record.get("amendments") or []),
                "revision": max(1, int(record.get("revision", 1))),
                "source_turn_ids": source_turn_ids,
                "job_id": job_id,
                "parent_job_id": record.get("parent_job_id"),
                "attempt": max(1, int(record.get("attempt", 1))),
                "execution_status": "queued",
                "delivery_status": "pending",
                "typed_constraints": deepcopy(record.get("typed_constraints") or []),
                "assumptions": deepcopy(record.get("assumptions") or []),
                "answer_format": deepcopy(record.get("answer_format") or {
                    "type": "concise_report", "detail": "match the user's requested form",
                }),
            },
        )
        job = BackgroundJob(job_id, snapshot)
        job.snapshot.task_record["acknowledged_at"] = time.time()
        with self._lock:
            self._jobs[job.id] = job
        try:
            self._queue.put_nowait(job.id)
        except queue.Full:
            with self._lock:
                self._jobs.pop(job.id, None)
            raise RuntimeError("thinking job queue is full") from None
        with self._lock:
            self._persist_task(job)
            self._persist_checkpoint(job)
        self._notify(job)
        self.wake()
        job.snapshot.task_record["acknowledgement_seconds"] = round(
            time.monotonic() - acknowledged_started, 4,
        )
        with self._lock:
            self._persist_task(job)
            self._persist_checkpoint(job)
        return job.id

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.view() if job else None

    def latest_for_thread(self, thread_id: str) -> BackgroundJob | None:
        """Return the newest conversation-origin job for exactly one thread."""
        if not isinstance(thread_id, str) or not thread_id:
            return None
        with self._lock:
            latest = max((
                job for job in self._jobs.values()
                if job.snapshot.origin_source == "conversation"
                and job.snapshot.origin_satellite_id == thread_id
            ), key=lambda job: (job.created_at, job.id), default=None)
            if latest is not None:
                return latest
            saved = self._persisted_tasks.get(thread_id)
            if saved is None:
                return None
            record = deepcopy(saved["task_record"])
            return BackgroundJob(
                str(record.get("job_id") or "persisted-task"),
                JobSnapshot(
                    saved["task"], origin_satellite_id=thread_id,
                    origin_source="conversation", task_record=record,
                ),
                status=JobStatus(saved.get("status", "FAILED")),
                created_at=float(saved.get("created_at", 0)),
            )

    def jobs_for_thread(self, thread_id: str) -> list[BackgroundJob]:
        """Return conversation jobs and the durable thread pointer, newest first."""
        if not isinstance(thread_id, str) or not thread_id:
            return []
        with self._lock:
            jobs = sorted(
                (job for job in self._jobs.values()
                 if job.snapshot.origin_source == "conversation"
                 and job.snapshot.origin_satellite_id == thread_id),
                key=lambda job: (job.created_at, job.id), reverse=True,
            )
        if jobs:
            return jobs
        saved = self.latest_for_thread(thread_id)
        return [saved] if saved is not None else []

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
            job.snapshot.task_record["execution_status"] = "running"
            self._persist_task(job)
            self._persist_checkpoint(job)
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
                job.snapshot.task_record["execution_status"] = "cancelled"
                self._persist_task(job)
                self._persist_checkpoint(job)
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
            job.snapshot.task_record["execution_status"] = "queued"
            self._persist_task(job)
            self._persist_checkpoint(job)
        self._queue.put_nowait(job_id)
        self.wake()
        return True

    def resume_with_input(self, job_id: str, user_input: str, handoff: dict | None = None) -> bool:
        """Queue a clarification-paused job while retaining its evidence ledger."""
        if not isinstance(user_input, str) or not user_input.strip():
            return False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.NEEDS_INPUT:
                return False
            job.continuation = user_input.strip()[:MAX_STATE_CHARS]
            if handoff:
                job.pending_handoffs.append(deepcopy(handoff))
            job.status = JobStatus.QUEUED
            job.stage = "Queued with user clarification"
            job.updated_at = time.time()
            job.snapshot.task_record["execution_status"] = "queued"
            self._persist_task(job)
            self._persist_checkpoint(job)
        self._queue.put_nowait(job_id)
        self._notify(job)
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
                    self._persist_checkpoint(job)
                self._notify(job)
                continue
            with self._lock:
                job.status = JobStatus.RUNNING
                job.stage = "Thinking"
                job.updated_at = time.time()
                job.snapshot.task_record["execution_status"] = "running"
                job.snapshot.task_record.setdefault("started_at", time.time())
                self._persist_task(job)
                self._persist_checkpoint(job)
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
                        job.summary = summary
                        job.state = state[:MAX_STATE_CHARS]
                        job.status = JobStatus.READY
                        assessment = job.synthesis_state.get("delivery_assessment")
                        delivery = assessment.get("status") if isinstance(assessment, dict) else None
                        if delivery not in {"complete", "partial", "evidence_only", "blocked"}:
                            delivery = "evidence_only" if job.synthesis_state.get("limited_evidence_fallback") else (
                                "partial" if job.synthesis_state.get("partial_report") else "ready"
                            )
                        job.stage = {
                            "complete": "Report ready",
                            "partial": "Report ready",
                            "evidence_only": "Research notes ready",
                            "blocked": "Report ready",
                            "ready": "Report ready",
                        }[delivery]
                        job.snapshot.task_record["execution_status"] = "completed"
                        job.snapshot.task_record["finished_at"] = time.time()
                        job.snapshot.task_record["delivery_status"] = delivery
                    if job.status is JobStatus.NEEDS_INPUT:
                        job.snapshot.task_record["execution_status"] = "needs_input"
                    elif job.status is JobStatus.CANCELLED:
                        job.snapshot.task_record["execution_status"] = "cancelled"
                    self._persist_task(job)
                    job.updated_at = time.time()
                    self._persist_checkpoint(job)
                self._notify(job)
                if job.status is JobStatus.PAUSED and not job.cancel_requested.is_set():
                    job.pause_requested.clear()
                    if self._wait_for_admission(job):
                        with self._lock:
                            job.status = JobStatus.QUEUED
                            job.stage = "Queued, waiting for a quiet moment"
                            job.updated_at = time.time()
                            job.snapshot.task_record["execution_status"] = "queued"
                            self._persist_task(job)
                            self._persist_checkpoint(job)
                        self._queue.put_nowait(job.id)
                        self._notify(job)
                        self.wake()
                    else:
                        with self._lock:
                            job.status = JobStatus.CANCELLED
                            job.updated_at = time.time()
                            job.snapshot.task_record["execution_status"] = "cancelled"
                            self._persist_task(job)
                            self._persist_checkpoint(job)
                        self._notify(job)
            except Exception as exc:  # Worker failures must not kill the scheduler.
                logger.exception("Background thinking job %s failed", job.id)
                with self._lock:
                    job.error = f"{type(exc).__name__}: {exc}"[:MAX_STATE_CHARS]
                    failed_stage = str(job.stage or "research")[:200]
                    job.synthesis_state["failed_stage"] = failed_stage
                    job.snapshot.task_record["failure_stage"] = failed_stage
                    job.snapshot.task_record["failure_reason"] = job.error
                    job.snapshot.task_record["finished_at"] = time.time()
                    if job.report_draft.strip():
                        job.summary = job.report_draft.strip()
                        job.synthesis_state["partial_report"] = True
                        job.snapshot.task_record["delivery_status"] = "partial"
                    else:
                        job.snapshot.task_record["delivery_status"] = "blocked"
                    job.status = JobStatus.FAILED
                    job.snapshot.task_record["execution_status"] = "failed"
                    job.updated_at = time.time()
                    self._persist_task(job)
                    self._persist_checkpoint(job)
                self._notify(job)
