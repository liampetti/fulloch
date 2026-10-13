"""Small deterministic task identity helpers for background research follow-ups."""

import hashlib
import json
import re
import uuid
from copy import deepcopy

_CONTINUATION_RE = re.compile(
    r"\s*(?:it|that|this|just do it|do it|go ahead|continue|try again|retry|start over)"
    r"\s*[.!?]*\s*\Z",
    re.I,
)
_UNRESOLVED_REFERENCE_RE = re.compile(
    r"\s*(?:it|that|this|just do it|do it|go ahead|continue|try again|retry|start over)"
    r"\s*[.!?]*\s*\Z",
    re.I,
)
_TASK_STATUS_RE = re.compile(
    r"\b(?:you didn['’]?t do anything|you haven't done anything|what happened to|"
    r"where is the research|is the research done|what's happening with|status update)\b",
    re.I,
)
_TASK_FAILURE_QUESTION_RE = re.compile(r"\s*(?:why not|why didn't you|why did it fail)\s*[?.!]*\s*\Z", re.I)
_WORD_RE = re.compile(r"[a-z0-9]{3,}", re.I)
_STOP_WORDS = frozenset({
    "about", "after", "also", "and", "are", "can", "could", "find", "for",
    "from", "give", "have", "into", "just", "latest", "look", "more", "please",
    "research", "that", "the", "their", "them", "there", "this", "with", "would",
})


def version_task_contract(record: dict | None, objective: str, *, thread_id=None, turn_id=None) -> dict:
    """Normalize legacy task snapshots to the durable v2 research contract."""
    value = deepcopy(record) if isinstance(record, dict) else {}
    constraints = value.get("constraints")
    if not isinstance(constraints, list):
        constraints = [constraints] if isinstance(constraints, str) and constraints else []
    typed = value.get("typed_constraints")
    if not isinstance(typed, list):
        typed = []
    normalized = []
    by_text = {}
    for item in [*constraints, *typed]:
        if isinstance(item, dict):
            text = str(item.get("text") or json.dumps(item, sort_keys=True, ensure_ascii=True)).strip()
            by_text[text] = {**by_text.get(text, {}), **deepcopy(item)}
        else:
            text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    typed_constraints = []
    for text in normalized:
        prior = deepcopy(by_text.get(text) or {})
        identifier = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()[:16]
        prior.update({
            "id": str(prior.get("id") or f"constraint-{identifier}"),
            "kind": str(prior.get("kind") or "user_constraint"),
            "text": text,
            "provenance": prior.get("provenance") or {
                "source": "user", "thread_id": thread_id, "turn_id": turn_id,
            },
        })
        if turn_id and isinstance(prior.get("provenance"), dict):
            if not prior["provenance"].get("turn_id"):
                prior["provenance"]["turn_id"] = turn_id
        typed_constraints.append(prior)
    prior_turns = value.get("source_turn_ids")
    if not isinstance(prior_turns, list):
        prior_turns = [prior_turns] if isinstance(prior_turns, str) and prior_turns else []
    value.update({
        "version": 2,
        "contract_version": 2,
        "task_id": value.get("task_id") or uuid.uuid4().hex,
        "objective": str(value.get("objective") or objective).strip(),
        # Keep the original compact field readable by older clients.
        "constraints": normalized,
        "typed_constraints": typed_constraints,
        "assumptions": value.get("assumptions") if isinstance(value.get("assumptions"), list) else [],
        "answer_format": value.get("answer_format") if isinstance(value.get("answer_format"), dict)
        else {"type": "concise_report", "detail": "match the user's requested form"},
        "thread_id": value.get("thread_id", thread_id),
        "source_turn_ids": list(dict.fromkeys([
            *prior_turns, *([turn_id] if turn_id else []),
        ])),
    })
    return value


def _related_topic(left: str, right: str) -> bool:
    left_terms = {item.casefold() for item in _WORD_RE.findall(left)} - _STOP_WORDS
    right_terms = {item.casefold() for item in _WORD_RE.findall(right)} - _STOP_WORDS
    return len(left_terms & right_terms) >= 2


def is_unresolved_task_reference(text: str) -> bool:
    """Whether the request contains no assignment without thread-local context."""
    return bool(_UNRESOLVED_REFERENCE_RE.fullmatch(str(text or "")))


def task_status_intent(text: str) -> str | None:
    """Recognize complaints/status questions that must not create a new task."""
    value = str(text or "").strip()
    if _TASK_FAILURE_QUESTION_RE.fullmatch(value):
        return "why_not"
    if _TASK_STATUS_RE.search(value):
        return "status"
    return None


def match_research_task(selector: str, jobs: list) -> tuple[object | None, bool]:
    """Match an explicit task ID/topic only when the result is unambiguous."""
    value = str(selector or "").strip().casefold()
    if not value:
        return None, False
    candidates = [job for job in jobs if getattr(job, "snapshot", None) is not None]
    exact_id = [job for job in candidates
                if str(job.snapshot.task_record.get("task_id") or "").casefold() == value
                or str(job.id).casefold() == value]
    if len(exact_id) == 1:
        return exact_id[0], False
    if len(exact_id) > 1:
        return None, True
    terms = {word.casefold() for word in _WORD_RE.findall(value)} - _STOP_WORDS
    matches = []
    for job in candidates:
        record = getattr(job.snapshot, "task_record", {}) or {}
        objective = str(record.get("objective") or job.snapshot.task)
        objective_terms = {word.casefold() for word in _WORD_RE.findall(objective)} - _STOP_WORDS
        if terms and (terms <= objective_terms or len(terms & objective_terms) >= max(2, len(terms) * 0.7)):
            matches.append(job)
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def resolve_research_task(proposed: str, prior_job=None) -> tuple[str, dict]:
    """Resolve an unambiguous imperative against its own thread's prior task."""
    prior_record = getattr(getattr(prior_job, "snapshot", None), "task_record", None)
    prior_objective = str((prior_record or {}).get("objective") or "").strip()
    prior_task = str(getattr(getattr(prior_job, "snapshot", None), "task", "")).strip()
    if prior_job is None:
        objective = str(proposed or "").strip()
        return objective, version_task_contract({
            "task_id": uuid.uuid4().hex, "constraints": [], "amendments": [],
            "parent_job_id": None, "attempt": 1,
        }, objective)

    objective = prior_objective or prior_task
    record = version_task_contract(prior_record, objective)
    request = str(proposed or "").strip()
    is_continuation = bool(_CONTINUATION_RE.fullmatch(request)) or request.casefold() in {
        str(record.get("task_id") or "").casefold(), str(prior_job.id).casefold(),
    }
    # Repeating the same objective retries/continues it; it adds no constraint.
    is_continuation = is_continuation or (
        " ".join(request.casefold().split()) == " ".join(objective.casefold().split())
    )
    is_refinement = not is_continuation and _related_topic(objective, request)
    if not is_continuation and not is_refinement:
        return request, version_task_contract({
            "task_id": uuid.uuid4().hex, "constraints": [], "amendments": [],
            "parent_job_id": None, "attempt": 1,
        }, request)
    record.update({
        "version": 2,
        "objective": objective,
        "parent_job_id": prior_job.id,
        "attempt": int(record.get("attempt", 1)) + 1,
    })
    record.setdefault("revision", 1)
    record.setdefault("task_id", None)
    record.setdefault("constraints", [])
    record.setdefault("amendments", [])
    if is_refinement:
        record["revision"] = int(record.get("revision", 1)) + 1
        record["constraints"] = list(dict.fromkeys([*record["constraints"], request]))
        record["amendments"].append({"text": request, "kind": "constraint_refinement"})
    resolved = objective
    for amendment in record["amendments"]:
        text = amendment.get("text") if isinstance(amendment, dict) else str(amendment)
        if text and f"Additional user constraints: {text}" not in resolved:
            resolved += f"\n\nAdditional user constraints: {text}"
    return resolved, version_task_contract(record, objective,
                                           thread_id=getattr(prior_job.snapshot, "origin_satellite_id", None))
