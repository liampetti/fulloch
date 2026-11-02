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
    r"\s*(?:it|that|this|just do it|do it|go ahead|continue|try again|retry|start over|yes|yeah|yep|sure|okay|ok|please do)"
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

_APPROVAL_RE = re.compile(r"\s*(?:yes|yeah|yep|sure|okay|ok|please do|go ahead|do it)\s*[.!?]*\s*\Z", re.I)


def resolve_approved_research(task: str, handoff: dict | None, conversation) -> str:
    """Resolve a bare approval without treating the approval itself as a topic."""
    if not _APPROVAL_RE.fullmatch(task or ""):
        return task
    turns = []
    for item in conversation or ():
        if isinstance(item, dict) and item.get("role") == "user":
            content = str(item.get("content") or "").strip()
            if content and not _APPROVAL_RE.fullmatch(content) and not is_unresolved_task_reference(content):
                turns.append(content[:2000])
    assignment = str((handoff or {}).get("research_task") or "").strip()
    if not assignment or _APPROVAL_RE.fullmatch(assignment) or is_unresolved_task_reference(assignment):
        assignment = turns[-1] if turns else ""
    if not assignment:
        return task
    context = [turn for turn in turns[-2:] if turn.casefold() != assignment.casefold()]
    return assignment + ("\n\nRecent user context:\n" + "\n".join(context) if context else "")


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
    # Only a topicless imperative ("just do it", "try again", "it"…) continues
    # the prior task. Any substantive request — even one that repeats the
    # objective verbatim or merely shares its topic — starts a fresh task: a
    # new analysis asks for current information, not the prior run's evidence.
    is_retry = bool(_CONTINUATION_RE.fullmatch(request)) or request.casefold() in {
        str(record.get("task_id") or "").casefold(), str(prior_job.id).casefold(),
    }
    if is_retry:
        record.update({
            "version": 2,
            "objective": objective,
            "parent_job_id": prior_job.id,
            "attempt": int(record.get("attempt", 1)) + 1,
            "fresh": False,
        })
        record.setdefault("revision", 1)
        record.setdefault("task_id", None)
        record.setdefault("constraints", [])
        record.setdefault("amendments", [])
        resolved = objective
        for amendment in record["amendments"]:
            text = amendment.get("text") if isinstance(amendment, dict) else str(amendment)
            if text and f"Additional user constraints: {text}" not in resolved:
                resolved += f"\n\nAdditional user constraints: {text}"
        return resolved, version_task_contract(record, objective,
                                               thread_id=getattr(prior_job.snapshot, "origin_satellite_id", None))
    # A new analysis: fresh task identity and a clean evidence ledger. The prior
    # objective anchors related requests; standing constraints are carried only
    # when the request refines the prior topic (they are requirements, not
    # evidence). An unrelated request starts from nothing. A related request is
    # recorded as an amendment of the current intent (it may add nuance) unless
    # it merely restates the objective verbatim, in which case it is a clean
    # re-run that adds no constraint.
    is_refinement = _related_topic(objective, request)
    base_objective = objective if is_refinement else request
    restates_objective = (
        " ".join(request.casefold().split()) == " ".join(objective.casefold().split())
    )
    prior_constraints: list = []
    prior_amendments: list = []
    prior_job_id = None
    if is_refinement:
        prior_constraints = [
            item.get("text") if isinstance(item, dict) else str(item)
            for item in (prior_record or {}).get("constraints") or []
        ]
        prior_amendments = [
            item for item in (prior_record or {}).get("amendments") or []
            if isinstance(item, dict) and item.get("kind") == "constraint_refinement"
        ]
        # Lineage: a related fresh task is derived from the prior task.
        prior_job_id = prior_job.id
        if not restates_objective:
            # The related request becomes a standing constraint + amendment.
            prior_constraints = list(dict.fromkeys([*prior_constraints, request]))
            prior_amendments = [*prior_amendments, {"text": request, "kind": "constraint_refinement"}]
    return base_objective, version_task_contract({
        "task_id": uuid.uuid4().hex,
        "constraints": prior_constraints,
        "amendments": prior_amendments,
        "parent_job_id": prior_job_id,
        "attempt": 1,
        "revision": 1,
        "fresh": True,
    }, base_objective, thread_id=getattr(prior_job.snapshot, "origin_satellite_id", None))
