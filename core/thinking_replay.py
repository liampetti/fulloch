"""Offline Deep Think replay; retained fixtures are inputs, never checkpoints to overwrite."""

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from utils.prompts import get_thinking_user_prompt

from .background_jobs import BackgroundJob, JobSnapshot
from .thinking_budget import (
    DEFAULT_STAGE_POLICIES,
    MAX_GENERATED_TOKENS,
    MAX_MODEL_CALLS,
    MAX_SYNTHESIS_SECONDS,
    STAGE_CALL_LIMITS,
    ThinkingBudget,
)
from .thinking_coverage import ensure_requirements, initial_requirements, refresh_coverage
from .thinking_evidence import restore_report_cards
from .thinking_fact_cards import build_fact_cards, report_card_ledger
from .thinking_scope import research_context
from .thinking_worker import ThinkingWorker, _record_completion_metrics

_REPORT_DIR = Path(__file__).resolve().parents[1] / "docs/report"
DEFAULT_FIXTURE = max(_REPORT_DIR.glob("*.evidence.json"), key=lambda path: path.stat().st_mtime,
                      default=_REPORT_DIR / "latest.evidence.json")


def benchmark_configurations(context_tokens=12_288):
    """Comparable section-writing policies under identical replay work limits."""
    configurations = [
        {"label": "thinking-2048", "section": {"output_tokens": 2048, "thinking_mode": True}},
        {"label": "non-thinking-4096", "section": {"output_tokens": 4096, "thinking_mode": False}},
        {"label": "thinking-4096", "section": {"output_tokens": 4096, "thinking_mode": True}},
    ]
    if context_tokens >= 12_288:
        configurations.append(
            {"label": "thinking-6144", "section": {"output_tokens": 6144, "thinking_mode": True}}
        )
    return configurations


def _acceptance_metrics(job, cards, report, elapsed_seconds):
    """Metrics that distinguish visible output from supported delivery."""
    eligible_ids = {card["id"] for card in cards}
    selected_ids = set(job.synthesis_state.get("selected_card_ids") or [])
    delivered_ids = {
        str(binding.get("card_id"))
        for binding in job.report_bindings
        if isinstance(binding, dict) and binding.get("card_id")
    }
    displayed_ids = set(job.synthesis_state.get("display_card_ids") or [])
    for requirement in job.evidence_requirements:
        evidence_ids = set(requirement.get("evidence_ids") or [])
        used = evidence_ids & delivered_ids
        requirement["delivered_synthesis"] = "partial" if used else "missing"
        if used and requirement.get("state") == "resolved" and evidence_ids <= delivered_ids:
            requirement["delivered_synthesis"] = "evidence_scoped"
    if not job.synthesis_state.get("limited_evidence_fallback"):
        job.synthesis_state["evidence_counts"] = {
            "ineligible": len(job.fact_cards) - len(cards),
            "unselected": len(cards) - len(selected_ids),
            "unsynthesized": len(selected_ids - delivered_ids),
            "display_omitted": 0,
        }
    evaluated_requirements = [
        item for item in job.evidence_requirements
        if not (item.get("identifier") == "answer" and item.get("broad"))
    ]
    important_ids = {
        card_id
        for item in evaluated_requirements
        if item.get("importance") == "high"
        for card_id in item.get("evidence_ids") or []
    }
    citation_issues = [
        issue for key in ("issues", "post_revision_issues")
        for issue in job.report_audit.get(key, [])
        if re.search(r"\b(?:citation|support|evidence_binding)\b", str(issue), re.I)
    ]
    responses = job.model_budget.get("responses", [])
    return {
        "visible_output_success": bool(report.strip()),
        "evidence_only_fallback": bool(job.synthesis_state.get("limited_evidence_fallback")),
        "partial_report": bool(job.synthesis_state.get("partial_report")),
        "response_counts": {
            "visible": sum(item.get("visible_chars", 0) > 0 for item in responses),
            "empty": sum(item.get("termination") == "empty" for item in responses),
            "truncated": sum(item.get("termination") == "truncated" for item in responses),
            "failed": sum(item.get("termination") in {"failed", "timed_out", "cancelled"} for item in responses),
        },
        "supported_answer_coverage": {
            "requirements": len(evaluated_requirements),
            "with_evidence": sum(bool(item.get("evidence_ids")) for item in evaluated_requirements),
            "fully_delivered": sum(item.get("delivered_synthesis") == "evidence_scoped" for item in evaluated_requirements),
            "partially_delivered": sum(item.get("delivered_synthesis") == "partial" for item in evaluated_requirements),
        },
        "citation_correctness": {
            "bound_cards": len(delivered_ids),
            "unknown_bound_cards": sorted(delivered_ids - eligible_ids),
            "support_issue_count": len(citation_issues),
            "passed": bool(delivered_ids) and not (delivered_ids - eligible_ids) and not citation_issues,
        },
        "important_record_retention": {
            "available": len(important_ids),
            "selected": len(important_ids & selected_ids),
            "delivered": len(important_ids & delivered_ids),
            "displayed_in_fallback": len(important_ids & displayed_ids),
        },
        "evidence_counts": dict(job.synthesis_state.get("evidence_counts") or {}),
        "latency_seconds": elapsed_seconds,
        "tokens": {
            "actual_generated": job.model_budget.get("actual_generated_tokens", 0),
            "estimated_visible": job.model_budget.get("estimated_visible_tokens", 0),
            "reserved": job.model_budget.get("reserved_tokens", 0),
        },
    }


def load_replay(path=DEFAULT_FIXTURE, *, reextract=False):
    data = json.loads(Path(path).read_bytes())
    job = BackgroundJob("replay", JobSnapshot(data["task"]))
    for field in ("evidence", "artifacts", "research_context", "source_registry"):
        setattr(job, field, data.get(field, getattr(job, field)))
    job.evidence_requirements = data.get("coverage_requirements", [])
    job.fact_cards = restore_report_cards(data) if data.get("schema_version") in {1, 2} else data.get("fact_cards", [])
    job.research_context = {
        **research_context(job.snapshot.task, as_of=job.research_context.get("as_of")),
        **job.research_context,
    }
    if reextract:
        if data.get("schema_version") in {1, 2}:
            raise ValueError("Compact report evidence already contains exact extracted facts; omit --reextract.")
        job.fact_cards = build_fact_cards(
            job.evidence, job.artifacts, job.snapshot.task, job.research_context,
            source_registry=job.source_registry, extraction_state=job.extraction_state,
        )
    if data.get("schema_version") in {1, 2} and job.evidence_requirements:
        # Compact exports omit runtime fields such as dimension/version. They
        # are still the user's saved questions, not an old plan to regenerate.
        for requirement in job.evidence_requirements:
            requirement.setdefault("version", 2)
            requirement.setdefault("dimension", requirement["identifier"].split("-q", 1)[0])
    else:
        job.evidence_requirements = ensure_requirements(job.evidence_requirements, job.snapshot.task,
                                                       job.research_context)
    refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
    return job


def restore_replay_chapters(job, titles=()):
    """Reconstruct saved question groups, not a research loop or a legacy writer.

    Compact report exports omit chapter runtime state. Keep the source contract,
    as-of instant and resolved subject, but start clean writing/audit budgets.
    No retrieval capabilities are provided by replay().
    """
    groups = {}
    identity_questions = {item["question"] for item in initial_requirements(job.snapshot.task, job.research_context)
                          if item.get("is_subject_identity")}
    for requirement in job.evidence_requirements:
        identifier = str(requirement["identifier"])
        chapter_id = identifier.split("-q", 1)[0] if identifier.startswith("chapter-") else identifier
        group = groups.setdefault(chapter_id, {
            "id": chapter_id, "title": str(requirement["question"]).rstrip(" .?!")[:100],
            "priority": requirement.get("importance", "medium"), "questions": [],
            "body": "", "digest": "", "tools": [], "queries": [],
        })
        requirement.update(chapter_id=chapter_id, subject="; ".join(job.research_context.get("entities", [])),
                           as_of=job.research_context.get("as_of"),
                           subject_resolution=job.research_context.get("subject_resolution", {}))
        requirement.setdefault("is_subject_identity", requirement["question"] in identity_questions)
        group["questions"].append({
            "id": identifier, "requirement_id": identifier, "question": requirement["question"],
            "priority": requirement.get("importance", "medium"), "state": "open",
            "supporting_card_ids": list(requirement.get("evidence_ids") or []),
        })
    job.chapter_plan = list(groups.values())
    if len(titles) == len(job.chapter_plan):
        for chapter, title in zip(job.chapter_plan, titles, strict=True):
            chapter["title"] = title


class ExtractiveReplayModel:
    """Controller smoke test only: exact extracts are not a model-quality benchmark."""

    model = "deterministic-extractive-smoke"

    def __init__(self, *, empty=False):
        self.empty = empty

    def generate(self, **kwargs):
        if self.empty:
            return ""
        if kwargs.get("user_prompt") == get_thinking_user_prompt("audit"):
            return '{"verdict":"pass","issues":[]}'
        match = re.search(
            r"Verified fact-card ledger:\n(.*?)\n\nMaterial evidence limitations:",
            kwargs.get("system_prompt", ""), re.S,
        )
        if not match:
            return ""
        packet = json.loads(match[1])
        return "\n\n".join(
            f"{claim.get('claim') or json.dumps(claim.get('record', {}))} "
            f"<!-- cards: {claim['id']} -->"
            for claim in packet.get("claims", [])[:3]
        )


def replay(path=DEFAULT_FIXTURE, *, model=None, reextract=False, context_tokens=12_288,
           stage_policies=None, configuration_label="", production=False):
    path = Path(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    job = load_replay(path, reextract=reextract)
    if production:
        sibling_report = path.with_name(path.name.removesuffix(".evidence.json") + ".md")
        titles = []
        if sibling_report != path and sibling_report.is_file():
            titles = [title for title in re.findall(r"(?m)^## (.+)$", sibling_report.read_text())
                      if title.casefold() not in {"summary", "references", "conclusions"}]
        restore_replay_chapters(job, titles)
    model = model or ExtractiveReplayModel()
    if stage_policies:
        job.model_budget["stage_policies"] = stage_policies
    worker = ThinkingWorker(
        model, None, SimpleNamespace(update_stage=lambda *_: None, checkpoint=lambda *_: None),
        lambda instance, **kwargs: instance.generate(**kwargs), lambda: {},
        SimpleNamespace(_schemas={}), lambda *_: "", threading.Lock(), 1,
        context_tokens=context_tokens,
    )
    started = time.monotonic()
    cards = report_card_ledger(job.fact_cards)
    budget = ThinkingBudget(model, job.model_budget, context_tokens)
    error = ""
    try:
        report, job.report_bindings = worker._synthesise(
            job, cards, budget, lambda: False, "",
        )
    except Exception as exc:
        report, error = "", f"{type(exc).__name__}: {exc}"
    elapsed_seconds = round(time.monotonic() - started, 3)
    cards = report_card_ledger(job.fact_cards)
    acceptance = _acceptance_metrics(job, cards, report, elapsed_seconds)
    _record_completion_metrics(job, report)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before, "Replay mutated its fixture"
    result = {
        "fixture_sha256": before, "mode": "reextraction" if reextract else "existing_cards",
        "configuration": {
            "label": configuration_label,
            "backend": f"{type(model).__module__}.{type(model).__name__}",
            "model": getattr(model, "model", type(model).__name__),
            "base_url": getattr(model, "base_url", None),
            "context_tokens": context_tokens,
            "requested_stage_policies": stage_policies,
            "effective_stage_policies": {
                stage: budget.policy(stage) for stage in DEFAULT_STAGE_POLICIES
            },
            "work_limits": {
                "model_calls": MAX_MODEL_CALLS,
                "generated_tokens": MAX_GENERATED_TOKENS,
                "synthesis_seconds": MAX_SYNTHESIS_SECONDS,
                "stage_calls": STAGE_CALL_LIMITS,
            },
        },
        "error": error, "elapsed_seconds": elapsed_seconds,
        "production_chapters": production,
        "cards": len(job.fact_cards), "eligible_cards": len(cards),
        "bound_cards": len(job.report_bindings), "report": report,
        "metrics": job.report_audit["metrics"], "checkpoint": job.checkpoint(),
        "acceptance_metrics": acceptance,
        "quality_review": {
            "status": "pending_manual_review",
            "questions": [
                "Does the answer directly address each important named question?",
                "Does every factual passage follow from its bound original record?",
                "Are conflicts, qualifications, temporal scope, and missing evidence visible?",
                "Would this be useful to the requester without consulting the evidence attachment?",
            ],
            "warning": "Bindings and automated audits are not proof of entailment or usefulness.",
        },
    }
    if hasattr(model, "calls"):
        result["model_trace"] = model.calls
    responses = job.model_budget.get("responses", [])
    result["generation_metrics"] = {
        "visible_responses": sum(item.get("visible_chars", 0) > 0 for item in responses),
        "truncated_responses": sum(item.get("termination") == "truncated" for item in responses),
        "empty_responses": sum(item.get("visible_chars", 0) == 0 for item in responses),
        "actual_tokens": job.model_budget.get("actual_generated_tokens", 0),
        "reserved_tokens": job.model_budget.get("reserved_tokens", 0),
        "estimated_visible_tokens": job.model_budget.get("estimated_visible_tokens", 0),
    }
    return result
