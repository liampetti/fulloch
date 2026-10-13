"""Bounded investigations with explicit model, scheduling and tool dependencies.

The worker never touches foreground history. Its snapshot and evidence belong to
the job manager. Single-slot generation acquires the supplied model lock; tools
run outside it. Two-slot generation uses the independently admitted server slot.
"""

import ast
import hashlib
import json
import logging
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Callable, ContextManager
from urllib.error import URLError
from urllib.parse import urlparse

from tools.capabilities import ToolCapability
from tools.thinking_context import reset_artifacts, set_artifacts
from tools.thinking_playbooks import matching_playbooks, scoped_capabilities
from tools.tool_registry import ThinkingResult, ToolRegistry, thinking_result_error
from utils import intents
from utils.prompts import (
    THINKING_DIGEST_LIMIT_ERROR,
    THINKING_FINAL_ASSEMBLY_OUTPUT_ERROR,
    get_idle_research_summary_prompt,
    get_thinking_chapter_analysis_prompt,
    get_thinking_chapter_card_prompt,
    get_thinking_chapter_coverage_prompt,
    get_thinking_chapter_merge_prompt,
    get_thinking_chapter_plan_prompt,
    get_thinking_chapter_repair_prompt,
    get_thinking_chapter_research_prompt,
    get_thinking_digest_repair_prompt,
    get_thinking_final_assembly_prompt,
    get_thinking_output_error,
    get_thinking_output_repair_prompt,
    get_thinking_reasoned_coverage_prompt,
    get_thinking_reconciliation_prompt,
    get_thinking_report_audit_prompt,
    get_thinking_report_prompt,
    get_thinking_user_prompt,
)

from .background_jobs import MAX_EVIDENCE_ENTRIES, BackgroundJob, BackgroundJobManager
from .idle_summaries import validate_summary
from .research_constraints import validate_constraint_records
from .research_task import is_unresolved_task_reference
from .slm import RemoteUnreachable
from .thinking_budget import (
    CHAPTER_TIME_RESERVE_SECONDS,
    STAGE_CALL_LIMITS,
    ModelBudgetExhausted,
    PromptOverflow,
    ThinkingBudget,
    estimate_tokens,
    is_context_overflow,
)
from .thinking_coverage import (
    apply_reasoned_coverage,
    apply_supported_coverage,
    chapter_requirements,
    classify_delivered_synthesis,
    direct_retrieval_to_gap,
    effective_capability_call_limit,
    ensure_requirements,
    fallback_chapter_plan,
    initial_requirements,
    is_near_duplicate_action,
    is_near_duplicate_query,
    matching_cards_for_requirement,
    open_requirement,
    record_action_selection,
    refresh_coverage,
    resolve_requirement,
    searchable_gap_query,
    validate_chapter_plan,
)
from .thinking_evidence import research_progress
from .thinking_extraction import MAX_EXTRACTION_CHARS
from .thinking_fact_cards import (
    assign_batch_findings,
    build_fact_cards,
    eligible_cards_for_requirement,
    link_chapter_cards,
    refresh_section_card,
    report_card_ledger,
    report_limitations,
    select_requirement_cards,
    serialize_synthesis_packet,
    source_record_view,
    validate_fact_card,
)
from .thinking_fallback import chapter_excerpts, evidence_fallback
from .thinking_scope import apply_subject_candidates, needs_subject_resolution, research_context
from .thinking_source_context import contextual_source_registry, with_source_passages

logger = logging.getLogger(__name__)

MAX_THINKING_WORKER_CALLS = 12
MAX_THINKING_TOOL_CALLS = 8
MAX_THINKING_CAPABILITY_CALLS = 3
MAX_CHAPTER_RESEARCH_DECISIONS = 4
MAX_THINKING_ARGUMENT_CORRECTIONS = 2
MAX_RETAINED_EXTRACTION_RECOVERIES = 4
MAX_THINKING_CAPABILITY_DESCRIPTION_CHARS = 600
DEEP_THINK_READ_TIMEOUT_S = 600.0
DEEP_THINK_GENERATION_TIMEOUT_S = 900.0
DEEP_THINK_STEP_MAX_TOKENS = 1024
DEEP_THINK_REPORT_AUDIT_MAX_TOKENS = 1024
DEEP_THINK_TRANSCRIPT_MAX_CHARS = 12_000
DEEP_THINK_OBSERVATION_MAX_CHARS = 3_000
_THINKING_OBSERVATION_RE = re.compile(r"\n\n(?=\[[^\]\n]+\]\n)")
_SOURCE_MARKER_RE = re.compile(r"\s*<!--\s*cards:\s*([\w.-]+(?:\s*,\s*[\w.-]+)*)\s*-->")
_MODEL_SOURCES_RE = re.compile(
    r"(?ims)^#{2,3}\s+(?:sources|references)\s*$.*?(?=^#{1,3}\s+|\Z)"
)


class ReportSynthesisError(RuntimeError):
    """An investigation could not collect verified material for a final report."""


def _thinking_excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[compacted]...\n"
    remaining = limit - len(marker)
    if remaining <= 0:
        return text[:limit]
    head = remaining * 2 // 3
    return text[:head].rstrip() + marker + text[-(remaining - head) :].lstrip()


def _compact_thinking_transcript(transcript: str) -> str:
    transcript = transcript.strip()
    if len(transcript) <= DEEP_THINK_TRANSCRIPT_MAX_CHARS:
        return transcript
    entries = _THINKING_OBSERVATION_RE.split(transcript)
    if len(entries) == 1:
        return _thinking_excerpt(transcript, DEEP_THINK_TRANSCRIPT_MAX_CHARS)
    labels = [entry.split("\n", 1)[0] for entry in entries]
    overhead = sum(len(label) + 2 for label in labels) + 2 * (len(entries) - 1)
    per_entry = max(160, (DEEP_THINK_TRANSCRIPT_MAX_CHARS - overhead) // len(entries))
    compacted = []
    for label, entry in zip(labels, entries, strict=True):
        body = entry[len(label) :].lstrip("\n")
        compacted.append(f"{label}\n{_thinking_excerpt(body, per_entry)}")
    return "\n\n".join(compacted)[:DEEP_THINK_TRANSCRIPT_MAX_CHARS]


def _append_thinking_observation(transcript: str, label: str, observation: str) -> str:
    observation = _thinking_excerpt(observation.strip(), DEEP_THINK_OBSERVATION_MAX_CHARS)
    entry = f"[{label}]\n{observation}"
    combined = f"{transcript.strip()}\n\n{entry}" if transcript.strip() else entry
    return _compact_thinking_transcript(combined)


def _describe_thinking_capability(name: str, schema) -> str:
    args = [
        param.name if param.required else f"{param.name}={param.default!r}"
        for param in schema.params
    ]
    description = " ".join(schema.description.split())
    if len(description) > MAX_THINKING_CAPABILITY_DESCRIPTION_CHARS:
        description = description[: MAX_THINKING_CAPABILITY_DESCRIPTION_CHARS - 3].rsplit(" ", 1)[0] + "..."
    return f"- {name}({', '.join(args)}): {description}"


def _validate_chapter_action(value: object, allowed_tools: list[str], registry):
    """Validate the research decision envelope before any capability can run."""
    if not isinstance(value, dict):
        return None, None, None, False, "Return one JSON object with a tool and args, or done=true."
    if value.get("done") is True:
        if set(value) != {"done"}:
            return None, None, None, False, "For completion, return exactly {\"done\":true}."
        return None, [], {}, True, None
    if set(value) != {"tool", "args"}:
        return None, None, None, False, "Return exactly the keys tool and args, or the single key done=true."
    raw_name, raw_args = value.get("tool"), value.get("args")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, None, None, False, "The tool field must name one listed read-only tool."
    if not isinstance(raw_args, (list, dict)):
        return None, None, None, False, "The args field must be a JSON array or named-argument object."
    name = registry.canonical_name(raw_name.strip())
    if name not in allowed_tools:
        return (None, None, None, False,
                f"Blocked unavailable or out-of-scope capability: {raw_name.strip()}.")
    args, kwargs = intents.coerce_args(raw_args)
    return name, args, kwargs, False, None


def _action_rejection_reason(emission, error, capabilities, registry):
    if emission is None:
        return "malformed_json"
    if isinstance(error, str) and error.startswith("Blocked unavailable or out-of-scope capability:"):
        name = emission.get("tool") if isinstance(emission, dict) else None
        if isinstance(name, str) and name.strip():
            canonical = registry.canonical_name(name.strip())
            return "unavailable_tool" if canonical not in capabilities else "out_of_scope_tool"
        return "invalid_action_shape"
    return "invalid_action_shape"


def _chapter_output_reason(error: str) -> str:
    lowered = str(error or "").casefold()
    if "citation markers" in lowered or "html comments" in lowered:
        return "model_authored_marker"
    if "support" in lowered and ("alias" in lowered or "ids" in lowered):
        return "missing_support_ids" if "at least one" in lowered or "must list" in lowered else "unknown_support_ids"
    if "paragraph" in lowered or "section headings" in lowered:
        return "invalid_passage_shape"
    if "complete json" in lowered or "one complete json" in lowered:
        return "malformed_json"
    if "digest" in lowered:
        return "digest_contract"
    return "invalid_output_shape"


def _record_chapter_output_rejection(job: BackgroundJob, error: str, record_ids: list[str]) -> None:
    reason = _chapter_output_reason(error)
    responses = job.model_budget.get("responses") or []
    finish_reason = responses[-1].get("finish_reason") if responses and isinstance(responses[-1], dict) else None
    rejection_log = job.synthesis_state.setdefault("output_rejections", [])
    rejection_log.append({"stage": "chapter_analysis", "subtype": reason,
                          "finish_reason": finish_reason})
    if len(rejection_log) > 100:
        del rejection_log[:-100]
    job.record_stage_event(
        "chapter_analysis", "output_rejection", reason,
        count=1, record_ids=record_ids,
    )
    citation_reasons = {
        "missing_support_ids": "missing_citation",
        "unknown_support_ids": "invalid_support_id",
        "model_authored_marker": "model_authored_marker",
    }
    if reason in citation_reasons:
        job.record_stage_event(
            "chapter_analysis", "citation_failure", citation_reasons[reason],
            count=1, record_ids=record_ids,
        )


def _record_synthesis_omissions(job: BackgroundJob) -> None:
    state = job.synthesis_state
    grouped: dict[str, list[str]] = {}
    reasons = state.get("omission_reasons") if isinstance(state.get("omission_reasons"), dict) else {}
    for card_id, reason in reasons.items():
        code = str(reason)
        if code not in {
            "context_limit", "server_context_limit", "time_budget", "model_call_or_token_budget",
            "generation_failed", "empty_generation", "no_verified_section",
        }:
            code = "not_synthesized"
        grouped.setdefault(code, []).append(str(card_id))

    covered = {identifier for values in grouped.values() for identifier in values}
    for card_id in state.get("omitted_card_ids", []):
        if str(card_id) not in covered:
            grouped.setdefault("not_selected_by_requirements", []).append(str(card_id))
            covered.add(str(card_id))
    for card_id in state.get("unsynthesized_card_ids", []):
        if str(card_id) not in covered:
            grouped.setdefault("not_synthesized", []).append(str(card_id))
            covered.add(str(card_id))
    if "display_card_ids" in state:
        selected = {str(value) for value in state.get("selected_card_ids", [])}
        displayed = {str(value) for value in state.get("display_card_ids", [])}
        for card_id in sorted(selected - displayed):
            grouped.setdefault("display_omitted", []).append(card_id)

    for reason, identifiers in grouped.items():
        unique_ids = list(dict.fromkeys(identifiers))
        if unique_ids:
            job.record_stage_event(
                "synthesis", "evidence_omission", reason,
                count=len(unique_ids), record_ids=unique_ids,
            )


def _normalise_thinking_args(schema, args: list, kwargs: dict, *, default_overrides: dict | None = None) -> tuple[list | None, str | None]:
    """Normalize keyword objects and schema-recognised assignment lists once.

    Only a fully named list or an assignment matching its signature position
    starts assignment parsing, so ordinary search text containing '=' survives.
    """
    known = {param.name: index for index, param in enumerate(schema.params)}
    kwargs = dict(kwargs)
    assignments = [re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=(.*)", value, re.S)
                   if isinstance(value, str) else None for value in args]
    named_list = len(args) > 1 and all(assignments) and any(match[1] in known for match in assignments)
    positional = []
    saw_named = False
    for value, assignment in zip(args, assignments, strict=True):
        expected = schema.params[len(positional)].name if len(positional) < len(schema.params) else None
        if assignment and (named_list or assignment[1] == expected or saw_named):
            key, raw = assignment[1], assignment[2].strip()
            if key not in known:
                return None, f"Unknown keyword argument for {schema.name}: {key}."
            try:
                value = json.loads(raw)
            except ValueError:
                value = raw
                if raw.startswith(("'", '"')):
                    try:
                        literal = ast.literal_eval(raw)
                    except (SyntaxError, ValueError):
                        return None, f"Invalid quoted argument for {key}."
                    if not isinstance(literal, str):
                        return None, f"Invalid quoted argument for {key}."
                    value = literal
            if key in kwargs and kwargs[key] != value:
                return None, f"Conflicting keyword values for {key}."
            kwargs[key] = value
            saw_named = True
        elif saw_named:
            return None, f"Positional argument after a named argument for {schema.name}."
        else:
            positional.append(value)
    unknown = sorted(str(key) for key in kwargs if key not in known)
    if unknown:
        return None, f"Unknown keyword arguments for {schema.name}: {', '.join(unknown)}."
    if len(positional) > len(schema.params):
        return None, f"Too many positional arguments for {schema.name}."
    positional_count = len(positional)
    for param in schema.params[len(positional):]:
        positional.append(kwargs.get(param.name, (default_overrides or {}).get(param.name, param.default)))
    for key, value in kwargs.items():
        index = known[key]
        if index < positional_count and positional[index] != value:
            return None, f"Conflicting positional and keyword value for {key}."
        positional[index] = value
    for param, value in zip(schema.params, positional, strict=True):
        if param.required and value is None:
            return None, f"Missing required argument {param.name} for {schema.name}."
    return positional, None


def _budgeted_generate(worker, budget, stage: str, **kwargs):
    request_kind = kwargs.pop("request_kind", "generation")
    job = getattr(worker, "_active_job", None)
    attempt = {"stage": stage, "kind": request_kind, "state": "attempted"}
    if job is not None:
        attempts = job.model_budget.setdefault("request_attempts", [])
        attempts.append(attempt)
        del attempts[:-128]
    if callable(check := getattr(worker, "_require_backend", None)):
        try:
            check()
        except RemoteUnreachable:
            attempt.update(state="blocked", reason="backend_unavailable")
            raise
    kwargs["thinking_mode"] = budget.policy(stage)["thinking_mode"]
    kwargs["output_policy"] = "research"
    stage_schemas = {
        "chapter_research": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "args": {"type": ["array", "object"]},
                    },
                    "required": ["tool", "args"],
                },
                {"type": "object", "properties": {"done": {"type": "boolean"}},
                 "required": ["done"]},
            ],
        },
        "chapter_planning": {
            "type": "object",
            "properties": {
                "chapters": {
                    "type": "array",
                    "minItems": 1, "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "maxLength": 70},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "questions": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"oneOf": [
                                {"type": "string", "maxLength": 180},
                                {"type": "object", "properties": {"question": {"type": "string", "maxLength": 180},
                                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "evidence_terms": {"type": "array", "maxItems": 3,
                                                       "items": {"type": "string", "maxLength": 40}}},
                                 "required": ["question"]},
                            ]}},
                            "tools": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                            "queries": {"type": "array", "minItems": 1, "maxItems": 2,
                                        "items": {"type": "string", "maxLength": 120}},
                        },
                        "required": ["title", "priority", "questions", "tools", "queries"],
                    },
                },
            },
            "required": ["chapters"],
        },
        "chapter_coverage": {
            "type": "object",
            "properties": {
                "findings": {"type": "array", "maxItems": 2, "items": {
                    "type": "object", "properties": {
                        "question_id": {"type": "string"},
                        "card_ids": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                    }, "required": ["question_id", "card_ids"],
                }},
                "subject_candidates": {"type": "array", "maxItems": 2, "items": {
                    "type": "object", "properties": {
                        "name": {"type": "string", "maxLength": 100},
                        "date": {"type": "string", "maxLength": 10},
                        "version": {"type": "string", "maxLength": 40},
                        "card_ids": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                    }, "required": ["name", "date", "card_ids"],
                }},
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "answered": {"type": "boolean"},
                            "answer_kind": {"type": "string"},
                            "card_ids": {"type": "array", "items": {"type": "string"}},
                            "answer": {"type": "string"},
                            "claim": {"type": "string"},
                            "premises": {"type": "array", "items": {"type": "string"}},
                            "missing_facts": {"type": "string"},
                            "unresolved_fields": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["question_id", "answered"],
                    },
                },
            },
            "required": ["answers"],
        },
    }
    kwargs.setdefault("response_schema", stage_schemas.get(stage))
    diagnostics = kwargs["diagnostics"] = {"request_kind": request_kind}
    timeout = budget.generation_timeout(stage)
    if timeout is not None:
        kwargs["generation_timeout"] = timeout
        kwargs["read_timeout"] = timeout
    prior_input = budget.state["stages"].get(stage, {}).get("input_tokens", 0)
    output, started = budget.begin(
        stage,
        system_prompt=kwargs.get("system_prompt", ""),
        user_prompt=kwargs.get("user_prompt"),
        history=kwargs.get("history"),
        thinking_mode=kwargs.get("thinking_mode", False),
    )
    kwargs["max_new_tokens"] = output
    logger.info("Deep-think model stage=%s call=%d input_tokens=%d output_limit=%d timeout=%s",
                stage, budget.state["calls"], budget.state["stages"][stage]["input_tokens"] - prior_input,
                kwargs["max_new_tokens"], timeout)
    try:
        attempt["state"] = "dispatched"
        response = worker._generate(**kwargs)
    except Exception as exc:
        diagnostics.setdefault("error_category", type(exc).__name__)
        budget.finish(stage, started, failed=True, diagnostics=diagnostics)
        attempt.update(state="failed", termination=diagnostics.get("termination", "failed"))
        logger.warning("Deep-think model stage=%s failed type=%s elapsed_ms=%d",
                       stage, type(exc).__name__, round((time.monotonic() - started) * 1000))
        if is_context_overflow(exc):
            raise PromptOverflow(f"The server rejected the {stage} context: {exc}") from exc
        raise
    logger.info("Deep-think model stage=%s completed visible_chars=%d elapsed_ms=%d",
                 stage, len(response or ""), round((time.monotonic() - started) * 1000))
    if diagnostics.get("termination") in {"truncated", "timed_out", "cancelled", "failed"}:
        attempt.update(state="rejected", termination=diagnostics["termination"])
        budget.finish(stage, started, response or "", failed=True, diagnostics=diagnostics)
        return ""
    if response_schema := kwargs.get("response_schema"):
        # Remote servers do not all enforce JSON Schema, so the controller
        # re-checks the exact contract it requested. A mismatch is a normal
        # parse failure the stage's repair path can see, not an exception.
        parsed = _parse_direct_json(response)
        if not _json_schema_match(parsed if parsed is not None else {}, response_schema):
            if (stage == "audit" and isinstance(parsed, dict)
                    and set(parsed) == {"verdict", "issues"}
                    and parsed.get("verdict") in {"pass", "revise"}
                    and isinstance(parsed.get("issues"), list)):
                # Old/unconstrained clients may still emit the legacy verdict.
                # Preserve it for the existing exact-extract-only checks, not
                # as positive support for novel prose. The actual request schema
                # still requires assessments on schema-enforcing backends.
                diagnostics["legacy_audit_verdict"] = True
                attempt["state"] = "completed"
                budget.finish(stage, started, response or "", diagnostics=diagnostics)
                return response
            diagnostics["schema_violation"] = stage
            attempt.update(state="rejected", reason="schema_violation")
            budget.finish(stage, started, response or "", failed=True, diagnostics=diagnostics)
            logger.info("Deep-think model stage=%s output failed the local schema check", stage)
            return ""
    attempt["state"] = "completed"
    budget.finish(stage, started, response or "", diagnostics=diagnostics)
    return response


def _recover_empty_section(worker, job, budget, fingerprint, prompt, user_prompt, cancelled):
    """One persisted strategy change per packet; two empty recoveries open the circuit."""
    recovery = job.synthesis_state.setdefault("empty_recovery", {"packets": [], "consecutive_empty": 0})
    if (cancelled() or fingerprint in recovery["packets"]
            or recovery["consecutive_empty"] >= 2):
        return ""
    recovery["packets"].append(fingerprint)
    prior = dict(budget.state["stage_policies"].get("section", {}))
    budget.state["stage_policies"]["section"] = {
        "output_tokens": 4096 if budget.context_tokens >= 8192 else budget.output_limit("section"),
        "thinking_mode": False,
    }
    try:
        # The original complete group must still fit the larger output reservation.
        if not budget.fits("section", system_prompt=prompt, user_prompt=user_prompt):
            return ""
        worker._update_stage(job, "Recovering empty research section")
        response = _budgeted_generate(worker, budget, "section", system_prompt=prompt,
                                     request_kind="empty_retry",
                                     user_prompt=user_prompt, temperature=0.2, cancel_check=cancelled) or ""
        recovery["consecutive_empty"] = 0 if response.strip() else recovery["consecutive_empty"] + 1
        return response
    finally:
        budget.state["stage_policies"]["section"] = prior


def _thinking_action_key(name: str, args: list) -> str:
    def normalise(value):
        if isinstance(value, str):
            return " ".join(value.split()).casefold()
        if isinstance(value, list):
            return [normalise(item) for item in value]
        if isinstance(value, dict):
            return {key: normalise(item) for key, item in value.items()}
        return value

    return (
        f"{name}:{json.dumps(normalise(args), sort_keys=True, separators=(',', ':'), default=str)}"
    )


def _parse_direct_json(response: str) -> dict | None:
    """Accept one complete JSON object, optionally in a Markdown code fence.

    Do not repair truncated JSON or fish an object out of surrounding prose.
    Formatting alone must not discard otherwise valid chapter work.
    """
    text = (response or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.S | re.I)
    if fenced:
        text = fenced[1].strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _json_schema_match(value: object, schema: dict) -> bool:
    """Bounded local check of the stage output against its item-level schema.

    Remote servers do not all enforce JSON Schema (some silently fall back to
    plain JSON mode), so the controller re-checks the exact contract it asked
    for. Only the keywords used by the stage schemas are implemented; anything
    richer stays the server's concern. A mismatch is a normal parse failure the
    repair path can see, never an exception.
    """
    if not isinstance(schema, dict):
        return True
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "type" in schema:
        expected = schema["type"]
        if isinstance(expected, list):
            if not any(_json_schema_match(value, {"type": item}) for item in expected):
                return False
        elif expected == "object":
            if not isinstance(value, dict):
                return False
        elif expected == "array":
            if not isinstance(value, list):
                return False
        elif expected == "string":
            if not isinstance(value, str):
                return False
        elif expected == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
        elif expected == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
        elif expected == "boolean":
            if not isinstance(value, bool):
                return False
        elif expected == "null" and value is not None:
            return False
    if "oneOf" in schema:
        if sum(_json_schema_match(value, branch) for branch in schema["oneOf"] if isinstance(branch, dict)) != 1:
            return False
    if "anyOf" in schema:
        if not any(_json_schema_match(value, branch) for branch in schema["anyOf"] if isinstance(branch, dict)):
            return False
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                return False
        properties = schema.get("properties") or {}
        for key, item_schema in properties.items():
            if key in value and not _json_schema_match(value[key], item_schema):
                return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            return False
        items = schema.get("items")
        if isinstance(items, dict):
            for item in value:
                if not _json_schema_match(item, items):
                    return False
    if isinstance(value, str) and (len(value) < schema.get("minLength", 0)
                                  or len(value) > schema.get("maxLength", len(value))):
        return False
    return True


def _chapter_output_error(parsed: dict | None) -> str:
    """Explain the actual rejected contract; do not repair/guess truncated prose."""
    if not isinstance(parsed, dict):
        return get_thinking_output_error("invalid_json")
    for name, limit in (("body", 20_000), ("digest", 4_000)):
        if not isinstance(parsed.get(name), str) or not parsed[name].strip():
            return get_thinking_output_error("empty_field", field=name)
        if len(parsed[name]) > limit:
            return get_thinking_output_error("field_limit", field=name, limit=limit)
    if len(_SOURCE_MARKER_RE.sub("", parsed["digest"]).split()) > 120:
        return get_thinking_output_error("digest_limit")
    return ""


def _render_chapter_passage(text: object, card_ids: object, aliases: dict) -> tuple[str, str]:
    """Attach citations only from explicit, packet-local supporting IDs."""
    if not isinstance(text, str) or not text.strip():
        return "", "Each passage needs non-empty text."
    clean = text.strip()
    if "<!--" in clean or "-->" in clean:
        return "", "Passage text must not contain model-written citation markers or HTML comments."
    if re.search(r"\((?:C\d+|src-[\w.-]+)(?:[,;\s]+(?:C\d+|src-[\w.-]+))*\)", clean):
        return "", "Put supporting aliases only in the cards array, never parenthesized in prose."
    if re.search(r"\n\s*\n", clean):
        return "", "Each passage must be one paragraph; return separate passages for separate paragraphs."
    if re.search(r"(?m)^#{1,6}\s+", clean):
        return "", "Passages must not contain section headings."
    if not isinstance(card_ids, list) or not card_ids:
        return "", "Each passage must list at least one supporting card alias."
    if any(not isinstance(identifier, str) for identifier in card_ids):
        return "", "Passage support IDs must be string aliases from the supplied source packet."
    selected = list(dict.fromkeys(card_ids))
    if any(identifier not in aliases for identifier in selected):
        return "", "Passage support IDs must be aliases from the supplied source packet."
    return f"{clean} <!-- cards: {', '.join(selected)} -->", ""


def _structured_chapter_output(value: dict | None, aliases: dict) -> tuple[dict | None, str]:
    """Validate passage/support arrays, then render internal citation markers."""
    if not isinstance(value, dict):
        return None, get_thinking_output_error("invalid_json")
    if set(value) != {"passages", "digest", "digest_cards"}:
        return None, "Return exactly passages, digest, and digest_cards in the required JSON object."
    passages = value.get("passages")
    if not isinstance(passages, list) or not passages:
        return None, "Return at least one passage with explicit supporting card aliases."
    rendered = []
    for row in passages:
        if not isinstance(row, dict) or set(row) != {"text", "cards"}:
            return None, "Each passage must contain exactly text and cards."
        passage, error = _render_chapter_passage(row.get("text"), row.get("cards"), aliases)
        if error:
            return None, error
        rendered.append(passage)
    digest, digest_error = _render_chapter_passage(
        value.get("digest"), value.get("digest_cards"), aliases,
    )
    if digest_error:
        return None, "The digest must contain text and explicit supporting card aliases: " + digest_error
    parsed = {"body": "\n\n".join(rendered), "digest": digest}
    error = _chapter_output_error(parsed)
    if error.startswith(THINKING_DIGEST_LIMIT_ERROR):
        return parsed, error
    return (parsed, "") if not error else (None, error)


def _structured_chapter_body(value: dict | None, aliases: dict) -> str:
    """Validate and retain the independently useful body of a bad digest output."""
    if not isinstance(value, dict) or set(value) != {"passages", "digest", "digest_cards"}:
        return ""
    passages = value.get("passages")
    if not isinstance(passages, list) or not passages:
        return ""
    rendered = []
    for row in passages:
        if not isinstance(row, dict) or set(row) != {"text", "cards"}:
            return ""
        passage, error = _render_chapter_passage(row.get("text"), row.get("cards"), aliases)
        if error:
            return ""
        rendered.append(passage)
    return "\n\n".join(rendered)


def _chapter_body_output_error(parsed: dict | None) -> str:
    """Validate the substantive body independently from its optional digest."""
    if not isinstance(parsed, dict):
        return get_thinking_output_error("invalid_body_json")
    body = parsed.get("body")
    if not isinstance(body, str) or not body.strip():
        return get_thinking_output_error("empty_field", field="body")
    if len(body) > 20_000:
        return get_thinking_output_error("field_limit", field="body", limit=20_000)
    return ""


def _fixed_scope_for_prompt(context: dict) -> str:
    keys = (
        "entities", "event_labels", "temporal_intent", "requested_dates",
        "ambiguous_dates", "as_of", "unresolved_references",
        "subject_resolution",
    )
    return json.dumps({key: context.get(key) for key in keys if context.get(key) not in (None, "", [], {})},
                       ensure_ascii=True, separators=(",", ":"))


def _chapter_evidence_packet(
    cards: list[dict], *, preferred_ids: set[str] | None = None, max_chars: int = 8_000,
    aliases: dict | None = None,
) -> str:
    """Serialize compact cards without letting a chart hide its quote or source."""
    preferred_ids = preferred_ids or set()
    # The selector already ranked relevance and diversified sources. Sorting IDs
    # here used to replace that order with arbitrary URL-hash order.
    ordered = sorted(cards, key=lambda card: str(card.get("id")) not in preferred_ids)
    packet = []
    used = 2
    for card in ordered:
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        record = source_record_view(card.get("fields") or {}) if card.get("kind") == "structured_record" else None
        if isinstance(record, dict) and isinstance(record.get("points"), list):
            # The quote identity and retrieval qualifications answer current-price
            # questions; serializing every chart sample can crowd that quote out
            # of the small research-decision context. Keep a provenance-linked
            # summary here. The complete record remains untouched for synthesis.
            points = [
                point for point in record["points"]
                if isinstance(point, dict) and isinstance(point.get("value"), (int, float))
            ]
            history = {key: record[key] for key in ("window",) if key in record}
            history["point_count"] = len(record["points"])
            if points:
                history.update({
                    "first": points[0],
                    "last": points[-1],
                    "low": min(points, key=lambda point: point["value"]),
                    "high": max(points, key=lambda point: point["value"]),
                })
            record = {key: value for key, value in record.items() if key != "points"}
            record["history_summary"] = history
        row = {
            "id": f"C{len(packet) + 1}" if aliases is not None else card.get("id"),
            "claim": card.get("statement") or None,
            "record": record,
            "status": card.get("evidence_status"),
            "scope": card.get("claim_identity"),
            "temporal_role": card.get("temporal_role"),
            "limitations": card.get("limitations"),
            "source": {
                "title": card.get("label"), "url": card.get("source_url"),
                "source_id": provenance.get("source_id"),
            },
        }
        encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        if used + len(encoded) + 1 > max_chars:
            continue
        if aliases is not None:
            aliases[row["id"]] = card["id"]
        packet.append(row)
        used += len(encoded) + 1
    return json.dumps(packet, ensure_ascii=True, separators=(",", ":"))


def _chapter_packet_from_cards(packet: dict, compact_facts: dict, chapter: dict,
                               extraction_disagreements: list[dict],
                               missing_information: list[dict]) -> str:
    """Writer view: concise facts plus complete selected records and local aliases."""
    aliases = packet.get("aliases") or {}
    reverse_aliases = {canonical: alias for alias, canonical in aliases.items()}
    source_by_id = {str(source.get("id")): source for source in packet.get("sources", [])
                    if isinstance(source, dict)}
    scope_by_id = {str(item.get("id")): item.get("scope") for item in packet.get("scopes", [])
                   if isinstance(item, dict)}
    writer_cards = []
    for claim in packet.get("claims", []):
        if not isinstance(claim, dict):
            continue
        identifier = str(claim.get("id") or "")
        alias = reverse_aliases.get(identifier)
        if not alias:
            continue
        fact = compact_facts.get(identifier)
        if not isinstance(fact, str) or not fact.strip():
            fact = str(claim.get("claim") or "")
        writer_cards.append({
            "alias": alias, "source_card_id": identifier,
            "source": claim.get("source"),
            "fact": fact,
            "record": claim.get("record"),
            "identity": claim.get("identity"),
            "status": claim.get("status"),
            "scope": claim.get("scope_id"),
            "limitations": claim.get("limitations"),
            "eligibility": claim.get("eligibility"),
            "finding": claim.get("finding"),
        })
    conflicts = []
    for item in [*(chapter.get("disagreements") or []), *extraction_disagreements]:
        card_ids = item.get("card_ids") if isinstance(item, dict) else None
        if not isinstance(card_ids, list):
            continue
        mapped = [reverse_aliases[value] for value in card_ids if value in reverse_aliases]
        if len(mapped) > 1:
            conflicts.append({"aliases": mapped, "detail": item.get("detail") or item.get("identity")})
    missing = [*missing_information, *(chapter.get("missing_information") or [])]
    return json.dumps({
        "cards": writer_cards, "sources": list(source_by_id.values()),
        "scopes": [{"id": key, "scope": value} for key, value in scope_by_id.items()],
        "disagreements": conflicts, "missing_information": missing,
        **({"passages": packet["passages"]} if packet.get("passages") else {}),
    }, ensure_ascii=True, separators=(",", ":"))


def _render_canonical_markers(report: str, aliases: dict[str, str]) -> str:
    """Translate canonical IDs back to a packet's local aliases for a merge call."""
    reverse = {canonical: alias for alias, canonical in aliases.items()}

    def replace(match):
        identifiers = [reverse.get(value.strip(), value.strip()) for value in match[1].split(",")]
        return " <!-- cards: " + ", ".join(identifiers) + " -->"

    return _SOURCE_MARKER_RE.sub(replace, report)


def _reader_source_language(text: str) -> str:
    """Keep internal citation vocabulary out of prose, not out of its markers.

    Restrict replacement to unmistakable pipeline terms, not physical playing,
    memory or graphics cards that may be the actual subject of a report.
    """
    def prose(value):
        value = re.sub(r"\b(?:fact|source|evidence)[ -]cards?\b",
                       lambda match: "sources" if match[0].casefold().endswith("cards") else "source",
                       value, flags=re.I)
        value = re.sub(r"\b(?:the|these|supplied|retained|cited) cards\b"
                       r"(?=\s+(?:establish|state|confirm|indicate|report|describe|document)\b)",
                       lambda match: "The sources" if match[0][0].isupper() else "the sources",
                       value, flags=re.I)
        value = re.sub(r"\bcard(?: ID| alias)?\s+C\d+\b", "the cited source", value, flags=re.I)
        value = re.sub(r"\bsrc-[a-f0-9]{16}-(?:c-\d+-\d+|t-\d+-r-\d+)(?:-g-\d+)?\b",
                       "the cited source", value)
        return value

    parts, cursor = [], 0
    for match in _SOURCE_MARKER_RE.finditer(text):
        parts.extend((prose(text[cursor:match.start()]), match[0]))
        cursor = match.end()
    parts.append(prose(text[cursor:]))
    return "".join(parts)


def _clean_report_block(text: object, seen_passages: set[str] | None = None) -> str:
    """Remove model-owned headings/diagnostics and exact repeated passages."""
    lines = []
    for line in _reader_source_language(str(text or "")).splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        if re.fullmatch(r"(?:summary|conclusions?|chapter body|digest)\s*:?\s*", stripped, re.I):
            continue
        lines.append(line.rstrip())
    paragraphs = []
    seen = seen_passages if seen_passages is not None else set()
    for paragraph in re.split(r"\n\s*\n", "\n".join(lines)):
        value = re.sub(r"\s+", " ", paragraph).strip()
        if not value:
            continue
        prose = re.sub(r"\*+", "", _SOURCE_MARKER_RE.sub("", value)).strip().casefold()
        if re.match(
            r"(?:audit (?:issues?|result)|generation limitations|synthesis limitations?|"
            r"evidence counts|unresolved requirement|additional eligible evidence was omitted|"
            r"partial (?:answer|findings)[:—-]?)",
            prose, re.I,
        ):
            continue
        if prose in seen:
            continue
        seen.add(prose)
        paragraphs.append(value)
    return "\n\n".join(paragraphs)


def _structured_record_text(card: dict) -> str:
    """Render a structured record's key fields as a readable finding sentence."""
    fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
    if not fields:
        return ""
    text = str(card.get("statement") or "").strip()
    if not text:
        text = _finding_prose(fields, card.get("retrieved_at"))
    if not text:
        keep = [
            f"{key}: {value}"
            for key, value in fields.items()
            if key in (
                "name", "symbol", "exchange", "currency", "price", "numeric_price",
                "quoted_at", "retrieved_at", "change", "change_percent",
                "reporting_period", "revenue", "eps", "guidance",
            )
            and value not in (None, "", [], {})
        ]
        text = " ".join(keep)
    return text


def _finding_prose(fields: dict, retrieved_at: object = None) -> str:
    """Turn a structured record's key fields into one readable finding sentence.

    A finance quote becomes "Acme Corp (ACME, NASDAQ) was quoted at USD 41.17 on
    Oct 09 2026, 04:00 PM UTC-04:00"; a results record becomes "Acme Corp
    reported revenue of ... for the ... period." Returns "" when nothing
    readable is available, so callers can fall back to the raw field dump.
    """
    retrieved = str(retrieved_at or fields.get("retrieved_at") or "").strip()
    period = str(fields.get("reporting_period") or "").strip()
    price = str(
        fields.get("price") or fields.get("numeric_price") or ""
    ).strip()
    if price:
        name = str(fields.get("name") or "").strip()
        symbol = str(fields.get("symbol") or fields.get("requested_symbol") or "").strip()
        exchange = str(fields.get("exchange") or "").strip()
        qualifier = ", ".join(part for part in (symbol, exchange) if part)
        identity = f"{name} ({qualifier})" if name and qualifier else (name or qualifier)
        quoted_at = str(fields.get("quoted_at") or "").strip()
        sentence = (
            f"{identity} was quoted at {price}" if identity
            else f"Quote: {price}"
        )
        if quoted_at:
            sentence += f" on {quoted_at}"
        if retrieved:
            sentence += f", retrieved {retrieved}"
        return sentence + "."
    revenue = str(
        fields.get("revenue") or fields.get("net_income") or fields.get("revenue_usd") or ""
    ).strip()
    if revenue:
        name = str(fields.get("name") or "").strip()
        sentence = f"{name} reported revenue of {revenue}" if name else f"Revenue was {revenue}"
        if period:
            sentence += f" for the {period} period"
        return sentence + "."
    return ""


def _reconcile_summary_with_findings(
    summary: str, requirements: list[dict], cards: list[dict], delivered_ids: set[str]
) -> str:
    """Replace summary denials with the finding the report body already delivers.

    When a high-priority requirement is resolved, its supporting card was
    actually cited in the chapter body, but the summary still denies the answer,
    the deterministic fallback rewrites the summary so the report is internally
    consistent. The finding text comes from the card itself, never from the
    summary.
    """
    by_id = {card["id"]: card for card in cards}
    for item in requirements:
        if item.get("importance") != "high" or item.get("state") != "resolved":
            continue
        evidence_ids = {str(value) for value in (item.get("evidence_ids") or [])}
        if not evidence_ids & delivered_ids:
            continue
        finding = ""
        for identifier in sorted(evidence_ids & delivered_ids):
            card = by_id.get(identifier)
            if card is None:
                continue
            text = _clean_report_block(
                str(card.get("statement") or "") or _structured_record_text(card)
            )
            if text:
                finding = text
                break
        if not finding:
            continue
        if _summary_denies_finding(summary, finding):
            question = str(item.get("question") or "").strip()
            replacement = (
                f"{'The report establishes: ' + question + ' — ' if question else ''}"
                + finding
            )
            if len(replacement) > 500:
                replacement = finding
            return replacement
    return summary


def _summary_denies_finding(summary: str, finding: str) -> bool:
    """Detect a summary that denies a delivered finding's key value.

    The check is deliberately conservative: it only fires when the summary
    carries an explicit refusal marker AND the finding's most salient value
    (a quoted amount, identifier or date) appears in the summary, so an
    ordinary restatement of the finding is never rewritten.
    """
    summary_text = re.sub(r"<!--\s*cards:[^>]*-->", "", summary).strip()
    if not summary_text:
        return False
    lowered = summary_text.casefold()
    refusal_markers = (
        "no verified", "no evidence", "not verified", "not established",
        "cannot be confirmed", "could not be confirmed", "unverified",
        "not stated", "not provided", "no authoritative", "no confirmed",
        "no direct evidence", "not answered", "no answer",
    )
    if not any(marker in lowered for marker in refusal_markers):
        return False
    key_values = _salient_finding_values(finding)
    if not key_values:
        return False
    # Collapse whitespace so "usd41.17" matches "usd 41.17" in prose.
    compact = re.sub(r"\s+", "", lowered)
    return any(value in compact for value in key_values)


def _salient_finding_values(finding: str) -> list[str]:
    """Extract the values most likely to be restated or denied in prose.

    Prefers explicit price/amount strings (with or without a currency code),
    then ISO dates, then tickers, so short or generic values never drive a
    rewrite.
    """
    candidates = []
    # Currency-prefixed amount: "USD 41.17", "$41.17" — keep the full string
    # so it can be matched verbatim in prose.
    currency = re.search(
        r"(?:\$\s?|USD\s?|EUR\s?|GBP\s?|CAD\s?|AUD\s?|HKD\s?|SGD\s?)"
        r"[\d][\d,]*(?:\.\d{1,2})?",
        finding,
        re.I,
    )
    if currency:
        candidates.append(re.sub(r"\s+", "", currency.group(0)))
    # ISO date
    date_match = re.search(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b", finding)
    if date_match:
        candidates.append(date_match.group(0))
    # Ticker / exchange symbol (e.g. NVDA:NASDAQ, ACME)
    ticker = re.search(r"\b[A-Z]{2,6}(?::[A-Z]{1,5})?\b", finding)
    if ticker:
        candidates.append(ticker.group(0))
    values = []
    for value in candidates:
        normalized = value.casefold()
        if normalized not in values:
            values.append(normalized)
    return values[:3]


def _concise_summary(summary: str, max_sentences: int = 4) -> str:
    """Bound a fallback summary to a short findings paragraph.

    A deterministic digest-fallback summary is the concatenation of chapter
    digests and can run long. Keep only the opening sentences so the report
    still opens with a concise summary, not a wall of text.
    """
    text = str(summary or "").strip()
    if not text:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= max_sentences:
        return text
    kept = " ".join(sentences[:max_sentences])
    if not kept.endswith((".", "!", "?")):
        kept += "."
    return kept


def _has_verified_evidence(job: BackgroundJob) -> bool:
    """Whether a job has retained factual material for a report.

    A planner reply, a failed tool call, or an empty typed envelope must not be
    mistaken for evidence.  This is deliberately controller-owned so every
    deep-think capability is held to the same completion rule.
    """
    return any(
        entry.get("status") == "evidence"
        and isinstance(entry.get("evidence"), dict)
        and bool(entry["evidence"])
        for entry in job.evidence
    )


def _refresh_active_manifest(job: BackgroundJob) -> dict:
    """Keep a compact navigation view over retained evidence and resumable work."""
    accepted = []
    for chapter in job.chapter_plan:
        if chapter.get("analysis_state") not in {"complete", "partial"}:
            continue
        card_ids = list(dict.fromkeys(
            identifier for identifier in chapter.get("analyzed_card_ids", [])
            if isinstance(identifier, str)
        ))
        accepted.append({
            "chapter_id": chapter.get("id"), "title": chapter.get("title"),
            "state": chapter.get("analysis_state"), "card_ids": card_ids,
            "has_body": bool(chapter.get("body")), "has_digest": bool(chapter.get("digest")),
            "fingerprint": chapter.get("analysis_fingerprint"),
        })
    source_refs = {}
    for card in job.fact_cards:
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        source = provenance.get("source_id")
        version = provenance.get("source_version")
        if source and version:
            source_refs.setdefault(str(source), set()).add(str(version))
    manifest = {
        "version": 1,
        "task_fingerprint": hashlib.sha256(job.snapshot.task.encode("utf-8")).hexdigest(),
        "coverage": [{
            "requirement_id": item.get("identifier"), "state": item.get("state"),
            "evidence_ids": list(item.get("evidence_ids") or []),
        } for item in job.evidence_requirements],
        "source_versions": {key: sorted(values) for key, values in source_refs.items()},
        "accepted_sections": accepted,
        "pending_work": [{
            "id": item.get("id"), "kind": item.get("kind"), "state": item.get("state"),
            "references": item.get("references", []),
            "requirement_ids": item.get("requirement_ids", []),
        } for item in job.work_queue if item.get("state") != "complete"],
    }
    job.synthesis_state["active_manifest"] = manifest
    return manifest


def _fallback_open_gaps(job: BackgroundJob) -> list[tuple[dict, dict]]:
    """Open requirements in chapters with remaining fallback retrieval allowance."""
    gaps = []
    task_question = " ".join(job.snapshot.task.casefold().split())
    for requirement in job.evidence_requirements:
        if requirement.get("state") != "open":
            continue
        question = " ".join(str(requirement.get("question") or "").casefold().split())
        chapter_id = requirement.get("chapter_id")
        has_specific_sibling = any(
            item is not requirement and item.get("chapter_id") == chapter_id
            for item in job.evidence_requirements
        )
        if question == task_question and has_specific_sibling:
            # The broad central request is covered by its specific chapter gaps;
            # repeating it as a fallback query cannot target a new evidence need.
            continue
        chapter = next((
            item for item in job.chapter_plan
            if item.get("id") == chapter_id
        ), None)
        attempted = any(
            selection.get("requirement_id") == requirement.get("identifier")
            for selection in job.action_selections
        )
        if (chapter and (not attempted or requirement.get("is_subject_identity"))
                and int(chapter.get("research_calls", 0)) < MAX_CHAPTER_RESEARCH_DECISIONS):
            gaps.append((requirement, chapter))
    return gaps


def _recover_retained_evidence(job: BackgroundJob) -> bool:
    """Spend a bounded, persisted extraction allowance before retrieving again."""
    if not open_requirement(job.evidence_requirements):
        return False
    recoveries = int(job.synthesis_state.get("extraction_recoveries", 0))
    if recoveries >= MAX_RETAINED_EXTRACTION_RECOVERIES:
        return False
    for state in job.extraction_state.values():
        if not state.get("unprocessed_spans"):
            continue
        allowance = int(state.get("allowance_chars") or MAX_EXTRACTION_CHARS)
        if allowance >= int(state.get("text_chars", 0)):
            continue
        state["allowance_chars"] = min(allowance + MAX_EXTRACTION_CHARS, int(state["text_chars"]))
        job.synthesis_state["extraction_recoveries"] = recoveries + 1
        job.fact_cards = build_fact_cards(
            job.evidence, job.artifacts, job.snapshot.task, job.research_context,
            source_registry=job.source_registry, extraction_state=job.extraction_state,
        )
        link_chapter_cards(job.fact_cards, job.chapter_plan)
        refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
        return True
    return False


def _fallback_thinking_report(task: str, evidence: list[dict], findings: str, *, cards: list[dict] | None = None, job=None) -> str:
    """Render only eligible, whole cards when generation fails, never raw prefixes."""
    retained = cards if cards is not None else (job.fact_cards if job is not None else None)
    retained = retained if retained is not None else build_fact_cards(evidence, {}, task)
    state = job.synthesis_state if job is not None else None
    report = evidence_fallback(
        task, retained,
        requirements=job.evidence_requirements if job is not None else (),
        state=state, max_chars=DEEP_THINK_TRANSCRIPT_MAX_CHARS,
    )
    if job is not None and cards is not None:
        eligible_inventory = report_card_ledger(job.fact_cards)
        state["evidence_counts"]["ineligible"] = len(job.fact_cards) - len(eligible_inventory)
    return report


def _fallback_report_bindings(job, cards: list[dict]) -> list[dict]:
    """Keep source metrics aligned with citations emitted by the evidence fallback."""
    displayed = job.synthesis_state.get("display_card_ids") or []
    marker_report = "\n\n".join(f"<!-- cards: {identifier} -->" for identifier in displayed)
    return _extract_report_bindings(marker_report, cards)[1]


def _safe_audit_fallback(task: str, issues: list[str]) -> str:
    """Return no inferred quality judgement when audit recovery fails."""
    del task, issues
    return "## Summary\n\nNo findings are available."
def _parse_report_audit(
    response: str, report: str = "", cards: list[dict] | None = None, ledger: str = "",
) -> list[str]:
    """Accept only corrections anchored to exact draft and cited source content."""
    audit = _parse_direct_json(response)
    if not isinstance(audit, dict) or audit.get("verdict") != "revise":
        return []
    issues = audit.get("issues")
    if not isinstance(issues, list):
        return []
    try:
        packet = json.loads(ledger)
    except (TypeError, ValueError):
        return []
    if not isinstance(packet, dict):
        return []
    card_by_id = {}
    for card in cards or []:
        card_by_id[str(card.get("id"))] = card
        for legacy_id in card.get("legacy_ids", []):
            card_by_id[str(legacy_id)] = card
    claim_by_id = {
        str(claim.get("id")): claim for claim in packet.get("claims", [])
        if isinstance(claim, dict) and claim.get("id")
    }
    for record in packet.get("records", []):
        if isinstance(record, dict) and record.get("card_id"):
            claim_by_id[str(record["card_id"])] = record

    def source_key_for(card_id):
        claim = claim_by_id.get(card_id) or {}
        card = card_by_id.get(card_id) or {}
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        return str(claim.get("source") or provenance.get("source_id")
                   or card.get("source_url") or (card.get("source_ids") or [""])[0])

    def source_values(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from source_values(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from source_values(nested)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            yield str(value)

    anchored = []
    passages = _passage_units(report)
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        fields = ("location", "draft_quote", "source_id", "source_quote", "reason", "fix")
        if any(not isinstance(issue.get(name), str) for name in fields):
            continue
        location = re.fullmatch(r"P(\d+)", issue["location"].strip(), re.I)
        draft_quote = issue["draft_quote"].strip()
        source_id = issue["source_id"].strip()
        source_quote = issue["source_quote"].strip()
        reason = issue["reason"].strip()
        fix = issue["fix"].strip()
        if (not location or len(draft_quote) < 5 or len(draft_quote) > 500
                or not source_id or len(source_quote) < 5 or len(source_quote) > 500
                or not reason or len(reason) > 500 or not fix or len(fix) > 500
                or source_quote == draft_quote):
            continue
        passage_index = int(location[1])
        if not 1 <= passage_index <= len(passages):
            continue
        passage = passages[passage_index - 1]
        visible = _SOURCE_MARKER_RE.sub("", passage)
        if draft_quote not in visible:
            continue
        cited_ids = list(dict.fromkeys(
            value.strip() for marker in _SOURCE_MARKER_RE.finditer(passage)
            for value in marker[1].split(",")
        ))
        cited_source_ids = {source_key_for(card_id) for card_id in cited_ids}
        if source_id not in cited_source_ids:
            continue
        source_content = []
        for source_passage in packet.get("passages", []):
            if (not isinstance(source_passage, dict)
                    or str(source_passage.get("source") or "") != source_id
                    or not set(source_passage.get("card_ids") or []) & set(cited_ids)):
                continue
            source_content.append(str(source_passage.get("text") or ""))
        for card_id in cited_ids:
            if source_key_for(card_id) != source_id:
                continue
            claim = claim_by_id.get(card_id) or {}
            card = card_by_id.get(card_id) or {}
            for value in (claim.get("claim"), card.get("statement")):
                if isinstance(value, str) and value:
                    source_content.append(value)
            record = claim.get("record")
            if record is None:
                record = card.get("fields")
            source_content.extend(source_values(record))
            source_content.extend(source_values(claim.get("evidence")))
            raw_record = card.get("evidence")
            source_content.extend(source_values(raw_record))
        if not any(source_quote in content for content in source_content):
            continue
        issue_type = issue.get("type") if isinstance(issue.get("type"), str) else "evidence_binding"
        anchored.append(
            f"[{issue_type}; P{passage_index}; cards: {', '.join(cited_ids)}] "
            f"Source-anchored finding: {reason} Source quote ({source_id}): {source_quote} "
            f"Proposed minimal fix: {fix}"
        )
    return anchored[:6]


def _report_support_assessments(
    verdict: dict, passages: list[str], cards: list[dict], source_packet: dict | None = None,
) -> dict[int, list[str]]:
    """Validate positive audit assessments and return blocking issues by passage.

    Legacy pass responses are accepted only for exact extractive sentences.
    Generated prose requires explicit status and exact, passage-bound support
    spans; citation markers alone never count as positive support.
    """
    by_id = {str(card.get("id")): card for card in cards}

    def source_values(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from source_values(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from source_values(child)

    def cited_ids(passage):
        return {
            value.strip()
            for match in _SOURCE_MARKER_RE.finditer(passage)
            for value in match[1].split(",")
        }

    assessment_rows = verdict.get("assessments")
    assessments: dict[int, list[dict]] = {}
    if isinstance(assessment_rows, list):
        for row in assessment_rows:
            if not isinstance(row, dict):
                continue
            location = re.fullmatch(r"P(\d+)", str(row.get("location") or ""), re.I)
            if location:
                assessments.setdefault(int(location[1]), []).append(row)

    issues: dict[int, list[str]] = {}
    for index, passage in enumerate(passages, 1):
        visible = _SOURCE_MARKER_RE.sub("", passage).strip()
        sentence_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", visible.strip())
        factual_sentences = [
            sentence.strip() for sentence in sentence_parts
            if sentence.strip() and not sentence.lstrip().startswith("#")
            and not sentence.casefold().startswith(("synthesis limitation:", "research limitation:"))
            and not (len(sentence.split()) <= 5 and re.search(
                r"\b(?:report|analysis|overview|summary)\.$", sentence.strip(), re.I,
            ))
            and not re.search(
                r"^(?:(?:the )?(?:selected )?(?:source|record|section)s? (?:support|retain|establish) "
                r"(?:a |its )?(?:scoped |qualified )?(?:source )?finding\.?|"
                r"(?:scoped|supported|completed) (?:source )?findings?(?: for [\w.-]+)?\.?|"
                r"scoped [\w -]{1,30} (?:report|analysis)\.?|"
                r"(?:partial )?report (?:for|about) [\w -]+\.?)$",
                sentence.strip(), re.I,
            )
        ]
        if not factual_sentences:
            continue
        cited = cited_ids(passage)
        rows = assessments.get(index, [])
        # Old model clients can safely pass an exact source extract, but cannot
        # bless a paraphrase without an explicit support assessment.
        if not rows:
            values = [
                text for identifier in cited if identifier in by_id
                for text in [str(by_id[identifier].get("statement") or ""),
                             *source_values(by_id[identifier].get("fields") or {})]
                if text.strip()
            ]
            for sentence in factual_sentences:
                if not any(
                    sentence in source
                    for source in values
                ):
                    issues.setdefault(index, []).append(
                        f"[support; P{index}; cards: {', '.join(sorted(cited))}] "
                        "No positive support assessment or exact source extract was supplied."
                    )
            continue

        matched_sentences: set[str] = set()
        for row in rows:
            claim = row.get("claim")
            status = row.get("status")
            if not isinstance(claim, str) or not claim.strip() or claim.strip() not in visible:
                issues.setdefault(index, []).append(
                    f"[support; P{index}; cards: {', '.join(sorted(cited))}] "
                    "Assessment does not bind to exact passage text."
                )
                continue
            matched_sentences.update(sentence for sentence in factual_sentences if sentence in claim.strip())
            if status != "supported":
                issues.setdefault(index, []).append(
                    f"[{status if status in {'contradicted', 'insufficient'} else 'insufficient'}; "
                    f"P{index}; cards: {', '.join(sorted(cited))}] "
                    f"{str(row.get('reason') or 'The claim is not affirmatively supported.')[:300]}"
                )
                continue
            supports = row.get("support")
            if not isinstance(supports, list) or not supports:
                issues.setdefault(index, []).append(
                    f"[insufficient; P{index}; cards: {', '.join(sorted(cited))}] "
                    "Supported verdict has no source spans."
                )
                continue
            valid_support = False
            for support in supports:
                if not isinstance(support, dict):
                    continue
                card_id, span = support.get("card_id"), support.get("span")
                if not isinstance(card_id, str) or card_id not in cited or card_id not in by_id:
                    continue
                card = by_id[card_id]
                content = [str(card.get("statement") or ""), *source_values(card.get("fields") or {})]
                span = span.strip() if isinstance(span, str) else ""
                # A contextual quote can preserve an antecedent/condition, but
                # must include this card's original assertion and match the
                # exact bound source version. Neighbour text alone is not proof.
                statement = str(card.get("statement") or "")
                provenance = card.get("provenance") or {}
                if span and statement and statement in span:
                    content.extend(
                        row.get("text", "") for row in (source_packet or {}).get("passages", [])
                        if card_id in row.get("card_ids", [])
                        and row.get("source") == provenance.get("source_id")
                        and row.get("source_version", "") == provenance.get("source_version", "")
                    )
                if span and any(
                    span in source
                    for source in content
                ):
                    valid_support = True
                    break
            if not valid_support:
                issues.setdefault(index, []).append(
                    f"[insufficient; P{index}; cards: {', '.join(sorted(cited))}] "
                    "Support span is not an exact field/claim from a cited card."
                )
        for sentence in factual_sentences:
            if sentence not in matched_sentences:
                issues.setdefault(index, []).append(
                    f"[insufficient; P{index}; cards: {', '.join(sorted(cited))}] "
                    f"No positive assessment covers: {sentence[:180]}"
                )
    return issues


def _deterministic_report_audit(report: str, cards: list[dict]) -> list[str]:
    """Check card integrity and field-aware copied values without model judgment."""
    issues = ["An invalid fact card was supplied to synthesis." for card in cards if validate_fact_card(card)]
    ledger = json.dumps(cards, ensure_ascii=True)
    for literal in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", report):
        if literal not in ledger:
            issues.append(f"The report contains `{literal}`, which is absent from the fact-card ledger.")
    issues.extend(_field_aware_monetary_audit(report, cards))
    issues.extend(_field_aware_timestamp_audit(report, cards))
    return issues[:6]


_MONEY_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MONEY_MENTION_RE = re.compile(
    rf"(?<![A-Z])(?P<code_before>[A-Z]{{3}})\s*(?P<amount_before>{_MONEY_NUMBER})(?!\d)"
    rf"|(?P<symbol>[$€£¥])\s*(?P<amount_symbol>{_MONEY_NUMBER})(?!\d)"
    rf"|(?<![\d.])(?P<amount_after>{_MONEY_NUMBER})\s*(?P<code_after>[A-Z]{{3}})(?![A-Z])",
)
_MONEY_VALUE_RE = re.compile(rf"(?<![\w.])-?{_MONEY_NUMBER}(?![\w.])")
_CURRENCY_SYMBOL_CODES = {
    "$": frozenset({"USD", "CAD", "AUD", "NZD", "SGD", "HKD"}),
    "€": frozenset({"EUR"}),
    "£": frozenset({"GBP"}),
    "¥": frozenset({"JPY", "CNY"}),
}
_MONEY_FIELD_NAMES = frozenset({
    "amount", "change", "cost", "fare", "fee", "price", "tax", "total", "numeric_price",
})


def _currency_codes(value: object) -> frozenset[str]:
    text = str(value or "").strip().upper()
    if text in _CURRENCY_SYMBOL_CODES:
        return _CURRENCY_SYMBOL_CODES[text]
    return frozenset({text}) if re.fullmatch(r"[A-Z]{3}", text) else frozenset()


def _decimal_value(value: object) -> Decimal | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return Decimal(str(value).replace(",", "")).normalize()
    except (InvalidOperation, ValueError):
        return None


def _money_mentions(text: str) -> list[tuple[frozenset[str], Decimal, str]]:
    output = []
    for match in _MONEY_MENTION_RE.finditer(text):
        symbol = match.group("symbol")
        raw_currency = match.group("code_before") or match.group("code_after") or symbol
        raw_amount = (match.group("amount_before") or match.group("amount_symbol")
                      or match.group("amount_after"))
        amount = _decimal_value(raw_amount)
        if amount is not None:
            output.append((_currency_codes(raw_currency), amount, match[0]))
    return output


def _card_money_values(card: dict) -> list[tuple[str, frozenset[str], Decimal]]:
    """Extract monetary facts only from typed fields of this cited card."""
    fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
    currency = _currency_codes(fields.get("currency") or fields.get("quote_currency"))
    values: list[tuple[str, frozenset[str], Decimal]] = []

    def collect_value(value, field_path):
        if isinstance(value, (dict, list, tuple)):
            children = value.values() if isinstance(value, dict) else value
            for child in children:
                collect_value(child, field_path)
            return
        if isinstance(value, str):
            explicit = _money_mentions(value)
            if explicit:
                values.extend((field_path, codes, amount) for codes, amount, _ in explicit)
                return
        amount = _decimal_value(value)
        if amount is None and isinstance(value, str):
            matches = _MONEY_VALUE_RE.findall(value)
            if len(matches) == 1:
                amount = _decimal_value(matches[0])
        if amount is not None:
            values.append((field_path, currency, amount))

    def visit(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold()
                child_path = (*path, normalized)
                if normalized in _MONEY_FIELD_NAMES:
                    collect_value(child, ".".join(child_path))
                elif normalized == "value" and any(
                    ancestor in {"points", "history_summary"} for ancestor in path
                ):
                    collect_value(child, ".".join(child_path))
                else:
                    visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, path)

    if card.get("kind") == "structured_record":
        visit(fields)
    elif card.get("statement"):
        # Article/table claims can retain an explicitly currency-qualified fact
        # even when the extractor has no typed numeric field for it.
        values.extend(("statement", codes, amount)
                      for codes, amount, _ in _money_mentions(str(card["statement"])))
    return values


def _money_field_group(path: str) -> str:
    if "points" in path or "history_summary" in path:
        return "history"
    if "change" in path:
        return "change"
    return "price"


_MONEY_CONTEXT_RE = re.compile(
    r"(?P<history>\b(?:historical|history|chart|points?|window|high|low|peak|trough|first|last)\b)"
    r"|(?P<change>\b(?:change|changed|up|down|increase|decrease|gain|loss|moved|movement)\b)"
    r"|(?P<price>\b(?:price|quote|quoted|cost|fare|amount|total)\b)",
    re.I,
)


def _report_money_group(prose: str, position: int, cards: list[dict]) -> str | None:
    before = prose[:position]
    boundary = max(before.rfind(value) for value in (".", ";", "!", "?", "\n"))
    segment = before[boundary + 1:]
    matches = list(_MONEY_CONTEXT_RE.finditer(segment))
    if matches:
        return next(name for name, value in matches[-1].groupdict().items() if value)
    if any((card.get("claim_identity") or {}).get("claim_type") == "market_snapshot"
           for card in cards):
        # An unlabeled amount on a market quote is the snapshot price, not one
        # of the chart's historical observations or a day-change field.
        return "price"
    return None


def _field_aware_monetary_audit(report: str, cards: list[dict]) -> list[str]:
    by_id = {str(card.get("id")): card for card in cards}
    for card in cards:
        for legacy_id in card.get("legacy_ids", []):
            by_id[str(legacy_id)] = card
    issues = []
    for paragraph in re.split(r"\n\s*\n", report):
        cited_ids = [identifier.strip() for marker in _SOURCE_MARKER_RE.finditer(paragraph)
                     for identifier in marker[1].split(",")]
        cited_cards = [by_id[identifier] for identifier in dict.fromkeys(cited_ids) if identifier in by_id]
        expected = [value for card in cited_cards for value in _card_money_values(card)]
        mentions = _money_mentions(_SOURCE_MARKER_RE.sub("", paragraph))
        if not cited_cards or not mentions:
            continue
        prose = _SOURCE_MARKER_RE.sub("", paragraph)
        cursor = 0
        for currencies, amount, raw in mentions:
            mention_position = prose.find(raw, cursor)
            cursor = max(cursor, mention_position + len(raw))
            group = _report_money_group(prose, mention_position, cited_cards)
            relevant = [
                (codes, value) for field_path, codes, value in expected
                if group is None or field_path == "statement" or _money_field_group(field_path) == group
            ]
            same_amount = [(codes, value) for codes, value in relevant if value == amount]
            if not same_amount:
                issues.append(
                    f"[monetary_value; paragraph; cards: {', '.join(cited_ids)}] "
                    f"The reported amount `{raw}` does not match a cited monetary field."
                )
            elif currencies and not any(currencies & codes for codes, _ in same_amount):
                issues.append(
                    f"[monetary_currency; paragraph; cards: {', '.join(cited_ids)}] "
                    f"The currency in `{raw}` does not match the cited monetary field."
                )
            if len(issues) >= 6:
                return issues
    return issues


_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s?Z|[+-]\d{2}:?\d{2})?)?"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}(?:,?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)"
    r"(?:\s*UTC(?:[+-]\d{2}:?\d{2})?)?)?\b",
    re.I,
)
_TIMESTAMP_LABEL_RE = re.compile(
    r"\bquote(?:d)?(?:[_ ](?:at|date|time|timestamp))?\b"
    r"|\bretriev(?:ed|al)(?:[_ ](?:at|date|time|timestamp))?\b|\bas[-_]of\b",
    re.I,
)


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip().rstrip(".,;)")
    if not text:
        return None
    iso = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed
    except ValueError:
        pass
    natural = re.sub(r"\s+UTC(?=[+-]\d{2}:?\d{2}$)", " ", text, flags=re.I)
    natural = re.sub(r"\s+UTC$", " +0000", natural, flags=re.I)
    natural = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", natural)
    for pattern in ("%b %d %Y, %I:%M:%S %p %z", "%b %d, %Y, %I:%M:%S %p %z",
                    "%B %d %Y, %I:%M:%S %p %z", "%B %d, %Y, %I:%M:%S %p %z",
                    "%b %d %Y, %I:%M %p %z", "%b %d, %Y, %I:%M %p %z",
                    "%B %d %Y, %I:%M %p %z", "%B %d, %Y, %I:%M %p %z",
                    "%b %d %Y, %I:%M:%S %p", "%b %d, %Y, %I:%M:%S %p",
                    "%B %d %Y, %I:%M:%S %p", "%B %d, %Y, %I:%M:%S %p",
                    "%b %d %Y, %I:%M %p", "%b %d, %Y, %I:%M %p",
                    "%B %d %Y, %I:%M %p", "%B %d, %Y, %I:%M %p",
                    "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y"):
        try:
            parsed = datetime.strptime(natural, pattern)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed
        except ValueError:
            continue
    return None


def _timestamp_role_before(report: str, position: int) -> str | None:
    before = report[:position]
    boundary = max(before.rfind(value) for value in (".", ";", "!", "?", "\n"))
    segment = before[boundary + 1:]
    labels = list(_TIMESTAMP_LABEL_RE.finditer(segment))
    if not labels or len(segment) - labels[-1].end() > 48:
        return None
    label = labels[-1][0].casefold()
    return "quote" if label.startswith("quote") or label.startswith("as") else "retrieval"


def _timestamp_values(card: dict, role: str) -> list[object]:
    fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
    provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
    if role == "quote":
        return [fields.get("quoted_at")] if fields.get("quoted_at") else []
    return [value for value in (
        card.get("retrieved_at"), fields.get("retrieved_at"), provenance.get("retrieved_at"),
    ) if value]


def _field_aware_timestamp_audit(report: str, cards: list[dict]) -> list[str]:
    by_id = {str(card.get("id")): card for card in cards}
    for card in cards:
        for legacy_id in card.get("legacy_ids", []):
            by_id[str(legacy_id)] = card
    issues = []
    for paragraph in re.split(r"\n\s*\n", report):
        cited_ids = [identifier.strip() for marker in _SOURCE_MARKER_RE.finditer(paragraph)
                     for identifier in marker[1].split(",")]
        cited_cards = [by_id[identifier] for identifier in dict.fromkeys(cited_ids) if identifier in by_id]
        prose = _SOURCE_MARKER_RE.sub("", paragraph)
        for mention in _TIMESTAMP_RE.finditer(prose):
            role = _timestamp_role_before(prose, mention.start())
            if role is None:
                continue
            stated = _parse_timestamp(mention[0])
            expected = [parsed for card in cited_cards for parsed in
                        (_parse_timestamp(value) for value in _timestamp_values(card, role)) if parsed]
            if stated is None or not expected or stated not in expected:
                role_label = "quote" if role == "quote" else "retrieval"
                issues.append(
                    f"[timestamp_role; paragraph; cards: {', '.join(cited_ids)}] "
                    f"The stated {role_label} time `{mention[0]}` does not match that timestamp field on the cited record."
                )
            if len(issues) >= 6:
                return issues
    return issues


def _invalid_report_bindings(report: str, cards: list[dict]) -> list[str]:
    valid = {str(card.get("id")) for card in cards}
    valid.update(str(value) for card in cards for value in card.get("legacy_ids", []))
    unknown = []
    for match in _SOURCE_MARKER_RE.finditer(report):
        for identifier in (item.strip() for item in match.group(1).split(",")):
            if identifier not in valid:
                unknown.append(identifier)
    return [f"[citation; report; cards: {item}] Unknown or ineligible card ID." for item in dict.fromkeys(unknown)][:6]


def issue_card_ids(issue: str) -> set[str]:
    match = re.search(r"\bcards:\s*([^\]]+)", str(issue))
    return {value.strip() for value in match.group(1).split(",") if value.strip()} if match else set()


def _canonical_markers(report: str, aliases: dict) -> str:
    """Translate only explicitly mapped packet aliases; unknown IDs remain rejectable."""
    def replace(match):
        identifiers = [aliases.get(value.strip(), value.strip()) for value in match[1].split(",")]
        return " <!-- cards: " + ", ".join(identifiers) + " -->"

    return _SOURCE_MARKER_RE.sub(replace, report)


def _audit_response_schema(cards: list[dict], aliases: dict | None = None) -> dict:
    """Constrain syntax and navigation, never the auditor's support verdict.

    Exact-original binding and full claim coverage remain independently checked
    after generation. Legacy extract-only verdicts remain readable.
    """
    return {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "assessments": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "location": {"type": "string"}, "claim": {"type": "string"},
            "status": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
            "support": {"type": "array", "items": {"type": "object", "properties": {
                "card_id": {"type": "string", "enum": [card["id"] for card in cards] + list(aliases or {})},
                "span": {"type": "string"},
            }, "required": ["card_id", "span"]}},
            "reason": {"type": "string"},
        }, "required": ["location", "claim", "status", "support"]}},
        "issues": {"type": "array", "items": {"oneOf": [
            {"type": "string"}, {"type": "object", "properties": {
                key: {"type": "string"} for key in
                ("location", "draft_quote", "source_id", "source_quote", "type", "reason", "fix")
            }},
        ]}},
    }, "required": ["verdict", "issues", "assessments"]}


def _passage_units(report: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", report) if part.strip()]


def _temporal_report_audit(report: str, cards: list[dict], research_context: dict) -> list[str]:
    """Flag an explicit upcoming/current framing of a sourced completed period."""
    try:
        as_of = str(research_context.get("as_of") or "")[:10]
    except AttributeError:
        as_of = ""
    by_id = {str(card.get("id")): card for card in cards}
    for card in cards:
        for identifier in card.get("legacy_ids", []):
            by_id[str(identifier)] = card
    issues = []
    for paragraph in re.split(r"\n\s*\n", report):
        if not re.search(r"\b(?:upcoming|will take place|is scheduled to|future event)\b", paragraph, re.I):
            continue
        ids = [part.strip() for match in _SOURCE_MARKER_RE.finditer(paragraph) for part in match.group(1).split(",")]
        for identifier in ids:
            card = by_id.get(identifier)
            if not card:
                continue
            statement = str(card.get("statement") or "")
            period = str((card.get("claim_identity") or {}).get("period") or "")
            # "To be held" and "is held annually" are not past tense. A
            # publication year is not the year of the event it describes either.
            past = bool(re.search(
                r"\b(?:finished|completed|won|ended|ran|competed|took place)\b|\b(?:was|were)\s+held\b",
                statement, re.I,
            ))
            dates = re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", statement)
            if not dates:
                context = card.get("context") if isinstance(card.get("context"), dict) else {}
                event_dates = context.get("event_dates") if isinstance(context.get("event_dates"), list) else []
                dates = [str(value) for value in event_dates
                         if re.fullmatch(r"(?:19|20)\d{2}-\d{2}-\d{2}", str(value))]
            old_date = bool(as_of and dates and all(date < as_of for date in dates))
            old_event_period = (
                card.get("temporal_role") in {"event_context", "itinerary_time"}
                and re.search(r"\b(?:19|20)\d{2}\b", period) and period[:4] < as_of[:4]
            )
            if past or old_date or old_event_period:
                issues.append(
                    f"[temporal_scope; paragraph; cards: {identifier}] The report calls a prior completed period upcoming."
                )
    return issues[:6]


def _raw_card_packet(cards: list[dict], sources: dict[str, dict], requirement_ids=()) -> dict:
    """Rollback representation with the same selected cards/context boundaries."""
    packet = serialize_synthesis_packet(cards, sources, requirement_ids=list(requirement_ids))
    views = {claim["id"]: claim for claim in packet.get("claims", [])}
    return {
        "schema": "bounded-raw-ledger-v1",
        "sources": packet.get("sources", []),
        "scopes": packet.get("scopes", []),
        "records": [
            {"tool": card.get("label"), "id": card["id"], "source": next(
                (item.get("id") for item in packet.get("sources", [])
                 if item.get("id") == str((card.get("provenance") or {}).get("source_id") or card.get("source_url") or card["source_ids"][0])),
                card["source_ids"][0] if card.get("source_ids") else "",
              ), "scope": card.get("scope"), "evidence": card.get("fields") if card.get("kind") == "structured_record" else card.get("statement") or card.get("fields"),
             "status": card.get("evidence_status"), "card_id": card["id"],
             "scope_id": views[card["id"]].get("scope_id"),
             "identity": views[card["id"]].get("identity"),
             "eligibility": views[card["id"]].get("eligibility"),
             "limitations": views[card["id"]].get("limitations")}
            for card in cards
        ],
        "requirement_ids": list(requirement_ids),
    }


def _conflict_scope_key(card: dict) -> str:
    identity = dict(card.get("claim_identity")) if isinstance(card.get("claim_identity"), dict) else {}
    # Polarity/value differs precisely because the assertions conflict; all
    # remaining identity fields define the boundary within which peers belong.
    identity.pop("polarity", None)
    return json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _indivisible_card_groups(cards: list[dict]) -> list[list[dict]]:
    """Keep all sides of one material disagreement in the same model request."""
    groups = []
    conflicts = {}
    for card in cards:
        if card.get("evidence_status") == "conflicted":
            key = _conflict_scope_key(card)
            if key not in conflicts:
                conflicts[key] = []
                groups.append(conflicts[key])
            conflicts[key].append(card)
        else:
            groups.append([card])
    return groups


def _extract_report_bindings(report: str, cards: list[dict]) -> tuple[str, list[dict]]:
    """Render concise numeric citations and retain their exact card bindings."""
    valid = {}
    for card in cards:
        valid[str(card.get("id"))] = card
        for legacy_id in card.get("legacy_ids", []):
            valid[str(legacy_id)] = card
    used = []
    seen = set()
    reference_numbers: dict[str, int] = {}

    def replace(match: re.Match) -> str:
        cited_numbers = []
        for card_id in (part.strip() for part in match.group(1).split(",")):
            card = valid.get(card_id)
            canonical_id = str(card.get("id")) if card else ""
            if card is None:
                continue
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
            context = card.get("context") if isinstance(card.get("context"), dict) else {}
            source_identity = context.get("source_identity")
            source_url = (source_identity if isinstance(source_identity, str)
                          and urlparse(source_identity).scheme in {"http", "https"}
                          else card.get("source_url", ""))
            if isinstance(source_url, str) and urlparse(source_url).scheme in {"http", "https"}:
                number = reference_numbers.setdefault(source_url, len(reference_numbers) + 1)
                if number not in cited_numbers:
                    cited_numbers.append(number)
            if canonical_id in seen:
                continue
            source_date = (
                context.get("updated_at")
                or context.get("published_at")
                or context.get("observed_at")
                or card.get("retrieved_at")
                or ""
            )
            used.append({
                "card_id": canonical_id, "source_ids": card.get("source_ids", []),
                "source_url": source_url,
                "source_title": fields.get("title") or card.get("label", ""),
                "source_date": source_date,
                "source_identity": source_identity or source_url,
                "canonical_identity": card.get("claim_identity") or {},
                "provenance": card.get("provenance") or {},
            })
            seen.add(canonical_id)
        return f" [{', '.join(str(number) for number in cited_numbers)}]" if cited_numbers else ""

    return _SOURCE_MARKER_RE.sub(replace, report).strip(), used


def _unbound_factual_paragraphs(report: str, cards: list[dict]) -> list[str]:
    """Return bounded narrative paragraphs that have no valid card marker.

    Marker validation is deliberately paragraph-level: one cited sentence
    cannot make unrelated prose elsewhere in the report appear sourced.  It
    is applied only when the ledger contains citeable URLs, so provider/local
    reports without URLs retain their scoped rendering path.
    """
    if not cards:
        return []
    valid_ids = {
        identifier for card in cards if isinstance(card, dict)
        for identifier in [str(card.get("id")), *[str(value) for value in card.get("legacy_ids", [])]]
    }
    missing = []
    for paragraph in re.split(r"\n\s*\n", report.strip()):
        stripped = paragraph.strip()
        if not stripped:
            continue
        prose_lines = [line for line in stripped.splitlines() if not line.lstrip().startswith("#")]
        prose = "\n".join(prose_lines).strip()
        if not prose:
            continue
        marker_ids = [part.strip() for match in _SOURCE_MARKER_RE.finditer(stripped) for part in match.group(1).split(",")]
        prose = _SOURCE_MARKER_RE.sub("", prose).strip()
        if not prose or re.fullmatch(r"informational only, not .*", prose, re.I | re.S):
            continue
        if re.match(r"(?:unresolved requirement:|additional eligible evidence was omitted|"
                    r"no report-wide conclusion was asserted|no evidence packet fit|synthesis limitation:)", prose, re.I):
            continue
        if not any(card_id in valid_ids for card_id in marker_ids):
            missing.append(re.sub(r"\s+", " ", prose)[:160])
    return missing[:6]


def _record_completion_metrics(job: BackgroundJob, report: str) -> None:
    """Persist and log aggregate investigation metrics without source/user content."""
    source_states: dict[str, dict] = {}
    for artifact in job.artifacts.values():
        data = artifact.get("data") if isinstance(artifact, dict) else None
        if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
            continue
        for source in data["sources"]:
            if not isinstance(source, dict):
                continue
            identity = str(source.get("canonical_url") or source.get("url") or "")
            if not identity:
                continue
            source_states[identity] = {
                "eligible": bool(source.get("eligible")),
                "dated": bool(source.get("published_at") or source.get("updated_at")),
            }
    for card in job.fact_cards:
        if not isinstance(card, dict):
            continue
        context = card.get("context") if isinstance(card.get("context"), dict) else {}
        identity = str(context.get("source_identity") or card.get("source_url") or "")
        if not identity:
            source_ids = card.get("source_ids") if isinstance(card.get("source_ids"), list) else []
            identity = f"artifact:{source_ids[0]}" if source_ids else ""
        if not identity:
            continue
        fields = card.get("fields") if isinstance(card.get("fields"), dict) else {}
        source_states.setdefault(
            identity,
            {
                "eligible": card.get("compatibility") == "compatible",
                "dated": bool(
                    context.get("published_at")
                    or context.get("updated_at")
                    or fields.get("published_at")
                    or fields.get("updated_at")
                ),
            },
        )
    issue_categories: dict[str, int] = {}
    for issue in [*job.report_audit.get("issues", []), *job.report_audit.get("post_revision_issues", [])]:
        issue_text = str(issue)
        match = re.match(r"\[([^;\]]+)", issue_text)
        category = match.group(1) if match else "post_revision_validation"
        issue_categories[category] = issue_categories.get(category, 0) + 1
    gap_states: dict[str, int] = {}
    for requirement in job.evidence_requirements:
        state = str(requirement.get("state") or "unknown")
        gap_states[state] = gap_states.get(state, 0) + 1
    metrics = dict(job.report_audit.get("metrics") or {})
    progress = research_progress(job)
    diagnostic_counts: dict[str, int] = {}
    for event in job.stage_log:
        key = f"{event.get('category', 'unknown')}:{event.get('reason', 'unknown')}"
        diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(event.get("count", 0))
    metrics.update(
        {
            **progress,
            "retrieved_sources": progress["fetched_documents"],
            "retrieval_artifact_count": progress["retrieval_artifacts"],
            "eligible_sources": progress["eligible_documents"],
            "rejected_sources": progress["fetched_documents"] - progress["eligible_documents"],
            "undated_sources": sum(not state["dated"] for state in source_states.values()),
            "cited_sources": progress["cited_documents"],
            "evidence_gap_states": gap_states,
            "audit_issue_categories": issue_categories,
            "revision_applied": bool(job.report_revision),
            "final_report_length": len(report),
            "model_budget": job.model_budget,
            "source_registry_count": len(job.source_registry),
            "extracted_claim_count": sum(int(item.get("claims", 0)) for item in job.extraction_state.values()),
            "extraction_input_chars": sum(int(item.get("text_chars", 0)) for item in job.extraction_state.values()),
            "extraction_input_tokens": sum(estimate_tokens(" " * int(item.get("text_chars", 0)))
                                            for item in job.extraction_state.values()),
            "extraction_model_calls": 0,
            "partial_extractions": sum(item.get("state") != "complete" for item in job.extraction_state.values()),
            "selected_card_count": len(job.synthesis_state.get("selected_card_ids", [])),
            "omitted_card_count": len(job.synthesis_state.get("omitted_card_ids", [])),
            "section_count": len(job.synthesis_state.get("sections", [])),
            "overflow_retries": int(job.synthesis_state.get("overflow_retries", 0)),
            "partial_report": bool(job.synthesis_state.get("partial_report")),
            "unsynthesized_card_count": len(job.synthesis_state.get("unsynthesized_card_ids", [])),
            "selection_reasons": job.synthesis_state.get("selection_reasons", {}),
            "evidence_counts": job.synthesis_state.get("evidence_counts", {}),
            "stage_diagnostics": diagnostic_counts,
            "requirement_coverage": {
                item["identifier"]: item.get("state") for item in job.evidence_requirements
            },
        }
    )
    job.report_audit["metrics"] = metrics
    budget = metrics.get("model_budget") if isinstance(metrics.get("model_budget"), dict) else {}
    stages = {
        stage: {key: values.get(key, 0) for key in ("calls", "input_tokens", "output_tokens", "latency_ms", "failures")}
        for stage, values in (budget.get("stages") or {}).items() if isinstance(values, dict)
    }
    aggregate = {
        key: metrics.get(key) for key in (
            "retrieved_sources", "processed_unique_documents", "retrieval_artifact_count",
            "eligible_sources", "rejected_sources", "cited_sources",
            "eligible_card_count", "evidence_gap_states", "audit_issue_categories",
            "revision_applied", "final_report_length", "selected_card_count",
            "omitted_card_count", "section_count", "overflow_retries", "partial_extractions",
            "partial_report", "unsynthesized_card_count",
            "stage_diagnostics",
            "tool_calls_used", "tool_call_budget", "duplicate_actions_avoided",
            "prompt_view_tokens", "compression_ratio",
        )
    }
    aggregate["model_stages"] = stages
    logger.info("Deep-think job %s aggregate metrics: %s", job.id, json.dumps(aggregate, sort_keys=True))


@dataclass
class ThinkingWorker:
    model: object
    grammar: object
    jobs: BackgroundJobManager
    generate: Callable[..., str]
    capabilities: Callable[[], dict[str, ToolCapability]]
    registry: ToolRegistry
    summarise_search: Callable[[str, Callable[[], bool]], str]
    model_lock: ContextManager
    server_slots: int
    fact_card_synthesis: bool = True
    context_tokens: int | None = None
    constraint_validator: Callable | None = None
    failure_capture: Callable[[dict], None] | None = None

    def _checkpoint_backend_loss(self, exc: Exception) -> None:
        """Persist accepted work once when generation loses its backend."""
        job = getattr(self, "_active_job", None)
        if job is None or job.synthesis_state.get("backend_recovery", {}).get("checkpointed"):
            return
        job.synthesis_state["backend_recovery"] = {
            "state": "waiting_for_health_check",
            "reason": type(exc).__name__,
            "process_generation": getattr(self.model, "_fulloch_process_generation", None),
            "checkpointed": True,
        }
        self.jobs.checkpoint(job)

    def _backend_health_check(self, job: BackgroundJob) -> bool:
        """Perform one bounded readiness check before resuming after an outage."""
        recovery = job.synthesis_state.get("backend_recovery")
        if not isinstance(recovery, dict):
            return True
        state = getattr(self.model, "_fulloch_backend_state", None)
        if state is not None:
            deadline = time.monotonic() + 5.0
            while state == "RECOVERING" and time.monotonic() < deadline:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                state = getattr(self.model, "_fulloch_backend_state", None)
            healthy = state == "READY"
            # A request timeout need not kill a healthy server. An explicit
            # resume may proceed once readiness is restored in the same process.
        else:
            # OpenAI-compatible endpoints expose a cheap model-list request. Do
            # not use a generation call as a health probe or replay an action.
            client = getattr(self.model, "_client", self.model)
            if callable(getattr(client, "with_options", None)):
                client = client.with_options(timeout=1.0, max_retries=0)
            models = getattr(client, "models", None)
            list_models = getattr(models, "list", None)
            if not callable(list_models):
                return False
            try:
                list_models(timeout=1.0)
                healthy = True
            except Exception:
                healthy = False
        if healthy:
            job.synthesis_state.pop("backend_recovery", None)
            self.jobs.checkpoint(job)
        return healthy

    def _update_stage(self, job: BackgroundJob, stage: str) -> None:
        self._require_backend()
        generation = getattr(self.model, "_fulloch_process_generation", None)
        if generation is not None:
            job.synthesis_state["process_generation"] = generation
        self.jobs.update_stage(job, stage)

    def _require_backend(self) -> None:
        """Stop unavailable work before preparing packets, probes or reservations."""
        if getattr(self, "_backend_lost", False):
            raise RemoteUnreachable("Research generation stopped after backend loss")
        state = getattr(self.model, "_fulloch_backend_state", None)
        if state in {"FAILED", "RECOVERING"}:
            exc = RemoteUnreachable(f"Local generation backend is {state.casefold()}")
            self._backend_lost = True
            self._checkpoint_backend_loss(exc)
            raise exc

    def _generate(self, **kwargs):
        if kwargs.get("history"):
            raise ValueError("Deep-think stages must use fresh context, not conversation history")
        self._require_backend()
        kwargs["history"] = []
        if self.failure_capture is not None:
            kwargs["failure_capture"] = self.failure_capture
        try:
            with nullcontext() if self.server_slots == 2 else self.model_lock:
                return self.generate(
                    self.model,
                    read_timeout=kwargs.pop("read_timeout", DEEP_THINK_READ_TIMEOUT_S),
                    generation_timeout=kwargs.pop("generation_timeout", DEEP_THINK_GENERATION_TIMEOUT_S),
                    recover_on_failure=False,
                    response_schema=kwargs.pop("response_schema", None),
                    **kwargs,
                )
        except Exception as exc:
            message = str(exc).casefold()
            backend_loss = isinstance(exc, (RemoteUnreachable, ConnectionError, TimeoutError, URLError)) or any(
                marker in message for marker in (
                    "connection reset", "connection refused", "connection is closed",
                    "server disconnected", "broken pipe", "backend unavailable",
                )
            )
            if backend_loss:
                self._backend_lost = True
                self._checkpoint_backend_loss(exc)
                if isinstance(exc, RemoteUnreachable):
                    raise
                raise RemoteUnreachable(f"Research generation backend unavailable: {type(exc).__name__}") from exc
            raise

    def _source_packet(self, job: BackgroundJob, cards: list[dict], **kwargs) -> dict:
        """Bounded source paragraphs accompany indexes in writing and audits."""
        self._require_backend()
        packet = serialize_synthesis_packet(cards, job.source_registry, **kwargs)
        sources = contextual_source_registry(job.evidence, job.artifacts, cards, job.source_registry)
        window = job.model_budget.get("context_tokens") or self.context_tokens or getattr(self.model, "context_tokens", 12_288)
        return with_source_passages(packet, sources, max_chars=min(96_000, max(512, int(window) * 4)))

    def _source_stage_fit(self, job, budget, stage, prompt, packet, *, verification=False) -> tuple[bool, str]:
        """Fit complete meaning-bearing work, not a claim after its context vanished."""
        self._require_backend()
        if packet.get("context_blockers"):
            return False, packet["context_blockers"][0]["reason"]
        try:
            budget.fit_packet(stage, system_prompt=prompt, user_prompt=get_thinking_user_prompt(stage))
            if verification:
                ledger = json.dumps(packet, ensure_ascii=True, separators=(",", ":"))
                audit_prompt = get_thinking_report_audit_prompt(
                    job.snapshot.task, ledger, "", compact=True,
                    constraint_checks=self._constraint_prompt_context(job),
                )
                count = budget.count_messages(system_prompt=audit_prompt,
                                              user_prompt=get_thinking_user_prompt("audit"), thinking_mode=False)
                # Reserve the forthcoming draft as well as the verifier's output.
                # Actual rendered passages are checked independently before audit;
                # unexpectedly large citation expansion is split, never clipped.
                required = count + budget.output_limit(stage) + 128 + budget.output_limit("audit") + budget.headroom
                if required > budget.context_tokens:
                    return False, "verification_context_does_not_fit"
        except PromptOverflow as exc:
            return False, exc.reason
        return True, ""

    def _queue_source_group(self, job, chapter, group, stage, *, reason="") -> dict:
        references = [card["id"] for card in group]
        fingerprint = hashlib.sha256(json.dumps(
            [{"id": card["id"], "provenance": card.get("provenance"), "finding": card.get("finding")}
             for card in group], ensure_ascii=True, sort_keys=True, default=str,
        ).encode()).hexdigest()
        unit = job.enqueue_work(f"{chapter['id']}:{stage}:{fingerprint[:16]}", "source_sample",
                                references=references, requirement_ids=chapter.get("requirement_ids", []),
                                fingerprint=fingerprint)
        if reason:
            job.transition_work(unit["id"], "blocked", reason="Indivisible source sample: " + reason)
        return unit

    def _summarize_research_notes(self, job: BackgroundJob, budget: ThinkingBudget,
                                  cancelled: Callable[[], bool], findings: str) -> str:
        """Optionally add one budgeted navigation summary between research and writing."""
        if not job.synthesis_state.get("idle_summaries_enabled"):
            return findings
        if estimate_tokens(findings) < 3000 or cancelled():
            return findings
        version = hashlib.sha256(json.dumps({
            "findings": findings, "scope": job.research_context,
            "evidence_ids": [item.get("artifact_id") for item in job.evidence],
        }, ensure_ascii=True, sort_keys=True, default=str).encode()).hexdigest()
        summaries = job.synthesis_state.setdefault("working_summaries", {})
        if version in summaries:
            return findings
        summaries[version] = {"state": "pending"}
        valid_refs = set(job.source_registry) | set(job.artifacts)
        prompt = get_idle_research_summary_prompt(findings)
        try:
            if not budget.fits("chapter_coverage", system_prompt=prompt,
                               user_prompt=get_thinking_user_prompt("summary_research"), thinking_mode=False):
                summaries[version] = {"state": "skipped", "reason": "budget_or_context"}
                return findings
            response = _budgeted_generate(
                self, budget, "chapter_coverage", system_prompt=prompt,
                user_prompt=get_thinking_user_prompt("summary_research"), temperature=0.0, cancel_check=cancelled,
                response_schema=None,  # validate_summary owns this distinct contract
            ) or ""
            parsed = _parse_direct_json(response)
            summary = validate_summary(parsed, valid_refs)
            current_version = hashlib.sha256(json.dumps({
                "findings": job.state, "scope": job.research_context,
                "evidence_ids": [item.get("artifact_id") for item in job.evidence],
            }, ensure_ascii=True, sort_keys=True, default=str).encode()).hexdigest()
            if summary is None or cancelled() or current_version != version:
                summaries[version] = {"state": "discarded_or_invalid"}
                return findings
            summaries[version] = {"state": "applied", "summary": summary}
            return json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
        except Exception as exc:
            summaries[version] = {"state": "failed", "reason": type(exc).__name__}
            self._require_backend()
            return findings

    def _constraint_audit(self, job: BackgroundJob, cards: list[dict]) -> None:
        """Deterministically check the task's typed constraints against retained records.

        Every structured field on a retained card is an explicit record value;
        a constraint that matches none of them is recorded as unknown, never
        as satisfied. A violated constraint is a hard finding: the delivered
        report cannot present the task as meeting a constraint its own
        evidence contradicts.
        """
        contract = job.snapshot.task_record or {}
        constraints = contract.get("typed_constraints")
        if not isinstance(constraints, list) or not constraints:
            job.synthesis_state.pop("constraint_audit", None)
            return
        records = []
        seen = set()
        for card in cards:
            fields = card.get("fields") if isinstance(card.get("fields"), dict) else None
            if not fields:
                continue
            # Identity is the field set, not the card ID: structurally identical
            # records from different cards are checked once.
            record = dict(fields)
            key = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                records.append(record)
        results = []
        for constraint in constraints[:24]:
            if not isinstance(constraint, dict):
                continue
            if not isinstance(constraint.get("field"), str):
                continue  # free-text constraints are advisory, not deterministically checkable
            results.append({
                "id": str(constraint.get("id") or constraint.get("text") or "")[:80],
                "text": str(constraint.get("text") or "")[:200],
                **validate_constraint_records(constraint, records, validator=self.constraint_validator),
            })
        if not results:
            job.synthesis_state.pop("constraint_audit", None)
            return
        previous_results = (job.synthesis_state.get("constraint_audit") or {}).get("results", [])
        job.synthesis_state["constraint_audit"] = {
            "results": results,
            "violated": [item["id"] for item in results if item["status"] == "violated"],
        }
        for item in results:
            if item["status"] == "violated" and item not in previous_results:
                job.state = _append_thinking_observation(
                    job.state, "constraint_check",
                    f"Constraint violated: {item['text'] or item['id']} — {item['reason']} "
                    f"(actual={item.get('actual')!r})",
                )

    def _constraint_prompt_context(self, job: BackgroundJob) -> str:
        constraints = job.snapshot.task_record.get("typed_constraints") or []
        if not constraints:
            return ""
        return json.dumps({
            "typed_constraints": constraints[:24],
            "results": (job.synthesis_state.get("constraint_audit") or {}).get("results", []),
        }, ensure_ascii=True, separators=(",", ":"))

    def _packet_prompt(
        self, job: BackgroundJob, cards: list[dict], requirement_ids: list[str],
        *, section_title: str = "", section_question: str = "", review_notes: str = "",
    ) -> tuple[str, str]:
        self._require_backend()
        packet = (
            self._source_packet(
                job, cards, requirement_ids=requirement_ids,
                reasons=job.synthesis_state.get("selection_reasons", {}),
            )
            if self.fact_card_synthesis
            else _raw_card_packet(cards, job.source_registry, requirement_ids)
        )
        ledger = json.dumps(packet, ensure_ascii=True, separators=(",", ":"))
        if packet.get("aliases"):
            key = hashlib.sha256(ledger.encode()).hexdigest()
            job.synthesis_state.setdefault("alias_maps", {})[key] = packet["aliases"]
        limitations = {
            # Card status, scope, missing fields and qualifications are already
            # in the authoritative packet. Do not duplicate them per card here.
            "evidence": report_limitations(job.evidence, []),
            "eligible_cards_not_selected": {
                "count": len(job.synthesis_state.get("omitted_card_ids", [])),
                "scope": "This packet is not exhaustive; cards without a named requirement match were not selected.",
            },
            "requirements": [
                {key: requirement.get(key) for key in ("identifier", "question", "state", "detail", "extraction_state")
                 if requirement.get(key) not in (None, "", [], {})}
                for requirement in job.evidence_requirements
                if requirement.get("identifier") in requirement_ids and requirement.get("state") != "resolved"
            ],
        }
        prompt = get_thinking_report_prompt(
            job.snapshot.task, "", ledger, review_notes=review_notes,
            limitations=json.dumps(limitations, ensure_ascii=True, separators=(",", ":")),
            section_title=section_title, section_question=section_question,
            constraint_checks=self._constraint_prompt_context(job),
        )
        return prompt, ledger

    def _audit_candidate(
        self, job: BackgroundJob, raw_report: str, cards: list[dict], ledger: str,
        budget: ThinkingBudget, cancelled: Callable[[], bool], *,
        section_title: str = "", section_question: str = "", allow_revision: bool = True,
        max_repairs: int = 4, chapter_repair: bool = False,
    ) -> tuple[str, list[dict], list[str], str]:
        return self._audit_passages(job, raw_report, cards, ledger, budget, cancelled,
                                    allow_revision=allow_revision, max_repairs=max_repairs,
                                    chapter_title=section_title, chapter_question=section_question,
                                    chapter_repair=chapter_repair)

    def _audit_passages(
        self, job, raw_report, cards, ledger, budget, cancelled, *, allow_revision,
        max_repairs=4, chapter_title="", chapter_question="", chapter_repair=False,
    ):
        """Audit the full draft once, then repair only rejected independently bound passages."""
        packet = json.loads(ledger)
        aliases = packet.get("aliases", {})
        original = _MODEL_SOURCES_RE.sub("", raw_report or "").strip()
        raw_report = _canonical_markers(original, aliases)
        passages = _passage_units(raw_report)
        numbered = "\n\n".join(f"[P{index}] {text}" for index, text in enumerate(passages, 1))
        audit_prompt = get_thinking_report_audit_prompt(
            job.snapshot.task, ledger, numbered, compact=True,
            constraint_checks=self._constraint_prompt_context(job),
        )
        fits, fit_reason = self._source_stage_fit(job, budget, "audit", audit_prompt, packet)
        factual_count = sum(len([sentence for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", visible)
                                 if sentence.strip()])
                             for passage in passages
                             if (visible := _SOURCE_MARKER_RE.sub("", passage).strip()) and not visible.startswith("#"))
        # Positive assessments repeat both the claim and its original support.
        # A context-fitting audit can still exceed its output allowance. Split
        # independent passages rather than losing a whole section to truncation.
        if len(passages) > 1 and factual_count > max(1, budget.output_limit("audit") // 320):
            fits, fit_reason = False, "verification_output_bound"
        if not fits:
            # Split at independent passage/support boundaries before dispatch,
            # preserving complete conflicting records and original context.
            if len(passages) > 1 and not packet.get("context_blockers"):
                retained, bindings, issues = [], [], []
                batches, batch, batch_facts = [], [], 0
                for passage in passages:
                    visible = _SOURCE_MARKER_RE.sub("", passage).strip()
                    count = (len([sentence for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", visible)
                                  if sentence.strip()]) if not visible.startswith("#") else 0)
                    if batch and batch_facts + count > max(1, budget.output_limit("audit") // 320):
                        batches.append("\n\n".join(batch))
                        batch, batch_facts = [], 0
                    batch.append(passage)
                    batch_facts += count
                if batch:
                    batches.append("\n\n".join(batch))
                # A context overflow rather than an output bound must strictly
                # reduce the packet too, so recursive audits always terminate.
                if len(batches) == 1:
                    middle = max(1, len(passages) // 2)
                    batches = ["\n\n".join(passages[:middle]), "\n\n".join(passages[middle:])]
                for passage in batches:
                    if cancelled():
                        return "", [], [], ""
                    if all(row.startswith("#") for row in _passage_units(passage)) and not _SOURCE_MARKER_RE.search(passage):
                        retained.append(passage)
                        continue
                    if not _SOURCE_MARKER_RE.search(passage) and not _unbound_factual_paragraphs(passage, cards):
                        retained.append(passage)
                        continue
                    ids = {identifier.strip() for match in _SOURCE_MARKER_RE.finditer(passage)
                           for identifier in match[1].split(",")}
                    subset = [card for group in _indivisible_card_groups(cards)
                              if any(card["id"] in ids for card in group) for card in group]
                    if not subset:
                        issues.append("[support; report; cards: unspecified] Missing valid passage support.")
                        continue
                    child_ledger = json.dumps(self._source_packet(job, subset), ensure_ascii=True, separators=(",", ":"))
                    _, child_bindings, child_issues, child_raw = self._audit_passages(
                        job, passage, subset, child_ledger, budget, cancelled,
                        allow_revision=allow_revision, max_repairs=max_repairs,
                        chapter_title=chapter_title, chapter_question=chapter_question, chapter_repair=chapter_repair,
                    )
                    if child_raw:
                        retained.append(child_raw)
                    bindings.extend(child_bindings)
                    issues.extend(child_issues)
                raw = "\n\n".join(retained)
                if issues:
                    job.synthesis_state["partial_report"] = True
                    # A dropped premise invalidates cross-passage conclusions.
                    raw = re.sub(r"(?ims)^## (?:Summary|Conclusions)\s*\n.*?(?=^## |\Z)", "", raw).strip()
                clean, bindings = _extract_report_bindings(raw, cards)
                return clean, bindings, list(dict.fromkeys(issues)), raw
            self._queue_source_group(job, {"id": chapter_title or "audit"}, cards, "audit", reason=fit_reason)
            return "", [], [f"[support; report; cards: unspecified] Indivisible verification packet: {fit_reason}."], ""
        diagnostics = job.synthesis_state.setdefault("passage_reviews", [])
        review = {"original": original, "canonical_draft": raw_report, "repairs": [], "rejections": []}
        diagnostics.append(review)
        audit_response = ""
        audit_ledger = ledger
        audit_ok = False
        try:
            audit_response = _budgeted_generate(
                self, budget, "audit", user_prompt=get_thinking_user_prompt("audit"),
                system_prompt=audit_prompt,
                response_schema=_audit_response_schema(cards, aliases),
                temperature=0.0, cancel_check=cancelled,
            ) or ""
            verdict = _parse_direct_json(audit_response)
            audit_ok = isinstance(verdict, dict) and verdict.get("verdict") in {"pass", "revise"} and isinstance(verdict.get("issues"), list)
        except (ValueError, TypeError):
            pass
        except PromptOverflow:
            bound_ids = {value.strip() for match in _SOURCE_MARKER_RE.finditer(raw_report)
                         for value in match[1].split(",")}
            compact_cards = [card for group in _indivisible_card_groups(cards)
                             if any(card["id"] in bound_ids for card in group) for card in group]
            if compact_cards and len(compact_cards) < len(cards):
                compact = json.dumps(self._source_packet(job, compact_cards), separators=(",", ":"))
                audit_ledger = compact
                try:
                    audit_response = _budgeted_generate(self, budget, "audit",
                        request_kind="context_retry",
                        user_prompt=get_thinking_user_prompt("audit"),
                        system_prompt=get_thinking_report_audit_prompt(
                            job.snapshot.task, compact, numbered,
                            constraint_checks=self._constraint_prompt_context(job),
                            compact=True,
                        ),
                        response_schema=_audit_response_schema(compact_cards, json.loads(compact).get("aliases")),
                        temperature=0.0, cancel_check=cancelled) or ""
                    verdict = _parse_direct_json(audit_response)
                    audit_ok = isinstance(verdict, dict) and verdict.get("verdict") in {"pass", "revise"} and isinstance(verdict.get("issues"), list)
                except (ValueError, TypeError, ModelBudgetExhausted, PromptOverflow):
                    pass
                except Exception as exc:
                    self._require_backend()
                    job.record_stage_event("report_audit", "citation_failure", "audit_unavailable")
                    logger.warning(
                        "Deep-think compact passage audit unavailable error_type=%s",
                        type(exc).__name__,
                    )
            job.synthesis_state.setdefault("stages_skipped", []).append("audit_overflow_or_budget")
        except ModelBudgetExhausted:
            job.synthesis_state.setdefault("stages_skipped", []).append("audit_overflow_or_budget")
        except Exception as exc:
            self._require_backend()
            job.record_stage_event("report_audit", "citation_failure", "audit_unavailable")
            logger.warning("Deep-think passage audit unavailable error_type=%s", type(exc).__name__)
        model_issues = _parse_report_audit(audit_response, raw_report, cards, audit_ledger)
        # A malformed revision verdict is not a successful support audit.
        if audit_ok and verdict.get("verdict") == "revise" and not model_issues:
            audit_ok = False
        # New-format audits must positively bind every factual sentence. Keep
        # the legacy verdict contract readable for saved/deployed model clients;
        # it remains subject to all deterministic field, time, identity and
        # binding checks above/below.
        support_issues = (
            _report_support_assessments(verdict, passages, cards, json.loads(audit_ledger))
            if audit_ok else {}
        )
        review["audit"] = audit_response
        accepted = []
        rejected = []
        by_id = {card["id"]: card for card in cards}
        validated = job.synthesis_state.setdefault("validated_passages", [])
        for index, passage in enumerate(passages, 1):
            if cancelled():
                return "", [], [], ""
            if not _SOURCE_MARKER_RE.sub("", passage).strip() or all(
                line.startswith("#") for line in passage.splitlines()
            ):
                accepted.append(passage)
                continue
            if not _unbound_factual_paragraphs(passage, cards) and not _SOURCE_MARKER_RE.search(passage):
                # Only controller-owned status vocabulary is exempt, never a model's
                # uncited claim labelled as a limitation or inference.
                accepted.append(passage)
                continue
            issues = _deterministic_report_audit(passage, cards)
            issues += _invalid_report_bindings(passage, cards)
            issues += _temporal_report_audit(passage, cards, job.research_context)
            issues += support_issues.get(index, [])
            issues += [f"[citation; P{index}; cards: unspecified] Missing valid citation." for _ in _unbound_factual_paragraphs(passage, cards)]
            ids = {value.strip() for match in _SOURCE_MARKER_RE.finditer(passage) for value in match[1].split(",")}
            passage_key = hashlib.sha256((passage + json.dumps(
                [by_id[value] for value in sorted(ids) if value in by_id], sort_keys=True,
            ) + self._constraint_prompt_context(job)).encode()).hexdigest()
            for issue in model_issues:
                location = re.search(r"\[.*?;\s*([^;]+);", issue)
                location_text = location[1].strip() if location else "report"
                location_match = re.search(r"\bP(\d+)\b", location_text)
                if (location_match and int(location_match[1]) == index) or (
                    not location_match and (not issue_card_ids(issue) or ids & issue_card_ids(issue)
                                            or not issue_card_ids(issue) & set(by_id))
                ):
                    issues.append(issue)
            citation_issues = [issue for issue in issues if issue.startswith("[citation;")]
            if citation_issues:
                reason = (
                    "unknown_card_id" if any("Unknown or ineligible card ID" in issue for issue in citation_issues)
                    else "missing_citation" if any("Missing valid citation" in issue for issue in citation_issues)
                    else "invalid_support_id"
                )
                valid_ids = sorted({
                    card_id for issue in citation_issues for card_id in issue_card_ids(issue)
                    if card_id in by_id
                })
                job.record_stage_event(
                    "report_audit", "citation_failure", reason,
                    count=len(citation_issues), record_ids=valid_ids,
                )
            if not audit_ok and passage_key not in validated:
                # Without a semantic audit, exact retained extracts alone can be
                # delivered. Valid-looking markers cannot authorize new prose.
                prose = _SOURCE_MARKER_RE.sub("", passage).strip().lstrip("- ")
                if not any(prose == str(by_id[identifier].get("statement") or "") for identifier in ids if identifier in by_id):
                    issues.append(f"[support; P{index}; cards: {', '.join(sorted(ids))}] Factual support audit unavailable.")
            if not issues:
                accepted.append(passage)
                if audit_ok and passage_key not in validated:
                    validated.append(passage_key)
                continue
            repaired = ""
            if allow_revision and len(review["repairs"]) < max_repairs:
                # Include supporting and conflicting records intact; an unknown
                # alias never silently selects unrelated evidence for recovery.
                supporting = [card for card in cards if card["id"] in ids]
                keys = {_conflict_scope_key(card) for card in supporting}
                repair_cards = [card for group in _indivisible_card_groups(cards)
                                if any(card["id"] in ids or (card.get("evidence_status") == "conflicted" and _conflict_scope_key(card) in keys) for card in group)
                                for card in group]
                if repair_cards:
                    repair_prompt, repair_ledger = self._packet_prompt(
                        job, repair_cards, [], section_title=f"P{index}",
                        section_question="Correct only the affected passage; do not add conclusions.",
                        review_notes="\n".join(issues) + "\nOriginal passage:\n" + passage,
                    )
                    if chapter_repair:
                        repair_prompt = get_thinking_chapter_repair_prompt(
                            job.snapshot.task, _fixed_scope_for_prompt(job.research_context),
                            chapter_title, chapter_question, repair_ledger, passage,
                            "\n".join(issues),
                            constraint_checks=self._constraint_prompt_context(job),
                        )
                    attempt = {"passage": index, "draft": passage, "issues": issues,
                               "card_ids": [card["id"] for card in repair_cards]}
                    review["repairs"].append(attempt)
                    try:
                        repair_schema = None
                        if chapter_repair:
                            repair_aliases = json.loads(repair_ledger).get("aliases") or {}
                            repair_schema = {"type": "object", "properties": {
                                "text": {"type": "string"},
                                "cards": {"type": "array", "minItems": 1, "items": {
                                    "type": "string", "enum": list(repair_aliases),
                                }},
                            }, "required": ["text", "cards"]}
                        revised_response = _budgeted_generate(self, budget, "revision",
                            request_kind="support_revision",
                            user_prompt=get_thinking_user_prompt("passage_repair"), system_prompt=repair_prompt,
                            temperature=0.2, cancel_check=cancelled,
                            response_schema=repair_schema) or ""
                        if chapter_repair:
                            repaired_json = _parse_direct_json(revised_response)
                            rendered_revision, repair_error = _render_chapter_passage(
                                (repaired_json or {}).get("text"),
                                (repaired_json or {}).get("cards"),
                                (json.loads(repair_ledger).get("aliases") or {}),
                            )
                            revised = rendered_revision if not repair_error else ""
                            if repair_error:
                                attempt["validation_error"] = repair_error
                        else:
                            revised = revised_response
                        attempt["output"] = revised
                        clean, bindings, remaining, checked = self._audit_passages(
                            job, revised, repair_cards, repair_ledger, budget, cancelled,
                            allow_revision=False, max_repairs=0,
                            chapter_title=chapter_title, chapter_question=chapter_question,
                            chapter_repair=chapter_repair,
                        ) if revised.strip() else ("", [], issues, "")
                        if clean and bindings and not remaining:
                            repaired = checked
                    except (ModelBudgetExhausted, PromptOverflow):
                        attempt["error"] = "repair_overflow_or_budget"
                        job.synthesis_state.setdefault("stages_skipped", []).append("revision_overflow_or_budget")
                    except Exception as exc:
                        attempt["error"] = type(exc).__name__
                        self._require_backend()
            if repaired:
                accepted.append(repaired)
            else:
                rejected.extend(issues)
                review["rejections"].append({"passage": index, "text": passage, "reasons": issues})
        # Conclusions have cross-passage dependencies. If any premise was removed,
        # do not publish a report-wide conclusion based on the original draft.
        if rejected:
            safe_passages = []
            conclusion = False
            for text in accepted:
                if text.startswith("#"):
                    conclusion = bool(re.search(r"\b(?:summary|conclusion)\b", text, re.I))
                    if not conclusion:
                        safe_passages.append(text)
                elif not conclusion and not re.match(
                    r"(?:overall\b|in conclusion\b|therefore\b|on balance\b|inference:)",
                    _SOURCE_MARKER_RE.sub("", text), re.I,
                ):
                    safe_passages.append(text)
            accepted = safe_passages
            job.synthesis_state["partial_report"] = True
        retained = "\n\n".join(accepted)
        clean, bindings = _extract_report_bindings(retained, cards)
        if rejected:
            if not bindings:
                clean = _safe_audit_fallback(job.snapshot.task, rejected)
        return clean.strip(), bindings, list(dict.fromkeys(rejected)), retained

    def _chapter_cards_for_synthesis(self, job: BackgroundJob, chapter: dict,
                                     cards: list[dict]) -> list[dict]:
        """Select evidence linked to the chapter, including reusable source cards."""
        requirements = [item for item in job.evidence_requirements
                        if item.get("chapter_id") == chapter.get("id")]
        if any(not item.get("is_subject_identity") for item in requirements):
            requirements = [item for item in requirements if not item.get("is_subject_identity")]
        selected, selection_reasons = select_requirement_cards(
            cards, requirements, source_registry=job.source_registry,
        )
        # Retrieval links are inventory, not permission to bypass selection caps.
        # Explicit supporting IDs are already prioritised by the selector, which
        # also retains whole records and conflict peers beyond its soft caps.
        for card in selected:
            card["chapter_ids"] = sorted({*(card.get("chapter_ids") or []), str(chapter.get("id") or "")})
            linked_requirements = {
                reason.partition(":")[0]
                for reason in selection_reasons.get(card["id"], [])
                if reason.partition(":")[0] in {item.get("identifier") for item in requirements}
            }
            if linked_requirements:
                card["requirement_ids"] = sorted({
                    *(card.get("requirement_ids") or []), *linked_requirements,
                })
        return selected

    def _review_source_documents(self, job, budget, cancelled):
        """Model-review all retained originals in bounded, cached page packets.

        No source-label, sentence-pattern, date or query-word filter owns this
        inventory. Only measured request limits partition it. Failed/unreviewed
        batches remain explicit and do not authorize deleting their evidence.
        """
        from utils.prompts import (
            THINKING_SOURCE_REVIEW_USER_PROMPT,
            get_thinking_source_review_prompt,
        )

        from .thinking_source_context import source_block_card, source_document_blocks

        sources = contextual_source_registry(job.evidence, job.artifacts, job.fact_cards, job.source_registry)
        blocks = source_document_blocks(sources)
        if not blocks:
            return
        # Persist the same originals/version locations used by reviews and audits.
        job.source_registry.update(sources)
        questions = json.dumps([{"id": item["identifier"], "question": item["question"]}
                                for item in job.evidence_requirements], ensure_ascii=False)
        question_ids = {item["identifier"] for item in job.evidence_requirements}
        cache = job.synthesis_state.setdefault("source_reviews", {})
        packets, pending = [], {}

        def prompt_for(packet):
            pages = {}
            for index, (_identifier, block) in enumerate(packet.items(), 1):
                source = sources[block["source"]]
                page = pages.setdefault(block["source"], {
                    "source": block["source"],
                    **{key: source[key] for key in ("title", "url", "published_at", "updated_at") if source.get(key)},
                    "blocks": {},
                })
                page["blocks"][f"B{index}"] = block["text"]
            return get_thinking_source_review_prompt(
                job.snapshot.task, _fixed_scope_for_prompt(job.research_context), questions,
                json.dumps(list(pages.values()), ensure_ascii=False, separators=(",", ":")),
            )

        for identifier, block in blocks.items():
            candidate = {**pending, identifier: block}
            # Soft size target only partitions work. Whole tables stay indivisible.
            if pending and (sum(len(row["text"]) for row in candidate.values()) > 14_000
                            or len({row["source"] for row in candidate.values()}) > 4):
                packets.append(pending)
                pending = {}
            pending[identifier] = block
        if pending:
            packets.append(pending)
        # Repartition by the complete tokenized prompt, not a raw text estimate.
        fitted = []
        for packet in packets:
            todo = [packet]
            while todo:
                part = todo.pop(0)
                if budget.fits("source_review", system_prompt=prompt_for(part),
                               user_prompt=THINKING_SOURCE_REVIEW_USER_PROMPT):
                    fitted.append(part)
                elif len(part) > 1:
                    rows = list(part.items())
                    middle = len(rows) // 2
                    todo[:0] = [dict(rows[:middle]), dict(rows[middle:])]
                else:
                    job.record_stage_event("source_review", "evidence_omission", "original_block_does_not_fit",
                                           record_ids=list(part))
        selected = {identifier: set() for identifier in question_ids}
        retained_cards = {card["id"]: card for card in job.fact_cards}
        reviewed_blocks = set()
        for packet in fitted:
            if cancelled():
                return
            fingerprint = hashlib.sha256(json.dumps({
                "task": job.snapshot.task, "questions": questions, "blocks": packet,
            }, sort_keys=True).encode()).hexdigest()
            aliases = {f"B{index}": identifier for index, identifier in enumerate(packet, 1)}
            page_ids = {row["source"] for row in packet.values()}
            result = cache.get(fingerprint)
            if result is None:
                self._update_stage(job, f"Reviewing retained sources ({len(reviewed_blocks)}/{len(blocks)} passages)")
                page_schemas = {}
                for source_id in page_ids:
                    local_blocks = [alias for alias, identifier in aliases.items() if packet[identifier]["source"] == source_id]
                    page_schemas[source_id] = {"type": "object", "properties": {
                        "summary": {"type": "string"},
                        "findings": {"type": "array", "maxItems": 4, "items": {
                            "type": "object", "properties": {
                                "summary": {"type": "string"},
                                "questions": {"type": "array", "items": {"type": "string", "enum": sorted(question_ids)}},
                                "blocks": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "enum": local_blocks}},
                            }, "required": ["summary", "questions", "blocks"],
                        }},
                    }, "required": ["summary", "findings"]}
                try:
                    response = _budgeted_generate(
                        self, budget, "source_review", system_prompt=prompt_for(packet),
                        user_prompt=THINKING_SOURCE_REVIEW_USER_PROMPT, temperature=0.0, cancel_check=cancelled,
                        response_schema={"type": "object", "properties": {"pages": {
                            "type": "object", "properties": page_schemas, "required": sorted(page_ids),
                        }}, "required": ["pages"]},
                    )
                    result = _parse_direct_json(response)
                except (ModelBudgetExhausted, PromptOverflow):
                    break
                pages = (result or {}).get("pages")
                if isinstance(pages, dict):
                    pages = [{"source": key, **value} for key, value in pages.items() if isinstance(value, dict)]
                    result = {"pages": pages}
                valid = isinstance(pages, list) and {row.get("source") for row in pages if isinstance(row, dict)} == page_ids
                if valid:
                    for page in pages:
                        if (not isinstance(page, dict) or not isinstance(page.get("summary"), str)
                                or not isinstance(page.get("findings"), list)):
                            valid = False
                            break
                        for finding in page["findings"]:
                            if (not isinstance(finding, dict) or not isinstance(finding.get("summary"), str)
                                    or not isinstance(finding.get("questions"), list)
                                    or not isinstance(finding.get("blocks"), list) or not finding["blocks"]
                                    or any(q not in question_ids for q in finding["questions"])
                                    or any(key not in aliases or packet[aliases[key]]["source"] != page["source"]
                                           for key in finding["blocks"])):
                                valid = False
                                break
                if not valid:
                    job.record_stage_event("source_review", "output_rejection", "invalid_original_bindings")
                    continue
                cache[fingerprint] = result
                self.jobs.checkpoint(job)
            reviewed_blocks.update(packet)
            for page in result["pages"]:
                for finding in page["findings"]:
                    for alias in finding["blocks"]:
                        identifier = aliases[alias]
                        block = packet[identifier]
                        existing = next((card for card in retained_cards.values()
                                         if card.get("statement") == block["text"]
                                         and (card.get("provenance") or {}).get("source_id") == block["source"]
                                         and (card.get("provenance") or {}).get("source_version") == block["source_version"]
                                         and (card.get("provenance") or {}).get("start") == block["start"]
                                         and (card.get("provenance") or {}).get("end") == block["end"]), None)
                        card = existing if existing is not None else retained_cards.setdefault(identifier, source_block_card(
                            identifier, block, sources[block["source"]],
                        ))
                        card["finding"]["question_ids"] = sorted(set(
                            card["finding"].get("question_ids", []) + finding["questions"],
                        ))
                        for question_id in finding["questions"]:
                            selected[question_id].add(card["id"])
        job.fact_cards = list(retained_cards.values())
        complete = reviewed_blocks == set(blocks)
        for requirement in job.evidence_requirements:
            requirement["model_selected_card_ids"] = sorted(selected[requirement["identifier"]])
            requirement["source_review_complete"] = complete
        job.synthesis_state["source_review_progress"] = {
            "sources": len({row["source"] for row in blocks.values()}),
            "blocks": len(blocks), "reviewed_blocks": len(reviewed_blocks), "complete": complete,
            "selected_blocks": len(set().union(*selected.values())),
        }

    def _extract_chapter_cards(
        self, job: BackgroundJob, chapter: dict, cards: list[dict],
        budget: ThinkingBudget, cancelled: Callable[[], bool], call_limit: int | None = None,
    ) -> tuple[dict, list[dict], list[dict]]:
        """Build bounded compact facts while retaining original cards for grounding."""
        requirements = [item for item in job.evidence_requirements
                        if item.get("chapter_id") == chapter.get("id")]
        if any(not item.get("is_subject_identity") for item in requirements):
            requirements = [item for item in requirements if not item.get("is_subject_identity")]
        question_input = json.dumps([{
            "question": item.get("question"), "priority": item.get("importance"),
        } for item in requirements], ensure_ascii=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(json.dumps(
            {"cards": cards, "questions": question_input,
             "source_context": self._source_packet(job, cards).get("passages", [])}, ensure_ascii=True,
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        if (chapter.get("card_extraction_fingerprint") == fingerprint
                and isinstance(chapter.get("compact_card_facts"), dict)):
            return (chapter["compact_card_facts"], chapter.get("compact_disagreements", []),
                    chapter.get("compact_missing_information", []))

        # Exact short claims are already compact, and complete records must be
        # passed intact anyway. Rephrasing every one consumed most of the live
        # writing budget without recovering the discarded article context.
        exact_facts = {
            card["id"]: str(card.get("statement") or "") or json.dumps(
                card.get("fields") or {}, ensure_ascii=True, separators=(",", ":"),
            ) for card in cards
            if chapter.get("coverage_checked") or card.get("kind") == "structured_record"
            or (card.get("finding") or {}).get("context_state") == "retained"
            or len(str(card.get("statement") or "")) <= 400
        }
        groups = [group for group in _indivisible_card_groups(cards)
                  if any(card["id"] not in exact_facts for card in group)]
        chunks: list[list[dict]] = []
        current: list[dict] = []

        def extraction_prompt(chunk):
            packet = self._source_packet(
                job, chunk,
                requirement_ids=[item["identifier"] for item in requirements],
            )
            ledger = json.dumps(packet, ensure_ascii=True, separators=(",", ":"))
            return get_thinking_chapter_card_prompt(chapter["title"], question_input, ledger), ledger

        def extraction_fits(prompt, ledger):
            return self._source_stage_fit(job, budget, "card_extraction", prompt, json.loads(ledger))[0]

        skipped_groups = []
        max_chunk_cards = max(1, budget.policy("card_extraction")["output_tokens"] // 128)
        for group in groups:
            candidate = [*current, *group]
            prompt, ledger = extraction_prompt(candidate)
            if current and (len(candidate) > max_chunk_cards or not extraction_fits(prompt, ledger)):
                chunks.append(current)
                current = []
                prompt, ledger = extraction_prompt(group)
            if extraction_fits(prompt, ledger):
                current = [*current, *group]
            else:
                skipped_groups.append(group)
                self._queue_source_group(job, chapter, group, "card_extraction", reason="required_context_or_output")
        if current:
            chunks.append(current)

        saved_groups = chapter.setdefault("card_extraction_groups", {})
        compact_facts: dict[str, str] = dict(exact_facts)
        disagreements: list[dict] = []
        missing_information: list[dict] = []
        for chunk in chunks:
            chunk_fingerprint = hashlib.sha256(json.dumps(
                {"cards": chunk, "source_context": self._source_packet(job, chunk).get("passages", [])},
                ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str,
            ).encode()).hexdigest()
            cached = saved_groups.get(chunk_fingerprint)
            if isinstance(cached, dict) and cached.get("state") in {"complete", "fallback"}:
                compact_facts.update(cached.get("facts") or {})
                disagreements.extend(cached.get("disagreements") or [])
                missing_information.extend(cached.get("missing_information") or [])
                continue
            prompt, ledger = extraction_prompt(chunk)
            packet = json.loads(ledger)
            aliases = packet.get("aliases") or {}
            alias_to_id = dict(aliases)
            response = ""
            state = "fallback"
            extraction_calls = budget.state["stages"].get("card_extraction", {}).get("calls", 0)
            if call_limit is None or extraction_calls < call_limit:
                try:
                    response = _budgeted_generate(
                        self, budget, "card_extraction",
                        system_prompt=prompt,
                        user_prompt=get_thinking_user_prompt("card_extraction"),
                        temperature=0.0, cancel_check=cancelled,
                    ) or ""
                except (PromptOverflow, ModelBudgetExhausted):
                    job.synthesis_state.setdefault("stages_skipped", []).append(
                        "chapter_card_extraction_budget_or_context",
                    )
                except Exception as exc:
                    self._require_backend()
                    logger.warning("Deep-think compact card extraction failed error_type=%s", type(exc).__name__)
            else:
                job.synthesis_state.setdefault("stages_skipped", []).append(
                    "chapter_card_extraction_calls_reserved_for_later_chapters",
                )
            parsed = _parse_direct_json(response)
            row_facts = {}
            row_disagreements = []
            row_missing = []
            if isinstance(parsed, dict) and isinstance(parsed.get("cards"), list):
                for row in parsed["cards"]:
                    if not isinstance(row, dict):
                        continue
                    alias, fact = row.get("alias"), row.get("fact")
                    if (isinstance(alias, str) and alias in alias_to_id
                            and isinstance(fact, str) and fact.strip() and len(fact) <= 1_000):
                        row_facts[alias_to_id[alias]] = fact.strip()
                disagreement_rows = parsed.get("disagreements")
                if not isinstance(disagreement_rows, list):
                    disagreement_rows = []
                for row in disagreement_rows:
                    if not isinstance(row, dict) or not isinstance(row.get("aliases"), list):
                        continue
                    ids = [alias_to_id[value] for value in row["aliases"]
                           if isinstance(value, str) and value in alias_to_id]
                    if len(ids) > 1:
                        row_disagreements.append({"card_ids": list(dict.fromkeys(ids)),
                                                  "detail": str(row.get("detail") or "")[:500]})
                missing_rows = parsed.get("missing_information")
                if not isinstance(missing_rows, list):
                    missing_rows = []
                for value in missing_rows:
                    if isinstance(value, str) and value.strip():
                        row_missing.append({"detail": value.strip()[:500]})
                state = "complete" if row_facts else "fallback"
            for card in chunk:
                if card["id"] not in row_facts:
                    row_facts[card["id"]] = (
                        str(card.get("statement") or "")
                        or json.dumps(card.get("fields") or {}, ensure_ascii=True, separators=(",", ":"))
                    )
            saved_groups[chunk_fingerprint] = {
                "state": state, "facts": row_facts,
                "disagreements": row_disagreements,
                "missing_information": row_missing,
            }
            compact_facts.update(row_facts)
            disagreements.extend(row_disagreements)
            missing_information.extend(row_missing)
            if cancelled():
                return compact_facts, disagreements, missing_information

        for group in skipped_groups:
            for card in group:
                compact_facts[card["id"]] = (
                    str(card.get("statement") or "")
                    or json.dumps(card.get("fields") or {}, ensure_ascii=True, separators=(",", ":"))
                )
        chapter["compact_card_facts"] = compact_facts
        chapter["compact_disagreements"] = disagreements
        chapter["compact_missing_information"] = missing_information
        chapter["card_extraction_fingerprint"] = fingerprint
        chapter["card_extraction_state"] = (
            "exact" if not groups else "complete" if chunks and not skipped_groups else "fallback"
        )
        return compact_facts, disagreements, missing_information

    def _assemble_final_sections(
        self, job: BackgroundJob, cards: list[dict], budget: ThinkingBudget,
        cancelled: Callable[[], bool],
    ) -> tuple[str, str]:
        """Generate summary/conclusions from compact digests, then audit cited cards."""
        by_id = {str(card.get("id")): card for card in cards}
        digest_rows = []
        digest_ids: set[str] = set()
        for chapter in job.chapter_plan:
            digest = str(chapter.get("digest") or "").strip()
            if not digest:
                continue
            supporting = {
                value.strip() for text in (digest,)
                for match in _SOURCE_MARKER_RE.finditer(text)
                for value in match[1].split(",")
                if value.strip() in by_id
            }
            if not supporting:
                continue
            digest_ids.update(supporting)
            digest_rows.append({
                "chapter": chapter.get("title"), "digest": digest,
                "supporting_card_ids": sorted(supporting),
            })

        uncertainty = []
        for item in job.evidence_requirements:
            if item.get("importance") != "high" or item.get("state") == "resolved":
                continue
            row = {"question": item.get("question")}
            detail = str(item.get("detail") or "")
            if detail.casefold().startswith("missing facts:"):
                row["specific_unknowns"] = detail.split(":", 1)[1].strip()[:400]
            uncertainty.append(row)
        # Delivered, resolved high-priority answers anchor the summary so it
        # cannot contradict a finding the report body already shows.
        resolved = []
        for chapter in job.chapter_plan:
            title = str(chapter.get("title") or "").strip()
            for identifier in chapter.get("analyzed_card_ids", []):
                card = by_id.get(str(identifier))
                if card is None:
                    continue
                for item in job.evidence_requirements:
                    if item.get("importance") != "high" or item.get("state") != "resolved":
                        continue
                    if identifier not in (item.get("evidence_ids") or []):
                        continue
                    text = _clean_report_block(
                        str(card.get("statement") or "")
                        or _structured_record_text(card)
                    )
                    if not text or len(text) > 500:
                        continue
                    resolved.append({
                        "question": str(item.get("question") or title),
                        "finding": text, "card_id": card["id"],
                    })
        resolved = resolved[:8]
        constraint_checks = self._constraint_prompt_context(job)
        fingerprint = hashlib.sha256(json.dumps({
            "task": job.snapshot.task,
            "scope": _fixed_scope_for_prompt(job.research_context),
            "digests": digest_rows,
            "support": [{key: by_id[identifier].get(key) for key in
                         ("id", "provenance", "statement", "fields", "evidence_status", "compatibility")}
                        for identifier in sorted(digest_ids & set(by_id))],
            "uncertainty": uncertainty,
            "resolved": resolved,
            **({"constraint_checks": constraint_checks} if constraint_checks else {}),
        }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cached_result = job.synthesis_state.get("final_assembly_result")
        if (isinstance(cached_result, dict) and cached_result.get("fingerprint") == fingerprint
                and cached_result.get("state") in {"complete", "digest_fallback"}):
            return str(cached_result.get("summary") or ""), str(cached_result.get("conclusions") or "")

        def save_result(summary_value: str, conclusions_value: str, state: str):
            job.synthesis_state["final_assembly_result"] = {
                "fingerprint": fingerprint, "state": state,
                "summary": summary_value, "conclusions": conclusions_value,
            }
            return summary_value, conclusions_value

        if not digest_rows:
            job.synthesis_state["final_assembly_state"] = "no_supported_digests"
            return save_result("", "", "digest_fallback")

        prompt = get_thinking_final_assembly_prompt(
            job.snapshot.task, _fixed_scope_for_prompt(job.research_context),
            json.dumps(digest_rows, ensure_ascii=True, separators=(",", ":")),
            json.dumps(uncertainty, ensure_ascii=True, separators=(",", ":")),
            json.dumps(resolved, ensure_ascii=True, separators=(",", ":")),
            constraint_checks=self._constraint_prompt_context(job),
        )
        parsed = None
        first_response = ""
        if budget.fits(
            "final_assembly", system_prompt=prompt,
            user_prompt=get_thinking_user_prompt("final_assembly"),
            thinking_mode=False,
        ):
            try:
                first_response = _budgeted_generate(
                    self, budget, "final_assembly", system_prompt=prompt,
                    user_prompt=get_thinking_user_prompt("final_assembly"),
                    temperature=0.2, cancel_check=cancelled,
                ) or ""
            except (PromptOverflow, ModelBudgetExhausted):
                job.synthesis_state["final_assembly_state"] = "budget_or_context"
            except Exception as exc:
                self._require_backend()
                logger.warning("Deep-think final assembly failed error_type=%s", type(exc).__name__)
            parsed = _parse_direct_json(first_response)
        valid = (
            isinstance(parsed, dict)
            and isinstance(parsed.get("summary"), str)
            and isinstance(parsed.get("conclusions"), str)
            and len(parsed.get("summary", "")) <= 8_000
            and len(parsed.get("conclusions", "")) <= 8_000
            and bool(parsed.get("summary", "").strip())
        )
        repair_prompt = get_thinking_output_repair_prompt(prompt, THINKING_FINAL_ASSEMBLY_OUTPUT_ERROR)
        if not valid and not cancelled() and budget.fits(
            "final_assembly", system_prompt=repair_prompt,
            user_prompt=get_thinking_user_prompt("final_assembly_repair"), thinking_mode=False,
        ):
            try:
                repair = _budgeted_generate(
                    self, budget, "final_assembly",
                    request_kind="format_retry",
                    system_prompt=repair_prompt,
                    user_prompt=get_thinking_user_prompt("final_assembly_repair"),
                    temperature=0.0, cancel_check=cancelled,
                ) or ""
            except (PromptOverflow, ModelBudgetExhausted):
                repair = ""
            except Exception as exc:
                self._require_backend()
                logger.warning("Deep-think final assembly repair failed: %s", type(exc).__name__)
                repair = ""
            parsed = _parse_direct_json(repair)
            valid = (
                isinstance(parsed, dict)
                and isinstance(parsed.get("summary"), str)
                and isinstance(parsed.get("conclusions"), str)
                and len(parsed.get("summary", "")) <= 8_000
                and len(parsed.get("conclusions", "")) <= 8_000
                and bool(parsed.get("summary", "").strip())
            )

        if cancelled():
            save_result(
                _clean_report_block("\n\n".join(row["digest"] for row in digest_rows)),
                "", "digest_fallback",
            )
            return "", ""
        if not valid:
            # Chapter digests have already passed source-card audits; they are a
            # safe deterministic summary, while unsupported conclusions stay out.
            summary = _concise_summary(_clean_report_block("\n\n".join(row["digest"] for row in digest_rows)))
            conclusions = ""
            job.synthesis_state["final_assembly_state"] = "digest_fallback"
        else:
            summary = _clean_report_block(parsed["summary"])
            conclusions = _clean_report_block(parsed["conclusions"])
            if re.search(r"\(C\d+(?:[,;\s]+C\d+)*\)", summary + conclusions):
                summary = _concise_summary(_clean_report_block("\n\n".join(row["digest"] for row in digest_rows)))
                conclusions = ""
                job.synthesis_state["final_assembly_state"] = "digest_fallback"
            else:
                job.synthesis_state["final_assembly_state"] = "structured"

        final_ids = {
            value.strip() for text in (summary, conclusions)
            for match in _SOURCE_MARKER_RE.finditer(text)
            for value in match[1].split(",")
        }
        if not (final_ids & digest_ids):
            summary = _concise_summary(_clean_report_block("\n\n".join(row["digest"] for row in digest_rows)))
            conclusions = ""
            final_ids = {
                value.strip() for match in _SOURCE_MARKER_RE.finditer(summary)
                for value in match[1].split(",")
            }
            job.synthesis_state["final_assembly_state"] = "digest_fallback"

        cited_cards = [by_id[identifier] for identifier in sorted(final_ids & digest_ids & set(by_id))]
        conflict_keys = {_conflict_scope_key(card) for card in cited_cards
                         if card.get("evidence_status") == "conflicted"}
        for card in cards:
            if (card.get("evidence_status") == "conflicted"
                    and _conflict_scope_key(card) in conflict_keys
                    and card["id"] not in {item["id"] for item in cited_cards}):
                cited_cards.append(card)
        if not cited_cards:
            return save_result(summary, conclusions, "digest_fallback")

        raw = "## Summary\n\n" + summary
        if conclusions:
            raw += "\n\n## Conclusions\n\n" + conclusions
        ledger_packet = self._source_packet(job, cited_cards)
        ledger = json.dumps(ledger_packet, ensure_ascii=True, separators=(",", ":"))
        audited, bindings, issues, audited_raw = self._audit_candidate(
            job, raw, cited_cards, ledger, budget, cancelled,
            section_title="Final answer",
            section_question=job.snapshot.task,
            max_repairs=1, chapter_repair=True,
        )
        if cancelled():
            save_result(
                _clean_report_block("\n\n".join(row["digest"] for row in digest_rows)),
                "", "digest_fallback",
            )
            return "", ""
        summary_match = re.search(
            r"(?ims)^## Summary\s*\n+(.*?)(?=^##\s|\Z)", audited_raw,
        )
        conclusions_match = re.search(
            r"(?ims)^## Conclusions\s*\n+(.*?)(?=^##\s|\Z)", audited_raw,
        )
        audited_summary = _clean_report_block(summary_match.group(1) if summary_match else "")
        audited_conclusions = _clean_report_block(
            conclusions_match.group(1) if conclusions_match else "",
        )
        if audited_summary and bindings:
            job.synthesis_state["final_assembly_audit"] = "passed" if not issues else "repaired_or_partial"
            return save_result(audited_summary, audited_conclusions, "complete")

        # If the final prose fails audit, do not discard already-audited chapter
        # digests or manufacture a report-wide conclusion.
        job.synthesis_state["final_assembly_audit"] = "digest_fallback"
        return save_result(
            _clean_report_block("\n\n".join(row["digest"] for row in digest_rows)),
            "", "digest_fallback",
        )

    def _synthesise_chapters(
        self, job: BackgroundJob, cards: list[dict], budget: ThinkingBudget,
        cancelled: Callable[[], bool], *, chapter_ids: set[str] | None = None,
        finalize: bool = True,
    ) -> tuple[str, list[dict]]:
        self._review_source_documents(job, budget, cancelled)
        cards = report_card_ledger(job.fact_cards)
        # Incremental section saves enter here before the final run() audit.
        # Refresh checks before writing or reusing any cached chapter output.
        self._constraint_audit(job, cards)
        constraint_checks = self._constraint_prompt_context(job)
        selected_global, reasons = select_requirement_cards(
            cards, job.evidence_requirements, source_registry=job.source_registry,
        )
        conflict_keys = {_conflict_scope_key(card) for card in selected_global
                         if card.get("evidence_status") == "conflicted"}
        for card in cards:
            if card.get("evidence_status") == "conflicted" and _conflict_scope_key(card) in conflict_keys:
                if card["id"] not in {item["id"] for item in selected_global}:
                    selected_global.append(card)
                    reasons[card["id"]] = ["conflict: preserve both sides"]

        job.synthesis_state.update({
            "selection_reasons": reasons,
            "chapter_card_ids": {},
            "overflow_retries": int(job.synthesis_state.get("overflow_retries", 0)),
        })
        delivered_card_ids: set[str] = set()
        report_parts: list[str] = []
        incomplete_chapters = []
        chapter_cards_by_id = {}
        for chapter in job.chapter_plan:
            chapter_cards = self._chapter_cards_for_synthesis(job, chapter, selected_global)
            chapter_cards_by_id[chapter["id"]] = chapter_cards
            chapter["card_ids"] = sorted(card["id"] for card in chapter_cards)

        analysis_priority = {"high": 3, "medium": 2, "low": 1}

        def reserved_chapter_call_limit(stage: str, used: int, weight: int,
                                        total_weight: int, later_count: int) -> int:
            available = max(0, STAGE_CALL_LIMITS[stage] - used)
            if not available:
                return used
            quota = min(
                max(1, available * weight // max(total_weight, 1)),
                max(0, available - later_count),
            )
            return used + quota

        for chapter_index, chapter in enumerate(job.chapter_plan):
            if chapter_ids is not None and chapter["id"] not in chapter_ids:
                continue
            if cancelled():
                return "", []
            chapter_cards = chapter_cards_by_id.get(chapter["id"], [])
            later_chapters = [
                item for item in job.chapter_plan[chapter_index + 1:]
                if chapter_cards_by_id.get(item["id"])
                and (chapter_ids is None or item["id"] in chapter_ids)
            ]
            budget.state["remaining_chapter_reserve_seconds"] = (
                len(later_chapters) * CHAPTER_TIME_RESERVE_SECONDS
            )
            analysis_calls = budget.state["stages"].get("chapter_analysis", {}).get("calls", 0)
            current_weight = analysis_priority.get(chapter.get("priority"), 1)
            total_weight = current_weight + sum(
                analysis_priority.get(item.get("priority"), 1) for item in later_chapters
            )
            chapter_analysis_call_limit = reserved_chapter_call_limit(
                "chapter_analysis", analysis_calls, current_weight, total_weight,
                len(later_chapters),
            )
            extraction_calls = budget.state["stages"].get("card_extraction", {}).get("calls", 0)
            chapter_extraction_call_limit = reserved_chapter_call_limit(
                "card_extraction", extraction_calls, current_weight, total_weight,
                len(later_chapters),
            )
            job.synthesis_state["chapter_card_ids"][chapter["id"]] = [
                card["id"] for card in chapter_cards
            ]
            job.synthesis_state["selection_reasons"].update({
                card_id: reasons.get(card_id, [f"{chapter['id']}: linked chapter evidence"])
                for card_id in (card["id"] for card in chapter_cards)
            })
            if not chapter_cards:
                chapter.update(analysis_state="missing", body="", digest="")
                refresh_section_card(chapter, cards)
                incomplete_chapters.append(chapter["id"])
                continue

            requirements = [item for item in job.evidence_requirements
                            if item.get("chapter_id") == chapter["id"] and not item.get("is_subject_identity")]
            if not requirements:
                requirements = [item for item in job.evidence_requirements if item.get("chapter_id") == chapter["id"]]
            compact_facts, extraction_disagreements, extraction_missing = self._extract_chapter_cards(
                job, chapter, chapter_cards, budget, cancelled, chapter_extraction_call_limit,
            )
            if cancelled():
                return "", []
            chapter_questions = []
            for item in requirements:
                detail = str(item.get("detail") or "")
                missing_facts = (
                    detail.split(":", 1)[1].strip()[:400]
                    if detail.casefold().startswith("missing facts:") else ""
                )
                chapter_questions.append({
                    "id": item["identifier"], "question": item["question"],
                    "priority": item.get("importance"),
                    "completion_criteria": item.get("completion_criteria"),
                    "state": item.get("state"), "missing_facts": missing_facts,
                })
            questions = json.dumps(chapter_questions, ensure_ascii=True, separators=(",", ":"))
            contract_fingerprint = hashlib.sha256(json.dumps({
                "task": job.snapshot.task, "scope": _fixed_scope_for_prompt(job.research_context),
                "constraints": constraint_checks,
                "questions": [{key: item.get(key) for key in ("identifier", "question", "completion_criteria", "dependencies")}
                              for item in requirements],
            }, sort_keys=True, default=str).encode()).hexdigest()
            source_fingerprints = {}
            prior_group_ids = {identifier for group in (chapter.get("analysis_groups") or {}).values()
                               if group.get("state") == "complete" for identifier in group.get("card_ids", [])}
            fingerprint_cards = {card["id"]: card for card in cards if card["id"] in prior_group_ids}
            fingerprint_cards.update({card["id"]: card for card in chapter_cards})
            for card in fingerprint_cards.values():
                stable = {key: value for key, value in card.items() if key not in {
                    "chapter_ids", "requirement_ids", "finding", "source_ids", "legacy_ids",
                }}
                finding = card.get("finding") or {}
                stable["finding"] = {key: value for key, value in finding.items()
                                     if key not in {"question_ids", "section_ids", "gaps", "assessment"}}
                context_packet = self._source_packet(job, [card])
                source_fingerprints[card["id"]] = hashlib.sha256(json.dumps({
                    "card": stable, "context": context_packet.get("passages", []),
                    "blockers": context_packet.get("context_blockers", []),
                }, sort_keys=True, default=str).encode()).hexdigest()
            retained_groups = [group for group in (chapter.get("analysis_groups") or {}).values()
                               if group.get("state") == "complete"
                               and group.get("contract_fingerprint") == contract_fingerprint
                               and group.get("source_fingerprints")
                               and all(source_fingerprints.get(identifier) == value
                                       for identifier, value in group["source_fingerprints"].items())]
            fingerprint = hashlib.sha256(json.dumps({
                "chapter": chapter["id"], "cards": chapter_cards,
                "questions": questions, "scope": _fixed_scope_for_prompt(job.research_context),
                "compact_facts": compact_facts,
                "source_context": self._source_packet(job, chapter_cards).get("passages", []),
                **({"constraint_checks": constraint_checks} if constraint_checks else {}),
            }, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            if (chapter.get("analysis_fingerprint") == fingerprint
                    and chapter.get("analysis_state") in {"complete", "partial"}):
                body, digest = chapter.get("body", ""), chapter.get("digest", "")
                if body:
                    report_parts.append(f"## {chapter['title']}\n\n{body}")
                    delivered_card_ids.update(
                        value.strip() for match in _SOURCE_MARKER_RE.finditer(body + "\n" + digest)
                        for value in match[1].split(",")
                    )
                    continue
            elif chapter.get("analysis_fingerprint"):
                # Do not let an old accepted section leak into a partial retry
                # when its cards, obligations, scope, source version or conflict
                # ledger changed. Budgets remain charged; only this dependent
                # analysis result is reopened.
                chapter["invalidated_fingerprint"] = chapter["analysis_fingerprint"]
                chapter.update({
                    "analysis_state": "partial" if retained_groups else "stale",
                    "body": "\n\n".join(group.get("body", "") for group in retained_groups),
                    "digest": "\n\n".join(group.get("digest", "") for group in retained_groups),
                    "analyzed_card_ids": sorted({identifier for group in retained_groups for identifier in group.get("card_ids", [])}),
                })
                refresh_section_card(chapter, cards)
                job.synthesis_state["accepted_chapter_ids"] = [
                    identifier for identifier in job.synthesis_state.get("accepted_chapter_ids", [])
                    if identifier != chapter["id"]
                ]
                job.report_draft = "\n\n".join(f"## {item['title']}\n\n{item['body']}"
                                               for item in job.chapter_plan if item.get("body"))
                job.report_bindings = _extract_report_bindings(job.report_draft, cards)[1]
                if retained_groups:
                    job.synthesis_state["accepted_chapter_ids"].append(chapter["id"])
                if retained_groups:
                    # Persist still-valid independently checked work before a
                    # new packet or backend failure can interrupt replacement.
                    self.jobs.checkpoint(job)

            retained_ids = {identifier for group in retained_groups for identifier in group.get("card_ids", [])}
            chapter["card_ids"] = sorted(set(chapter.get("card_ids") or []) | retained_ids)
            groups = _indivisible_card_groups([card for card in chapter_cards if card["id"] not in retained_ids])
            analysis_chunks: list[list[dict]] = []
            pending_chunk: list[dict] = []

            def writer_prompt(
                chunk, correction="", *, chapter=chapter, requirements=requirements,
                compact_facts=compact_facts,
                extraction_disagreements=extraction_disagreements,
                extraction_missing=extraction_missing, questions=questions,
            ):
                packet = self._source_packet(
                    job, chunk,
                    requirement_ids=[item["identifier"] for item in requirements],
                    reasons=job.synthesis_state.get("selection_reasons", {}),
                )
                writer_view = _chapter_packet_from_cards(
                    packet, compact_facts, chapter, extraction_disagreements, extraction_missing,
                )
                prompt = get_thinking_chapter_analysis_prompt(
                    job.snapshot.task, _fixed_scope_for_prompt(job.research_context),
                    chapter["title"], questions, writer_view, correction,
                    constraint_checks=self._constraint_prompt_context(job),
                    compact=True,
                )
                return prompt, json.dumps(packet, ensure_ascii=True, separators=(",", ":")), packet

            def writer_schema(packet):
                aliases = list((packet.get("aliases") or {}).keys())
                support = {"type": "array", "minItems": 1,
                           "items": {"type": "string", "enum": aliases}}
                return {"type": "object", "properties": {
                    "passages": {"type": "array", "minItems": 1, "items": {
                        "type": "object", "properties": {"text": {"type": "string"}, "cards": support},
                        "required": ["text", "cards"],
                    }},
                    "digest": {"type": "string"}, "digest_cards": support,
                }, "required": ["passages", "digest", "digest_cards"]}

            oversized_ids = []
            source_units = []
            for group in groups:
                unit = self._queue_source_group(job, chapter, group, "chapter_analysis")
                source_units.append(unit)
                candidate = [*pending_chunk, *group]
                candidate_prompt, _, candidate_packet = writer_prompt(candidate)
                fit, reason = self._source_stage_fit(job, budget, "chapter_analysis", candidate_prompt,
                                                     candidate_packet, verification=True)
                if pending_chunk and not fit:
                    analysis_chunks.append(pending_chunk)
                    pending_chunk = []
                    candidate_prompt, _, candidate_packet = writer_prompt(group)
                    fit, reason = self._source_stage_fit(job, budget, "chapter_analysis", candidate_prompt,
                                                         candidate_packet, verification=True)
                if fit:
                    pending_chunk = [*pending_chunk, *group]
                else:
                    oversized_ids.extend(card["id"] for card in group)
                    self._queue_source_group(job, chapter, group, "chapter_analysis", reason=reason)
            if pending_chunk:
                analysis_chunks.append(pending_chunk)

            saved_groups = chapter.setdefault("analysis_groups", {})
            completed_groups = list(retained_groups)
            for chunk in analysis_chunks:
                if cancelled():
                    return "", []
                chunk_ids = [card["id"] for card in chunk]
                chunk_key = hashlib.sha256(json.dumps({
                    "cards": chunk, "questions": questions,
                    "compact_facts": {key: compact_facts.get(key) for key in chunk_ids},
                    "source_context": self._source_packet(job, chunk).get("passages", []),
                    **({"constraint_checks": constraint_checks} if constraint_checks else {}),
                }, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
                cached = saved_groups.get(chunk_key, {})
                prompt, ledger, packet = writer_prompt(chunk)
                if cached.get("state") == "complete":
                    completed_groups.append(cached)
                    continue
                if cached.get("state") == "failed":
                    continue
                if cached.get("state") == "draft":
                    canonical_body = cached.get("body", "")
                    canonical_digest = cached.get("digest", "")
                else:
                    if budget.state["stages"].get("chapter_analysis", {}).get("calls", 0) >= chapter_analysis_call_limit:
                        saved_groups[chunk_key] = {
                            "state": "failed", "card_ids": chunk_ids,
                            "reason": "chapter_call_allowance_reserved_for_later_chapters",
                        }
                        continue
                    self._update_stage(job, f"Analysing {chapter['title'][:70]}")
                    first_response = ""
                    try:
                        first_response = _budgeted_generate(
                            self, budget, "chapter_analysis",
                            system_prompt=prompt,
                            user_prompt=get_thinking_user_prompt("chapter_analysis"),
                            temperature=0.2, cancel_check=cancelled,
                            response_schema=writer_schema(packet),
                        ) or ""
                    except (PromptOverflow, ModelBudgetExhausted):
                        saved_groups[chunk_key] = {"state": "failed", "card_ids": chunk_ids,
                                                   "reason": "context_or_budget"}
                        continue
                    except Exception as exc:
                        self._require_backend()
                        logger.warning(
                            "Deep-think chapter writing failed chapter=%s error_type=%s",
                            chapter["id"], type(exc).__name__,
                        )
                        saved_groups[chunk_key] = {"state": "failed", "card_ids": chunk_ids,
                                                   "reason": type(exc).__name__}
                        continue
                    aliases = packet.get("aliases") or {}
                    parsed, output_error = _structured_chapter_output(
                        _parse_direct_json(first_response), aliases,
                    )
                    first_parsed = parsed
                    valid = not output_error
                    digest_omitted = False
                    if not valid:
                        _record_chapter_output_rejection(job, output_error, chunk_ids)
                    raw_value = _parse_direct_json(first_response)
                    body_only = _structured_chapter_body(raw_value, aliases)
                    if not valid and body_only and "digest" in output_error.casefold():
                        body_ids = sorted({
                            identifier.strip()
                            for match in _SOURCE_MARKER_RE.finditer(body_only)
                            for identifier in match[1].split(",")
                        })
                        digest_prompt = get_thinking_digest_repair_prompt(body_ids, body_only)
                        digest_response = ""
                        if budget.fits(
                            "chapter_analysis", system_prompt=digest_prompt,
                            user_prompt=get_thinking_user_prompt("digest_repair"), thinking_mode=False,
                        ):
                            try:
                                digest_response = _budgeted_generate(
                                    self, budget, "chapter_analysis", system_prompt=digest_prompt,
                                    request_kind="digest_retry",
                                    user_prompt=get_thinking_user_prompt("digest_repair"), temperature=0.0,
                                    cancel_check=cancelled,
                                ) or ""
                            except Exception as exc:
                                self._require_backend()
                                logger.warning("Deep-think digest-only repair failed: %s", type(exc).__name__)
                        digest_value = _parse_direct_json(digest_response)
                        digest, digest_error = _render_chapter_passage(
                            (digest_value or {}).get("digest"),
                            (digest_value or {}).get("digest_cards"), aliases,
                        )
                        if digest_error or len(_SOURCE_MARKER_RE.sub("", digest).split()) > 120:
                            digest = ""
                            digest_omitted = True
                        parsed = {"body": body_only, "digest": digest}
                        valid = True
                    if not valid and not cancelled():
                        saved_groups[chunk_key] = {"state": "repairing", "card_ids": chunk_ids,
                                                   "first_output": first_response, "repair_attempts": 1}
                        recovery_prompt, _, _ = writer_prompt(chunk, output_error)
                        if (budget.state["stages"].get("chapter_analysis", {}).get("calls", 0)
                                < chapter_analysis_call_limit and budget.fits(
                            "chapter_analysis", system_prompt=recovery_prompt,
                            user_prompt=get_thinking_user_prompt("chapter_output_repair"), thinking_mode=False,
                        )):
                            try:
                                second_response = _budgeted_generate(
                                    self, budget, "chapter_analysis",
                                    request_kind="format_retry",
                                    system_prompt=recovery_prompt,
                                    user_prompt=get_thinking_user_prompt("chapter_output_repair"),
                                    temperature=0.0, cancel_check=cancelled,
                                    response_schema=writer_schema(packet),
                                ) or ""
                            except (PromptOverflow, ModelBudgetExhausted):
                                second_response = ""
                            except Exception as exc:
                                self._require_backend()
                                logger.warning("Deep-think chapter output repair failed: %s", type(exc).__name__)
                                second_response = ""
                            parsed, output_error = _structured_chapter_output(
                                _parse_direct_json(second_response), aliases,
                            )
                            valid = not output_error
                            if not valid:
                                _record_chapter_output_rejection(job, output_error, chunk_ids)
                    if not valid:
                        # A too-long digest is a summary-contract failure, not
                        # grounds to discard the independently auditable body.
                        # Keep the repair allowance above, then audit the body
                        # unchanged with no digest if that was the sole defect.
                        candidates = (parsed, first_parsed)
                        for candidate in candidates:
                            if (_chapter_output_error(candidate).startswith(
                                    THINKING_DIGEST_LIMIT_ERROR)
                                    and not _chapter_body_output_error(candidate)):
                                parsed = {**candidate, "digest": ""}
                                output_error = ""
                                valid = True
                                digest_omitted = True
                                logger.warning(
                                    "Deep-think chapter %s retained its body after digest repair failed",
                                    chapter["id"],
                                )
                                break
                    if not valid:
                        logger.warning(
                            "Deep-think chapter output rejected chapter=%s reason=%s",
                            chapter["id"], _chapter_output_reason(output_error),
                        )
                        saved_groups[chunk_key] = {"state": "failed", "card_ids": chunk_ids,
                                                   "reason": "empty_or_invalid_output", "detail": output_error,
                                                   "repair_attempts": 1}
                        continue
                    canonical_body = _canonical_markers(_clean_report_block(parsed["body"]), aliases)
                    canonical_digest = _canonical_markers(_clean_report_block(parsed["digest"]), aliases)
                    saved_groups[chunk_key] = {
                        "state": "draft", "card_ids": chunk_ids,
                        "body": canonical_body, "digest": canonical_digest,
                        "digest_omitted": digest_omitted,
                        "aliases": aliases, "packet_fingerprint": hashlib.sha256(ledger.encode()).hexdigest(),
                    }
                    cached = saved_groups[chunk_key]
                if cancelled():
                    return "", []
                combined = (
                    "## Chapter body\n\n" + canonical_body
                    + "\n\n## Digest\n\n" + canonical_digest
                )
                audited, bindings, issues, audited_raw = self._audit_candidate(
                    job, combined, chunk, ledger, budget, cancelled,
                    section_title=chapter["title"], section_question="; ".join(
                        item["question"] for item in requirements
                    ), max_repairs=1, chapter_repair=True,
                )
                if cancelled():
                    return "", []
                body_match = re.search(
                    r"(?ims)^## Chapter body\s*\n+(.*?)(?=^## Digest\s*$|\Z)", audited_raw,
                )
                digest_match = re.search(r"(?ims)^## Digest\s*\n+(.*?)(?=^##\s|\Z)", audited_raw)
                audited_body = body_match.group(1).strip() if body_match else ""
                audited_digest = digest_match.group(1).strip() if digest_match else ""
                if audited_body and bindings:
                    completed = {
                        "state": "complete", "card_ids": chunk_ids,
                        "body": audited_body, "digest": audited_digest,
                        "digest_omitted": bool(cached.get("digest_omitted")),
                        "issues": issues, "bindings": bindings,
                        "contract_fingerprint": contract_fingerprint,
                        "source_fingerprints": {identifier: source_fingerprints[identifier] for identifier in chunk_ids},
                    }
                    saved_groups[chunk_key] = completed
                    completed_groups.append(completed)
                else:
                    saved_groups[chunk_key] = {
                        "state": "failed", "card_ids": chunk_ids,
                        "reason": "no_verified_chapter_body", "issues": issues,
                    }

            if cancelled():
                return "", []
            completed_ids = {identifier for group in completed_groups for identifier in group.get("card_ids", [])}
            for unit in source_units:
                if set(unit["references"]) <= completed_ids:
                    job.transition_work(unit["id"], "complete")
            if not completed_groups:
                chapter.update(analysis_state="missing", body="", digest="",
                               oversized_card_ids=oversized_ids)
                refresh_section_card(chapter, cards)
                incomplete_chapters.append(chapter["id"])
                continue

            body, digest = "", ""
            merge_input = (
                {"groups": completed_groups, "constraint_checks": constraint_checks}
                if constraint_checks else [item.get("card_ids") for item in completed_groups]
            )
            merge_fingerprint = hashlib.sha256(json.dumps(
                merge_input, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            cached_merge = chapter.get("chapter_merge") or {}
            if len(completed_groups) == 1:
                body = completed_groups[0].get("body", "")
                digest = completed_groups[0].get("digest", "")
            elif (cached_merge.get("fingerprint") == merge_fingerprint
                  and cached_merge.get("state") in {"complete", "deterministic_merge"}):
                body, digest = cached_merge.get("body", ""), cached_merge.get("digest", "")
            else:
                used_ids = {
                    value.strip() for group in completed_groups
                    for text in (group.get("body", ""), group.get("digest", ""))
                    for match in _SOURCE_MARKER_RE.finditer(text)
                    for value in match[1].split(",")
                }
                merge_cards = [card for card in chapter_cards if card["id"] in used_ids]
                merge_conflicts = {_conflict_scope_key(card) for card in merge_cards
                                   if card.get("evidence_status") == "conflicted"}
                for card in chapter_cards:
                    if (card.get("evidence_status") == "conflicted"
                            and _conflict_scope_key(card) in merge_conflicts
                            and card["id"] not in {item["id"] for item in merge_cards}):
                        merge_cards.append(card)
                merged_success = False
                if merge_cards:
                    merge_packet = self._source_packet(
                        job, merge_cards,
                        requirement_ids=[item["identifier"] for item in requirements],
                    )
                    merge_aliases = merge_packet.get("aliases") or {}
                    merge_view = _chapter_packet_from_cards(
                        merge_packet, compact_facts, chapter,
                        extraction_disagreements, extraction_missing,
                    )
                    group_digests = json.dumps([{
                        "group": index,
                        "body": _render_canonical_markers(item.get("body", ""), merge_aliases),
                        "digest": _render_canonical_markers(
                            item.get("digest") or item.get("body", ""), merge_aliases,
                        ),
                    } for index, item in enumerate(completed_groups, 1)],
                        ensure_ascii=True, separators=(",", ":"))
                    merge_prompt = get_thinking_chapter_merge_prompt(
                        job.snapshot.task, _fixed_scope_for_prompt(job.research_context),
                        chapter["title"], questions, group_digests, merge_view,
                        constraint_checks=self._constraint_prompt_context(job),
                    )
                    if (budget.state["stages"].get("chapter_analysis", {}).get("calls", 0)
                            < chapter_analysis_call_limit and budget.fits(
                        "chapter_analysis", system_prompt=merge_prompt,
                        user_prompt=get_thinking_user_prompt("chapter_merge"), thinking_mode=False,
                    )):
                        try:
                            merged_response = _budgeted_generate(
                                self, budget, "chapter_analysis",
                                system_prompt=merge_prompt,
                                user_prompt=get_thinking_user_prompt("chapter_merge"),
                                temperature=0.2, cancel_check=cancelled,
                            ) or ""
                        except (PromptOverflow, ModelBudgetExhausted):
                            merged_response = ""
                        except Exception as exc:
                            self._require_backend()
                            logger.warning("Deep-think chapter merge failed: %s", type(exc).__name__)
                            merged_response = ""
                        parsed_merge, merge_error = _structured_chapter_output(
                            _parse_direct_json(merged_response), merge_aliases,
                        )
                        valid_merge = not merge_error
                        merge_repair_prompt = get_thinking_output_repair_prompt(merge_prompt, merge_error)
                        if (not valid_merge and not cancelled()
                                and budget.state["stages"].get("chapter_analysis", {}).get("calls", 0)
                                < chapter_analysis_call_limit and budget.fits(
                            "chapter_analysis", system_prompt=merge_repair_prompt,
                            user_prompt=get_thinking_user_prompt("chapter_merge_repair"), thinking_mode=False,
                        )):
                            try:
                                retry = _budgeted_generate(
                                    self, budget, "chapter_analysis",
                                    request_kind="format_retry",
                                    system_prompt=merge_repair_prompt,
                                    user_prompt=get_thinking_user_prompt("chapter_merge_repair"),
                                    temperature=0.0, cancel_check=cancelled,
                                ) or ""
                            except Exception:
                                self._require_backend()
                                retry = ""
                            parsed_merge, merge_error = _structured_chapter_output(
                                _parse_direct_json(retry), merge_aliases,
                            )
                            valid_merge = not merge_error
                        if valid_merge and not cancelled():
                            raw_merge = (
                                "## Chapter body\n\n" + _canonical_markers(parsed_merge["body"], merge_aliases)
                                + "\n\n## Digest\n\n" + _canonical_markers(parsed_merge["digest"], merge_aliases)
                            )
                            merge_ledger = json.dumps(merge_packet, ensure_ascii=True, separators=(",", ":"))
                            audited, merge_bindings, issues, audited_raw = self._audit_candidate(
                                job, raw_merge, merge_cards, merge_ledger, budget, cancelled,
                                section_title=chapter["title"], section_question="; ".join(
                                    item["question"] for item in requirements
                                ), max_repairs=1, chapter_repair=True,
                            )
                            body_match = re.search(
                                r"(?ims)^## Chapter body\s*\n+(.*?)(?=^## Digest\s*$|\Z)", audited_raw,
                            )
                            digest_match = re.search(
                                r"(?ims)^## Digest\s*\n+(.*?)(?=^##\s|\Z)", audited_raw,
                            )
                            body = body_match.group(1).strip() if body_match else ""
                            digest = digest_match.group(1).strip() if digest_match else ""
                            merged_success = bool(body and merge_bindings)
                            chapter["chapter_merge"] = {
                                "fingerprint": merge_fingerprint,
                                "state": "complete" if merged_success else "failed",
                                "body": body, "digest": digest, "issues": issues,
                            }
                if not merged_success:
                    # Every component was already audited against original cards;
                    # a deterministic merge cannot introduce new factual prose.
                    unique_bodies = list(dict.fromkeys(
                        item.get("body", "").strip() for item in completed_groups
                        if item.get("body", "").strip()
                    ))
                    unique_digests = list(dict.fromkeys(
                        (item.get("digest") or item.get("body", "")).strip()
                        for item in completed_groups
                        if (item.get("digest") or item.get("body", "")).strip()
                    ))
                    body = "\n\n".join(unique_bodies)
                    digest = "\n\n".join(unique_digests)
                    chapter["chapter_merge"] = {
                        "fingerprint": merge_fingerprint, "state": "deterministic_merge",
                        "body": body, "digest": digest,
                    }

            chapter["body"] = body
            chapter["digest"] = digest
            chapter["analysis_fingerprint"] = fingerprint
            chapter["analysis_state"] = "partial" if oversized_ids or len(completed_groups) < len(analysis_chunks) + len(retained_groups) else "complete"
            chapter["oversized_card_ids"] = oversized_ids
            chapter["analyzed_card_ids"] = sorted({
                identifier for group in completed_groups for identifier in group.get("card_ids", [])
            })
            if any(group.get("digest_omitted") for group in completed_groups):
                job.synthesis_state.setdefault("digest_omitted_chapters", []).append(chapter["id"])
            chapter["state"] = "analyzed" if finalize or all(item.get("state") == "resolved" for item in requirements) else "researching"
            refresh_section_card(chapter, cards)
            _refresh_active_manifest(job)
            if body:
                report_parts.append(f"## {chapter['title']}\n\n{body}")
                delivered_card_ids.update(
                    value.strip() for match in _SOURCE_MARKER_RE.finditer(body + "\n" + digest)
                    for value in match[1].split(",")
                )
            if chapter["analysis_state"] == "partial":
                incomplete_chapters.append(chapter["id"])
            # Checkpoint only material that has passed passage audit. This gives
            # the job manager enough verified work to export if a later chapter
            # or final assembly fails.
            accepted_chapters = [
                f"## {item['title']}\n\n{item['body']}"
                for item in job.chapter_plan if item.get("body")
            ]
            job.report_draft = "\n\n".join(accepted_chapters)
            job.report_bindings = _extract_report_bindings(job.report_draft, cards)[1]
            job.synthesis_state["accepted_chapter_ids"] = [
                item["id"] for item in job.chapter_plan if item.get("body")
            ]
            job.synthesis_state["partial_report"] = True
            self.jobs.checkpoint(job)

        if cancelled():
            return "", []
        if not finalize:
            accepted_chapters = [
                f"## {item['title']}\n\n{item['body']}"
                for item in job.chapter_plan if item.get("body")
            ]
            job.report_draft = "\n\n".join(accepted_chapters)
            job.report_bindings = _extract_report_bindings(job.report_draft, cards)[1]
            job.synthesis_state["accepted_chapter_ids"] = [
                item["id"] for item in job.chapter_plan if item.get("body")
            ]
            job.synthesis_state["partial_report"] = True
            return job.report_draft, job.report_bindings
        budget.state["remaining_chapter_reserve_seconds"] = 0
        # A good timing chapter must not hide all retrieved participant/form
        # evidence just because those chapters failed writing. Keep scoped,
        # whole extracts, clearly distinguished from completed analysis.
        if any(chapter.get("body") for chapter in job.chapter_plan):
            for chapter in job.chapter_plan:
                if chapter.get("body"):
                    continue
                extracts = chapter_excerpts(chapter_cards_by_id.get(chapter["id"], []))
                if extracts:
                    chapter.update(body=extracts, digest="", analysis_state="evidence_only")
                    refresh_section_card(chapter, cards)
        summary, conclusions = self._assemble_final_sections(job, cards, budget, cancelled)
        if cancelled():
            return "", []
        # A summary intentionally overlaps its chapters. Deduplicating against
        # it used to erase the only substantive body from a partial report.
        summary = _clean_report_block(summary)
        conclusions = _clean_report_block(conclusions)
        if not summary:
            safe_chapter_material = "\n\n".join(
                str(chapter.get("digest") or chapter.get("body") or "")
                for chapter in job.chapter_plan
            )
            summary = _clean_report_block(safe_chapter_material)
            job.synthesis_state["final_assembly_state"] = "chapter_fallback"
        # Count only rendered material, not hidden digests or removed paragraphs.
        delivered_card_ids = {
            value.strip() for text in (summary, conclusions)
            for match in _SOURCE_MARKER_RE.finditer(text)
            for value in match[1].split(",")
        }
        analyzed_delivered_ids = set(delivered_card_ids)
        extractive_ids = set()
        report_parts = []
        rendered_chapters = []
        title_counts: dict[str, int] = {}
        for chapter in job.chapter_plan:
            body = _clean_report_block(chapter.get("body", ""))
            if not body:
                continue
            body_ids = {value.strip() for match in _SOURCE_MARKER_RE.finditer(body)
                        for value in match[1].split(",")}
            delivered_card_ids.update(body_ids)
            if chapter.get("analysis_state") == "evidence_only":
                extractive_ids.update(body_ids)
            else:
                analyzed_delivered_ids.update(body_ids)
            title = str(chapter.get("title") or "Research findings").strip() or "Research findings"
            title_counts[title.casefold()] = title_counts.get(title.casefold(), 0) + 1
            if title_counts[title.casefold()] > 1:
                title += f" ({title_counts[title.casefold()]})"
            report_parts.append(f"## {title}\n\n{body}")
            rendered_chapters.append(chapter)
        # A resolved, high-priority answer whose supporting card is delivered
        # must not be denied by the summary. Catch the common denial phrasings
        # deterministically and replace them with the reconciled finding.
        summary = _reconcile_summary_with_findings(
            summary, job.evidence_requirements, cards, analyzed_delivered_ids,
        )
        essential_gaps = [
            item for item in job.evidence_requirements
            if item.get("importance") == "high" and (
                item.get("state") != "resolved"
                or not item.get("evidence_ids")
                or not (set(item.get("evidence_ids") or []) & analyzed_delivered_ids)
            )
        ]
        selected_ids = {card_id for chapter in job.chapter_plan
                        for card_id in chapter.get("card_ids", [])}
        report = f"## Summary\n\n{summary}" if summary else ""
        if report_parts:
            report += ("\n\n" if report else "") + "\n\n".join(report_parts)
        if conclusions:
            report += ("\n\n" if report else "") + f"## Conclusions\n\n{conclusions}"
        job.synthesis_state.update({
            "selected_card_ids": sorted(selected_ids),
            "omitted_card_ids": sorted(card["id"] for card in cards if card["id"] not in selected_ids),
            "unsynthesized_card_ids": sorted(selected_ids - analyzed_delivered_ids),
            "extractive_only_card_ids": sorted(extractive_ids - analyzed_delivered_ids),
            "partial_report": bool(essential_gaps or incomplete_chapters),
            "essential_unresolved_requirement_ids": [item["identifier"] for item in essential_gaps],
            "incomplete_chapters": incomplete_chapters,
            "sections": [{"id": chapter["id"], "title": chapter["title"],
                          "analysis_state": chapter.get("analysis_state"),
                          "card_ids": chapter.get("analyzed_card_ids", []),
                          "requirement_ids": [question["requirement_id"]
                                              for question in chapter.get("questions", [])]}
                         for chapter in rendered_chapters],
            "evidence_counts": {
                "ineligible": len(job.fact_cards) - len(cards),
                "unselected": len(cards) - len(selected_ids),
                "unsynthesized": len(selected_ids - analyzed_delivered_ids),
                "display_omitted": 0,
            },
        })
        job.synthesis_state["synthesis_packet_tokens"] = sum(
            estimate_tokens(json.dumps(chapter.get("compact_card_facts", {}), ensure_ascii=True))
            for chapter in job.chapter_plan
        )

        if not report.strip():
            fallback = _fallback_thinking_report(job.snapshot.task, job.evidence, "", cards=cards, job=job)
            bindings = _fallback_report_bindings(job, cards)
            job.report_bindings = bindings
            _record_synthesis_omissions(job)
            return fallback, bindings
        clean, bindings = _extract_report_bindings(report, cards)
        job.report_bindings = bindings
        _record_synthesis_omissions(job)
        return clean.strip(), bindings

    def _synthesise(
        self, job: BackgroundJob, cards: list[dict], budget: ThinkingBudget,
        cancelled: Callable[[], bool], findings: str,
    ) -> tuple[str, list[dict]]:
        """One production chapter path; pre-plan saved reports retain compatibility."""
        if job.chapter_plan:
            return self._synthesise_chapters(job, cards, budget, cancelled)
        return self._synthesise_legacy(job, cards, budget, cancelled, findings)

    def _synthesise_legacy(
        self, job: BackgroundJob, cards: list[dict], budget: ThinkingBudget,
        cancelled: Callable[[], bool], findings: str,
    ) -> tuple[str, list[dict]]:
        """Offline/replay compatibility for historical evidence without chapter plans.

        Live run() establishes a plan before retrieval. This writer has no tool
        dispatch and is not an alternative production research loop.
        """
        selected, reasons = select_requirement_cards(cards, job.evidence_requirements,
                                                     source_registry=job.source_registry)
        conflict_keys = {
            _conflict_scope_key(card) for card in selected
            if card.get("evidence_status") == "conflicted"
        }
        for peer in cards:
            if (peer.get("evidence_status") == "conflicted"
                    and _conflict_scope_key(peer) in conflict_keys
                    and peer["id"] not in {card["id"] for card in selected}):
                selected.append(peer)
                reasons[peer["id"]] = ["conflict: preserve both sides"]
        selected_ids = {card["id"] for card in selected}
        job.synthesis_state.update({
            "selected_card_ids": sorted(selected_ids),
            "omitted_card_ids": sorted(card["id"] for card in cards if card["id"] not in selected_ids),
            "selection_reasons": reasons,
            "overflow_retries": int(job.synthesis_state.get("overflow_retries", 0)),
            "unselected_reasons": {card["id"]: "no_substantive_requirement_match" for card in cards if card["id"] not in selected_ids},
        })
        eligible_requirements = [
            str(item["identifier"]) for item in job.evidence_requirements
            if item.get("state") not in {"unavailable", "exhausted"}
        ]
        prompt, ledger = self._packet_prompt(job, selected, eligible_requirements)
        job.synthesis_state["synthesis_packet_tokens"] = estimate_tokens(ledger)
        if not job.evidence:
            prompt = get_thinking_report_prompt(
                job.snapshot.task, findings, ledger,
                constraint_checks=self._constraint_prompt_context(job),
            )
        raw_report = ""
        can_single_pass = budget.fits(
            "synthesis", system_prompt=prompt, user_prompt=get_thinking_user_prompt("synthesis"), thinking_mode=True,
        )
        if not can_single_pass:
            job.synthesis_state["overflow_retries"] += 1
        if can_single_pass:
            try:
                raw_report = _budgeted_generate(
                    self, budget, "synthesis", user_prompt=get_thinking_user_prompt("synthesis"),
                    system_prompt=prompt, max_new_tokens=8192, temperature=0.4,
                    cancel_check=cancelled, thinking_mode=True,
                ) or ""
            except PromptOverflow:
                job.synthesis_state["overflow_retries"] += 1
            except ModelBudgetExhausted:
                job.synthesis_state.setdefault("stages_skipped", []).append("single_pass_call_budget")
            except Exception as exc:
                self._require_backend()
                logger.warning(
                    "Deep-think single-pass synthesis failed; retaining evidence fallback error_type=%s",
                    type(exc).__name__,
                )
                job.synthesis_state.setdefault("stages_skipped", []).append("single_pass_generation_failed")
                fallback = _fallback_thinking_report(
                    job.snapshot.task, job.evidence, findings, cards=selected, job=job,
                )
                bindings = _fallback_report_bindings(job, selected)
                job.report_bindings = bindings
                _record_synthesis_omissions(job)
                return fallback, bindings
        if cancelled():
            return "", []

        if raw_report.strip():
            final, bindings, issues, _audited_raw = self._audit_candidate(
                job, raw_report, selected, ledger, budget, cancelled
            )
            job.report_draft = raw_report
            if final and bindings:
                job.report_bindings = bindings
                if issues:
                    job.report_revision = final
            job.report_audit.update({"issues": issues, "revised": bool(issues)})
            _record_synthesis_omissions(job)
            return final, bindings

        # A non-fitting whole prompt, or an actual server overflow, moves to
        # requirement-driven complete-card sections. No identical retry occurs.
        sections_by_id: dict[str, dict] = job.synthesis_state.setdefault("section_results", {})
        job.synthesis_state["oversized_card_ids"] = []
        job.synthesis_state["omitted_section_card_ids"] = []
        omission_reasons = job.synthesis_state["omission_reasons"] = {}
        section_specs = []
        assigned_ids = set()
        # Specific questions own their evidence first. Broad answer/current
        # requirements must not synthesize the same cards again in separate
        # requests. Cross-section reasoning can expand original evidence later.
        requirements = sorted(job.evidence_requirements, key=lambda item: (
            item["identifier"] == "answer", item["identifier"] == "current-context",
            not bool(item.get("dimension")),
        ))
        for requirement in requirements:
            req_id = str(requirement["identifier"])
            group_cards = [
                card for card in selected
                if card["id"] not in assigned_ids
                if any(str(reason).partition(":")[0] == req_id
                       for reason in reasons.get(card["id"], []))
            ]
            if not group_cards:
                continue
            # Conflicting peers are always co-located even when only one side
            # matched a lexical coverage term.
            identities = {_conflict_scope_key(card) for card in group_cards}
            for peer in selected:
                if (peer.get("evidence_status") == "conflicted"
                        and _conflict_scope_key(peer) in identities
                        and peer["id"] not in assigned_ids
                        and peer not in group_cards):
                    group_cards.append(peer)
            section_specs.append((req_id, str(requirement.get("question") or req_id), group_cards))
            assigned_ids.update(card["id"] for card in group_cards)
        if not section_specs and selected:
            section_specs.append(("evidence", "Summarise the selected relevant evidence.", selected))

        raw_sections = []
        audited_sections: list[dict] = []
        all_bindings: dict[str, dict] = {}
        section_generation_failed = False
        for req_id, question, group_cards in section_specs:
            title = next((str(item.get("dimension") or item.get("identifier"))
                          for item in job.evidence_requirements if item.get("identifier") == req_id), req_id)
            chunks: list[list[dict]] = []
            current: list[dict] = []
            for card_group in _indivisible_card_groups(group_cards):
                candidate = [*current, *card_group]
                candidate_prompt, _ = self._packet_prompt(
                    job, candidate, [req_id], section_title=title, section_question=question,
                )
                if current and not budget.fits(
                    "section", system_prompt=candidate_prompt,
                    user_prompt=get_thinking_user_prompt("section", section_title=title), thinking_mode=True,
                ):
                    chunks.append(current)
                    current = card_group
                else:
                    current = candidate
                if current == card_group:
                    one_prompt, _ = self._packet_prompt(
                        job, current, [req_id], section_title=title, section_question=question,
                    )
                    if not budget.fits("section", system_prompt=one_prompt,
                                        user_prompt=get_thinking_user_prompt("section", section_title=title), thinking_mode=True):
                        current = []
                        job.synthesis_state.setdefault("oversized_card_ids", []).extend(card["id"] for card in card_group)
                        omission_reasons.update({card["id"]: "context_limit" for card in card_group})
            if current:
                chunks.append(current)
            for chunk_index, initial_chunk in enumerate(chunks, 1):
                pending_chunks = [(initial_chunk, str(chunk_index), 0)]
                while pending_chunks:
                    chunk, chunk_suffix, retry_depth = pending_chunks.pop(0)
                    packet_requirements = [
                        item["identifier"] for item in job.evidence_requirements
                        if any(str(reason).partition(":")[0] == item["identifier"]
                               for card in chunk for reason in reasons.get(card["id"], []))
                    ]
                    section_id = f"{req_id}:{chunk_suffix}"
                    section_prompt, section_ledger = self._packet_prompt(
                        job, chunk, [req_id], section_title=title, section_question=question,
                    )
                    fingerprint = hashlib.sha256(
                        (job.snapshot.task + section_ledger + title + question
                         + self._constraint_prompt_context(job)).encode()
                    ).hexdigest()
                    job.enqueue_work(
                        section_id, "section_write", references=[card["id"] for card in chunk],
                        requirement_ids=packet_requirements or [req_id], fingerprint=fingerprint,
                    )
                    cached = sections_by_id.get(section_id, {})
                    if cached.get("fingerprint") == fingerprint and cached.get("raw") is not None:
                        raw = cached.get("raw", "")
                        already_audited = bool(cached.get("audited"))
                        if already_audited:
                            clean = cached.get("report", "")
                            section_bindings = cached.get("bindings", [])
                            issues = cached.get("issues", [])
                    else:
                        if job.synthesis_state.get("empty_recovery", {}).get("consecutive_empty", 0) >= 2:
                            omission_reasons.update({card["id"]: "empty_generation" for card in chunk})
                            continue
                        try:
                            self._update_stage(job, f"Writing research section {len(audited_sections) + 1}")
                            job.transition_work(section_id, "running")
                            raw = _budgeted_generate(
                                self, budget, "section", user_prompt=get_thinking_user_prompt("section", section_title=title),
                                system_prompt=section_prompt, max_new_tokens=4096, temperature=0.4,
                                cancel_check=cancelled, thinking_mode=True,
                            ) or ""
                            if not raw.strip() and not cancelled():
                                raw = _recover_empty_section(
                                    self, job, budget, fingerprint, section_prompt,
                                    get_thinking_user_prompt("section", section_title=title), cancelled,
                                )
                        except PromptOverflow:
                            job.synthesis_state["overflow_retries"] += 1
                            job.transition_work(section_id, "packet_full", reason="server_context_limit")
                            indivisible = _indivisible_card_groups(chunk)
                            if len(indivisible) > 1 and retry_depth < 3:
                                midpoint = len(indivisible) // 2
                                split_chunks = [
                                    ([card for group in indivisible[:midpoint] for card in group], f"{chunk_suffix}.1"),
                                    ([card for group in indivisible[midpoint:] for card in group], f"{chunk_suffix}.2"),
                                ]
                                for child_cards, child_suffix in split_chunks:
                                    child_id = f"{req_id}:{child_suffix}"
                                    job.enqueue_work(
                                        child_id, "section_write",
                                        references=[card["id"] for card in child_cards],
                                        requirement_ids=packet_requirements or [req_id],
                                    )
                                pending_chunks[0:0] = [
                                    (child_cards, child_suffix, retry_depth + 1)
                                    for child_cards, child_suffix in split_chunks
                                ]
                            else:
                                job.transition_work(section_id, "blocked", reason="indivisible_packet_exceeds_model_context")
                                job.synthesis_state.setdefault("oversized_section_ids", []).append(section_id)
                                job.synthesis_state.setdefault("omitted_section_card_ids", []).extend(
                                    card["id"] for card in chunk
                                )
                                omission_reasons.update({card["id"]: "server_context_limit" for card in chunk})
                            continue
                        except ModelBudgetExhausted as exc:
                            job.transition_work(section_id, "budget_limited", reason="generation_budget_exhausted")
                            job.synthesis_state.setdefault("omitted_section_card_ids", []).extend(
                                card["id"] for card in chunk
                            )
                            job.synthesis_state.setdefault("stages_skipped", []).append("section_call_budget")
                            reason = "time_budget" if "elapsed-time" in str(exc) else "model_call_or_token_budget"
                            omission_reasons.update({card["id"]: reason for card in chunk})
                            continue
                        except Exception as exc:
                            self._require_backend()
                            job.transition_work(section_id, "blocked", reason="generation_failed")
                            logger.warning(
                                "Deep-think section generation failed section=%s error_type=%s",
                                section_id, type(exc).__name__,
                            )
                            section_generation_failed = True
                            job.synthesis_state.setdefault("stages_skipped", []).append("empty_or_failed_section")
                            omission_reasons.update({card["id"]: "generation_failed" for card in chunk})
                            continue
                        if not raw or not raw.strip():
                            job.transition_work(section_id, "blocked", reason="empty_generation")
                            section_generation_failed = True
                            job.synthesis_state.setdefault("stages_skipped", []).append("empty_section_output")
                            omission_reasons.update({card["id"]: "empty_generation" for card in chunk})
                            continue
                        # Checkpoint exact packet membership before audit so a
                        # cancellation/resume reuses completed section work.
                        sections_by_id[section_id] = {
                            "fingerprint": fingerprint, "card_ids": [card["id"] for card in chunk],
                            "requirement_ids": packet_requirements or [req_id], "raw": raw,
                            "packet_tokens": estimate_tokens(section_ledger),
                            "aliases": json.loads(section_ledger).get("aliases", {}),
                        }
                        cached = sections_by_id[section_id]
                        already_audited = False
                    if cancelled():
                        return "", []
                    if not already_audited:
                        self._update_stage(job, f"Checking research section {len(audited_sections) + 1}")
                        clean, section_bindings, issues, audited_raw = self._audit_candidate(
                            job, raw, chunk, section_ledger, budget, cancelled,
                            section_title=title, section_question=question,
                        )
                        if cancelled():
                            return "", []
                        cached = sections_by_id[section_id]
                        cached.update({"report": clean, "bindings": section_bindings,
                                       "issues": issues, "raw": audited_raw, "audited": True})
                    raw = cached.get("raw", raw)
                    if clean and section_bindings:
                        job.transition_work(section_id, "complete")
                        _refresh_active_manifest(job)
                        self.jobs.checkpoint(job)
                        raw_sections.append((title, raw))
                        audited_sections.append({"id": section_id, "title": title,
                                                  "question": question, "cards": chunk,
                                                  "requirement_ids": packet_requirements or [req_id],
                                                 "raw": raw, "report": clean, "bindings": section_bindings})
                        for binding in section_bindings:
                            all_bindings[binding["card_id"]] = binding
                    else:
                        job.transition_work(section_id, "needs_evidence", reason="no_verified_section")
                        omission_reasons.update({card["id"]: "no_verified_section" for card in chunk})

        job.synthesis_state["sections"] = [
            {"id": section["id"], "card_ids": [card["id"] for card in section["cards"]],
             "requirement_ids": section["requirement_ids"],
             "input_tokens": sections_by_id.get(section["id"], {}).get("packet_tokens", 0)}
            for section in audited_sections
        ]
        if audited_sections:
            job.synthesis_state["synthesis_packet_tokens"] = sum(
                int(sections_by_id.get(section["id"], {}).get("packet_tokens", 0))
                for section in audited_sections
            )
        synthesized_ids = set(all_bindings)
        unsafe_omissions = selected_ids - synthesized_ids
        job.synthesis_state["unsynthesized_card_ids"] = sorted(unsafe_omissions)
        job.synthesis_state["partial_report"] = bool(unsafe_omissions)
        missing_requirements = {
            str(reason).partition(":")[0] for card_id in unsafe_omissions
            for reason in reasons.get(card_id, [])
        }
        job.synthesis_state["unsynthesized_requirement_ids"] = sorted(missing_requirements)
        if unsafe_omissions and not audited_sections and all(
            omission_reasons.get(identifier) in {"context_limit", "server_context_limit"}
            for identifier in unsafe_omissions
        ):
            raise ReportSynthesisError(
                "Required evidence cards could not fit any safe synthesis packet: "
                + ", ".join(sorted(unsafe_omissions)[:20])
                + ". Retained records were not truncated; use a larger-context model or narrower request."
            )
        if not audited_sections:
            if section_generation_failed or unsafe_omissions:
                fallback = _fallback_thinking_report(job.snapshot.task, job.evidence, findings, cards=selected, job=job)
                job.synthesis_state["limited_evidence_fallback"] = True
                bindings = _fallback_report_bindings(job, selected)
                job.report_bindings = bindings
                _record_synthesis_omissions(job)
                return fallback, bindings
            raise ReportSynthesisError(
                "No safe evidence packet fits this model's context. Reduce the source/record size or use a larger-context model; retained evidence was not truncated."
            )

        # Bounded cross-section reasoning sees findings plus the original cited
        # cards, never a replacement source made from an earlier model summary.
        cited_ids = set(all_bindings)
        cited_cards = [card for card in selected if card["id"] in cited_ids]
        unresolved_requirements = [item for item in job.evidence_requirements
                                   if item.get("state") != "resolved"]
        cross_packet = serialize_synthesis_packet(cited_cards, job.source_registry)
        cross_input = {
            "sections": [{"id": section["id"], "title": section["title"],
                          "findings": section["raw"],
                          "original_card_ids": [card["id"] for card in section["cards"]]}
                         for section in audited_sections],
            "original_evidence": cross_packet,
            "unresolved_requirements": [
                {"id": item["identifier"], "question": item["question"],
                 "detail": item.get("detail"), "state": item.get("state")}
                for item in job.evidence_requirements if item.get("state") != "resolved"
            ],
        }
        reconcile_prompt = get_thinking_reconciliation_prompt(
            json.dumps(cross_input, ensure_ascii=True, separators=(",", ":")),
        )
        reconciliation = ""
        if not unsafe_omissions and budget.fits("reconciliation", user_prompt=get_thinking_user_prompt("reconciliation"),
                       system_prompt=reconcile_prompt, thinking_mode=False):
            try:
                reconciliation = _budgeted_generate(
                    self, budget, "reconciliation", user_prompt=get_thinking_user_prompt("reconciliation"),
                    system_prompt=reconcile_prompt, max_new_tokens=2048, temperature=0.0,
                    cancel_check=cancelled, thinking_mode=False,
                ) or ""
            except PromptOverflow:
                # Retain the section-to-original-card map and only claims shared
                # across sections or explicitly conflicted; do not retry a
                # summary-only reconciliation packet.
                memberships: dict[str, int] = {}
                for section in audited_sections:
                    for card in section["cards"]:
                        memberships[card["id"]] = memberships.get(card["id"], 0) + 1
                shared_cards = [card for card in cited_cards
                                if memberships.get(card["id"], 0) > 1
                                or card.get("evidence_status") == "conflicted"]
                if shared_cards:
                    smaller_input = {
                        "sections": [{"id": section["id"], "requirement": section["question"],
                                      "original_card_ids": [card["id"] for card in section["cards"]]}
                                     for section in audited_sections],
                        "shared_original_evidence": serialize_synthesis_packet(
                            shared_cards, job.source_registry,
                        ),
                        "unresolved_requirements": [item["identifier"] for item in unresolved_requirements],
                    }
                    smaller_prompt = get_thinking_reconciliation_prompt(
                        json.dumps(smaller_input, ensure_ascii=True, separators=(",", ":")), shared_only=True,
                    )
                    if budget.fits("reconciliation", user_prompt=get_thinking_user_prompt("shared_reconciliation"),
                                   system_prompt=smaller_prompt, thinking_mode=False):
                        try:
                            reconciliation = _budgeted_generate(
                                self, budget, "reconciliation", user_prompt=get_thinking_user_prompt("shared_reconciliation"),
                                request_kind="context_retry",
                                system_prompt=smaller_prompt, max_new_tokens=1024, temperature=0.0,
                                cancel_check=cancelled, thinking_mode=False,
                            ) or ""
                            job.synthesis_state["overflow_retries"] += 1
                        except (PromptOverflow, ModelBudgetExhausted):
                            pass
                if not reconciliation:
                    job.synthesis_state.setdefault("stages_skipped", []).append("global_reconciliation_overflow")
            except ModelBudgetExhausted:
                job.synthesis_state.setdefault("stages_skipped", []).append("global_reconciliation_budget")
        if cancelled():
            return "", []
        if reconciliation.strip():
            raw_report = "## Summary\n\n" + reconciliation.strip() + "\n\n" + "\n\n".join(
                f"## {title}\n\n{raw}" for title, raw in raw_sections
            )
        else:
            raw_report = (
                "## Summary\n\n"
                + "\n\n".join(f"## {title}\n\n{raw}" for title, raw in raw_sections)
            )
        # Do not add automated completeness or correctness judgments to the
        # report; the cited findings and retained evidence remain inspectable.
        if unsafe_omissions:
            counts = {}
            for identifier in unsafe_omissions:
                reason = omission_reasons.get(identifier, "no_verified_section")
                counts[reason] = counts.get(reason, 0) + 1
            logger.warning("Deep-think job %s delivering partial report: sections=%d unsynthesized=%d reasons=%s",
                           job.id, len(audited_sections), len(unsafe_omissions), counts)
        # The source-aligned section audits above are not a global audit. Check
        # the assembled conclusion against all original cited cards as a final
        # bounded pass; any revision is constrained to the concise summary.
        global_ledger = json.dumps(
            serialize_synthesis_packet(cited_cards, job.source_registry),
            ensure_ascii=True, separators=(",", ":"),
        )
        global_clean, global_bindings, global_issues, _global_audited_raw = self._audit_candidate(
            job, raw_report, cited_cards, global_ledger, budget, cancelled, allow_revision=True,
        )
        # The final assembled report is authoritative. A repaired section can
        # remove citations; stale pre-repair bindings must not become references.
        job.report_bindings = global_bindings
        job.report_draft = raw_report
        job.report_audit.update({"issues": global_issues, "revised": bool(global_issues),
                                 "section_count": len(audited_sections)})
        report = global_clean or "\n\n".join(f"## {title}\n\n{clean}" for title, clean, _ in [
            (section["title"], section["report"], section["bindings"]) for section in audited_sections
        ])
        _record_synthesis_omissions(job)
        return report, job.report_bindings

    def _reasoned_report(self, job: BackgroundJob, cancelled: Callable[[], bool]) -> tuple[str, str]:
        """Reason from contract-owned premises without inventing source evidence."""
        contract = job.snapshot.task_record
        premises = [value for value in contract.get("assumptions", []) if isinstance(value, str) and value.strip()]
        if not premises:
            job.pending_clarification = {"kind": "missing_reasoning_premises"}
            return "Reactive question: Which premises should I use for this reasoning task?", job.state
        if job.pending_clarification.get("kind") == "missing_reasoning_premises":
            job.pending_clarification = {}
        context = research_context(job.snapshot.task, job.snapshot.conversation)
        if not job.evidence_requirements:
            job.evidence_requirements = initial_requirements(job.snapshot.task, context)
        for item in job.evidence_requirements:
            item.update(reasoning_allowed=True, explicit_premises=premises)
        refresh_coverage(job.evidence_requirements, [], {})
        pending = [item for item in job.evidence_requirements if item.get("state") != "resolved"]
        if cancelled():
            return "", job.state
        if pending:
            budget = ThinkingBudget(self.model, job.model_budget, context_tokens=self.context_tokens,
                                    availability_check=self._require_backend)
            prompt = get_thinking_reasoned_coverage_prompt(
                json.dumps(pending, ensure_ascii=True),
                json.dumps(premises, ensure_ascii=True),
            )
            response = _budgeted_generate(
                self, budget, "chapter_coverage", system_prompt=prompt,
                user_prompt=get_thinking_user_prompt("reasoned_coverage"), temperature=0.0, cancel_check=cancelled,
            )
            apply_reasoned_coverage(job.evidence_requirements, _parse_direct_json(response))
            refresh_coverage(job.evidence_requirements, [], {})
        if cancelled():
            return "", job.state
        accepted = [item for item in job.evidence_requirements if item.get("state") == "resolved"]
        if not accepted:
            raise ReportSynthesisError("No answer was grounded in the declared reasoning premises.")
        report = "## Reasoning from supplied premises\n\n" + "\n\n".join(
            str(item["accepted_answer"]["text"]) for item in accepted
        ) + "\n\n### Premises\n\n" + "\n".join(f"- {value}" for value in premises)
        unresolved = [item for item in job.evidence_requirements if item.get("state") != "resolved"]
        if unresolved:
            report += "\n\n### Unresolved\n\n" + "\n".join(f"- {item['question']}" for item in unresolved)
        job.synthesis_state["partial_report"] = bool(unresolved)
        job.report_draft = report
        self.jobs.checkpoint(job)
        return report, job.state

    def run(self, job: BackgroundJob, cancelled: Callable[[], bool]) -> tuple[str, str]:
        self._active_job = job
        if not self._backend_health_check(job):
            raise RemoteUnreachable("Research backend did not pass its bounded recovery health check")
        self._backend_lost = False
        self._require_backend()
        if is_unresolved_task_reference(job.snapshot.task):
            job.pending_clarification = {"kind": "unresolved_task_reference"}
            job.record_stage_event("planning", "action_rejection", "invalid_action_shape")
            return "Reactive question: What research topic should I continue?", job.state
        if self.model is None:
            raise RuntimeError("local language model is not loaded")
        if (job.snapshot.task_record.get("answer_format") or {}).get("type") == "reasoned_conversation":
            return self._reasoned_report(job, cancelled)
        capabilities = {
            name: capability
            for name, capability in self.capabilities().items()
            if capability.access_class == "read"
        }
        allowed = scoped_capabilities(job.snapshot.task, capabilities)
        capabilities = {name: capability for name, capability in capabilities.items() if name in allowed}
        playbooks = matching_playbooks(job.snapshot.task, capabilities)
        descriptions = {
            name: _describe_thinking_capability(name, self.registry._schemas[name])
            for name in capabilities
            if name in self.registry._schemas
        }
        # Handoffs contain actual controller-owned tool results, never facts
        # reconstructed from assistant prose or the short web summariser.
        handoffs = list(job.pending_handoffs)
        if not job.synthesis_state.get("handoff_imported"):
            handoffs.insert(0, job.snapshot.handoff)
        for handoff in handoffs:
            observations = handoff.get("observations", [])
            observations = observations if isinstance(observations, (list, tuple)) else []
            for observation in observations[:MAX_EVIDENCE_ENTRIES]:
                if not isinstance(observation, dict):
                    continue
                name = self.registry.canonical_name(str(observation.get("tool") or ""))
                if name not in capabilities:
                    continue
                result = ThinkingResult(
                    str(observation.get("text") or ""),
                    status=observation.get("status"), scope=observation.get("scope"),
                    evidence=observation.get("evidence"),
                    next_actions=tuple(observation.get("next_actions") or ())
                    if isinstance(observation.get("next_actions", ()), (list, tuple)) else (),
                    artifact=observation.get("artifact"), recovery=observation.get("recovery"),
                )
                if thinking_result_error(result):
                    continue
                artifact_id = job.record_outcome(
                    name, result.thinking_status, result.scope, result.evidence,
                    result.next_actions, result.artifact, result.recovery,
                    retrieved_at=observation.get("retrieved_at"),
                    metadata=observation.get("metadata"),
                )
                job.state = _append_thinking_observation(job.state, f"foreground {name}", json.dumps({
                    "status": result.thinking_status, "scope": result.scope,
                    "artifact_id": artifact_id,
                    "note": "Retained original evidence is available; review relevance and freshness before fetching again.",
                }))
            for step in handoff.get("plan", {}).get("steps", [])[:MAX_EVIDENCE_ENTRIES]:
                if isinstance(step, dict) and isinstance(step.get("action"), dict) and step.get("status") != "pending":
                    summary = {"tool": step["action"].get("intent"), "status": step.get("status")}
                    if summary not in job.foreground_steps and len(job.foreground_steps) < MAX_EVIDENCE_ENTRIES:
                        job.foreground_steps.append(summary)
        job.pending_handoffs.clear()
        job.synthesis_state["handoff_imported"] = True
        if not job.research_context or job.research_context.get("scope_version", 1) < 2:
            job.research_context = research_context(job.snapshot.task, job.snapshot.conversation,
                as_of=job.research_context.get("as_of"))
        findings = job.state
        base_requirements = ensure_requirements(
            job.evidence_requirements, job.snapshot.task, job.research_context
        )
        budget = ThinkingBudget(self.model, job.model_budget, self.context_tokens,
                                availability_check=self._require_backend)
        job.fact_cards = build_fact_cards(
            job.evidence, job.artifacts, job.snapshot.task, job.research_context,
            source_registry=job.source_registry, extraction_state=job.extraction_state,
        )
        if not job.chapter_plan:
            self._update_stage(job, "Planning research")
            plan_prompt = get_thinking_chapter_plan_prompt(
                job.snapshot.task,
                _fixed_scope_for_prompt(job.research_context),
                json.dumps([{
                    "id": item.get("identifier"),
                    "question": item.get("question"),
                    "importance": item.get("importance"),
                    "completion_criteria": item.get("completion_criteria"),
                } for item in base_requirements], ensure_ascii=True, separators=(",", ":")),
                "\n".join(descriptions.values()),
                prior_work=json.dumps({
                    "research_assignment": job.snapshot.handoff.get("research_task", job.snapshot.task),
                    "completed_foreground_steps": job.foreground_steps,
                    "retained_evidence": [{"tool": entry["tool"], "status": entry["status"],
                                           "scope": entry["scope"]} for entry in job.evidence],
                }, ensure_ascii=True, separators=(",", ":")) if job.evidence or job.foreground_steps else "",
            )
            planned = None
            plan_response = ""
            plan_generation_failure = ""
            plan_fits = budget.fits(
                "chapter_planning", system_prompt=plan_prompt,
                user_prompt=get_thinking_user_prompt("chapter_planning"), thinking_mode=False,
            )
            if not plan_fits:
                plan_generation_failure = "context_limit"
            if not cancelled() and plan_fits:
                try:
                    plan_response = _budgeted_generate(
                        self, budget, "chapter_planning",
                        system_prompt=plan_prompt,
                        user_prompt=get_thinking_user_prompt("chapter_planning"),
                        temperature=0.2, cancel_check=cancelled,
                    )
                    parsed = _parse_direct_json(plan_response)
                    planned = validate_chapter_plan(
                        parsed, job.snapshot.task, base_requirements, capabilities,
                    )
                    if planned is None and not cancelled():
                        # One targeted repair is charged to the same bounded
                        # planning stage. A rejected plan is never dispatched;
                        # deterministic fallback remains the final safe path.
                        repair_reason = (
                            "Return one complete JSON object matching the requested chapter-plan schema. "
                            "Include chapters with title, priority, questions, tools, and queries; "
                            "each question must include its question and priority. Do not add prose."
                        )
                        repair_prompt = get_thinking_output_repair_prompt(plan_prompt, repair_reason)
                        job.synthesis_state["chapter_plan_repair"] = {
                            "attempted": True, "initial_output_valid": False,
                        }
                        plan_response = _budgeted_generate(
                            self, budget, "chapter_planning",
                            request_kind="format_retry",
                            system_prompt=repair_prompt,
                            user_prompt=get_thinking_user_prompt("chapter_planning"),
                            temperature=0.0, cancel_check=cancelled,
                        ) or ""
                        repaired_plan = _parse_direct_json(plan_response)
                        planned = validate_chapter_plan(
                            repaired_plan, job.snapshot.task, base_requirements, capabilities,
                        )
                        job.synthesis_state["chapter_plan_repair"]["succeeded"] = planned is not None
                except (ModelBudgetExhausted, PromptOverflow) as exc:
                    plan_generation_failure = (
                        "context_limit" if isinstance(exc, PromptOverflow) else "model_budget"
                    )
                    logger.info("Deep-think chapter planning used its deterministic fallback")
                except Exception as exc:
                    self._require_backend()
                    plan_generation_failure = "model_error"
                    logger.warning(
                        "Deep-think structured chapter planning failed error_type=%s",
                        type(exc).__name__,
                    )
            if cancelled():
                return "", findings
            if planned is None:
                if plan_generation_failure:
                    job.record_stage_event(
                        "chapter_planning", "generation_failure", plan_generation_failure,
                    )
                elif plan_response:
                    job.record_stage_event(
                        "chapter_planning", "output_rejection",
                        "malformed_json" if _parse_direct_json(plan_response) is None
                        else "invalid_output_shape",
                    )
                else:
                    job.record_stage_event(
                        "chapter_planning", "generation_failure", "empty_output",
                    )
                planned = fallback_chapter_plan(
                    job.snapshot.task, base_requirements, capabilities,
                )
                job.synthesis_state["chapter_planning"] = "deterministic_fallback"
            else:
                job.synthesis_state["chapter_planning"] = "structured"
            job.chapter_plan = planned
            job.evidence_requirements = chapter_requirements(
                job.chapter_plan, job.snapshot.task, job.research_context,
            )
            job.planning_complete = True
        else:
            job.planning_complete = True
            if not job.evidence_requirements:
                job.evidence_requirements = chapter_requirements(
                    job.chapter_plan, job.snapshot.task, job.research_context,
                )
            for chapter in job.chapter_plan:
                chapter["tools"] = [name for name in chapter.get("tools", []) if name in capabilities]
        for chapter in job.chapter_plan:
            # Older checkpoints counted only dispatched operations. Migrate
            # conservatively while retaining the new decision counter durably.
            chapter.setdefault("research_decisions", int(chapter.get("research_calls", 0)))
        if needs_subject_resolution(job.research_context):
            apply_subject_candidates(job.research_context, [], job.fact_cards)
            identity_question = next((question for chapter in job.chapter_plan for question in chapter["questions"]
                                      if question.get("source_requirement_id") == "subject-identity"), None)
            if identity_question is None:
                # Migrate pending older/seeded plans without resetting completed
                # actions or renumbering their existing evidence/question IDs.
                identity_question = {
                    "id": "subject-identity", "requirement_id": "subject-identity",
                    "question": "Resolve the exact upcoming subject, date and version from original source evidence.",
                    "priority": "high", "state": "open", "source_requirement_id": "subject-identity",
                    "supporting_card_ids": [], "dependencies": [],
                }
                job.chapter_plan[0]["questions"].insert(0, identity_question)
                # The chapter is reopened for the newly required prerequisite;
                # accepted bodies and spent budgets remain saved independently.
                job.chapter_plan[0]["state"] = "planned"
            identity_id = identity_question["requirement_id"]
            if not any(item["identifier"] == identity_id for item in job.evidence_requirements):
                identity_requirement = next(item for item in chapter_requirements(
                    job.chapter_plan, job.snapshot.task, job.research_context,
                ) if item["identifier"] == identity_id)
                job.evidence_requirements.insert(0, identity_requirement)
            for requirement in job.evidence_requirements:
                requirement["subject_resolution_required"] = True
                requirement["is_subject_identity"] = requirement["identifier"] == identity_id
                requirement["subject_resolution"] = job.research_context.get("subject_resolution") or {"state": "unresolved"}
                if not requirement["is_subject_identity"] and identity_id not in requirement.get("dependencies", []):
                    requirement["dependencies"] = [*requirement.get("dependencies", []), identity_id]
        contract = job.snapshot.task_record
        for requirement in job.evidence_requirements:
            requirement["typed_constraints"] = contract.get("typed_constraints") or []
            requirement["reasoning_allowed"] = (contract.get("answer_format") or {}).get("type") == "reasoned_conversation"
            requirement["explicit_premises"] = [
                value for value in contract.get("assumptions", []) if isinstance(value, str)
            ]
            requirement["as_of"] = str(job.research_context.get("as_of") or "")
        job.section_plan = [
            {"id": chapter["id"], "title": chapter["title"],
             "requirement_ids": [question["requirement_id"] for question in chapter["questions"]]}
            for chapter in job.chapter_plan
        ]
        link_chapter_cards(job.fact_cards, job.chapter_plan)
        refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
        # Persist budgets across a clarification pause so confirmation cannot
        # reset the investigation's bounded work allowance.
        capability_calls = job.capability_calls
        attempted_actions = job.attempted_actions
        exhausted_capabilities = job.exhausted_capabilities
        needs_input = None
        active_requirement = None
        active_chapter = None

        def normalise_action(name, args, kwargs):
            schema = self.registry._schemas.get(name or "")
            if schema is None:
                return None, f"Capability schema unavailable for {name}"
            overrides = {"query": job.snapshot.task, "profile": "research"} if name == "external_information" else {}
            values, error = _normalise_thinking_args(schema, args, kwargs, default_overrides=overrides)
            if not error and name == "external_information":
                for index, param in enumerate(schema.params):
                    if param.name == "query" and values[index] == param.default:
                        # A copied signature default is not a task-specific
                        # research query. Keep the actual assigned request.
                        values[index] = job.snapshot.task
                if (active_requirement or {}).get("is_subject_identity") and not re.search(
                    r"\b(?:date|calendar|schedule|release|departure|when|time)\b", str(values[0]), re.I,
                ):
                    values[0] = searchable_gap_query(job.snapshot.task, active_requirement)
                values = direct_retrieval_to_gap(
                    name, values, active_requirement or open_requirement(job.evidence_requirements)
                )
            return values, error

        def fallback_action() -> tuple[str, list, dict, dict] | None:
            """Choose a validated retrieval for an open chapter with no dispatch."""
            self._require_backend()
            nonlocal active_requirement, active_chapter
            if sum(capability_calls.values()) >= MAX_THINKING_TOOL_CALLS:
                job.tool_budget_exhausted = True
                return None
            gaps = _fallback_open_gaps(job)
            if needs_subject_resolution(job.research_context) and job.research_context.get("subject_resolution", {}).get("state") != "resolved":
                gaps = [pair for pair in gaps if pair[0].get("is_subject_identity")]
            if not gaps:
                return None
            priority = {"high": 3, "medium": 2, "low": 1}
            requirement, chapter = max(
                gaps,
                key=lambda pair: (
                    priority.get(pair[0].get("importance"), 1),
                    -job.chapter_plan.index(pair[1]),
                ),
            )
            candidates = []
            if "external_information" in chapter.get("tools", []):
                candidates.append("external_information")
            candidates.extend(
                playbook.fallback_capability for playbook in playbooks
                if playbook.fallback_capability in chapter.get("tools", [])
            )
            targeted_query = searchable_gap_query(job.snapshot.task, requirement)
            for name in dict.fromkeys(candidates):
                if name not in capabilities or name in exhausted_capabilities:
                    continue
                if capability_calls.get(name, 0) >= effective_capability_call_limit(
                    name, job.action_selections, MAX_THINKING_CAPABILITY_CALLS
                ):
                    continue
                schema = self.registry._schemas.get(name)
                if schema is None:
                    continue
                required = sum(param.required for param in schema.params)
                if name == "external_information" and schema.params:
                    args = [targeted_query]
                elif required == 0:
                    args = []
                elif required == 1:
                    args = [targeted_query]
                else:
                    continue
                previous_requirement, previous_chapter = active_requirement, active_chapter
                active_requirement, active_chapter = requirement, chapter
                args, argument_error = normalise_action(name, args, {})
                active_requirement, active_chapter = previous_requirement, previous_chapter
                if argument_error or args is None:
                    continue
                if _thinking_action_key(name, args) not in attempted_actions:
                    return name, args, chapter, requirement
            return None

        def observe(label, text):
            nonlocal findings
            findings = _append_thinking_observation(findings, label, text)
            job.state = findings

        def run_capability(name: str, args: list) -> bool:
            self._require_backend()
            nonlocal needs_input
            stop_for_preliminary_report = False
            validation_refunded = False
            capability = capabilities.get(name)
            if capability is None:
                job.record_stage_event(
                    "capability_dispatch", "tool_blocked", "unavailable_tool",
                )
                observe("worker", f"Unavailable capability: {name}")
                return True
            if active_chapter is not None and name not in active_chapter.get("tools", []):
                job.record_stage_event(
                    "capability_dispatch", "tool_blocked", "out_of_scope_tool",
                )
                observe("worker", f"Capability {name} is outside the active chapter's tool scope.")
                return True
            if name in exhausted_capabilities:
                job.record_stage_event(
                    "capability_dispatch", "tool_blocked", "capability_exhausted",
                )
                observe("worker", f"Capability unavailable or exhausted for {name}; do not retry the same provider outage.")
                return any(other not in exhausted_capabilities for other in capabilities)
            schema = self.registry._schemas.get(name)
            if schema is not None:
                required = sum(param.required for param in schema.params)
                if not required <= len(args) <= len(schema.params):
                    job.record_stage_event(
                        "capability_dispatch", "tool_blocked", "invalid_arguments",
                    )
                    observe(
                        "worker",
                        f"Invalid arguments for {name}; expected {required}-{len(schema.params)} positional values.",
                    )
                    return True
            if sum(capability_calls.values()) >= MAX_THINKING_TOOL_CALLS:
                job.tool_budget_exhausted = True
                job.record_stage_event("capability_dispatch", "tool_blocked", "tool_budget")
                observe(
                    "worker",
                    "Overall tool-call budget reached; synthesise only the eligible evidence collected so far.",
                )
                return False
            capability_limit = effective_capability_call_limit(
                name, job.action_selections, MAX_THINKING_CAPABILITY_CALLS
            )
            if capability_calls.get(name, 0) >= capability_limit:
                exhausted_capabilities.add(name)
                job.record_stage_event("capability_dispatch", "tool_blocked", "capability_budget")
                observe(
                    "worker",
                    f"Capability budget reached for {name}; choose another available capability for the open gap.",
                )
                return any(
                    other != name and capability_calls.get(other, 0) < effective_capability_call_limit(
                        other, job.action_selections, MAX_THINKING_CAPABILITY_CALLS
                    )
                    for other in capabilities
                )
            # Every dispatch path normalizes exactly once before recording its
            # identity. Do not reparse canonical values containing '=' here.
            requirement = active_requirement or open_requirement(job.evidence_requirements)
            # Count progress against the prior eligible working set. Scope or
            # freshness changes can make retained cards newly useful without
            # adding card IDs, and must not terminate retrieval prematurely.
            prior_card_ids = {card["id"] for card in report_card_ledger(job.fact_cards)}
            prior_source_ids = set(job.source_registry)
            prior_covered = {item["identifier"] for item in job.evidence_requirements if item.get("state") == "resolved"}
            record_action_selection(
                job.action_selections, tool=name, args=args, requirement=requirement
            )
            job.action_selections[-1]["chapter_id"] = active_chapter.get("id") if active_chapter else ""
            job.action_selections[-1]["progress_before"] = {
                "coverage": sorted(prior_covered),
                "synthesized": sorted({identifier for chapter in job.chapter_plan
                                       for identifier in chapter.get("analyzed_card_ids", [])}),
            }
            capability_calls[name] = capability_calls.get(name, 0) + 1
            self._update_stage(
                job,
                "Searching flights"
                if name == "search_flights"
                else "Comparing accommodation"
                if name == "search_hotels"
                else "Searching sources"
                if name in {"external_information", "search_papers"}
                else "Reviewing information",
            )
            logger.debug("Deep-think job %s dispatching approved capability=%s", job.id, name)
            artifact_token = set_artifacts(job.artifacts)
            try:
                result = capability.invoke(args, {})
            except Exception as exc:
                text = f"Tool {name} failed: {type(exc).__name__}: {exc}"
                result = (
                    ThinkingResult(
                        text,
                        status="failed",
                        scope="The requested tool operation raised an exception.",
                    )
                    if schema is not None and schema.thinking_outcome
                    else text
                )
            finally:
                reset_artifacts(artifact_token)
            if cancelled():
                return False
            artifact_id = None
            if schema is not None and schema.thinking_outcome:
                error = (
                    "The tool did not provide its required typed evidence envelope."
                    if not isinstance(result, ThinkingResult)
                    else thinking_result_error(result)
                )
                if error:
                    result = ThinkingResult(
                        f"Tool {name} returned an invalid deep-think outcome: {error}",
                        status="failed",
                        scope="The tool result was rejected before it entered the evidence ledger.",
                    )
            next_actions = []
            if isinstance(result, ThinkingResult):
                result_status = result.thinking_status
                result_scope = result.scope
                if result_status == "unavailable":
                    exhausted_capabilities.add(name)
                    job.record_stage_event(
                        "capability_dispatch", "tool_blocked", "unavailable_tool",
                    )
                next_actions = result.next_actions
                artifact_id = job.record_outcome(
                    name,
                    result.thinking_status,
                    result.scope,
                    result.evidence,
                    result.next_actions,
                    result.artifact,
                    result.recovery,
                    metadata={
                        "record_identity": result.record_identity,
                        "provenance": result.provenance,
                        "temporal": result.temporal,
                        "qualifications": result.qualifications,
                    },
                )
                job.fact_cards = build_fact_cards(
                    job.evidence, job.artifacts, job.snapshot.task, job.research_context,
                    source_registry=job.source_registry, extraction_state=job.extraction_state,
                )
                link_chapter_cards(job.fact_cards, job.chapter_plan)
                refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
                self.jobs.checkpoint(job)
                matching_cards = matching_cards_for_requirement(job.fact_cards, requirement, artifact_id)
                eligible = bool(matching_cards) and not any(
                    card.get("evidence_status") == "conflicted" for card in matching_cards
                )
                if artifact_id:
                    result = ThinkingResult(
                        f"{result}\nArtifact reference: {artifact_id}",
                        status=result.thinking_status,
                        evidence=result.evidence,
                        scope=result.scope,
                        next_actions=result.next_actions,
                        artifact=result.artifact,
                        recovery=result.recovery,
                        record_identity=result.record_identity,
                        provenance=result.provenance,
                        temporal=result.temporal,
                        qualifications=result.qualifications,
                    )
                if result.thinking_status == "needs_input":
                    recovery = result.recovery
                    if recovery.get("requires_user") is False:
                        if recovery.get("kind") == "argument_validation":
                            job.action_selections[-1]["validation_only"] = True
                            corrections = job.synthesis_state.setdefault("argument_corrections", {})
                            if corrections.get(name, 0) < MAX_THINKING_ARGUMENT_CORRECTIONS:
                                # No retrieval took place. In the live run an
                                # invalid profile consumed the final search slot,
                                # blocking its own required internal correction.
                                capability_calls[name] -= 1
                                corrections[name] = corrections.get(name, 0) + 1
                                validation_refunded = True
                        job.synthesis_state["action_feedback"] = {
                            "tool": name, "args": args, "message": str(result),
                            "scope": result.scope, "recovery": recovery,
                        }
                        result = str(result)
                    elif _has_verified_evidence(job):
                        stop_for_preliminary_report = True
                        result = str(result)
                    elif recovery.get("requires_user", not recovery):
                        job.pending_clarification = recovery
                        needs_input = (
                            "Reactive question: "
                            + str(result).removeprefix("Reactive question:").strip()
                        )
                        result = str(result)
                    else:
                        # The planner receives recoverable validation failures
                        # and can choose an identifier lookup or corrected call.
                        result = str(result)
                else:
                    feedback = job.synthesis_state.get("action_feedback", {})
                    if feedback.get("tool") == name:
                        job.synthesis_state.pop("action_feedback", None)
                        if feedback.get("recovery") and result_status in {"evidence", "rejected"}:
                            job.resolve_pending_clarification(feedback["recovery"], "corrected")
                    if result.thinking_status in {"failed", "unavailable"}:
                        result = str(result)
                resolve_requirement(
                    job.evidence_requirements,
                    requirement,
                    status=result_status,
                    scope=result_scope,
                    eligible=eligible,
                )
                if requirement and requirement.get("state") == "resolved":
                    requirement["evidence_ids"] = [
                        card["id"] for card in matching_cards
                    ]
            elif schema is not None:
                # Older read-only capabilities predate the typed evidence
                # envelope. Preserve their bounded observation as a clearly
                # scoped generic record so calculations, notes, and read-state
                # tools can participate in the same fact-card pipeline.
                preview = intents.classify_step(result)
                plain_status = "needs_input" if preview.kind is intents.StepKind.REACTIVE else "evidence"
                plain_scope = (
                    "An unstructured read-only tool observation; use only its stated content."
                    if plain_status == "evidence"
                    else "The read-only tool requested clarification."
                )
                artifact_id = job.record_outcome(
                    name,
                    plain_status,
                    plain_scope,
                    {"text": str(result)},
                    (),
                    {"type": "tool_observation", "tool": name, "text": str(result)},
                )
                job.fact_cards = build_fact_cards(
                    job.evidence, job.artifacts, job.snapshot.task, job.research_context,
                    source_registry=job.source_registry, extraction_state=job.extraction_state,
                )
                link_chapter_cards(job.fact_cards, job.chapter_plan)
                refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
                resolve_requirement(
                    job.evidence_requirements,
                    requirement,
                    status=plain_status,
                    scope=plain_scope,
                    eligible=bool(matching_cards_for_requirement(job.fact_cards, requirement, artifact_id)),
                )
                if requirement and requirement.get("state") == "resolved":
                    requirement["evidence_ids"] = [
                        card["id"] for card in matching_cards_for_requirement(
                            job.fact_cards, requirement, artifact_id,
                        )
                    ]
            step = intents.classify_step(result)
            if step.kind is intents.StepKind.REACTIVE and not validation_refunded:
                capability_calls[name] -= 1
            if step.artifact is not None and job.artifact is None:
                job.artifact = step.artifact
            if step.kind is intents.StepKind.WEB_SEARCH and name != "external_information":
                summary = self.summarise_search(step.text, cancelled)
                if summary and summary.strip():
                    result = summary
                else:
                    logger.warning("Deep-think job %s received an empty web summary", job.id)
            logger.info(
                "Deep-think job %s received %d characters from %s", job.id, len(result), name
            )
            observe(f"tool:{name}", result)
            if next_actions:
                observe(
                    "worker",
                    "Suggested distinct next capabilities: " + ", ".join(next_actions) + ".",
                )
            job.action_selections[-1].update({
                "artifact_id": artifact_id,
                "new_eligible_cards": len({card["id"] for card in report_card_ledger(job.fact_cards)} - prior_card_ids),
                "new_sources": len(set(job.source_registry) - prior_source_ids),
                "coverage_gain": len({item["identifier"] for item in job.evidence_requirements if item.get("state") == "resolved"} - prior_covered),
                "repeated_source_only": bool(job.fact_cards) and set(job.source_registry) == prior_source_ids,
                "effective_capability_limit": effective_capability_call_limit(
                    name, job.action_selections[:-1], MAX_THINKING_CAPABILITY_CALLS
                ),
            })
            if name == "evaluate_itinerary" and "not feasible" in result.lower():
                observe(
                    "worker",
                    "This rejects one evaluated itinerary only. Inspect untested retrieved combinations or take a materially different available search before making a broader feasibility conclusion.",
                )
            if stop_for_preliminary_report:
                observe(
                    "worker",
                    "Further input would refine the result, but sufficient evidence exists for a preliminary scoped report now.",
                )
                return False
            return True

        requirements_by_chapter = {
            chapter["id"]: [item for item in job.evidence_requirements
                            if item.get("chapter_id") == chapter["id"]]
            for chapter in job.chapter_plan
        }

        def judge_chapter_coverage(chapter: dict) -> None:
            """Verify direct answers with evidence IDs, not keyword overlap."""
            self._require_backend()
            if (job.research_context.get("subject_resolution") or {}).get("state") == "resolved":
                self._review_source_documents(job, budget, cancelled)
            chapter_reqs = requirements_by_chapter.get(chapter["id"], [])
            if needs_subject_resolution(job.research_context) and (
                job.research_context.get("subject_resolution") or {}
            ).get("state") != "resolved":
                chapter_reqs = [item for item in chapter_reqs if item.get("is_subject_identity")]
            eligible = report_card_ledger(job.fact_cards)
            if not chapter_reqs or not eligible:
                refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
                return
            preferred, _ = select_requirement_cards(
                eligible, chapter_reqs, source_registry=job.source_registry,
            )
            # Discovery must not undo per-question eligibility by refilling from
            # the global ledger. In particular, an unresolved prerequisite must
            # never yield a tutorial-heavy dependent coverage packet.
            allowed_ids = {card["id"] for item in chapter_reqs
                           for card in eligible_cards_for_requirement(eligible, item)}
            preferred_ids = {card["id"] for card in preferred}
            ordered = [*preferred, *(card for card in eligible if card["id"] in allowed_ids
                                    and card["id"] not in preferred_ids)]
            if not ordered:
                job.record_stage_event("chapter_coverage", "evidence_omission", "no_question_eligible_cards")
                refresh_coverage(chapter_reqs, job.fact_cards, job.extraction_state)
                return
            questions = json.dumps([{
                "question_id": item["identifier"], "question": item["question"],
                "completion_criteria": item.get("completion_criteria"),
                "current_gap": item.get("detail", ""),
            } for item in chapter_reqs if item.get("state") != "resolved"],
                ensure_ascii=True, separators=(",", ":"))
            if questions == "[]":
                return
            assessment_fingerprint = hashlib.sha256(json.dumps({
                "questions": [{key: item.get(key) for key in ("identifier", "question", "state", "dependencies")}
                              for item in chapter_reqs],
                "scope": _fixed_scope_for_prompt(job.research_context),
                "cards": [{key: card.get(key) for key in ("id", "statement", "fields", "provenance", "evidence_status", "compatibility")}
                          for card in ordered],
            }, sort_keys=True, default=str).encode()).hexdigest()
            if chapter.get("batch_assessment_fingerprint") == assessment_fingerprint:
                return
            # This limit is characters, not tokens. Still check the complete
            # request against the actual context budget below.
            packet_limit = min(24_000, max(512, budget.context_tokens * 2))
            response = ""
            shown_cards = []
            while packet_limit >= 512:
                coverage_aliases = {}
                packet = _chapter_evidence_packet(ordered, max_chars=packet_limit, aliases=coverage_aliases)
                shown_ids = set(coverage_aliases.values())
                shown_cards = [card for card in ordered if card.get("id") in shown_ids]
                context_packet = self._source_packet(job, shown_cards)
                context = context_packet.get("passages") or []
                prompt = get_thinking_chapter_coverage_prompt(
                    job.snapshot.task, _fixed_scope_for_prompt(job.research_context),
                    questions, packet, json.dumps(context, ensure_ascii=True, separators=(",", ":")) if context else "",
                )
                if context_packet.get("context_blockers") or not budget.fits(
                    "chapter_coverage", system_prompt=prompt,
                    user_prompt=get_thinking_user_prompt("chapter_coverage"), thinking_mode=False,
                ):
                    packet_limit //= 2
                    continue
                try:
                    response = _budgeted_generate(
                        self, budget, "chapter_coverage", system_prompt=prompt,
                        user_prompt=get_thinking_user_prompt("chapter_coverage"), temperature=0.0,
                        cancel_check=cancelled,
                    ) or ""
                except (PromptOverflow, ModelBudgetExhausted):
                    job.synthesis_state.setdefault("stages_skipped", []).append(
                        "chapter_coverage_budget_or_context",
                    )
                except Exception as exc:
                    self._require_backend()
                    logger.warning(
                        "Deep-think chapter coverage check failed error_type=%s",
                        type(exc).__name__,
                    )
                break
            if cancelled():
                return
            parsed = _parse_direct_json(response)
            if isinstance(parsed, dict):
                # Packet-local navigation is never evidence. Decode before all
                # original-ID eligibility, scope and dependency checks below.
                for field in ("answers", "findings", "subject_candidates"):
                    for row in parsed.get(field, []) if isinstance(parsed.get(field), list) else []:
                        if isinstance(row, dict) and isinstance(row.get("card_ids"), list):
                            row["card_ids"] = [coverage_aliases.get(identifier, identifier)
                                               if isinstance(identifier, str) else identifier
                                               for identifier in row["card_ids"]]
            if parsed is not None:
                chapter["batch_assessment_fingerprint"] = assessment_fingerprint
            if needs_subject_resolution(job.research_context):
                proposals = (parsed or {}).get("subject_candidates") or []
                shown_ids = {card["id"] for card in shown_cards}
                proposals = [row for row in proposals if isinstance(row, dict)
                             and isinstance(row.get("card_ids"), list)
                             and all(identifier in shown_ids for identifier in row["card_ids"]
                                     if isinstance(identifier, str))]
                resolution = apply_subject_candidates(
                    job.research_context, proposals, job.fact_cards,
                )
                for item in job.evidence_requirements:
                    item["subject_resolution"] = resolution
                    if item.get("is_subject_identity"):
                        if resolution["state"] == "resolved":
                            item.update(state="resolved", answer_coverage="evidence_scoped",
                                        evidence_ids=resolution["card_ids"], accepted_answer={
                                            "text": f"{resolution['name']} — {resolution['date']}",
                                            "supporting_card_ids": resolution["card_ids"],
                                        })
                        else:
                            item.update(state="open", answer_coverage="unsupported", evidence_ids=[],
                                        detail=resolution["detail"])
                chapter_reqs_to_assess = [item for item in chapter_reqs if not item.get("is_subject_identity")]
            else:
                chapter_reqs_to_assess = chapter_reqs
            apply_supported_coverage(chapter_reqs_to_assess, shown_cards, parsed)
            apply_reasoned_coverage(chapter_reqs, parsed)
            refresh_coverage(chapter_reqs, job.fact_cards, job.extraction_state)
            for question in chapter.get("questions", []):
                requirement = next((item for item in chapter_reqs
                                    if item.get("identifier") == question.get("requirement_id")), None)
                if requirement is None:
                    continue
                question["state"] = "answered" if requirement.get("state") == "resolved" else "open"
                question["supporting_card_ids"] = list(requirement.get("evidence_ids") or [])
                question["missing_facts"] = "" if question["state"] == "answered" else str(
                    requirement.get("detail") or "Direct answer not established by retained evidence."
                )
            chapter["coverage_checked"] = True
            assign_batch_findings(chapter, chapter_reqs, shown_cards, parsed)
            link_chapter_cards(job.fact_cards, job.chapter_plan)

        def save_supported_subsection(chapter):
            """Save checked useful facts while independent questions remain open."""
            self._require_backend()
            assigned = {identifier for question in chapter.get("questions", [])
                        for identifier in [*(question.get("supporting_card_ids") or []),
                                           *(question.get("finding_bindings") or {})]}
            eligible = report_card_ledger(job.fact_cards)
            new = assigned - set(chapter.get("analyzed_card_ids") or [])
            if not new.intersection(card["id"] for card in eligible) or cancelled():
                return
            self._update_stage(job, f"Saving supported subsection: {chapter['title'][:70]}")
            try:
                self._synthesise_chapters(job, eligible, budget, cancelled,
                                         chapter_ids={chapter["id"]}, finalize=False)
            except Exception as exc:
                self._require_backend()
                logger.warning("Deep-think incremental section save failed chapter=%s error_type=%s",
                               chapter["id"], type(exc).__name__)
                job.synthesis_state.setdefault("incremental_section_failures", []).append(
                    {"chapter_id": chapter["id"], "error_type": type(exc).__name__})

        def finish_batch_progress(chapter):
            selections = [row for row in job.action_selections if row.get("chapter_id") == chapter["id"]
                          and row.get("progress_before")]
            if not selections:
                return
            selection = selections[-1]
            previous = selection.pop("progress_before")
            synthesized = {identifier for item in job.chapter_plan for identifier in item.get("analyzed_card_ids", [])}
            selection["useful_finding_gain"] = len(synthesized - set(previous["synthesized"]))
            selection["coverage_gain"] = len({item["identifier"] for item in job.evidence_requirements
                                              if item.get("state") == "resolved"} - set(previous["coverage"]))

        if job.continuation:
            user_input, job.continuation = job.continuation, ""
            observe("user clarification", user_input)
            recovery = job.pending_clarification
            resume_action = recovery.get("resume_action") if isinstance(recovery, dict) else None
            if (
                isinstance(resume_action, dict)
                and re.fullmatch(
                    r"\s*(?:yes|yeah|yep|yes please|please do|go ahead)\s*[.!]?\s*",
                    user_input,
                    re.I,
                )
            ):
                name = self.registry.canonical_name(str(resume_action.get("intent") or ""))
                args, kwargs = intents.coerce_args(resume_action.get("args"))
                schema = self.registry._schemas.get(name or "")
                if schema is not None:
                    args, argument_error = normalise_action(name, args, kwargs)
                else:
                    argument_error = "The saved correction references an unavailable capability."
                job.resolve_pending_clarification(recovery, "confirmed")
                job.pending_clarification = {}
                if name and args is not None and not argument_error:
                    attempted_actions.add(_thinking_action_key(name, args))
                    observe("worker", f"User confirmed the proposed correction; retrying {name}.")
                    run_capability(name, args)
                    if needs_input:
                        return needs_input, findings
                else:
                    observe("worker", "The saved correction could not be resumed; replanning from the clarification.")
            else:
                # The next planning turn sees the user's replacement or refusal
                # and can select a new, materially different action.
                job.resolve_pending_clarification(recovery, "replaced")
                job.pending_clarification = {}

        # First judge retained material on resume; a query/source keyword match
        # alone never closes a question.
        for chapter in job.chapter_plan:
            if chapter.get("state") not in {"research_complete", "analyzed"}:
                judge_chapter_coverage(chapter)
                if all(item.get("state") == "resolved"
                       for item in requirements_by_chapter.get(chapter["id"], [])):
                    chapter["state"] = "research_complete"
                save_supported_subsection(chapter)
                finish_batch_progress(chapter)

        priority_weight = {"high": 3, "medium": 2, "low": 1}
        while sum(capability_calls.values()) < MAX_THINKING_TOOL_CALLS:
            self._require_backend()
            if cancelled():
                return "", findings
            refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
            _recover_retained_evidence(job)
            for chapter in job.chapter_plan:
                if (chapter.get("state") not in {"research_complete", "analyzed"}
                        and int(chapter.get("research_decisions", 0)) >= MAX_CHAPTER_RESEARCH_DECISIONS
                        and any(item.get("state") == "open"
                                for item in requirements_by_chapter.get(chapter["id"], []))):
                    chapter["state"] = "research_complete"
                    chapter["research_note"] = (
                        f"Chapter decision allowance reached ({MAX_CHAPTER_RESEARCH_DECISIONS}); "
                        "remaining questions are reported as open."
                    )
                    observe("worker", f"{chapter['title']}: {chapter['research_note']}")
            active_chapters = [chapter for chapter in job.chapter_plan
                               if chapter.get("state") not in {"research_complete", "analyzed"}
                               and int(chapter.get("research_decisions", 0)) < MAX_CHAPTER_RESEARCH_DECISIONS
                               and any(item.get("state") == "open"
                                        for item in requirements_by_chapter.get(chapter["id"], []))]
            if needs_subject_resolution(job.research_context) and (
                job.research_context.get("subject_resolution") or {}
            ).get("state") != "resolved":
                active_chapters = [chapter for chapter in active_chapters if any(
                    item.get("is_subject_identity") for item in requirements_by_chapter.get(chapter["id"], []))]
            if not active_chapters:
                break
            active_chapter = max(
                active_chapters,
                key=lambda chapter: (
                    any(item.get("is_subject_identity") and item.get("state") != "resolved"
                        for item in requirements_by_chapter.get(chapter["id"], [])),
                    priority_weight.get(chapter.get("priority"), 1)
                    / (1 + int(chapter.get("research_decisions", 0))),
                    -job.chapter_plan.index(chapter),
                ),
            )
            chapter_requirements_open = [item for item in requirements_by_chapter[active_chapter["id"]]
                                          if item.get("state") == "open"]
            if any(item.get("is_subject_identity") for item in chapter_requirements_open):
                chapter_requirements_open = [item for item in chapter_requirements_open if item.get("is_subject_identity")]
            if not chapter_requirements_open:
                active_chapter["state"] = "research_complete"
                continue
            active_requirement = max(
                chapter_requirements_open,
                key=lambda item: (bool(item.get("is_subject_identity")), priority_weight.get(item.get("importance"), 1)),
            )
            chapter_tools = [name for name in active_chapter.get("tools", [])
                             if name in capabilities and name not in exhausted_capabilities
                             and capability_calls.get(name, 0) < effective_capability_call_limit(
                                 name, job.action_selections, MAX_THINKING_CAPABILITY_CALLS,
                             )]
            if not chapter_tools:
                active_chapter["state"] = "research_complete"
                exhausted_tools = [
                    name for name in active_chapter.get("tools", [])
                    if name in capabilities and capability_calls.get(name, 0) >=
                    effective_capability_call_limit(
                        name, job.action_selections, MAX_THINKING_CAPABILITY_CALLS,
                    )
                ]
                if exhausted_tools:
                    active_chapter["research_note"] = (
                        "Capability budget reached for " + ", ".join(exhausted_tools) + "."
                    )
                    observe("worker", active_chapter["research_note"])
                else:
                    active_chapter["research_note"] = (
                        "No listed read-only tool remains available for this chapter."
                    )
                continue
            chapter_reqs = requirements_by_chapter[active_chapter["id"]]
            questions = json.dumps([{
                "question_id": item["identifier"], "question": item["question"],
                "priority": item.get("importance"), "missing_facts": item.get("detail", ""),
            } for item in chapter_requirements_open], ensure_ascii=True, separators=(",", ":"))
            eligible_cards = report_card_ledger(job.fact_cards)
            preferred_cards, _ = select_requirement_cards(
                eligible_cards, chapter_requirements_open, source_registry=job.source_registry,
            )
            ordered_cards = preferred_cards
            evidence_packet = _chapter_evidence_packet(
                ordered_cards, max_chars=min(4_000, max(512, budget.context_tokens // 5)),
            )
            prior_queries = [{"tool": item.get("tool"), "args": item.get("args")}
                             for item in job.action_selections
                             if item.get("chapter_id") == active_chapter["id"]]
            scoped_descriptions = "\n".join(descriptions[name] for name in chapter_tools
                                              if name in descriptions)
            chapter_descriptor = json.dumps(
                {"id": active_chapter["id"], "title": active_chapter["title"],
                 "priority": active_chapter["priority"]},
                ensure_ascii=True, separators=(",", ":"),
            )

            def research_prompt(
                controller_feedback, *, active_chapter=active_chapter,
                chapter_descriptor=chapter_descriptor, questions=questions,
                prior_queries=prior_queries, evidence_packet=evidence_packet,
                scoped_descriptions=scoped_descriptions,
                active_requirement=active_requirement,
            ):
                return get_thinking_chapter_research_prompt(
                    job.snapshot.task, _fixed_scope_for_prompt(job.research_context),
                    chapter_descriptor, questions,
                    json.dumps({"seeds": [searchable_gap_query(job.snapshot.task, active_requirement)]
                                if (active_requirement or {}).get("is_subject_identity")
                                else active_chapter.get("queries", []),
                                "already_used": prior_queries,
                                "controller_feedback": controller_feedback,
                                "chapter_note": active_chapter.get("research_note", "")}, ensure_ascii=True,
                               separators=(",", ":")),
                    evidence_packet, scoped_descriptions,
                )

            prompt = research_prompt(job.synthesis_state.get("action_feedback"))
            self._update_stage(job, f"Researching {active_chapter['title'][:70]}")
            if not budget.fits(
                "chapter_research", system_prompt=prompt,
                user_prompt=get_thinking_user_prompt("chapter_research"), thinking_mode=False,
            ):
                job.record_stage_event(
                    "chapter_research", "generation_failure", "context_limit",
                )
                active_chapter["state"] = "research_complete"
                active_chapter["research_note"] = "The chapter research request did not fit the available context."
                continue
            try:
                active_chapter["research_decisions"] = int(
                    active_chapter.get("research_decisions", 0)
                ) + 1
                response = _budgeted_generate(
                    self, budget, "chapter_research", user_prompt=get_thinking_user_prompt("chapter_research"),
                    system_prompt=prompt, temperature=0.2, cancel_check=cancelled,
                )
            except (PromptOverflow, ModelBudgetExhausted) as exc:
                job.record_stage_event(
                    "chapter_research", "generation_failure",
                    "context_limit" if isinstance(exc, PromptOverflow) else "model_budget",
                )
                active_chapter["state"] = "research_complete"
                active_chapter["research_note"] = type(exc).__name__
                continue
            except Exception as exc:
                self._require_backend()
                job.record_stage_event(
                    "chapter_research", "action_rejection", "generation_failed",
                )
                logger.warning(
                    "Deep-think chapter research action generation failed error_type=%s",
                    type(exc).__name__,
                )
                active_chapter["state"] = "research_complete"
                active_chapter["research_note"] = "No usable retrieval action was produced."
                continue
            if cancelled():
                return "", findings
            emission = _parse_direct_json(response)
            name, args, kwargs, done, action_error = _validate_chapter_action(
                emission, chapter_tools, self.registry,
            )
            if action_error:
                rejection_reason = _action_rejection_reason(
                    emission, action_error, capabilities, self.registry,
                )
                job.record_stage_event(
                    "chapter_research", "action_rejection", rejection_reason,
                )
                if rejection_reason in {"unavailable_tool", "out_of_scope_tool"}:
                    observe("worker", "The selected tool is unavailable or outside this chapter's approved tool scope.")
                correction_feedback = {
                    "kind": "action_format_validation",
                    "message": action_error,
                }
                correction_prompt = research_prompt(correction_feedback)
                correction_fits = budget.fits(
                    "chapter_research", system_prompt=correction_prompt,
                    user_prompt=get_thinking_user_prompt("chapter_research"), thinking_mode=False,
                )
                if not correction_fits:
                    job.record_stage_event(
                        "chapter_research", "generation_failure", "context_limit",
                    )
                    response = ""
                elif int(active_chapter.get("research_decisions", 0)) >= MAX_CHAPTER_RESEARCH_DECISIONS:
                    job.record_stage_event(
                        "chapter_research", "action_rejection", "decision_budget",
                    )
                    response = ""
                else:
                    try:
                        active_chapter["research_decisions"] = int(
                            active_chapter.get("research_decisions", 0)
                        ) + 1
                        response = _budgeted_generate(
                            self, budget, "chapter_research",
                            request_kind="format_retry",
                            user_prompt=get_thinking_user_prompt("chapter_research"),
                            system_prompt=correction_prompt, temperature=0.0,
                            cancel_check=cancelled,
                        ) or ""
                    except (PromptOverflow, ModelBudgetExhausted) as exc:
                        job.record_stage_event(
                            "chapter_research", "generation_failure",
                            "context_limit" if isinstance(exc, PromptOverflow) else "model_budget",
                        )
                        response = ""
                    except Exception as exc:
                        self._require_backend()
                        job.record_stage_event(
                            "chapter_research", "generation_failure", "model_error",
                        )
                        logger.warning(
                            "Deep-think chapter action correction failed error_type=%s",
                            type(exc).__name__,
                        )
                        response = ""
                if cancelled():
                    return "", findings
                emission = _parse_direct_json(response)
                name, args, kwargs, done, action_error = _validate_chapter_action(
                    emission, chapter_tools, self.registry,
            )
            if action_error:
                rejection_reason = _action_rejection_reason(
                    emission, action_error, capabilities, self.registry,
                )
                job.record_stage_event(
                    "chapter_research", "action_rejection", rejection_reason,
                )
                active_chapter["state"] = "research_complete"
                active_chapter["research_note"] = (
                    "The chapter action was rejected after one format correction "
                    f"({rejection_reason})."
                )
                observe("worker", active_chapter["research_note"])
                continue
            if done:
                active_chapter["state"] = "research_complete"
                active_chapter["research_note"] = "No further distinct read-only action was selected."
                continue
            args, argument_error = normalise_action(name, args, kwargs)
            if argument_error or args is None:
                job.record_stage_event(
                    "chapter_research", "action_rejection", "invalid_arguments",
                )
                active_chapter["research_note"] = "The action arguments did not match the approved tool schema."
                job.synthesis_state["action_feedback"] = {
                    "tool": name, "args": emission.get("args"),
                    "message": str(argument_error or active_chapter["research_note"]),
                }
                observe("worker", active_chapter["research_note"])
                continue
            action_key = _thinking_action_key(name, args)
            if action_key in attempted_actions:
                job.duplicate_actions_avoided += 1
                job.record_stage_event(
                    "chapter_research", "action_rejection", "duplicate_action",
                )
                active_chapter["research_note"] = "The next action duplicated a prior request."
                if not (active_requirement or {}).get("is_subject_identity"):
                    active_chapter["state"] = "research_complete"
                job.synthesis_state["action_feedback"] = {
                    "tool": name, "args": args, "kind": "duplicate_action",
                    "message": "Duplicate request rejected. Choose a distinct lookup for the still-open question.",
                }
                observe("worker", f"Duplicate capability request for {name}; the repeated action was not dispatched.")
                continue
            feedback = job.synthesis_state.get("action_feedback") or {}
            corrective_retry = (feedback.get("tool") == name and bool(feedback.get("message"))
                                and feedback.get("kind") != "duplicate_action")
            if not corrective_retry and is_near_duplicate_action(
                name, args, job.action_selections, active_requirement,
            ) or (not corrective_retry and is_near_duplicate_query(name, args, job.action_selections)):
                job.duplicate_actions_avoided += 1
                job.record_stage_event(
                    "chapter_research", "action_rejection", "near_duplicate_action",
                )
                active_chapter["research_note"] = "The next action was too similar to an earlier query."
                if not (active_requirement or {}).get("is_subject_identity"):
                    active_chapter["state"] = "research_complete"
                job.synthesis_state["action_feedback"] = {
                    "tool": name, "args": args, "kind": "duplicate_action",
                    "message": "Near-duplicate rejected. Choose a materially distinct lookup for the still-open question.",
                }
                observe("worker", f"Duplicate capability request for {name}; the near-repeat was not dispatched.")
                continue
            attempted_actions.add(action_key)
            active_chapter["state"] = "researching"
            active_chapter["research_calls"] = int(active_chapter.get("research_calls", 0)) + 1
            active_chapter.setdefault("completed_actions", []).append(action_key)
            if not run_capability(name, args):
                break
            if needs_input:
                return needs_input, findings
            refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
            judge_chapter_coverage(active_chapter)
            if all(item.get("state") == "resolved" for item in chapter_reqs):
                active_chapter["state"] = "research_complete"
            save_supported_subsection(active_chapter)
            finish_batch_progress(active_chapter)
        if cancelled():
            return "", findings
        active_requirement = None
        active_chapter = None
        refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
        if sum(capability_calls.values()) >= MAX_THINKING_TOOL_CALLS:
            job.tool_budget_exhausted = True
            observe(
                "worker",
                "Overall tool-call budget reached; synthesis is limited to evidence collected so far.",
            )
        while not needs_input and (fallback := fallback_action()) is not None:
            fallback_name, fallback_args, fallback_chapter, fallback_requirement = fallback
            active_chapter, active_requirement = fallback_chapter, fallback_requirement
            fallback_key = _thinking_action_key(fallback_name, fallback_args)
            attempted_actions.add(fallback_key)
            active_chapter["research_calls"] = int(active_chapter.get("research_calls", 0)) + 1
            active_chapter.setdefault("completed_actions", []).append(fallback_key)
            observe(
                "worker",
                f"No retrieval was dispatched for open question {fallback_requirement['identifier']}; "
                + (
                    f"using the playbook fallback {fallback_name}."
                    if any(book.fallback_capability == fallback_name for book in playbooks)
                    else f"using targeted fallback retrieval {fallback_name}."
                ),
            )
            run_capability(fallback_name, fallback_args)
            if not cancelled():
                judge_chapter_coverage(active_chapter)
                save_supported_subsection(active_chapter)
                finish_batch_progress(active_chapter)
            active_chapter, active_requirement = None, None
            if needs_input:
                return needs_input, findings
        if cancelled():
            return "", findings
        if not _has_verified_evidence(job):
            raise ReportSynthesisError(
                "No verified evidence was retrieved; refusing to mark a source-less report ready."
            )
        refresh_coverage(job.evidence_requirements, job.fact_cards, job.extraction_state)
        for chapter in job.chapter_plan:
            judge_chapter_coverage(chapter)
        job.fact_cards = build_fact_cards(
            job.evidence, job.artifacts, job.snapshot.task, job.research_context,
            source_registry=job.source_registry, extraction_state=job.extraction_state,
        )
        link_chapter_cards(job.fact_cards, job.chapter_plan)
        if not report_card_ledger(job.fact_cards):
            raise ReportSynthesisError(
                "No eligible compatible fact cards were retained; refusing to synthesise a positive report."
            )
        self._update_stage(job, "Synthesising report")
        eligible_cards = report_card_ledger(job.fact_cards)
        statuses: dict[str, int] = {}
        for card in job.fact_cards:
            status = str(card.get("evidence_status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        job.report_audit.setdefault("metrics", {}).update({
            "card_count": len(job.fact_cards),
            "eligible_card_count": len(eligible_cards),
            "source_diversity": len({card.get("independence_key") for card in eligible_cards}),
            "card_statuses": statuses,
            "tool_calls_used": sum(capability_calls.values()),
            "tool_call_budget": MAX_THINKING_TOOL_CALLS,
            "capability_call_budget": MAX_THINKING_CAPABILITY_CALLS,
            "capability_effective_limits": {
                name: effective_capability_call_limit(
                    name, job.action_selections, MAX_THINKING_CAPABILITY_CALLS
                )
                for name in capability_calls
            },
            "exhausted_capabilities": sorted(exhausted_capabilities),
            "tool_budget_exhausted": job.tool_budget_exhausted,
            "original_evidence_tokens": estimate_tokens(json.dumps(job.evidence, ensure_ascii=True)),
            "original_artifact_tokens": estimate_tokens(json.dumps(job.artifacts, ensure_ascii=True)),
            "original_fact_card_tokens": estimate_tokens(json.dumps(eligible_cards, ensure_ascii=True)),
            "synthesis_packet_tokens": 0,
            "prompt_view_tokens": 0,
            "selected_card_count": 0,
            "omitted_card_count": 0,
            "duplicate_actions_avoided": job.duplicate_actions_avoided,
        })
        job.report_bindings = []
        logger.info(
            "Deep-think job %s synthesis=%s cards=%d eligible=%d evidence=%d",
            job.id,
            "chapters" if job.chapter_plan else "legacy_saved_evidence",
            len(job.fact_cards),
            len(report_card_ledger(job.fact_cards)),
            len(job.evidence),
        )
        job.state = findings
        self._constraint_audit(job, eligible_cards)
        findings = self._summarize_research_notes(job, budget, cancelled, job.state)
        report, job.report_bindings = self._synthesise(
            job, eligible_cards, budget, cancelled, findings,
        )
        constraint_results = (job.synthesis_state.get("constraint_audit") or {}).get("results", [])
        if constraint_results:
            report += "\n\n## Constraint checks\n\n" + "\n".join(
                f"- {item['text'] or item['id']}: {item['status']} in retained records"
                f" ({item['reason']})." for item in constraint_results
            )
        delivery = classify_delivered_synthesis(
            job.evidence_requirements, job.report_bindings,
            job.synthesis_state.get("extractive_only_card_ids") or [],
        )
        job.synthesis_state["delivery_assessment"] = {
            **delivery,
            "task_revision": int(job.snapshot.task_record.get("revision", 1) or 1),
        }
        if delivery["status"] in {"blocked", "evidence_only"}:
            job.synthesis_state["partial_report"] = True
            job.synthesis_state["delivery_assessment"]["reason"] = "no_answered_essential_with_delivered_support"
        displayed = {binding["card_id"] for binding in job.report_bindings}
        delivered = displayed - set(job.synthesis_state.get("extractive_only_card_ids") or [])
        if not job.synthesis_state.get("limited_evidence_fallback"):
            selected_ids = set(job.synthesis_state.get("selected_card_ids") or [])
            job.synthesis_state["evidence_counts"] = {
                "ineligible": len(job.fact_cards) - len(eligible_cards),
                "unselected": len(eligible_cards) - len(selected_ids),
                "unsynthesized": len(selected_ids - delivered),
                "display_omitted": 0,
            }
        job.report_audit["metrics"]["prompt_view_tokens"] = sum(
            int(stage.get("input_tokens", 0)) for stage in budget.state.get("stages", {}).values()
            if isinstance(stage, dict)
        )
        job.report_audit["metrics"]["synthesis_packet_tokens"] = int(
            job.synthesis_state.get("synthesis_packet_tokens", 0)
        )
        job.report_audit["metrics"]["compression_ratio"] = round(
            job.report_audit["metrics"]["original_fact_card_tokens"]
            / max(1, job.report_audit["metrics"]["synthesis_packet_tokens"]), 3,
        )
        if cancelled():
            return "", findings
        report = (report or "").strip()
        if report:
            _record_completion_metrics(job, report)
            logger.info("Deep-think job %s produced a %d-character report", job.id, len(report))
            return report, findings
        logger.warning(
            "Deep-think job %s returned an empty final report; preserving collected evidence",
            job.id,
        )
        fallback = _fallback_thinking_report(
            job.snapshot.task, job.evidence, findings, cards=eligible_cards, job=job
        )
        job.report_bindings = _fallback_report_bindings(job, eligible_cards)
        _record_completion_metrics(job, fallback)
        return fallback, findings
