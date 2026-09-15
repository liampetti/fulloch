"""Bounded investigations with explicit model, scheduling and tool dependencies.

The worker never touches foreground history. Its snapshot and evidence belong to
the job manager. Single-slot generation acquires the supplied model lock; tools
run outside it. Two-slot generation uses the independently admitted server slot.
"""

import json
import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager

from tools.capabilities import ToolCapability
from tools.thinking_context import reset_artifacts, set_artifacts
from tools.thinking_playbooks import matching_playbooks
from tools.tool_registry import ThinkingResult, ToolRegistry, thinking_result_error
from utils import intents
from utils.prompts import get_thinking_report_prompt, get_thinking_worker_prompt

from .background_jobs import BackgroundJob, BackgroundJobManager

logger = logging.getLogger(__name__)

MAX_THINKING_WORKER_CALLS = 12
MAX_THINKING_CAPABILITY_CALLS = 3
DEEP_THINK_READ_TIMEOUT_S = 600.0
DEEP_THINK_GENERATION_TIMEOUT_S = 900.0
DEEP_THINK_STEP_MAX_TOKENS = 1024
DEEP_THINK_TRANSCRIPT_MAX_CHARS = 12_000
DEEP_THINK_OBSERVATION_MAX_CHARS = 3_000
_THINKING_OBSERVATION_RE = re.compile(r"\n\n(?=\[[^\]\n]+\]\n)")


class ReportSynthesisError(RuntimeError):
    """An investigation could not collect source material for a final report."""


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
    if len(description) > 240:
        description = description[:237].rstrip() + "..."
    return f"- {name}({', '.join(args)}): {description}"


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


def _fallback_thinking_report(task: str, evidence: list[dict], findings: str) -> str:
    source = (
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        if evidence
        else findings
    )
    source = source[:DEEP_THINK_TRANSCRIPT_MAX_CHARS].rstrip()
    return (
        "## Summary\n\n"
        "The investigation collected the evidence below, but the final report model "
        "returned no visible response. This is the retrieved material without additional synthesis.\n\n"
        f"## Task\n\n{task}\n\n## Collected evidence\n\n"
        f"{source or 'No usable evidence was retained.'}"
    )


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

    def _generate(self, **kwargs):
        with nullcontext() if self.server_slots == 2 else self.model_lock:
            return self.generate(
                self.model,
                read_timeout=DEEP_THINK_READ_TIMEOUT_S,
                generation_timeout=DEEP_THINK_GENERATION_TIMEOUT_S,
                recover_on_failure=False,
                **kwargs,
            )

    def run(self, job: BackgroundJob, cancelled: Callable[[], bool]) -> tuple[str, str]:
        if self.model is None:
            raise RuntimeError("local language model is not loaded")
        capabilities = {
            name: capability
            for name, capability in self.capabilities().items()
            if capability.access_class == "read"
        }
        playbooks = matching_playbooks(job.snapshot.task, capabilities)
        descriptions = [
            _describe_thinking_capability(name, self.registry._schemas[name])
            for name in capabilities
            if name in self.registry._schemas
        ]
        findings = job.state
        capability_calls: dict[str, int] = {}
        attempted_actions: set[str] = set()
        needs_input = None

        def observe(label, text):
            nonlocal findings
            findings = _append_thinking_observation(findings, label, text)
            job.state = findings

        def run_capability(name: str, args: list) -> bool:
            nonlocal needs_input
            stop_for_preliminary_report = False
            capability = capabilities.get(name)
            if capability is None:
                observe("worker", f"Unavailable capability: {name}")
                return False
            schema = self.registry._schemas.get(name)
            if schema is not None:
                required = sum(param.required for param in schema.params)
                if not required <= len(args) <= len(schema.params):
                    observe(
                        "worker",
                        f"Invalid arguments for {name}; expected {required}-{len(schema.params)} positional values.",
                    )
                    return False
            if capability_calls.get(name, 0) >= MAX_THINKING_CAPABILITY_CALLS:
                observe(
                    "worker",
                    f"Capability budget reached for {name}; synthesise the evidence collected so far.",
                )
                return False
            capability_calls[name] = capability_calls.get(name, 0) + 1
            self.jobs.update_stage(
                job,
                "Searching flights"
                if name == "search_flights"
                else "Comparing accommodation"
                if name == "search_hotels"
                else "Searching sources"
                if name in {"external_information", "search_papers"}
                else "Reviewing information",
            )
            logger.debug("Deep-think job %s dispatching %s with args=%r", job.id, name, args)
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
                next_actions = result.next_actions
                artifact_id = job.record_outcome(
                    name,
                    result.thinking_status,
                    result.scope,
                    result.evidence,
                    result.next_actions,
                    result.artifact,
                )
                if artifact_id:
                    result = ThinkingResult(
                        f"{result}\nArtifact reference: {artifact_id}",
                        status=result.thinking_status,
                        evidence=result.evidence,
                        scope=result.scope,
                        next_actions=result.next_actions,
                        artifact=result.artifact,
                    )
                if result.thinking_status == "needs_input":
                    if any(item.get("status") == "evidence" for item in job.evidence):
                        stop_for_preliminary_report = True
                        result = str(result)
                    else:
                        needs_input = (
                            "Reactive question: "
                            + str(result).removeprefix("Reactive question:").strip()
                        )
                elif result.thinking_status in {"failed", "unavailable"}:
                    result = str(result)
            step = intents.classify_step(result)
            if step.kind is intents.StepKind.REACTIVE:
                capability_calls[name] -= 1
            if step.artifact is not None and job.artifact is None:
                job.artifact = step.artifact
            if step.kind is intents.StepKind.WEB_SEARCH:
                summary = self.summarise_search(step.text, cancelled)
                if summary and summary.strip():
                    result = summary
                else:
                    logger.warning("Deep-think job %s received an empty web summary", job.id)
            logger.info(
                "Deep-think job %s received %d characters from %s", job.id, len(result), name
            )
            observe(f"tool:{name}", result)
            if isinstance(result, ThinkingResult) and next_actions:
                observe(
                    "worker",
                    "Suggested distinct next capabilities: " + ", ".join(next_actions) + ".",
                )
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

        for _ in range(MAX_THINKING_WORKER_CALLS):
            if cancelled():
                return "", findings
            self.jobs.update_stage(job, "Analysing findings")
            prompt = get_thinking_worker_prompt(
                job.snapshot.task,
                list(job.snapshot.conversation),
                notes=job.snapshot.notes,
                job_state=findings,
                capabilities="\n".join(descriptions),
                capability_playbooks="\n\n".join(playbook.render() for playbook in playbooks),
            )
            # JSON grammar and Qwen reasoning tokens cannot compose. Deliberation
            # comes from the bounded loop; synthesis below uses thinking mode.
            response = self._generate(
                user_prompt=prompt,
                grammar=self.grammar,
                max_new_tokens=DEEP_THINK_STEP_MAX_TOKENS,
                temperature=0.4,
                cancel_check=cancelled,
                thinking_mode=False,
            )
            if cancelled():
                return "", findings
            response = (response or "").strip()
            if not response:
                observe(
                    "worker",
                    "Worker stopped without a next action; synthesise only the evidence collected so far.",
                )
                break
            try:
                emission = intents.parse_agent_emission(response)
            except (TypeError, ValueError):
                observe(
                    "worker",
                    "Worker returned an invalid planning response; select the next capability or reply sufficient findings collected.",
                )
                continue
            reply = emission.get("reply")
            if isinstance(reply, str) and reply.strip():
                reply = reply.strip()
                if reply.startswith("Reactive question:"):
                    return reply, findings
                observe("worker", reply)
                break
            plan = emission.get("plan")
            if isinstance(plan, str) and plan.strip():
                logger.debug("Deep-think job %s plan: %s", job.id, plan.strip())
            actions = emission.get("actions") or []
            if not actions and isinstance(plan, str) and plan.strip():
                fallback_name = next(
                    (
                        p.fallback_capability
                        for p in playbooks
                        if p.fallback_capability in capabilities
                    ),
                    None,
                )
                if fallback_name is not None:
                    action_key = _thinking_action_key(fallback_name, [job.snapshot.task])
                    if action_key not in attempted_actions:
                        attempted_actions.add(action_key)
                        observe(
                            "worker",
                            f"Worker supplied a plan without an action; using {fallback_name}.",
                        )
                        run_capability(fallback_name, [job.snapshot.task])
                        if needs_input:
                            return needs_input, findings
                        continue
            if len(actions) != 1 or not isinstance(actions[0], dict):
                observe(
                    "worker",
                    "Worker did not select exactly one next capability; synthesise current findings.",
                )
                break
            action = actions[0]
            name = self.registry.canonical_name(action.get("intent", ""))
            if capabilities.get(name or "") is None:
                observe("worker", "Blocked unavailable capability: " + str(action.get("intent")))
                break
            args, kwargs = intents.coerce_args(action.get("args"))
            if kwargs:
                observe("worker", f"Unsupported keyword arguments for {name}")
                break
            action_key = _thinking_action_key(name, args)
            if action_key in attempted_actions:
                observe(
                    "worker",
                    f"Duplicate capability request for {name}; synthesise the evidence collected so far.",
                )
                break
            attempted_actions.add(action_key)
            if not run_capability(name, args):
                break
            if needs_input:
                return needs_input, findings
        if cancelled():
            return "", findings
        if not findings.strip():
            raise ReportSynthesisError("No source material was retrieved for the report.")
        self.jobs.update_stage(job, "Synthesising report")
        report_prompt = get_thinking_report_prompt(
            job.snapshot.task,
            findings if not job.evidence else "",
            json.dumps(job.evidence, ensure_ascii=True, sort_keys=True),
        )
        report = self._generate(
            user_prompt="Produce the final report now.",
            system_prompt=report_prompt,
            max_new_tokens=8192,
            temperature=0.4,
            cancel_check=cancelled,
            thinking_mode=True,
        )
        if cancelled():
            return "", findings
        report = (report or "").strip()
        if report:
            logger.info("Deep-think job %s produced a %d-character report", job.id, len(report))
            return report, findings
        logger.warning(
            "Deep-think job %s returned an empty final report; preserving collected evidence",
            job.id,
        )
        return _fallback_thinking_report(job.snapshot.task, job.evidence, findings), findings
