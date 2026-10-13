"""Turn-local plan progress, duplicate protection, and research handoffs.

A replan replaces arguments, not the obligation to address pending work. Stale
dependent arguments are never executed automatically; only the model can revise
them after seeing the prerequisite observation.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone

from utils.intents import LOOKUP_TOOLS, StepKind, coerce_args, reactive_to_speech


def action_key(action, registry) -> str:
    """Make aliases and positional/keyword/default arguments equivalent."""
    name = registry.canonical_name(action.get("intent", "")) or action.get("intent", "")
    args, kwargs = coerce_args(action.get("args"))
    schema = registry.get_schema(name)
    if schema is not None:
        values = {param.name: param.default for param in schema.params if not param.required}
        values.update({param.name: value for param, value in zip(schema.params, args, strict=False)})
        values.update(kwargs)
        if len(args) <= len(schema.params):
            return json.dumps([name, values], sort_keys=True, default=str)
    return json.dumps([name, args, kwargs], sort_keys=True, default=str)


@dataclass
class PlannedStep:
    id: int
    action: dict
    key: str
    dependencies: list[int] = field(default_factory=list)
    needs_revision: bool = False
    status: str = "pending"
    result: object = None
    reason: str = ""
    generation: int = 0
    observation_ref: int | None = None


class TurnPlan:
    def __init__(self, task, registry, access_class):
        self.task = task
        self.registry = registry
        self.access_class = access_class
        self.steps: list[PlannedStep] = []
        self.observations: list[dict] = []
        self._results: dict[str, object] = {}
        self._result_origins: dict[str, PlannedStep] = {}
        self._generation = 0
        self._last_emission_fingerprint: str | None = None
        self._last_emission_observation_count = 0
        self._unchanged_emissions = 0

    def repeated_without_progress(self, actions) -> bool:
        """Stop repeated equivalent action plans after one correction attempt."""
        fingerprint = json.dumps(
            sorted(action_key(action, self.registry) for action in actions),
            separators=(",", ":"),
        )
        observation_count = len(self.observations)
        if fingerprint == self._last_emission_fingerprint:
            if observation_count == self._last_emission_observation_count:
                self._unchanged_emissions += 1
            else:
                self._unchanged_emissions = 0
        else:
            self._unchanged_emissions = 0
        self._last_emission_fingerprint = fingerprint
        self._last_emission_observation_count = observation_count
        return self._unchanged_emissions >= 2

    def name(self, action):
        return self.registry.canonical_name(action.get("intent", "")) or action.get("intent", "")

    def stage(self, actions) -> list[PlannedStep]:
        self._generation += 1
        batch = []
        claimed = set()
        for action in actions:
            key = action_key(action, self.registry)
            pending = [step for step in self.steps if step.id not in claimed
                       and step.status in {"pending", "blocked"}]
            requested_id = action.get("step_id")
            step = next((step for step in pending if step.id == requested_id), None) if requested_id is not None else None
            invalid_id = requested_id is not None and (
                not isinstance(requested_id, int) or isinstance(requested_id, bool) or step is None
            )
            if requested_id is None:
                pending = [step for step in pending if self.name(step.action) == self.name(action)]
                step = next((step for step in pending if step.key == key), None)
                step = step or next(iter(pending), None)
            if step is None:
                step = PlannedStep(len(self.steps) + 1, deepcopy(action), key, generation=self._generation)
                self.steps.append(step)
            else:
                # A fresh emission can repair arguments using actual observations.
                step.action, step.key = deepcopy(action), key
                step.needs_revision, step.reason = False, ""
                step.status, step.result = "pending", None
            claimed.add(step.id)
            if invalid_id:
                step.needs_revision = True
                step.reason = "Unknown or completed step_id; select a pending step from plan_progress."
            dependencies = action.get("depends_on")
            if dependencies is None:
                # Legacy arrays do not express dependencies. Be conservative
                # after retrievals, especially research-then-save placeholders.
                step.dependencies = [prior.id for prior in batch
                                     if self.access_class(self.name(prior.action)) == "read"]
            elif isinstance(dependencies, list) and all(
                isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(batch)
                for index in dependencies
            ):
                step.dependencies = [batch[index].id for index in dependencies]
            else:
                step.needs_revision = True
                step.reason = "Invalid dependencies; revise this step before execution."
            batch.append(step)
        return batch

    def cached(self, step):
        result = self._results.get(step.key)
        original = self._result_origins.get(step.key)
        # Distinct identical actions deliberately listed together are separate
        # goals (e.g. two timers). Deduplicate repeats on *replans*, not those
        # original siblings. An uncertain mutation is never repeated blindly.
        if original is not None and original.id != step.id and original.generation == step.generation:
            if original.status == "completed":
                return None
        return result

    def record(self, step, result, *, reused=False):
        step.result = result
        failed = result.kind is StepKind.ERROR or (
            result.kind is StepKind.REACTIVE and self.access_class(self.name(step.action)) != "read"
        ) or (
            result.outcome is not None and result.outcome["status"] != "evidence"
        )
        step.status = "blocked" if failed else "completed"
        step.reason = "The tool did not establish a successful result." if failed else ""
        # Repeating an uncertain mutation risks doing it twice. Successful
        # reads can also be reused; failed reads may be retried within the cap.
        if not failed or self.access_class(self.name(step.action)) != "read":
            self._results[step.key] = result
            self._result_origins[step.key] = step
        if reused:
            return
        outcome = deepcopy(result.outcome)
        observed_text = result.raw_text if result.raw_text is not None else str(result.text)
        if outcome is None:
            outcome = {
                "status": "failed" if failed else "needs_input" if result.kind is StepKind.REACTIVE else "evidence",
                "scope": f"One foreground {self.name(step.action)} tool observation.",
                "evidence": {"text": observed_text} if not failed and result.kind is not StepKind.REACTIVE and observed_text else {},
                "next_actions": [], "artifact": deepcopy(result.artifact), "recovery": {},
            }
        self.observations.append({
            "tool": self.name(step.action), "args": deepcopy(step.action.get("args", [])),
            "text": observed_text, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "access_class": self.access_class(self.name(step.action)), **outcome,
        })
        step.observation_ref = len(self.observations)

    def defer(self, batch, completed):
        """Keep the suffix, but require fresh arguments for dependent steps."""
        for step in batch[batch.index(completed) + 1:]:
            if step.status != "pending":
                continue
            if self.name(step.action) == "deep_think":
                # Research consumes observations, including failures, itself.
                continue
            if step.dependencies or "depends_on" not in step.action:
                step.needs_revision = True
                step.reason = "Recompose arguments from prerequisite observations, not the old plan."

    def ready(self):
        """Only explicitly independent suffixes or research handoffs auto-resume."""
        ready = []
        for step in self.steps:
            if step.status != "pending" or step.needs_revision:
                continue
            earlier = [prior for prior in self.steps if prior.id in step.dependencies]
            research = self.name(step.action) == "deep_think"
            if research:
                if any(prior.status == "pending" for prior in self.steps if prior.id < step.id):
                    continue
            elif any(prior.status != "completed" for prior in earlier):
                continue
            ready.append(step)
            if research:
                break
        return ready[:3]

    def pending(self):
        return [step for step in self.steps if step.status == "pending"]

    def progress(self):
        return {
            "task": self.task,
            "steps": [{
            "id": step.id, "action": step.action, "status": step.status,
            "depends_on": step.dependencies, "needs_revision": step.needs_revision,
            "reason": step.reason, "outcome_ref": step.observation_ref,
            } for step in self.steps],
            "outcome_refs": [{
                "id": index, "tool": observation.get("tool"),
                "status": observation.get("status"), "scope": observation.get("scope"),
            } for index, observation in enumerate(self.observations, 1)],
        }

    def handoff(self, research_task=None):
        return deepcopy({
            "original_request": self.task,
            "research_task": research_task or self.task,
            "plan": self.progress(),
            "observations": self.observations,
        })

    def output(self, *, include_lookups=False):
        parts = []
        for step in self.steps:
            result = step.result
            if result is None or not result.in_output:
                continue
            name = self.name(step.action)
            # Raw lookup records are composed by the model, not spoken verbatim.
            if name in LOOKUP_TOOLS and not include_lookups:
                continue
            if result.kind is StepKind.REACTIVE and self.access_class(name) == "read":
                continue
            text = reactive_to_speech(result.text) if result.kind is StepKind.REACTIVE else result.text.strip()
            if text and text not in parts:
                parts.append(text)
        return parts

    def unfinished(self):
        labels = []
        queued_research = [step.id for step in self.steps
                           if step.status == "completed" and self.name(step.action) == "deep_think"]
        for step in self.steps:
            if step.status == "completed":
                continue
            if step.status == "blocked" and step.result is not None and step.result.in_output:
                continue
            # Unexecuted context lookups after a research handoff are internal
            # plan details, not unfinished user operations. Keep their state for
            # diagnostics without adding tool names to the acknowledgement.
            if (step.result is None and self.name(step.action) in LOOKUP_TOOLS
                    and any(identifier < step.id for identifier in queued_research)):
                continue
            label = self.name(step.action).replace("_", " ")
            if label not in labels:
                labels.append(label)
        return "Not completed: " + ", ".join(labels) + "." if labels else ""

    def finish(self, text="", *, include_results=False):
        parts = self.output() if include_results else []
        for step in self.steps:
            if step.status == "blocked" and step.result is not None and step.result.in_output:
                failure = reactive_to_speech(step.result.text) if step.result.kind is StepKind.REACTIVE else step.result.text.strip()
                if failure and failure not in " ".join([*parts, text]):
                    parts.append(failure)
        for part in (text.strip(), self.unfinished()):
            if part and part not in parts:
                parts.append(part)
        return " ".join(parts)
