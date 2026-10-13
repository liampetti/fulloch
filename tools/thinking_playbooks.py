"""Tool-owned applicability and capability policy for central research prompts."""

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from utils.prompts import get_thinking_playbook_contract, render_thinking_playbook_prompt


@dataclass(frozen=True)
class ThinkingPlaybook:
    """A concise, domain-owned evidence path for a family of read tools."""

    name: str
    triggers: tuple[str, ...]
    capabilities: tuple[str, ...]
    solve_path: tuple[str, ...]
    completion_rule: str
    prohibited_shortcuts: tuple[str, ...] = ()
    matcher: Callable[[str], bool] | None = None
    fallback_capability: str | None = None
    task_scoped: bool = False

    def applies_to(self, task: str, available_capabilities: Iterable[str]) -> bool:
        available = set(available_capabilities)
        return (
            bool(set(self.capabilities) & available)
            and (
                self.matcher(task)
                if self.matcher is not None
                else any(re.search(pattern, task, re.IGNORECASE) for pattern in self.triggers)
            )
        )

    def render(self) -> str:
        return render_thinking_playbook_prompt(
            self.name, self.solve_path, self.completion_rule, self.prohibited_shortcuts,
        )


_PLAYBOOKS: dict[str, ThinkingPlaybook] = {}
_PUBLIC_MODEL_RESEARCH_RE = re.compile(
    r"\b(?:local|on[- ]device|open[- ]weight)\b.{0,100}\bmodels?\b|"
    r"\bmodels?\b.{0,100}\b(?:local|on[- ]device|open[- ]weight)\b",
    re.I,
)
_HOUSEHOLD_HISTORY_CAPABILITIES = frozenset({
    "get_entity_history", "get_conversation_history", "conversation_history",
})
_EXPLICIT_HOUSEHOLD_RESEARCH_RE = re.compile(
    r"\b(?:home assistant|\bha\b|household|entity history|conversation history|my notes?)\b",
    re.I,
)


def thinking_playbook(
    *,
    name: str,
    triggers: tuple[str, ...],
    capabilities: tuple[str, ...],
    solve_path: tuple[str, ...] | None = None,
    completion_rule: str | None = None,
    prohibited_shortcuts: tuple[str, ...] | None = None,
    matcher: Callable[[str], bool] | None = None,
    fallback_capability: str | None = None,
    task_scoped: bool = False,
) -> ThinkingPlaybook:
    """Register applicability beside its tool, loading instructions from prompts."""
    contract = get_thinking_playbook_contract(name) if solve_path is None or completion_rule is None else {}
    playbook = ThinkingPlaybook(
        name,
        triggers,
        capabilities,
        solve_path if solve_path is not None else contract["solve_path"],
        completion_rule if completion_rule is not None else contract["completion_rule"],
        prohibited_shortcuts if prohibited_shortcuts is not None else contract.get("prohibited_shortcuts", ()),
        matcher,
        fallback_capability,
        task_scoped,
    )
    _PLAYBOOKS[name] = playbook
    return playbook


def matching_playbooks(task: str, available_capabilities: Iterable[str]) -> list[ThinkingPlaybook]:
    """Return enabled tool playbooks relevant to one research task."""
    return [
        playbook
        for playbook in _PLAYBOOKS.values()
        if playbook.applies_to(task, available_capabilities)
    ]


def scoped_capabilities(task: str, available_capabilities: Iterable[str]) -> set[str]:
    """Apply opt-in, tool-owned task scope before exposing specialist tools."""
    available = set(available_capabilities)
    scoped = set().union(*(set(book.capabilities) for book in _PLAYBOOKS.values() if book.task_scoped))
    applicable = set().union(*(
        set(book.capabilities) for book in _PLAYBOOKS.values()
        if book.task_scoped and book.applies_to(task, available)
    ))
    result = available - (scoped - applicable)
    if (_PUBLIC_MODEL_RESEARCH_RE.search(task)
            and not _EXPLICIT_HOUSEHOLD_RESEARCH_RE.search(task)):
        result -= _HOUSEHOLD_HISTORY_CAPABILITIES
    return result
