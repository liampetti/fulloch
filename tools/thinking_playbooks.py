"""Tool-owned research contracts for the generic background thinking worker."""

import re
from dataclasses import dataclass
from typing import Callable, Iterable


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
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(self.solve_path, 1))
        shortcuts = "\n".join(f"- {item}" for item in self.prohibited_shortcuts)
        return (
            f"[{self.name}]\nStandard solve path:\n{steps}\n"
            f"Completion rule: {self.completion_rule}"
            + (f"\nDo not:\n{shortcuts}" if shortcuts else "")
        )


_PLAYBOOKS: dict[str, ThinkingPlaybook] = {}


def thinking_playbook(
    *,
    name: str,
    triggers: tuple[str, ...],
    capabilities: tuple[str, ...],
    solve_path: tuple[str, ...],
    completion_rule: str,
    prohibited_shortcuts: tuple[str, ...] = (),
    matcher: Callable[[str], bool] | None = None,
    fallback_capability: str | None = None,
) -> ThinkingPlaybook:
    """Register a playbook beside the tool module that owns its policy."""
    playbook = ThinkingPlaybook(
        name,
        triggers,
        capabilities,
        solve_path,
        completion_rule,
        prohibited_shortcuts,
        matcher,
        fallback_capability,
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
