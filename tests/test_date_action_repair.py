"""Bounded repair of model-authored date calculations."""

from unittest.mock import Mock

import pytest

from core.agent_emission import repair_nested_date_actions
from tools.tool_registry import tool_registry
from utils import intents


@pytest.mark.parametrize("args", [
    ['days_between("today", "2026-12-25")'],
    {"expression": 'days_between("today", "2026-12-25")'},
    'days_between("today", "2026-12-25")',
])
def test_repair_preserves_plan_metadata_and_input(args):
    action = {"intent": "calc", "args": args, "depends_on": [0], "step_id": 2}
    emission = {"actions": [action], "delivery": "ignored"}
    result = repair_nested_date_actions(
        emission, registry=tool_registry, intent_services=intents, requires_deep_think=lambda _: False,
    )
    assert result["actions"] == [{
        "intent": "days_between", "args": ["today", "2026-12-25"],
        "depends_on": [0], "step_id": 2,
    }]
    assert emission["actions"][0] == action
    assert result["delivery"] == "ignored"


@pytest.mark.parametrize("disabled", ["calculate", "days_between", "background_calculate", "background_days_between"])
def test_repair_does_not_bypass_availability_or_background_policy(monkeypatch, disabled):
    original = tool_registry.is_available
    monkeypatch.setattr(tool_registry, "is_available", lambda name: name != disabled and original(name))
    emission = {"actions": [{"intent": "calculate", "args": ['days_between("today", "2026-12-25")']}]}
    result = repair_nested_date_actions(
        emission, registry=tool_registry, intent_services=intents,
        requires_deep_think=lambda name: disabled == f"background_{name}",
    )
    assert result is emission


def test_invalid_nested_call_is_not_rewritten_or_executed():
    expression = 'days_between("today", __import__("os"))'
    emission = {"actions": [{"intent": "calculate", "args": [expression]}]}
    registry = Mock(wraps=tool_registry)
    result = repair_nested_date_actions(
        emission, registry=registry, intent_services=intents, requires_deep_think=lambda _: False,
    )
    assert result is emission
    registry.execute_tool.assert_not_called()
    assert intents.handle_action(emission["actions"][0]).startswith("Reactive question:")
