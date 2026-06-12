"""Tests for the unified action dispatcher + replan predicate."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_registry import UnknownToolError, tool_registry  # noqa: E402
from utils import intents  # noqa: E402

# --- handle_action ------------------------------------------------------

class TestHandleAction:
    def test_dispatches_known_tool_returns_string(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok"):
            result = intents.handle_action({"intent": "foo", "args": ["x"]})
        assert result == "ok"

    def test_passes_args_to_registry(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok") as mock_exec:
            intents.handle_action({"intent": "foo", "args": [1, "two"]})
        mock_exec.assert_called_once_with("foo", args=[1, "two"])

    def test_missing_args_defaults_empty(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok") as mock_exec:
            intents.handle_action({"intent": "foo"})
        mock_exec.assert_called_once_with("foo", args=[])

    def test_unknown_tool_returns_reactive_sentinel(self):
        # Unknown tools return a Reactive question: sentinel so the agent
        # loop replans with the failure visible in history.
        with patch.object(
            tool_registry, "execute_tool",
            side_effect=UnknownToolError("nope"),
        ):
            result = intents.handle_action({"intent": "nope", "args": []})
        assert result is not None
        assert result.startswith("Reactive question:")
        assert "'nope'" in result
        # Sanity: should_replan picks it up.
        assert intents.should_replan(result) is True

    def test_tool_exception_returns_none(self):
        with patch.object(
            tool_registry, "execute_tool",
            side_effect=RuntimeError("boom"),
        ):
            result = intents.handle_action({"intent": "foo", "args": []})
        assert result is None

    def test_missing_intent_key_returns_none(self):
        result = intents.handle_action({"args": []})
        assert result is None

    def test_non_dict_returns_none(self):
        assert intents.handle_action("not a dict") is None
        assert intents.handle_action(None) is None


# --- is_web_search ------------------------------------------------------

class TestIsWebSearch:
    def test_true_for_canonical_web_search_tool(self):
        with patch.object(
            tool_registry, "canonical_name",
            return_value=intents.WEB_SEARCH_TOOL,
        ):
            assert intents.is_web_search("web_search") is True

    def test_false_for_other_tool(self):
        with patch.object(tool_registry, "canonical_name", return_value="turn_on"):
            assert intents.is_web_search("turn_on") is False

    def test_false_for_unknown_tool(self):
        with patch.object(tool_registry, "canonical_name", return_value=None):
            assert intents.is_web_search("nope") is False

    def test_falsy_intent_short_circuits(self):
        # No registry lookup needed for empty / None intent names.
        with patch.object(tool_registry, "canonical_name") as m:
            assert intents.is_web_search("") is False
            assert intents.is_web_search(None) is False
        m.assert_not_called()


# --- should_replan ------------------------------------------------------

class TestShouldReplan:
    def test_none_triggers_replan(self):
        assert intents.should_replan(None) is True

    def test_plain_string_does_not_replan(self):
        assert intents.should_replan("Set Lounge to 20 percent") is False
        assert intents.should_replan("") is False

    @pytest.mark.parametrize("prefix", [
        "User question:",
        "Thinking question:",
        "Summary question:",
        "Reactive question:",
    ])
    def test_sentinel_prefix_triggers_replan(self, prefix):
        assert intents.should_replan(f"{prefix} some payload") is True

    def test_sentinel_with_leading_whitespace_still_triggers(self):
        assert intents.should_replan("   User question: payload") is True

    def test_sentinel_must_be_at_start(self):
        # A sentinel appearing mid-string is not a routing signal.
        assert intents.should_replan("note says 'User question: x'") is False

    def test_non_string_non_none_does_not_replan(self):
        assert intents.should_replan(42) is False
        assert intents.should_replan([]) is False


# --- classify_step (typed boundary) -------------------------------------

class TestClassifyStep:
    def test_none_is_error_and_replans(self):
        step = intents.classify_step(None)
        assert step.kind is intents.StepKind.ERROR
        assert step.should_replan is True
        assert step.in_output is False
        assert step.text == "<error>"

    def test_plain_string_is_normal_and_spoken(self):
        step = intents.classify_step("Set Lounge to 20 percent")
        assert step.kind is intents.StepKind.NORMAL
        assert step.should_replan is False
        assert step.in_output is True
        assert step.text == "Set Lounge to 20 percent"

    def test_non_string_is_normal_but_not_spoken(self):
        # Non-str, non-None tool output: kept in history as str(), not spoken,
        # and does not replan.
        step = intents.classify_step(42)
        assert step.kind is intents.StepKind.NORMAL
        assert step.in_output is False
        assert step.should_replan is False
        assert step.text == "42"

    @pytest.mark.parametrize("prefix,kind", [
        ("User question:", "WEB_SEARCH"),
        ("Thinking question:", "THINKING"),
        ("Summary question:", "SUMMARY"),
        ("Reactive question:", "REACTIVE"),
    ])
    def test_sentinel_prefix_maps_to_kind(self, prefix, kind):
        step = intents.classify_step(f"{prefix} payload")
        assert step.kind is intents.StepKind[kind]
        assert step.should_replan is True

    def test_leading_whitespace_still_matches(self):
        step = intents.classify_step("   Thinking question:\nwhy is the sky blue")
        assert step.kind is intents.StepKind.THINKING

    def test_sentinel_must_be_at_start(self):
        # A note whose body merely contains a sentinel mid-string must NOT route.
        step = intents.classify_step("note says 'User question: x'")
        assert step.kind is intents.StepKind.NORMAL
        assert step.should_replan is False

    def test_should_replan_delegates_to_classify(self):
        # The legacy raw-string predicate is now a thin wrapper.
        assert intents.should_replan("Reactive question: oops") is True
        assert intents.should_replan("all good") is False


# --- module surface -----------------------------------------------------

class TestModuleSurface:
    def test_describe_tools_proxies_to_registry(self):
        with patch.object(tool_registry, "describe_tools", return_value="<list>") as m:
            assert intents.describe_tools() == "<list>"
        m.assert_called_once()

    def test_max_agent_calls_constant_set(self):
        # Sanity check that the loop cap is a positive int.
        assert isinstance(intents.MAX_AGENT_CALLS_PER_TURN, int)
        assert intents.MAX_AGENT_CALLS_PER_TURN >= 2


class TestAgentGrammarParses:
    """The agent grammar file must parse via llama.cpp. Multi-line `|`
    alternations at the top level break the parser silently — this test
    catches that on every test run."""

    def test_grammar_file_loads(self):
        # Needs the real llama.cpp grammar parser — skip when it's been stubbed
        # (CI without the GPU stack). The grammar still gets validated for real
        # on any machine with llama_cpp installed.
        from tests.conftest import STUBBED_MODULES
        if "llama_cpp" in STUBBED_MODULES:
            pytest.skip("requires real llama_cpp (grammar parser)")

        from llama_cpp import LlamaGrammar

        from core import slm
        # Will raise on a parse error.
        LlamaGrammar.from_file(slm.GRAMMAR_FILE)
