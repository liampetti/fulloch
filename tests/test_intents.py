"""Tests for the unified action dispatcher + replan predicate."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_registry import UnknownToolError, tool_registry  # noqa: E402
from utils import intents  # noqa: E402

# --- parse_agent_emission (tolerant JSON) -------------------------------


class TestParseAgentEmission:
    def test_plain_json(self):
        assert intents.parse_agent_emission('{"reply": "hi"}') == {"reply": "hi"}

    def test_strips_think_block(self):
        raw = '<think>let me see</think>\n{"reply": "hi"}'
        assert intents.parse_agent_emission(raw) == {"reply": "hi"}

    def test_ignores_trailing_stray_close_tag_and_repeats(self):
        # The real failure: a valid object, then </think>, then the same object
        # repeated. We take the first balanced object and ignore the rest.
        obj = '{"actions": [{"intent": "external_information", "args": ["news"]}]}'
        raw = obj + "\n</think>\n\n" + obj + "\n</think>\n"
        out = intents.parse_agent_emission(raw)
        assert out == {"actions": [{"intent": "external_information", "args": ["news"]}]}

    def test_braces_inside_string_dont_miscount(self):
        assert intents.parse_agent_emission('{"reply": "a } b { c"}') == {"reply": "a } b { c"}

    def test_no_object_raises(self):
        with pytest.raises(ValueError):
            intents.parse_agent_emission("just some prose, no json")

    def test_unclosed_object_raises(self):
        # A genuinely truncated object has no balanced match -> caller falls back.
        with pytest.raises(ValueError):
            intents.parse_agent_emission('{"actions": [{"intent": "x"}]')


class TestStripUnfoundedSaveClaim:
    def test_strips_claim_when_no_write(self):
        reply = (
            "The latest news includes Egypt's first-ever win. I've saved this to your daily log."
        )
        out = intents.strip_unfounded_save_claim(reply, note_written=False)
        assert "saved this to your daily log" not in out
        assert "Egypt's first-ever win" in out

    def test_keeps_claim_when_write_happened(self):
        reply = "Done. I've added that to your notes."
        assert intents.strip_unfounded_save_claim(reply, note_written=True) == reply

    def test_various_save_phrasings_stripped(self):
        for claim in [
            "I've noted that in your journal.",
            "Logged it to your to-do list.",
            "I made a note of it in your diary.",
            "That's recorded in your notes now.",
        ]:
            reply = "Here's the answer. " + claim
            out = intents.strip_unfounded_save_claim(reply, note_written=False)
            assert claim not in out and "Here's the answer." in out

    def test_no_claim_unchanged(self):
        reply = "Egypt won their first World Cup match thanks to Salah."
        assert intents.strip_unfounded_save_claim(reply, note_written=False) == reply

    def test_unrelated_save_word_not_stripped(self):
        # "saved a goal" has no note-noun nearby -> left alone.
        reply = "The keeper saved a penalty in the final minute."
        assert intents.strip_unfounded_save_claim(reply, note_written=False) == reply

    def test_never_blanks_whole_reply(self):
        reply = "I've saved this to your notes."
        # Only sentence is the claim -> fall back to the original, don't return "".
        assert intents.strip_unfounded_save_claim(reply, note_written=False) == reply


# --- handle_action ------------------------------------------------------


class TestHandleAction:
    def test_dispatches_known_tool_returns_string(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok"):
            result = intents.handle_action({"intent": "foo", "args": ["x"]})
        assert result == "ok"

    def test_passes_args_to_registry(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok") as mock_exec:
            intents.handle_action({"intent": "foo", "args": [1, "two"]})
        mock_exec.assert_called_once_with("foo", args=[1, "two"], kwargs={})

    def test_missing_args_defaults_empty(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok") as mock_exec:
            intents.handle_action({"intent": "foo"})
        mock_exec.assert_called_once_with("foo", args=[], kwargs={})

    def test_unknown_tool_returns_reactive_sentinel(self):
        # Unknown tools return a Reactive question: sentinel so the agent
        # loop replans with the failure visible in history.
        with patch.object(
            tool_registry,
            "execute_tool",
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
            tool_registry,
            "execute_tool",
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

    def test_dict_args_dispatched_as_kwargs(self):
        # A grammar-less remote model may emit args as a kwargs object instead
        # of the list the grammar forces; route it through as kwargs rather than
        # crashing (the dict indexed as args[0] raised KeyError(0)).
        with patch.object(tool_registry, "execute_tool", return_value="ok") as mock_exec:
            intents.handle_action({"intent": "external_information", "args": {"query": "news"}})
        mock_exec.assert_called_once_with("external_information", args=[], kwargs={"query": "news"})

    def test_scalar_args_wrapped_positional(self):
        with patch.object(tool_registry, "execute_tool", return_value="ok") as mock_exec:
            intents.handle_action({"intent": "foo", "args": "bar"})
        mock_exec.assert_called_once_with("foo", args=["bar"], kwargs={})


class TestCoerceReplyText:
    def test_list_arg(self):
        assert intents.coerce_reply_text(["Done!"]) == "Done!"

    def test_dict_arg(self):
        assert intents.coerce_reply_text({"text": "Saved it."}) == "Saved it."

    def test_bare_string(self):
        assert intents.coerce_reply_text("Okay.") == "Okay."

    def test_empty_returns_none(self):
        assert intents.coerce_reply_text([]) is None
        assert intents.coerce_reply_text("  ") is None
        assert intents.coerce_reply_text(None) is None

    def test_reply_pseudo_intents_membership(self):
        assert "reply" in intents.REPLY_PSEUDO_INTENTS
        assert "say" in intents.REPLY_PSEUDO_INTENTS


class TestCoerceArgs:
    def test_list_stays_positional(self):
        assert intents.coerce_args([1, "two"]) == ([1, "two"], {})

    def test_dict_becomes_kwargs(self):
        assert intents.coerce_args({"query": "x"}) == ([], {"query": "x"})

    def test_none_is_empty(self):
        assert intents.coerce_args(None) == ([], {})

    def test_scalar_wrapped(self):
        assert intents.coerce_args("solo") == (["solo"], {})
        assert intents.coerce_args(5) == ([5], {})


# --- is_registered_tool (hallucinated-tool guard) -----------------------


class TestIsRegisteredTool:
    def test_true_for_loaded_tool(self):
        # Direct registry match — a loaded tool (resolved via canonical_name).
        with patch.object(tool_registry, "canonical_name", return_value="turn_on"):
            assert intents.is_registered_tool("lights_on") is True

    def test_false_for_unloaded_tool(self):
        # canonical_name returns None for a name the registry doesn't know.
        with patch.object(tool_registry, "canonical_name", return_value=None):
            assert intents.is_registered_tool("get_weather_forecast") is False

    def test_false_for_empty_or_none(self):
        # Never even hits the registry; a malformed action is "not registered".
        assert intents.is_registered_tool(None) is False
        assert intents.is_registered_tool("") is False


# --- is_web_search ------------------------------------------------------


class TestIsWebSearch:
    def test_true_for_canonical_web_search_tool(self):
        with patch.object(
            tool_registry,
            "canonical_name",
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


# --- is_lookup ----------------------------------------------------------


class TestIsLookup:
    def test_true_for_lookup_tool(self):
        for canonical in intents.LOOKUP_TOOLS:
            with patch.object(
                tool_registry,
                "canonical_name",
                return_value=canonical,
            ):
                assert intents.is_lookup("any_alias") is True

    def test_entity_history_is_a_lookup(self):
        # The reported bug: "when did X last turn on" read the whole dump aloud.
        assert "get_entity_history" in intents.LOOKUP_TOOLS

    def test_false_for_action_tool(self):
        with patch.object(tool_registry, "canonical_name", return_value="turn_on"):
            assert intents.is_lookup("turn_on") is False

    def test_false_for_unknown_tool(self):
        with patch.object(tool_registry, "canonical_name", return_value=None):
            assert intents.is_lookup("nope") is False

    def test_falsy_intent_short_circuits(self):
        with patch.object(tool_registry, "canonical_name") as m:
            assert intents.is_lookup("") is False
            assert intents.is_lookup(None) is False
        m.assert_not_called()


# --- should_replan ------------------------------------------------------


class TestShouldReplan:
    def test_none_triggers_replan(self):
        assert intents.should_replan(None) is True

    def test_plain_string_does_not_replan(self):
        assert intents.should_replan("Set Lounge to 20 percent") is False
        assert intents.should_replan("") is False

    @pytest.mark.parametrize(
        "prefix",
        [
            "User question:",
            "Thinking question:",
            "Summary question:",
            "Reactive question:",
        ],
    )
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

    @pytest.mark.parametrize(
        "prefix,kind",
        [
            ("User question:", "WEB_SEARCH"),
            ("Thinking question:", "THINKING"),
            ("Summary question:", "SUMMARY"),
            ("Reactive question:", "REACTIVE"),
        ],
    )
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


class TestReactiveToSpeech:
    def test_strips_prefix_keeps_user_observation(self):
        raw = (
            "Reactive question: Couldn't find an entity matching "
            "'light.downstairs_office_lights'. Try a different name or be more specific."
        )
        assert intents.reactive_to_speech(raw) == (
            "Couldn't find an entity matching 'light.downstairs_office_lights'. "
            "Try a different name or be more specific."
        )

    def test_drops_agent_directive_sentences(self):
        assert (
            intents.reactive_to_speech(
                "Reactive question: Could not fetch history for 'x'. Tell the user there was an error."
            )
            == "Could not fetch history for 'x'."
        )
        assert (
            intents.reactive_to_speech(
                "Reactive question: Could not parse start 'foo'. Ask the user to clarify the date."
            )
            == "Could not parse start 'foo'."
        )

    def test_empty_after_stripping_falls_back(self):
        assert (
            intents.reactive_to_speech("Reactive question: Ask the user to clarify.")
            == "Sorry, I couldn't do that."
        )


class TestNoLlmReactive:
    """No-LLM tier: a tool that ran and produced a reactive observation is spoken
    directly; kinds that genuinely need the SLM still fall back to the no-AI phrase."""

    def _host(self, spoken):
        import types

        return types.SimpleNamespace(
            llm_enabled=False,
            _history=[],
            _history_for=lambda session: [],
            _trim_history=lambda: None,
            _emit_agent_event=lambda *a, **k: None,
            _record_spoken=lambda s: spoken.__setitem__("said", s),
            _speak_no_ai_fallback=lambda session, source, satellite_id=None: (
                spoken.__setitem__("said", "NO_AI") or "NO_AI"
            ),
        )

    def test_reactive_observation_spoken_not_no_ai(self, monkeypatch):
        import core.agent_loop as al

        spoken = {}
        loop = al.AgentLoop(self._host(spoken), source="voice")
        monkeypatch.setattr(
            intents,
            "handle_action",
            lambda a: (
                "Reactive question: Couldn't find an entity matching 'light.x'. "
                "Try a different name or be more specific."
            ),
        )
        out = loop._run_without_llm(
            "dim x", {"actions": [{"intent": "ha_set_brightness", "args": ["x", 30]}]}
        )
        assert "Couldn't find an entity" in out and "AI model" not in out
        assert spoken["said"] == out

    def test_web_search_still_falls_back_to_no_ai(self, monkeypatch):
        import core.agent_loop as al

        spoken = {}
        loop = al.AgentLoop(self._host(spoken), source="voice")
        # A WEB_SEARCH-kind result genuinely needs the SLM to summarise → no-AI.
        monkeypatch.setattr(intents, "handle_action", lambda a: "User question: what's the weather")
        out = loop._run_without_llm(
            "weather", {"actions": [{"intent": "external_information", "args": ["x"]}]}
        )
        assert out == "NO_AI"
