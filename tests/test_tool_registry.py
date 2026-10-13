"""Tests for the tool registry."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_registry import (  # noqa: E402
    ThinkingResult,
    ToolRegistry,
    ToolSchema,
    UnknownToolError,
    thinking_result_error,
)


class TestRegistration:
    def test_foreground_schema_omits_background_argument_contracts(self, mock_tool_registry):
        def lookup(query, window="today"):
            return "evidence"

        mock_tool_registry.register_tool(lookup, name="lookup", description="Long research contract", deep_think_only=True)
        foreground = mock_tool_registry.describe_tools(foreground=True)
        background = mock_tool_registry.describe_tools()
        assert "lookup" in foreground and "through deep_think" in foreground
        assert "Long research contract" not in foreground and "Args:" not in foreground
        assert "Long research contract" in background and "window" in background

    def test_selected_schema_retains_full_signatures_and_respects_availability(self, mock_tool_registry):
        mock_tool_registry.register_tool(lambda query: "", name="search", description="Search")
        mock_tool_registry.register_tool(lambda: "", name="off", available=lambda: False)
        mock_tool_registry.register_tool(lambda: "", name="unrelated")
        selected = mock_tool_registry.describe_tools(names={"search", "off"})
        assert "Args: query" in selected
        assert "off" not in selected and "unrelated" not in selected

    def test_register_simple_function(self, mock_tool_registry):
        @mock_tool_registry.register_tool
        def test_func():
            return "hello"

        assert mock_tool_registry.get_tool("test_func") is not None

    def test_register_with_custom_name(self, mock_tool_registry):
        def my_function():
            return "world"

        mock_tool_registry.register_tool(my_function, name="custom_name")
        assert mock_tool_registry.get_tool("custom_name") is not None

    def test_register_with_aliases(self, mock_tool_registry):
        def greet():
            return "hi"

        mock_tool_registry.register_tool(greet, name="greet", aliases=["hello", "hi_there"])

        assert mock_tool_registry.get_tool("greet") is not None
        assert mock_tool_registry.get_tool("hello") is not None
        assert mock_tool_registry.get_tool("hi_there") is not None

    def test_registers_deep_think_only_tool(self, mock_tool_registry):
        def investigate():
            return "evidence"

        mock_tool_registry.register_tool(investigate, name="investigate", deep_think_only=True)

        assert mock_tool_registry.is_deep_think_only("investigate") is True
        assert mock_tool_registry.is_deep_think_only("unknown") is False

    def test_name_collision_is_skipped_with_warning(self, mock_tool_registry, caplog):
        def first():
            return "first"

        def second():
            return "second"

        mock_tool_registry.register_tool(first, name="dup")
        mock_tool_registry.register_tool(second, name="dup")

        # First registration wins; second is ignored.
        assert mock_tool_registry.execute_tool("dup") == "first"
        assert "collision" in caplog.text.lower()

    def test_alias_collision_is_skipped_with_warning(self, mock_tool_registry, caplog):
        def a():
            return "a"

        def b():
            return "b"

        mock_tool_registry.register_tool(a, name="a", aliases=["shared"])
        mock_tool_registry.register_tool(b, name="b", aliases=["shared"])

        # Alias still resolves to the first registrant.
        assert mock_tool_registry.execute_tool("shared") == "a"
        assert "collision" in caplog.text.lower()


class TestCanonicalName:
    def test_canonical_of_registered_name(self, mock_tool_registry):
        def fn():
            return "x"

        mock_tool_registry.register_tool(fn, name="real")
        assert mock_tool_registry.canonical_name("real") == "real"

    def test_canonical_resolves_alias(self, mock_tool_registry):
        def fn():
            return "x"

        mock_tool_registry.register_tool(fn, name="real", aliases=["nick"])
        assert mock_tool_registry.canonical_name("nick") == "real"

    def test_canonical_unknown_returns_none(self, mock_tool_registry):
        assert mock_tool_registry.canonical_name("ghost") is None


class TestExecution:
    def test_execute_with_kwargs(self, mock_tool_registry):
        def add(a: int, b: int) -> int:
            return a + b

        mock_tool_registry.register_tool(add, name="add")
        assert mock_tool_registry.execute_tool("add", kwargs={"a": 2, "b": 3}) == 5

    def test_execute_with_positional_args(self, mock_tool_registry):
        def multiply(x: int, y: int) -> int:
            return x * y

        mock_tool_registry.register_tool(multiply, name="multiply")
        assert mock_tool_registry.execute_tool("multiply", args=[4, 5]) == 20

    def test_execute_unknown_tool_raises(self, mock_tool_registry):
        with pytest.raises(UnknownToolError):
            mock_tool_registry.execute_tool("nonexistent")

    def test_execute_returns_empty_string_for_none(self, mock_tool_registry):
        def noisy():
            return None

        mock_tool_registry.register_tool(noisy, name="noisy")
        assert mock_tool_registry.execute_tool("noisy") == ""

    def test_execute_propagates_tool_exception(self, mock_tool_registry):
        """A tool that raises propagates the exception to the caller instead of
        being swallowed into "". `handle_action` maps it to None -> ERROR ->
        replan; swallowing it would surface an empty observation and a
        misleading "Done." reply (the tool never actually ran)."""

        def boom():
            raise RuntimeError("nope")

        mock_tool_registry.register_tool(boom, name="boom")
        with pytest.raises(RuntimeError):
            mock_tool_registry.execute_tool("boom")


class TestThinkingResult:
    def test_typed_outcome_accepts_the_standard_evidence_envelope(self):
        result = ThinkingResult(
            "Found one source.",
            evidence={"source": "example"},
            scope="One retrieved source.",
            next_actions=("search_again",),
            artifact={"type": "source"},
        )

        assert thinking_result_error(result) is None

    def test_standard_metadata_survives_arbitrary_adapter_to_fact_cards(self):
        from core.background_jobs import BackgroundJob, JobSnapshot
        from core.thinking_fact_cards import build_fact_cards

        result = ThinkingResult(
            "A synthetic adapter returned one observation.", scope="One synthetic record.",
            evidence={"type": "structured", "records": [{"label": "sample", "value": "12"}]},
            artifact={"type": "structured", "records": [{"label": "sample", "value": "12"}]},
            record_identity={"id": "record-12", "version": "v2"},
            provenance={"source_id": "fixture", "record_path": "/records/0",
                        "publisher_role": "primary_publisher"},
            temporal={"observed_at": "2026-10-01T00:00:00Z", "effective_at": "2026-09-01"},
            qualifications={"unit": "items", "scope": "fixture-only"},
        )
        job = BackgroundJob("synthetic", JobSnapshot("Analyze sample records"))
        artifact_id = job.record_outcome("synthetic_adapter", result.thinking_status, result.scope,
            result.evidence, result.next_actions, result.artifact, result.recovery, metadata={
                "record_identity": result.record_identity, "provenance": result.provenance,
                "temporal": result.temporal, "qualifications": result.qualifications,
            })
        cards = build_fact_cards(job.evidence, job.artifacts, job.snapshot.task)

        assert artifact_id
        assert cards
        assert cards[0]["record_identity"] == {"id": "record-12", "version": "v2"}
        assert cards[0]["context"]["effective_at"] == "2026-09-01"
        assert cards[0]["qualifications"]["unit"] == "items"
        original_key = cards[0]["record_key"]
        assert '"fixture"' in original_key
        job.evidence[0]["metadata"]["record_identity"]["id"] = "record-13"
        changed = build_fact_cards(job.evidence, job.artifacts, job.snapshot.task)
        assert changed[0]["record_key"] != original_key
        job.evidence[0]["metadata"]["record_identity"] = {"version": "v2", "id": "record-12"}
        reordered = build_fact_cards(job.evidence, job.artifacts, job.snapshot.task)
        assert reordered[0]["record_key"] == original_key

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"status": "exhausted", "scope": "One search."}, "unknown status"),
            ({"scope": ""}, "scope must be a non-empty string"),
            ({"scope": "One search.", "evidence": []}, "evidence must be a mapping"),
            ({"scope": "One search.", "next_actions": ["search"]}, "next_actions must be a tuple"),
            ({"scope": "One search.", "artifact": "source"}, "artifact must be a mapping"),
        ],
    )
    def test_typed_outcome_rejects_invalid_contract_fields(self, kwargs, error):
        assert error in thinking_result_error(ThinkingResult("Result", **kwargs))


class TestDescribeTools:
    def test_describe_includes_name_and_description(self, mock_tool_registry):
        def search(query: str, limit: int = 10) -> str:
            return ""

        mock_tool_registry.register_tool(search, name="search", description="Search for items")

        description = mock_tool_registry.describe_tools()
        assert "search" in description
        assert "Search for items" in description
        assert "query" in description
        assert "limit" in description
        assert "optional" in description
        assert "default: 10" in description

    def test_describe_hides_unavailable_tool(self):
        registry = ToolRegistry()
        registry.register_tool(lambda: "", name="hidden", available=lambda: False)

        assert "hidden" not in registry.describe_tools()


class TestSchema:
    def test_schema_records_required_and_default(self, mock_tool_registry):
        def fn(required_arg: str, optional_arg: int = 5):
            pass

        mock_tool_registry.register_tool(fn, name="fn")
        schema: ToolSchema = mock_tool_registry._schemas["fn"]

        params = {p.name: p for p in schema.params}
        assert params["required_arg"].required is True
        assert params["optional_arg"].required is False
        assert params["optional_arg"].default == 5
