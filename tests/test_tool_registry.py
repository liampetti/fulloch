"""Tests for the tool registry."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_registry import ToolSchema, UnknownToolError  # noqa: E402


class TestRegistration:
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

    def test_execute_returns_empty_string_on_exception(self, mock_tool_registry):
        """Exceptions are logged but not surfaced to the user as text — the
        assistant falls through to its random 'sorry, can you repeat that'
        fallback instead of speaking 'Error executing X: ...' aloud."""
        def boom():
            raise RuntimeError("nope")

        mock_tool_registry.register_tool(boom, name="boom")
        assert mock_tool_registry.execute_tool("boom") == ""


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
