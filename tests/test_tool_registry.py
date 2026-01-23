"""
Tests for the tool registry module.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_registry import (
    ToolRegistry,
    tool,
    ParameterType,
    ParameterSchema,
    FunctionSchema,
)


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_simple_function(self, mock_tool_registry):
        """Test registering a simple function."""

        @mock_tool_registry.register_tool
        def test_func():
            return "hello"

        assert "test_func" in mock_tool_registry._tools
        assert mock_tool_registry.get_tool("test_func") is not None

    def test_register_with_custom_name(self, mock_tool_registry):
        """Test registering with a custom name."""

        def my_function():
            return "world"

        mock_tool_registry.register_tool(my_function, name="custom_name")

        assert "custom_name" in mock_tool_registry._tools
        assert mock_tool_registry.get_tool("custom_name") is not None

    def test_register_with_aliases(self, mock_tool_registry):
        """Test registering with aliases."""

        def greet():
            return "hi"

        mock_tool_registry.register_tool(
            greet,
            name="greet",
            aliases=["hello", "hi"]
        )

        assert mock_tool_registry.get_tool("greet") is not None
        assert mock_tool_registry.get_tool("hello") is not None
        assert mock_tool_registry.get_tool("hi") is not None

    def test_execute_tool(self, mock_tool_registry):
        """Test executing a registered tool."""

        def add(a: int, b: int) -> int:
            return a + b

        mock_tool_registry.register_tool(add, name="add")
        result = mock_tool_registry.execute_tool("add", kwargs={"a": 2, "b": 3})

        assert result == 5

    def test_execute_tool_with_args(self, mock_tool_registry):
        """Test executing a tool with positional args."""

        def multiply(x: int, y: int) -> int:
            return x * y

        mock_tool_registry.register_tool(multiply, name="multiply")
        result = mock_tool_registry.execute_tool("multiply", args=[4, 5])

        assert result == 20

    def test_execute_unknown_tool(self, mock_tool_registry):
        """Test executing an unknown tool raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            mock_tool_registry.execute_tool("nonexistent")

    def test_get_schema(self, mock_tool_registry):
        """Test getting a function schema."""

        def my_func(name: str, count: int = 1) -> str:
            return f"{name} x {count}"

        mock_tool_registry.register_tool(
            my_func,
            name="my_func",
            description="Test function"
        )

        schema = mock_tool_registry.get_schema("my_func")

        assert schema is not None
        assert schema.name == "my_func"
        assert schema.description == "Test function"
        assert len(schema.parameters) == 2

    def test_get_all_schemas(self, mock_tool_registry):
        """Test getting all schemas."""

        def func1():
            pass

        def func2():
            pass

        mock_tool_registry.register_tool(func1, name="func1")
        mock_tool_registry.register_tool(func2, name="func2")

        schemas = mock_tool_registry.get_all_schemas()
        assert len(schemas) == 2

    def test_to_openai_schema(self, mock_tool_registry):
        """Test converting to OpenAI function calling format."""

        def search(query: str, limit: int = 10) -> str:
            return f"Results for {query}"

        mock_tool_registry.register_tool(
            search,
            name="search",
            description="Search for items"
        )

        openai_schemas = mock_tool_registry.to_openai_schema()

        assert len(openai_schemas) == 1
        schema = openai_schemas[0]
        assert schema["name"] == "search"
        assert schema["description"] == "Search for items"
        assert "parameters" in schema
        assert "query" in schema["parameters"]["properties"]

    def test_parameter_type_detection(self, mock_tool_registry):
        """Test that parameter types are correctly detected."""

        def typed_func(
            text: str,
            number: int,
            decimal: float,
            flag: bool,
            items: list
        ):
            pass

        mock_tool_registry.register_tool(typed_func, name="typed_func")
        schema = mock_tool_registry.get_schema("typed_func")

        param_types = {p.name: p.type for p in schema.parameters}
        assert param_types["text"] == ParameterType.STRING
        assert param_types["number"] == ParameterType.INTEGER
        assert param_types["decimal"] == ParameterType.FLOAT
        assert param_types["flag"] == ParameterType.BOOLEAN
        assert param_types["items"] == ParameterType.ARRAY


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""
        registry = ToolRegistry()

        # Create a local decorator using the registry
        def local_tool(**kwargs):
            def decorator(func):
                return registry.register_tool(func, **kwargs)
            return decorator

        @local_tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert registry.get_tool("greet") is not None
        result = registry.execute_tool("greet", kwargs={"name": "World"})
        assert result == "Hello, World!"

    def test_decorator_with_aliases(self):
        """Test decorator with aliases."""
        registry = ToolRegistry()

        def local_tool(**kwargs):
            def decorator(func):
                return registry.register_tool(func, **kwargs)
            return decorator

        @local_tool(
            name="toggle_light",
            description="Toggle a light",
            aliases=["light", "switch"]
        )
        def toggle_light(room: str) -> str:
            return f"Toggled light in {room}"

        assert registry.get_tool("toggle_light") is not None
        assert registry.get_tool("light") is not None
        assert registry.get_tool("switch") is not None


class TestParameterSchema:
    """Tests for ParameterSchema dataclass."""

    def test_create_parameter(self):
        param = ParameterSchema(
            name="query",
            type=ParameterType.STRING,
            description="Search query",
            required=True
        )

        assert param.name == "query"
        assert param.type == ParameterType.STRING
        assert param.required is True

    def test_optional_parameter(self):
        param = ParameterSchema(
            name="limit",
            type=ParameterType.INTEGER,
            description="Max results",
            required=False,
            default=10
        )

        assert param.required is False
        assert param.default == 10


class TestFunctionSchema:
    """Tests for FunctionSchema dataclass."""

    def test_create_schema(self):
        params = [
            ParameterSchema(
                name="text",
                type=ParameterType.STRING,
                description="Input text",
                required=True
            )
        ]

        schema = FunctionSchema(
            name="process",
            description="Process text",
            parameters=params,
            returns="string"
        )

        assert schema.name == "process"
        assert len(schema.parameters) == 1
        assert schema.returns == "string"
