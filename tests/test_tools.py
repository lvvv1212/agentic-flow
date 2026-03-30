"""Tests for the tool system."""

import math
from agentic_flow.tools import tool, Tool, ToolResult, calculator, file_reader, python_executor


class TestToolDecorator:
    def test_basic_decoration(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "Say hello."

    def test_custom_name_and_description(self) -> None:
        @tool(name="my_tool", description="A custom tool.")
        def foo(x: int) -> str:
            return str(x)

        assert foo.name == "my_tool"
        assert foo.description == "A custom tool."

    def test_schema_generation(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        schema = add.parameters
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert schema["properties"]["a"]["type"] == "integer"
        assert set(schema["required"]) == {"a", "b"}

    def test_optional_parameter(self) -> None:
        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search."""
            return query

        assert "query" in search.parameters.get("required", [])
        assert "limit" not in search.parameters.get("required", [])

    def test_tool_invocation(self) -> None:
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply."""
            return a * b

        result = multiply(a=3, b=7)
        assert isinstance(result, ToolResult)
        assert result.output == "21"
        assert not result.error

    def test_tool_error_handling(self) -> None:
        @tool
        def fail(x: int) -> str:
            """Fail."""
            raise ValueError("bad input")

        result = fail(x=1)
        assert result.error
        assert "bad input" in result.output

    def test_openai_schema(self) -> None:
        @tool
        def do_thing(text: str) -> str:
            """Do a thing."""
            return text

        schema = do_thing.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "do_thing"
        assert "parameters" in schema["function"]


class TestBuiltinTools:
    def test_calculator(self) -> None:
        result = calculator(expression="2 + 3 * 4")
        assert result.output == "14"

    def test_calculator_math_functions(self) -> None:
        result = calculator(expression="sqrt(16)")
        assert result.output == "4.0"

    def test_calculator_error(self) -> None:
        result = calculator(expression="import os")
        assert result.error

    def test_python_executor(self) -> None:
        result = python_executor(code="print(2 + 2)")
        assert result.output.strip() == "4"

    def test_python_executor_result_variable(self) -> None:
        result = python_executor(code="result = 42")
        assert result.output.strip() == "42"

    def test_python_executor_error(self) -> None:
        result = python_executor(code="raise ValueError('test')")
        assert "Error" in result.output

    def test_file_reader_not_found(self) -> None:
        result = file_reader(path="/nonexistent/file.txt")
        assert result.error


class TestToolResult:
    def test_str_normal(self) -> None:
        r = ToolResult(output="hello")
        assert str(r) == "hello"

    def test_str_error(self) -> None:
        r = ToolResult(output="oops", error=True)
        assert str(r) == "[ERROR] oops"
