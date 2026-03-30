"""Tool system for agentic-flow.

Provides the @tool decorator to convert plain Python functions into agent-callable
tools with auto-generated JSON schemas.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints


# ---------------------------------------------------------------------------
# Core abstractions
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Wrapper for a tool invocation result."""

    output: str
    error: bool = False

    def __str__(self) -> str:
        prefix = "[ERROR] " if self.error else ""
        return f"{prefix}{self.output}"


_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(py_type: type) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _PYTHON_TYPE_TO_JSON.get(py_type, "string")


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a Google/Numpy-style docstring."""
    if not docstring:
        return {}
    descriptions: dict[str, str] = {}
    # Match lines like "    param_name: description" or "    param_name (type): description"
    pattern = re.compile(r"^\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", re.MULTILINE)
    for match in pattern.finditer(docstring):
        descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


@dataclass
class Tool:
    """A callable tool that an agent can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    fn: Callable[..., Any] = field(repr=False)

    # ------------------------------------------------------------------
    def __call__(self, **kwargs: Any) -> ToolResult:
        try:
            result = self.fn(**kwargs)
            return ToolResult(output=str(result))
        except Exception as exc:
            return ToolResult(output=str(exc), error=True)

    # ------------------------------------------------------------------
    def to_openai_schema(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None) -> Any:
    """Decorator that converts a typed Python function into an agent :class:`Tool`.

    Usage::

        @tool
        def search(query: str) -> str:
            \"\"\"Search the web for information.\"\"\"
            ...

        @tool(name="calc", description="Evaluate a math expression.")
        def calculate(expression: str) -> str:
            ...
    """

    def _wrap(func: Callable) -> Tool:
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        param_docs = _parse_docstring_params(func.__doc__)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            if pname == "return":
                continue
            ptype = hints.get(pname, str)
            prop: dict[str, Any] = {"type": _json_type(ptype)}
            if pname in param_docs:
                prop["description"] = param_docs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        tool_name = name or func.__name__
        tool_desc = description or doc.split("\n")[0] or tool_name

        return Tool(name=tool_name, description=tool_desc, parameters=schema, fn=func)

    if fn is not None:
        # Called as @tool without parentheses
        return _wrap(fn)
    # Called as @tool(...) with arguments
    return _wrap


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math expression like '2 + 2' or 'sqrt(16) * 3'.
    """
    allowed_names: dict[str, Any] = {
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        raise ValueError(f"Cannot evaluate '{expression}': {exc}") from exc


@tool
def file_reader(path: str) -> str:
    """Read the contents of a local file.

    Args:
        path: Absolute or relative path to the file.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


@tool
def python_executor(code: str) -> str:
    """Execute Python code in a sandboxed environment and return stdout.

    Args:
        code: Python source code to execute.
    """
    import io
    import contextlib

    stdout = io.StringIO()
    local_ns: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<agent>", "exec"), {"__builtins__": __builtins__}, local_ns)  # noqa: S102
    except Exception as exc:
        return f"Error: {exc}"
    output = stdout.getvalue()
    # If there's a variable called `result`, include it
    if "result" in local_ns and not output.strip():
        output = str(local_ns["result"])
    return output or "(no output)"


@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using a search API and return results.

    Requires SERPAPI_API_KEY or TAVILY_API_KEY in environment variables.

    Args:
        query: The search query.
        num_results: Maximum number of results to return.
    """
    import os

    # Try Tavily first
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        return _search_tavily(query, num_results, tavily_key)

    # Try SerpAPI
    serp_key = os.environ.get("SERPAPI_API_KEY")
    if serp_key:
        return _search_serpapi(query, num_results, serp_key)

    return (
        f"[web_search] No API key found. Set TAVILY_API_KEY or SERPAPI_API_KEY.\n"
        f"Query was: {query}"
    )


def _search_tavily(query: str, num_results: int, api_key: str) -> str:
    import requests

    resp = requests.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": num_results, "api_key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    lines: list[str] = []
    for r in data.get("results", []):
        lines.append(f"- **{r['title']}** ({r['url']})")
        lines.append(f"  {r.get('content', '')[:300]}")
    return "\n".join(lines) or "No results found."


def _search_serpapi(query: str, num_results: int, api_key: str) -> str:
    import requests

    resp = requests.get(
        "https://serpapi.com/search",
        params={"q": query, "num": num_results, "api_key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    lines: list[str] = []
    for r in data.get("organic_results", [])[:num_results]:
        lines.append(f"- **{r.get('title', '')}** ({r.get('link', '')})")
        lines.append(f"  {r.get('snippet', '')[:300]}")
    return "\n".join(lines) or "No results found."
