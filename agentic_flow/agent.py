"""Core Agent implementation for agentic-flow.

Implements a ReAct-style (Thought -> Action -> Observation) reasoning loop
that works with any OpenAI-compatible API via :class:`LLMClient`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generator

from agentic_flow.llm import LLMClient
from agentic_flow.memory import MemoryManager, ShortTermMemory, LongTermMemory
from agentic_flow.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """The final output of an agent run."""

    output: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tool_calls_made: int = 0


@dataclass
class Agent:
    """A ReAct-style AI agent with tool use, memory, and streaming.

    Parameters
    ----------
    name : str
        Human-readable agent name.
    instructions : str
        System prompt that defines the agent's persona and behaviour.
    tools : list[Tool]
        Tools available to the agent.
    model : str
        LLM model identifier.
    api_key : str | None
        API key (falls back to ``OPENAI_API_KEY``).
    base_url : str | None
        Custom base URL for OpenAI-compatible endpoints.
    temperature : float
        Sampling temperature.
    max_iterations : int
        Safety limit on reasoning loop iterations.
    max_tokens : int | None
        Maximum tokens per LLM completion.
    memory : MemoryManager | None
        If *None* a fresh :class:`MemoryManager` is created automatically.
    verbose : bool
        When *True*, print each reasoning step to stdout.
    """

    name: str = "agent"
    instructions: str = "You are a helpful AI assistant."
    tools: list[Tool] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_iterations: int = 10
    max_tokens: int | None = None
    memory: MemoryManager | None = None
    verbose: bool = False

    # internal
    _llm: LLMClient = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._llm = LLMClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.memory is None:
            self.memory = MemoryManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> AgentResult:
        """Run the agent on *prompt* and return the final result.

        The agent follows a ReAct loop:

        1. **Thought** — the LLM decides what to do next.
        2. **Action** — the LLM invokes a tool (if needed).
        3. **Observation** — the tool result is fed back.
        4. Repeat until the LLM produces a final answer or the iteration
           limit is reached.
        """
        assert self.memory is not None
        # Build initial messages
        messages = self._build_messages(prompt)

        tool_schemas = [t.to_openai_schema() for t in self.tools] if self.tools else None
        tool_map = {t.name: t for t in self.tools}

        iterations = 0
        total_tool_calls = 0

        while iterations < self.max_iterations:
            iterations += 1

            response = self._llm.chat(messages=messages, tools=tool_schemas)
            content = response["content"]
            tool_calls = response.get("tool_calls")

            if self.verbose and content:
                print(f"[{self.name}] {content}")

            # No tool calls — we have a final answer
            if not tool_calls:
                self.memory.add_message("assistant", content)
                return AgentResult(
                    output=content,
                    messages=messages + [{"role": "assistant", "content": content}],
                    iterations=iterations,
                    tool_calls_made=total_tool_calls,
                )

            # Process tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                if self.verbose:
                    print(f"[{self.name}] -> {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

                tool_obj = tool_map.get(fn_name)
                if tool_obj is None:
                    result = ToolResult(output=f"Unknown tool: {fn_name}", error=True)
                else:
                    result = tool_obj(**fn_args)

                if self.verbose:
                    preview = str(result)[:200]
                    print(f"[{self.name}] <- {preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })
                total_tool_calls += 1

        # Exhausted iterations — ask for a summary
        messages.append({
            "role": "user",
            "content": "You have reached the maximum number of iterations. Please provide your best answer now.",
        })
        response = self._llm.chat(messages=messages)
        final = response["content"]
        self.memory.add_message("assistant", final)
        return AgentResult(
            output=final,
            messages=messages + [{"role": "assistant", "content": final}],
            iterations=iterations,
            tool_calls_made=total_tool_calls,
        )

    # ------------------------------------------------------------------
    def run_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream the agent's final response token-by-token.

        This is a simplified streaming mode — the agent first resolves all
        tool calls in a non-streaming inner loop, then streams the final
        answer.
        """
        assert self.memory is not None
        messages = self._build_messages(prompt)
        tool_schemas = [t.to_openai_schema() for t in self.tools] if self.tools else None
        tool_map = {t.name: t for t in self.tools}

        for _ in range(self.max_iterations):
            response = self._llm.chat(messages=messages, tools=tool_schemas)
            tool_calls = response.get("tool_calls")

            if not tool_calls:
                # Stream the final answer
                yield from self._llm.chat_stream(messages=messages)
                return

            # Process tool calls (non-streaming)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response["content"] or "",
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                tool_obj = tool_map.get(fn_name)
                result = tool_obj(**fn_args) if tool_obj else ToolResult(output=f"Unknown tool: {fn_name}", error=True)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})

        # Fallback: stream a summary
        messages.append({"role": "user", "content": "Provide your best answer now."})
        yield from self._llm.chat_stream(messages=messages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str) -> list[dict[str, Any]]:
        """Build the message list for the LLM from memory + new prompt."""
        assert self.memory is not None
        messages: list[dict[str, Any]] = []

        # System message
        messages.append({"role": "system", "content": self.instructions})

        # Conversation history (excluding system message already added)
        for msg in self.memory.get_messages():
            if msg.get("role") != "system":
                messages.append(msg)

        # Optionally inject relevant long-term memories
        relevant = self.memory.retrieve_relevant(prompt, top_k=3)
        if relevant:
            context_parts = [f"- {r['text'][:300]}" for r in relevant if r["score"] > 0.3]
            if context_parts:
                context = "Relevant context from memory:\n" + "\n".join(context_parts)
                messages.append({"role": "system", "content": context})

        # Current user message
        messages.append({"role": "user", "content": prompt})
        self.memory.add_message("user", prompt)

        return messages
