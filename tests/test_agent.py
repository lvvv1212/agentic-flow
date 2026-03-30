"""Tests for the Agent class.

These tests mock the LLM to avoid requiring an API key.
"""

from unittest.mock import patch, MagicMock
from agentic_flow.agent import Agent, AgentResult
from agentic_flow.tools import tool, Tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestAgent:
    def test_creation(self) -> None:
        agent = Agent(name="test", instructions="Be helpful.")
        assert agent.name == "test"
        assert agent.memory is not None

    def test_creation_with_tools(self) -> None:
        agent = Agent(name="test", tools=[add])
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "add"

    @patch("agentic_flow.agent.LLMClient")
    def test_run_simple(self, mock_llm_cls: MagicMock) -> None:
        """Agent returns a direct answer when no tool calls are made."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "content": "The answer is 42.",
            "tool_calls": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        agent = Agent(name="simple", instructions="Answer questions.")
        agent._llm = mock_llm

        result = agent.run("What is the meaning of life?")
        assert isinstance(result, AgentResult)
        assert result.output == "The answer is 42."
        assert result.tool_calls_made == 0

    @patch("agentic_flow.agent.LLMClient")
    def test_run_with_tool_call(self, mock_llm_cls: MagicMock) -> None:
        """Agent invokes a tool and then returns the final answer."""
        mock_llm = MagicMock()

        # First call: LLM wants to use the add tool
        # Second call: LLM provides the final answer
        mock_llm.chat.side_effect = [
            {
                "content": "Let me calculate.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 3, "b": 4}',
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
            {
                "content": "The sum of 3 and 4 is 7.",
                "tool_calls": None,
                "usage": {"prompt_tokens": 30, "completion_tokens": 10},
            },
        ]

        agent = Agent(name="calc", instructions="Calculate.", tools=[add])
        agent._llm = mock_llm

        result = agent.run("What is 3 + 4?")
        assert result.output == "The sum of 3 and 4 is 7."
        assert result.tool_calls_made == 1
        assert result.iterations == 2

    @patch("agentic_flow.agent.LLMClient")
    def test_max_iterations(self, mock_llm_cls: MagicMock) -> None:
        """Agent stops after max_iterations."""
        mock_llm = MagicMock()

        # Always return a tool call to force iteration
        tool_response = {
            "content": "Thinking...",
            "tool_calls": [
                {
                    "id": "call_x",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        final_response = {
            "content": "I give up, the answer is 2.",
            "tool_calls": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        # 3 iterations of tool calls + 1 final forced answer
        mock_llm.chat.side_effect = [tool_response, tool_response, tool_response, final_response]

        agent = Agent(name="loop", instructions="Loop.", tools=[add], max_iterations=3)
        agent._llm = mock_llm

        result = agent.run("Loop forever")
        assert result.iterations == 3
        assert "2" in result.output
