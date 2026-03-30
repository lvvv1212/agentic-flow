"""Tests for orchestration patterns."""

from unittest.mock import patch, MagicMock
from agentic_flow.agent import Agent, AgentResult
from agentic_flow.orchestrator import SequentialPipeline, ParallelExecutor


class TestSequentialPipeline:
    @patch.object(Agent, "run")
    def test_pipeline(self, mock_run: MagicMock) -> None:
        mock_run.return_value = AgentResult(output="processed")

        a1 = Agent(name="a1", instructions="first")
        a2 = Agent(name="a2", instructions="second")
        pipeline = SequentialPipeline(agents=[a1, a2])

        result = pipeline.run("input")
        assert result == "processed"
        assert mock_run.call_count == 2


class TestParallelExecutor:
    @patch.object(Agent, "run")
    def test_parallel_concat(self, mock_run: MagicMock) -> None:
        mock_run.return_value = AgentResult(output="result")

        a1 = Agent(name="a1", instructions="first")
        a2 = Agent(name="a2", instructions="second")
        executor = ParallelExecutor(agents=[a1, a2], combiner="concat")

        result = executor.run("input")
        assert "a1" in result or "a2" in result
        assert "result" in result
