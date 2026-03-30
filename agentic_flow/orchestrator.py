"""Multi-agent orchestration patterns for agentic-flow.

Provides several orchestration strategies:

- **SequentialPipeline** — agents run one after another; each receives the
  previous agent's output.
- **ParallelExecutor** — agents run concurrently on the same input.
- **DebateOrchestrator** — multiple agents discuss/debate and a judge
  synthesises the final answer.
- **SupervisorOrchestrator** — a supervisor agent delegates sub-tasks to
  worker agents.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from agentic_flow.agent import Agent, AgentResult
from agentic_flow.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Orchestrator(ABC):
    """Abstract base for orchestration strategies."""

    @abstractmethod
    def run(self, prompt: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Sequential pipeline
# ---------------------------------------------------------------------------

@dataclass
class SequentialPipeline(Orchestrator):
    """Run agents in sequence: output of agent *N* becomes input for agent *N+1*.

    Parameters
    ----------
    agents : list[Agent]
        Ordered list of agents.
    verbose : bool
        Print intermediate outputs.
    """

    agents: list[Agent] = field(default_factory=list)
    verbose: bool = False

    def run(self, prompt: str) -> str:
        current_input = prompt
        for i, agent in enumerate(self.agents):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Pipeline step {i+1}/{len(self.agents)}: {agent.name}")
                print(f"{'='*60}")
            result = agent.run(current_input)
            current_input = result.output
            if self.verbose:
                print(f"[{agent.name}] Output: {current_input[:200]}...")
        return current_input


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

@dataclass
class ParallelExecutor(Orchestrator):
    """Run agents in parallel on the same input, then combine results.

    Parameters
    ----------
    agents : list[Agent]
        Agents to run concurrently.
    combiner : str
        How to combine results: ``"concat"`` (default) or ``"vote"``.
    max_workers : int
        Thread pool size.
    """

    agents: list[Agent] = field(default_factory=list)
    combiner: str = "concat"
    max_workers: int = 4

    def run(self, prompt: str) -> str:
        results: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(agent.run, prompt): agent.name for agent in self.agents}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res: AgentResult = future.result()
                    results[name] = res.output
                except Exception as exc:
                    results[name] = f"[error] {exc}"

        if self.combiner == "vote":
            return self._majority_vote(results)
        return self._concat(results)

    @staticmethod
    def _concat(results: dict[str, str]) -> str:
        parts = [f"## {name}\n{output}" for name, output in results.items()]
        return "\n\n".join(parts)

    @staticmethod
    def _majority_vote(results: dict[str, str]) -> str:
        from collections import Counter

        counts = Counter(results.values())
        winner, _ = counts.most_common(1)[0]
        return winner


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

@dataclass
class DebateOrchestrator(Orchestrator):
    """Multiple agents debate a topic over several rounds.

    A judge agent synthesises the final answer from the debate transcript.

    Parameters
    ----------
    agents : list[Agent]
        Debating agents.
    judge : Agent | None
        If *None*, a simple LLM call is used as the judge.
    rounds : int
        Number of debate rounds.
    model : str
        Model for the default judge.
    verbose : bool
        Print the debate transcript.
    """

    agents: list[Agent] = field(default_factory=list)
    judge: Agent | None = None
    rounds: int = 2
    model: str = "gpt-4o-mini"
    verbose: bool = False

    def run(self, prompt: str) -> str:
        transcript: list[str] = []

        for round_num in range(1, self.rounds + 1):
            if self.verbose:
                print(f"\n--- Debate Round {round_num} ---")

            for agent in self.agents:
                # Each agent sees the original prompt + transcript so far
                debate_context = (
                    f"Topic: {prompt}\n\n"
                    f"Debate transcript so far:\n{''.join(transcript) or '(none yet)'}\n\n"
                    f"Please provide your perspective."
                )
                result = agent.run(debate_context)
                entry = f"\n[{agent.name} — Round {round_num}]\n{result.output}\n"
                transcript.append(entry)

                if self.verbose:
                    print(entry)

        # Judge synthesises
        full_transcript = "".join(transcript)
        if self.judge:
            judge_prompt = (
                f"You are the judge. Here is a debate on: {prompt}\n\n"
                f"{full_transcript}\n\n"
                f"Synthesise a balanced, well-reasoned final answer."
            )
            judge_result = self.judge.run(judge_prompt)
            return judge_result.output

        # Default judge via raw LLM
        llm = LLMClient(model=self.model)
        judge_messages = [
            {
                "role": "system",
                "content": (
                    "You are an impartial judge. Synthesise the arguments from the "
                    "debate into a single, balanced, well-reasoned answer."
                ),
            },
            {
                "role": "user",
                "content": f"Debate topic: {prompt}\n\nTranscript:\n{full_transcript}",
            },
        ]
        response = llm.chat(messages=judge_messages)
        return response["content"]


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

@dataclass
class SupervisorOrchestrator(Orchestrator):
    """A supervisor agent delegates sub-tasks to worker agents.

    The supervisor receives a task, decides which worker to call (and with
    what input), collects results, and produces a final answer.

    Parameters
    ----------
    supervisor : Agent
        The coordinating agent.
    workers : dict[str, Agent]
        Named worker agents the supervisor can delegate to.
    max_rounds : int
        Safety limit on delegation rounds.
    verbose : bool
        Print delegation steps.
    """

    supervisor: Agent = field(default_factory=Agent)
    workers: dict[str, Agent] = field(default_factory=dict)
    max_rounds: int = 5
    verbose: bool = False

    def __post_init__(self) -> None:
        # Build the supervisor's system prompt to include worker descriptions
        worker_info = "\n".join(
            f"- **{name}**: {w.instructions[:120]}" for name, w in self.workers.items()
        )
        delegation_instructions = (
            f"{self.supervisor.instructions}\n\n"
            f"You have the following worker agents available:\n{worker_info}\n\n"
            f"To delegate a task, respond with JSON:\n"
            f'{{"delegate_to": "<worker_name>", "task": "<task description>"}}\n\n'
            f"When you have the final answer, respond normally without JSON delegation."
        )
        self.supervisor.instructions = delegation_instructions

    def run(self, prompt: str) -> str:
        context = prompt
        for round_num in range(self.max_rounds):
            result = self.supervisor.run(context)
            output = result.output.strip()

            # Try to parse delegation JSON
            delegation = self._parse_delegation(output)
            if delegation is None:
                # Final answer
                return output

            worker_name = delegation["delegate_to"]
            task = delegation["task"]

            if self.verbose:
                print(f"[supervisor] Delegating to '{worker_name}': {task[:100]}")

            worker = self.workers.get(worker_name)
            if worker is None:
                context = (
                    f"Worker '{worker_name}' does not exist. "
                    f"Available workers: {list(self.workers.keys())}. "
                    f"Please try again."
                )
                continue

            worker_result = worker.run(task)
            context = (
                f"Original task: {prompt}\n\n"
                f"Worker '{worker_name}' returned:\n{worker_result.output}\n\n"
                f"Continue coordinating or provide the final answer."
            )

            if self.verbose:
                print(f"[{worker_name}] Result: {worker_result.output[:200]}...")

        return f"Supervisor could not complete the task within {self.max_rounds} rounds."

    @staticmethod
    def _parse_delegation(text: str) -> dict[str, str] | None:
        """Try to extract a delegation JSON from the supervisor's response."""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            if "delegate_to" in data and "task" in data:
                return data
        except (ValueError, json.JSONDecodeError):
            pass
        return None
