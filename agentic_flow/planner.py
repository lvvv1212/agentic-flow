"""Planning module for agentic-flow.

Implements a plan-and-execute pattern where an LLM decomposes a goal into
sub-tasks, executes them with an agent, and dynamically replans on failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentic_flow.llm import LLMClient

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single sub-task within a plan."""

    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    depends_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result[:300] if self.result else "",
            "depends_on": self.depends_on,
        }


@dataclass
class Plan:
    """An ordered list of tasks produced by the planner."""

    goal: str
    tasks: list[Task] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for t in self.tasks)

    @property
    def next_task(self) -> Task | None:
        for t in self.tasks:
            if t.status == TaskStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self._task_by_id(d).status == TaskStatus.COMPLETED
                    for d in t.depends_on
                    if self._task_by_id(d) is not None
                )
                if deps_met:
                    return t
        return None

    def _task_by_id(self, task_id: int) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def summary(self) -> str:
        lines = [f"Plan: {self.goal}", ""]
        for t in self.tasks:
            status_icon = {
                TaskStatus.PENDING: "[ ]",
                TaskStatus.IN_PROGRESS: "[~]",
                TaskStatus.COMPLETED: "[x]",
                TaskStatus.FAILED: "[!]",
                TaskStatus.SKIPPED: "[-]",
            }[t.status]
            deps = f" (depends on: {t.depends_on})" if t.depends_on else ""
            lines.append(f"  {status_icon} {t.id}. {t.description}{deps}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT = """\
You are a planning assistant. Given a goal, decompose it into a clear list of
sequential sub-tasks. Each task should be concrete and actionable.

Respond ONLY with a JSON array of objects, each with:
- "id": integer starting at 1
- "description": what needs to be done
- "depends_on": list of task ids this depends on (empty list if none)

Example:
[
  {"id": 1, "description": "Search for recent papers on GRPO", "depends_on": []},
  {"id": 2, "description": "Summarize key findings", "depends_on": [1]}
]
"""

_REPLAN_SYSTEM_PROMPT = """\
You are a replanning assistant. A task in the current plan has failed.
Review the plan status and produce a REVISED JSON task list that works around
the failure. Keep completed tasks as-is and adjust remaining tasks.

Respond ONLY with a JSON array of task objects (same format as before).
"""


@dataclass
class Planner:
    """Decomposes goals into plans and supports dynamic replanning.

    Parameters
    ----------
    model : str
        LLM model for planning.
    api_key : str | None
        API key.
    base_url : str | None
        Custom API base URL.
    max_replans : int
        Maximum number of replan attempts on failure.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    max_replans: int = 2

    _llm: LLMClient = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._llm = LLMClient(model=self.model, api_key=self.api_key, base_url=self.base_url, temperature=0.3)

    # ------------------------------------------------------------------
    def create_plan(self, goal: str) -> Plan:
        """Ask the LLM to decompose *goal* into a :class:`Plan`."""
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Goal: {goal}"},
        ]
        response = self._llm.chat(messages=messages)
        tasks = self._parse_tasks(response["content"])
        plan = Plan(goal=goal, tasks=tasks)
        logger.info("Created plan with %d tasks for: %s", len(tasks), goal)
        return plan

    # ------------------------------------------------------------------
    def replan(self, plan: Plan, failed_task: Task, error: str) -> Plan:
        """Produce a revised plan after *failed_task* encountered an error."""
        context = (
            f"Original goal: {plan.goal}\n\n"
            f"Current plan status:\n{plan.summary()}\n\n"
            f"Task {failed_task.id} failed with error: {error}\n\n"
            "Produce a revised plan."
        )
        messages = [
            {"role": "system", "content": _REPLAN_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        response = self._llm.chat(messages=messages)
        new_tasks = self._parse_tasks(response["content"])

        # Merge: keep completed tasks, replace the rest
        completed = {t.id: t for t in plan.tasks if t.status == TaskStatus.COMPLETED}
        merged: list[Task] = []
        for nt in new_tasks:
            if nt.id in completed:
                merged.append(completed[nt.id])
            else:
                merged.append(nt)
        plan.tasks = merged
        logger.info("Replanned: now %d tasks", len(merged))
        return plan

    # ------------------------------------------------------------------
    def execute(
        self,
        plan: Plan,
        executor: Any,  # Agent — kept as Any to avoid circular import
    ) -> str:
        """Execute a plan using *executor* (an Agent) for each task.

        Returns the final combined result.
        """
        replans = 0
        results: list[str] = []

        while not plan.is_complete:
            task = plan.next_task
            if task is None:
                # All remaining tasks have unmet dependencies — cannot proceed
                logger.warning("No executable tasks remaining.")
                break

            task.status = TaskStatus.IN_PROGRESS
            logger.info("Executing task %d: %s", task.id, task.description)

            try:
                agent_result = executor.run(task.description)
                task.result = agent_result.output
                task.status = TaskStatus.COMPLETED
                results.append(f"Task {task.id}: {task.result}")
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.result = str(exc)
                logger.error("Task %d failed: %s", task.id, exc)

                if replans < self.max_replans:
                    replans += 1
                    plan = self.replan(plan, task, str(exc))
                else:
                    logger.error("Max replans reached. Stopping.")
                    break

        return "\n\n".join(results) if results else "Plan could not be completed."

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tasks(text: str) -> list[Task]:
        """Extract a list of Task objects from LLM JSON output."""
        # Find JSON array in the response
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            # Fallback: single task
            return [Task(id=1, description=text)]
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return [Task(id=1, description=text)]

        tasks: list[Task] = []
        for item in data:
            tasks.append(
                Task(
                    id=item.get("id", len(tasks) + 1),
                    description=item.get("description", ""),
                    depends_on=item.get("depends_on", []),
                )
            )
        return tasks
