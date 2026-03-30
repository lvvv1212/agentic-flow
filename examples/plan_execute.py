"""Example: Plan-and-execute pattern.

The planner decomposes a complex goal into sub-tasks, then an executor
agent works through them one by one. If a task fails, the planner
dynamically replans.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/plan_execute.py
"""

from agentic_flow import Agent, Planner
from agentic_flow.tools import web_search, python_executor, calculator


def main() -> None:
    # The executor agent that will carry out individual tasks
    executor = Agent(
        name="executor",
        instructions=(
            "You are a capable assistant that completes tasks step by step. "
            "Use your tools when needed. Be thorough and precise."
        ),
        tools=[web_search, python_executor, calculator],
        model="gpt-4o-mini",
        verbose=True,
    )

    # The planner decomposes goals
    planner = Planner(model="gpt-4o-mini", max_replans=2)

    goal = (
        "Research the top 3 most popular Python web frameworks in 2024, "
        "compare their GitHub stars and key features, then write a short "
        "recommendation for a startup building a REST API."
    )

    print(f"Goal: {goal}\n")
    print("Creating plan...")
    plan = planner.create_plan(goal)
    print(plan.summary())

    print("\nExecuting plan...\n")
    result = planner.execute(plan, executor)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)
    print("\nPlan status:")
    print(plan.summary())


if __name__ == "__main__":
    main()
