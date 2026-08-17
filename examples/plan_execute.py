"""Example: Plan-and-execute pattern (DeepSeek-configured).

The planner decomposes a complex goal into sub-tasks, then an executor
agent works through them one by one. If a task fails, the planner
dynamically replans.

Configuration (read from environment / .env automatically):
    DEEPSEEK_API_KEY   (required) API key
    DEEPSEEK_MODEL     (optional, default deepseek-chat)
    DEEPSEEK_BASE_URL  (optional, default https://api.deepseek.com/v1)
    # For *real* web research, also set one of:
    TAVILY_API_KEY / SERPAPI_API_KEY   (consumed by tools.web_search)

Usage:
    python examples/plan_execute.py
"""

import os

from agentic_flow import Agent, Planner
from agentic_flow.tools import web_search, python_executor, calculator
from agentic_flow.deepseek import _load_dotenv

# Load DEEPSEEK_* / OPENAI_* from a local .env (if present) into os.environ
# *before* we read them below. (DeepSeekClient does this lazily; the script
# reads credentials directly, so we trigger it up front.)
_load_dotenv()

# Prefer the already-configured DeepSeek credentials; fall back to OpenAI.
_ds_key = os.environ.get("DEEPSEEK_API_KEY")
if _ds_key:
    API_KEY = _ds_key
    BASE_URL = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
    MODEL = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
else:
    API_KEY = os.environ.get("OPENAI_API_KEY")
    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-4o-mini"


def main() -> None:
    # The executor agent that will carry out individual tasks.
    # web_search needs TAVILY_API_KEY / SERPAPI_API_KEY; if those are not set we
    # fall back to local tools so the example still runs end-to-end.
    executor_tools = [python_executor, calculator]
    if os.environ.get("TAVILY_API_KEY") or os.environ.get("SERPAPI_API_KEY"):
        executor_tools.append(web_search)

    executor = Agent(
        name="executor",
        instructions=(
            "You are a capable assistant that completes tasks step by step. "
            "Use your tools when needed. Be thorough and precise."
        ),
        tools=executor_tools,
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        verbose=True,
    )

    # The planner decomposes goals
    planner = Planner(model=MODEL, api_key=API_KEY, base_url=BASE_URL, max_replans=2)

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
