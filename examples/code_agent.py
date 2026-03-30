"""Example: Code agent that writes and executes Python code.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/code_agent.py
"""

from agentic_flow import Agent
from agentic_flow.tools import python_executor, calculator, file_reader


def main() -> None:
    agent = Agent(
        name="coder",
        instructions=(
            "You are an expert Python programmer. When asked to solve a problem:\n"
            "1. Think about the approach step by step.\n"
            "2. Write Python code to solve it.\n"
            "3. Execute the code using the python_executor tool.\n"
            "4. Verify the output and explain the result.\n\n"
            "Always write clean, well-commented code."
        ),
        tools=[python_executor, calculator, file_reader],
        model="gpt-4o-mini",
        verbose=True,
    )

    result = agent.run(
        "Write a Python function that finds the longest palindromic substring "
        "in a given string. Test it with the input 'babad' and 'cbbd'."
    )
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.output)


if __name__ == "__main__":
    main()
