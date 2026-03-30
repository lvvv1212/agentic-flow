"""Example: Research agent that searches the web and synthesises information.

Usage:
    export OPENAI_API_KEY=sk-...
    export TAVILY_API_KEY=tvly-...   # optional, for web search
    python examples/research_agent.py
"""

from agentic_flow import Agent, tool
from agentic_flow.tools import web_search, calculator


@tool
def summarize_notes(notes: str) -> str:
    """Condense raw notes into a concise summary.

    Args:
        notes: The raw notes to summarize.
    """
    # In a real implementation this could call an LLM — here we just
    # demonstrate tool composition.
    paragraphs = notes.strip().split("\n\n")
    if len(paragraphs) <= 2:
        return notes
    return "\n\n".join(paragraphs[:2]) + "\n\n[...truncated for brevity]"


def main() -> None:
    agent = Agent(
        name="researcher",
        instructions=(
            "You are an expert research assistant. When asked a question:\n"
            "1. Search the web for relevant information.\n"
            "2. Synthesise the findings into a clear, well-structured answer.\n"
            "3. Cite your sources.\n"
            "If the search tool is unavailable, reason from your own knowledge."
        ),
        tools=[web_search, calculator, summarize_notes],
        model="gpt-4o-mini",
        verbose=True,
    )

    result = agent.run("What are the latest advances in GRPO for LLM training?")
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.output)
    print(f"\n(iterations: {result.iterations}, tool calls: {result.tool_calls_made})")


if __name__ == "__main__":
    main()
