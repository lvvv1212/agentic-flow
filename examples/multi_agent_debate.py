"""Example: Multi-agent debate on a topic.

Three agents with different perspectives debate, and a judge synthesises
the final answer.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/multi_agent_debate.py
"""

from agentic_flow import Agent, DebateOrchestrator


def main() -> None:
    optimist = Agent(
        name="Optimist",
        instructions=(
            "You are an optimistic technology analyst. You focus on the benefits, "
            "opportunities, and positive potential of new technologies. Support your "
            "arguments with examples and evidence."
        ),
        model="gpt-4o-mini",
    )

    skeptic = Agent(
        name="Skeptic",
        instructions=(
            "You are a cautious technology critic. You focus on risks, limitations, "
            "and potential downsides. You are not anti-technology, but you insist on "
            "rigorous evaluation. Support your arguments with evidence."
        ),
        model="gpt-4o-mini",
    )

    pragmatist = Agent(
        name="Pragmatist",
        instructions=(
            "You are a pragmatic technology strategist. You balance benefits and "
            "risks, focusing on practical implementation and real-world constraints. "
            "You seek middle-ground positions backed by evidence."
        ),
        model="gpt-4o-mini",
    )

    judge = Agent(
        name="Judge",
        instructions=(
            "You are an impartial judge. Synthesise the debate arguments into "
            "a balanced, well-structured final answer. Acknowledge the strongest "
            "points from each side."
        ),
        model="gpt-4o-mini",
    )

    debate = DebateOrchestrator(
        agents=[optimist, skeptic, pragmatist],
        judge=judge,
        rounds=2,
        verbose=True,
    )

    topic = "Should AI agents be given the ability to autonomously execute code in production environments?"
    print(f"Debate topic: {topic}\n")

    final_answer = debate.run(topic)

    print("\n" + "=" * 60)
    print("JUDGE'S VERDICT")
    print("=" * 60)
    print(final_answer)


if __name__ == "__main__":
    main()
