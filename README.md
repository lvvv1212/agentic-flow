# agentic-flow

A lightweight, modular framework for building AI agents with **tool use**, **planning**, and **multi-agent orchestration** — in pure Python.

> Build powerful AI agents in 10 lines of code.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **Tool use** with a simple `@tool` decorator — auto-generates JSON schemas from type hints
- **Built-in memory** — short-term (conversation) + long-term (vector similarity)
- **Plan-and-execute** — LLM-powered task decomposition with dynamic replanning
- **Multi-agent orchestration** — pipeline, parallel, debate, and supervisor patterns
- **Any LLM backend** — works with OpenAI, Azure, Ollama, vLLM, LiteLLM, or any OpenAI-compatible API
- **Minimal dependencies** — just `openai`, `tiktoken`, `requests`, and `numpy`

## Quick Start

### Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

### Your First Agent

```python
from agentic_flow import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = Agent(
    name="researcher",
    instructions="You are a helpful research assistant.",
    tools=[search],
    model="gpt-4o-mini",
)

result = agent.run("What are the latest advances in GRPO?")
print(result.output)
```

That's it. The agent uses a **ReAct-style reasoning loop** (Thought → Action → Observation) under the hood, automatically deciding when to call tools and when to respond.

### Built-in Tools

```python
from agentic_flow.tools import calculator, python_executor, file_reader, web_search

agent = Agent(
    name="coder",
    instructions="You write and run Python code to solve problems.",
    tools=[python_executor, calculator, file_reader],
)
```

| Tool | Description |
|------|-------------|
| `calculator` | Safe math expression evaluation (`sqrt(16) * 3`) |
| `python_executor` | Execute Python code and capture output |
| `file_reader` | Read local files |
| `web_search` | Search via Tavily or SerpAPI |

### Custom Tools

The `@tool` decorator auto-generates the JSON schema from your function's type hints and docstring:

```python
@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: The city name.
        units: Temperature units (celsius or fahrenheit).
    """
    return fetch_weather(city, units)

# Inspect the generated schema
print(get_weather.to_openai_schema())
```

## Multi-Agent Orchestration

### Sequential Pipeline

Chain agents together — each agent's output becomes the next agent's input:

```python
from agentic_flow import Agent, SequentialPipeline

researcher = Agent(name="researcher", instructions="Research the topic thoroughly.")
writer = Agent(name="writer", instructions="Write a clear, engaging article.")
editor = Agent(name="editor", instructions="Polish the article for publication.")

pipeline = SequentialPipeline(agents=[researcher, writer, editor])
result = pipeline.run("The impact of AI on healthcare")
```

### Multi-Agent Debate

Multiple agents with different perspectives debate, then a judge synthesises:

```python
from agentic_flow import Agent, DebateOrchestrator

optimist = Agent(name="Optimist", instructions="Focus on benefits and opportunities.")
skeptic = Agent(name="Skeptic", instructions="Focus on risks and limitations.")
judge = Agent(name="Judge", instructions="Synthesise a balanced answer.")

debate = DebateOrchestrator(
    agents=[optimist, skeptic],
    judge=judge,
    rounds=2,
)
result = debate.run("Should AI agents be given autonomous code execution?")
```

### Supervisor Pattern

A supervisor delegates sub-tasks to specialised workers:

```python
from agentic_flow import Agent, SupervisorOrchestrator

supervisor = Agent(name="supervisor", instructions="Coordinate the team.")
coder = Agent(name="coder", instructions="Write Python code.", tools=[python_executor])
researcher = Agent(name="researcher", instructions="Research topics.", tools=[web_search])

orch = SupervisorOrchestrator(
    supervisor=supervisor,
    workers={"coder": coder, "researcher": researcher},
)
result = orch.run("Build a data pipeline that fetches weather data and plots trends")
```

### Parallel Execution

Run multiple agents concurrently and combine results:

```python
from agentic_flow import Agent, ParallelExecutor

agents = [
    Agent(name="analyst_1", instructions="Analyse from a financial perspective."),
    Agent(name="analyst_2", instructions="Analyse from a technical perspective."),
    Agent(name="analyst_3", instructions="Analyse from a market perspective."),
]

executor = ParallelExecutor(agents=agents, combiner="concat")
result = executor.run("Evaluate the potential of quantum computing startups")
```

## Plan-and-Execute

Decompose complex goals into sub-tasks with automatic replanning on failure:

```python
from agentic_flow import Agent, Planner
from agentic_flow.tools import web_search, python_executor

executor = Agent(
    name="executor",
    instructions="Complete tasks step by step.",
    tools=[web_search, python_executor],
)

planner = Planner(model="gpt-4o-mini", max_replans=2)
plan = planner.create_plan("Compare Python web frameworks and recommend one for a startup")

print(plan.summary())
# Plan: Compare Python web frameworks...
#   [ ] 1. Research top Python web frameworks
#   [ ] 2. Compare features, performance, and community
#   [ ] 3. Write a recommendation

result = planner.execute(plan, executor)
```

## Memory

Agents automatically maintain conversation history. Long-term memory uses vector similarity for context retrieval:

```python
from agentic_flow import Agent
from agentic_flow.memory import MemoryManager

memory = MemoryManager()
agent = Agent(name="assistant", instructions="You remember everything.", memory=memory)

# Conversation history is maintained across runs
agent.run("My name is Alice and I work on robotics.")
result = agent.run("What do I work on?")
# Agent recalls: "You work on robotics."
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Orchestrator                    │
│  (Pipeline / Parallel / Debate / Supervisor)     │
├─────────────────────────────────────────────────┤
│                    Planner                       │
│           (Decompose → Execute → Replan)         │
├─────────────────────────────────────────────────┤
│                     Agent                        │
│          ReAct loop: Think → Act → Observe       │
├──────────┬──────────────────┬───────────────────┤
│  Tools   │     Memory       │    LLM Client     │
│ @tool    │ Short + Long     │ OpenAI-compatible  │
│ decorator│ term storage     │ retry + streaming  │
└──────────┴──────────────────┴───────────────────┘
```

## Why agentic-flow?

| Feature | agentic-flow | LangChain | CrewAI | AutoGen |
|---------|:------------:|:---------:|:------:|:-------:|
| Lines to build an agent | **~10** | ~50 | ~30 | ~40 |
| Core dependencies | **4** | 20+ | 10+ | 10+ |
| Tool decorator | **Yes** | Varies | Yes | No |
| Built-in planning | **Yes** | Via chains | Yes | No |
| Multi-agent patterns | **4 built-in** | Manual | 1 | 2 |
| OpenAI-compatible | **Any backend** | Adapters | OpenAI | OpenAI |
| Learning curve | **Minutes** | Days | Hours | Hours |
| Package size | **< 50 KB** | ~50 MB | ~10 MB | ~5 MB |

**agentic-flow** is for developers who want the power of an agent framework without the complexity. No sprawling abstractions, no framework lock-in — just clean Python that does what you need.

## Project Structure

```
agentic-flow/
├── agentic_flow/
│   ├── __init__.py          # Public API
│   ├── agent.py             # Core Agent with ReAct loop
│   ├── tools.py             # @tool decorator + built-in tools
│   ├── memory.py            # Short-term + long-term memory
│   ├── planner.py           # Plan-and-execute with replanning
│   ├── orchestrator.py      # Multi-agent orchestration patterns
│   └── llm.py               # LLM client with retry + streaming
├── examples/
│   ├── research_agent.py    # Web research agent
│   ├── code_agent.py        # Code writing + execution
│   ├── multi_agent_debate.py# Multi-perspective debate
│   └── plan_execute.py      # Plan-and-execute demo
├── tests/
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_agent.py
│   ├── test_planner.py
│   └── test_orchestrator.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
