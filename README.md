# agentic-flow

> ⚠️ **Fork Notice** — This repository is a **fork** of [`JialiangFan/agentic-flow`](https://github.com/JialiangFan/agentic-flow) (MIT licensed). All original credit belongs to the author **JialiangFan**; the original `LICENSE` and commit history are preserved. This fork adds a **DeepSeek integration** (`agentic_flow/deepseek.py`, `examples/deepseek_agent.py`, `tests/test_deepseek.py`) on top of the upstream framework.

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

> ⚠️ **离线 vs 语义检索（为什么这么设计）**：`LongTermMemory` 在**没有 embedding API**
> 时会退化为 hash 伪向量（SHA-256），这些向量**无语义**——同样的意思换种说法不会命中。
> 之所以保留这个降级，是为了让框架在**零外部服务**时也能端到端跑通（开发 / 单测 / CI），
> 契合「最小依赖」的定位；但它**只证明流程能跑通，不代表检索质量**。要真正的语义检索，
> 请配置 embedding API（如 `OPENAI_API_KEY` 或兼容端点）。详见 `agentic_flow/memory.py`。

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
│   ├── deepseek.py          # DeepSeek (OpenAI-compatible) client
│   └── llm.py               # LLM client with retry + streaming
├── examples/
│   ├── research_agent.py    # Web research agent
│   ├── code_agent.py        # Code writing + execution
│   ├── multi_agent_debate.py# Multi-perspective debate
│   ├── plan_execute.py      # Plan-and-execute demo
│   └── deepseek_agent.py    # DeepSeek-powered agent
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

## 接入 DeepSeek（OpenAI 兼容）

agentic-flow 的 `LLMClient` 本身就是 OpenAI 兼容封装，因此 DeepSeek（兼容端点
`https://api.deepseek.com/v1`）无需额外适配即可使用。本仓库额外提供了一个
**可复用、带类型化错误处理** 的 `DeepSeekClient` 与便捷工厂 `create_deepseek_agent`，
并把 API Key 统一收口到环境变量，**绝不硬编码到源码**。

### 1. API Key 管理（不硬编码）

密钥通过环境变量提供，二选一：

```bash
# 方式 A：直接在 shell 导出
export DEEPSEEK_API_KEY=sk-xxxx

# 方式 B：写入 .env 文件（已被 .gitignore 忽略，不会误提交）
cp .env.example .env
# 然后编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxxx
```

> ⚠️ 切勿把真实密钥写进 `.py` 源码或提交到仓库。`.env` 已在 `.gitignore` 中忽略。

### 2. 快速上手

**低层客户端**（适合直接发请求 / 做连通性校验）：

```python
from agentic_flow import DeepSeekClient

client = DeepSeekClient()          # 自动读取 DEEPSEEK_API_KEY（与 .env）
client.verify_connection()         # 发最小请求验证密钥是否有效
print(client.chat_text("你好，介绍一下你自己。"))
```

**一行式创建 DeepSeek Agent**：

```python
from agentic_flow import create_deepseek_agent

agent = create_deepseek_agent(
    name="assistant",
    instructions="你是一个简洁、有帮助的助手。",
)
print(agent.run("用一句话解释什么是 ReAct 框架。").output)
```

### 3. 错误处理

所有失败都会抛出类型化异常，便于针对性处理：

| 异常 | 触发场景 |
|------|----------|
| `DeepSeekAuthError` | 缺少或错误的 API Key（HTTP 401） |
| `DeepSeekRateLimitError` | 触发限流（HTTP 429） |
| `DeepSeekNetworkError` | 网络/连接/超时问题 |
| `DeepSeekServerError` | DeepSeek 服务端 5xx 错误 |
| `DeepSeekError` | 其他 API 错误（基类） |

```python
from agentic_flow import DeepSeekClient, DeepSeekAuthError, DeepSeekError

try:
    DeepSeekClient().verify_connection()
except DeepSeekAuthError:
    print("密钥无效或缺失")
except DeepSeekError as e:
    print("其他错误:", e)
```

### 4. 完整运行步骤

```bash
# (1) 安装依赖（已 editable 安装则跳过）
pip install -e .

# (2) 配置密钥
export DEEPSEEK_API_KEY=sk-xxxx        # 或写入 .env

# (3) 运行示例（会自动先 verify_connection 验证接入）
python examples/deepseek_agent.py

# (4) 验证是否接入成功
#     示例脚本若打印 [OK] DeepSeek 已连通 即表示 Key 有效、端点可达；
#     若打印 [认证失败] 请检查 DEEPSEEK_API_KEY；[API 错误] 多为网络/限流。
```

### 5. 关于模型名

默认模型为 `deepseek-chat`。若你的 DeepSeek 账户 / 私有端点支持特定模型
（例如 `DeepSeek-V4-Pro`），请在 `.env` 中设置：

```bash
DEEPSEEK_MODEL=DeepSeek-V4-Pro
```

也可在代码中显式传入：`DeepSeekClient(model="DeepSeek-V4-Pro")`。

## License

MIT — see [LICENSE](LICENSE).
