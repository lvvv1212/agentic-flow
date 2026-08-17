"""Minimal DeepSeek-powered agent example for agentic-flow.

Setup
-----
1. Copy `.env.example` to `.env` and set DEEPSEEK_API_KEY.
2. (Optional) set DEEPSEEK_MODEL / DEEPSEEK_BASE_URL in `.env`.
3. Run:  python examples/deepseek_agent.py
"""

from agentic_flow import DeepSeekClient, DeepSeekAuthError, DeepSeekError, create_deepseek_agent


def main() -> None:
    # 1) Verify the API key / connectivity before doing real work.
    try:
        probe = DeepSeekClient()
        info = probe.verify_connection()
        print(f"[OK] DeepSeek 已连通: model={info['model']!r} reply={info['response']!r}")
    except DeepSeekAuthError as exc:
        print(f"[认证失败] {exc}")
        return
    except DeepSeekError as exc:
        print(f"[API 错误] {exc}")
        return

    # 2) Build a DeepSeek-backed agent and run it.
    agent = create_deepseek_agent(
        name="deepseek-assistant",
        instructions="你是一个简洁、有帮助的中文助手。",
    )
    result = agent.run("用一句话解释什么是 ReAct 框架。")
    print("\nAgent 输出:\n", result.output)


if __name__ == "__main__":
    main()
