"""DeepSeek API integration for agentic-flow.

Provides a thin, reusable client around DeepSeek's OpenAI-compatible API
(endpoint: ``https://api.deepseek.com/v1``).  API keys are read from the
``DEEPSEEK_API_KEY`` environment variable (or a ``.env`` file) -- never
hard-coded in source code.

Typical usage
-------------
    from agentic_flow import create_deepseek_agent

    agent = create_deepseek_agent(
        name="assistant",
        instructions="You are a helpful assistant.",
    )
    print(agent.run("Hello!").output)

Or use the low-level client directly::

    from agentic_flow import DeepSeekClient

    client = DeepSeekClient()      # reads DEEPSEEK_API_KEY from env / .env
    client.verify_connection()     # raises typed errors on failure
    print(client.chat_text("Hi"))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from agentic_flow.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults (all overridable via environment variables)
# ---------------------------------------------------------------------------

#: DeepSeek OpenAI-compatible base URL.
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
#: Default model id. Override with ``DEEPSEEK_MODEL`` if your account/endpoint
#: exposes a different model (e.g. ``DeepSeek-V4-Pro``).
DEFAULT_MODEL = "deepseek-chat"

ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"


def _load_dotenv() -> None:
    """Best-effort ``.env`` loader (uses python-dotenv if present, else a
    minimal built-in parser). Only populates variables that are not already
    set in the environment, so shell exports always win.
    """
    env_file = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(env_file):
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_file)
        return
    except ImportError:
        pass
    # Minimal fallback parser (no external dependency)
    try:
        with open(env_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logger.warning("Could not read .env file: %s", exc)


# ---------------------------------------------------------------------------
# Exceptions -- typed errors so callers can handle each failure distinctly
# ---------------------------------------------------------------------------


class DeepSeekError(Exception):
    """Base class for all DeepSeek client errors."""


class DeepSeekAuthError(DeepSeekError):
    """Authentication failed: missing or invalid API key (HTTP 401)."""


class DeepSeekRateLimitError(DeepSeekError):
    """Rate limit exceeded (HTTP 429)."""


class DeepSeekNetworkError(DeepSeekError):
    """Network / connectivity problem (DNS, connection refused, timeout)."""


class DeepSeekServerError(DeepSeekError):
    """DeepSeek server returned a 5xx error."""


def _map_openai_error(exc: Exception) -> DeepSeekError:
    """Translate an OpenAI SDK exception into a typed DeepSeek error."""
    try:
        from openai import (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            OpenAIError,
        )
    except ImportError:
        return DeepSeekError(f"DeepSeek request failed: {exc}")

    if isinstance(exc, AuthenticationError):
        return DeepSeekAuthError(
            f"DeepSeek authentication failed (check DEEPSEEK_API_KEY): {exc}"
        )
    if isinstance(exc, RateLimitError):
        return DeepSeekRateLimitError(f"DeepSeek rate limit exceeded: {exc}")
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return DeepSeekNetworkError(f"DeepSeek network error: {exc}")
    if isinstance(exc, InternalServerError):
        return DeepSeekServerError(f"DeepSeek server error: {exc}")
    if isinstance(exc, OpenAIError):
        return DeepSeekError(f"DeepSeek API error: {exc}")
    return DeepSeekError(f"DeepSeek request failed: {exc}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class DeepSeekClient:
    """Reusable DeepSeek API client (OpenAI-compatible).

    Configuration is resolved from environment variables by default, so the
    API key is never hard-coded in source code.

    Environment variables
    ----------------------
    ``DEEPSEEK_API_KEY``  (required) API key.
    ``DEEPSEEK_MODEL``    (optional) model id, default ``deepseek-chat``.
    ``DEEPSEEK_BASE_URL`` (optional) API base URL.

    Raises
    ------
    DeepSeekAuthError
        If no API key can be resolved (missing env var / ``.env`` entry).
    """

    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    max_tokens: int | None = None
    timeout: float = 60.0

    # internal
    _llm: LLMClient = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _load_dotenv()
        self.api_key = self.api_key or os.environ.get(ENV_API_KEY)
        self.model = self.model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.base_url = self.base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL
        if not self.api_key:
            raise DeepSeekAuthError(
                "Missing DeepSeek API key. Set the DEEPSEEK_API_KEY environment "
                "variable (or add it to a .env file). Do NOT hard-code it in source."
            )

    # ------------------------------------------------------------------
    @property
    def llm(self) -> LLMClient:
        """Lazily build the underlying OpenAI-compatible client."""
        if self._llm is None:
            self._llm = LLMClient(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                max_tokens=self.max_tokens,
            )
            # Apply the request timeout (LLMClient does not expose it directly)
            try:
                self._llm.client.timeout = self.timeout  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                logger.debug("Could not set client timeout")
        return self._llm

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns the same dict shape as :meth:`LLMClient.chat` (``content``,
        ``tool_calls``, ``usage``). Raises typed :class:`DeepSeekError`
        subclasses on failure.
        """
        try:
            return self.llm.chat(
                messages=messages, tools=tools, tool_choice=tool_choice, **kwargs
            )
        except DeepSeekError:
            raise
        except Exception as exc:  # openai SDK exceptions, etc.
            raise _map_openai_error(exc) from exc

    # ------------------------------------------------------------------
    def chat_text(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        """Convenience method: send a single user prompt, return the text."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages=messages, **kwargs)["content"]

    # ------------------------------------------------------------------
    def verify_connection(self) -> dict[str, Any]:
        """Ping the API with a minimal request to confirm the key works.

        Returns ``{"ok": True, "model": ..., "base_url": ..., "response": ...}``
        on success. Raises a typed :class:`DeepSeekError` on any failure
        (e.g. :class:`DeepSeekAuthError` for a bad key,
        :class:`DeepSeekNetworkError` if the endpoint is unreachable).
        """
        try:
            reply = self.chat_text(
                "Reply with the single word: OK",
                system="You are a terse assistant.",
                max_tokens=8,
            )
        except DeepSeekError:
            raise
        except Exception as exc:
            raise _map_openai_error(exc) from exc
        return {
            "ok": True,
            "model": self.model,
            "base_url": self.base_url,
            "response": reply.strip(),
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_deepseek_agent(
    name: str = "deepseek-agent",
    instructions: str = "You are a helpful AI assistant.",
    tools: list | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_iterations: int = 10,
    max_tokens: int | None = None,
    verbose: bool = False,
) -> "Agent":
    """Build an :class:`~agentic_flow.agent.Agent` pre-configured for DeepSeek.

    Reads DeepSeek credentials from the environment unless explicitly passed,
    and pins them on the agent so it never falls back to ``OPENAI_API_KEY``.
    """
    from agentic_flow.agent import Agent

    client = DeepSeekClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools or [],
        model=client.model,
        api_key=client.api_key,  # explicit: avoid OPENAI_API_KEY fallback
        base_url=client.base_url,
        temperature=client.temperature,
        max_iterations=max_iterations,
        max_tokens=client.max_tokens,
        verbose=verbose,
    )
