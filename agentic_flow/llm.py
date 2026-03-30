"""LLM client interface for agentic-flow.

Wraps the OpenAI Python SDK to provide retry logic, rate-limit handling,
token counting, and streaming support.  Works with any OpenAI-compatible
endpoint (OpenAI, Azure, Ollama, vLLM, LiteLLM, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
import tiktoken
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return an approximate token count for *text* using tiktoken."""
    try:
        enc_name = tiktoken.encoding_for_model(model).name
    except KeyError:
        enc_name = "cl100k_base"
    if enc_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[enc_name] = tiktoken.get_encoding(enc_name)
    return len(_ENCODER_CACHE[enc_name].encode(text))


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "gpt-4o-mini") -> int:
    """Rough token count for a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # overhead per message
        for value in msg.values():
            if isinstance(value, str):
                total += count_tokens(value, model)
    return total


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    """Thin wrapper around the OpenAI chat completions API.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``"gpt-4o-mini"``).
    api_key : str | None
        API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url : str | None
        Custom base URL for OpenAI-compatible APIs.
    temperature : float
        Sampling temperature.
    max_retries : int
        Number of retries on transient errors (rate-limit, server error).
    retry_delay : float
        Base delay in seconds between retries (exponential back-off).
    max_tokens : int | None
        Maximum tokens in the completion.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    max_tokens: int | None = None

    # internal
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Lazily initialise the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required. Install it with: pip install openai"
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request with retry logic.

        Returns the raw response as a dict with keys ``content``,
        ``tool_calls``, and ``usage``.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **kwargs,
        }
        if self.max_tokens:
            request["max_tokens"] = self.max_tokens
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                msg = response.choices[0].message
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                return {
                    "content": msg.content or "",
                    "tool_calls": tool_calls,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                }
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                retryable = any(
                    kw in err_str
                    for kw in ("rate limit", "rate_limit", "429", "500", "502", "503", "timeout")
                )
                if retryable and attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning("LLM request failed (attempt %d): %s — retrying in %.1fs", attempt + 1, exc, delay)
                    time.sleep(delay)
                else:
                    raise

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream a chat completion, yielding text chunks."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            **kwargs,
        }
        if self.max_tokens:
            request["max_tokens"] = self.max_tokens
        if tools:
            request["tools"] = tools

        response = self.client.chat.completions.create(**request)
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
