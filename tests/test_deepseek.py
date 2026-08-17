"""Offline tests for the DeepSeek client integration.

These tests are fully **offline**: they never touch the network and never need
a real API key. Two techniques keep them hermetic:

1. OpenAI SDK exceptions are constructed locally with mock ``httpx`` objects, so
   the real mapping logic in :func:`agentic_flow.deepseek._map_openai_error` can
   be exercised with genuine exception instances.
2. The underlying :class:`LLMClient` is replaced by a :class:`MagicMock`, so
   ``DeepSeekClient.chat / chat_text / verify_connection`` run against canned
   return values or canned exceptions instead of hitting ``api.deepseek.com``.

Run with::

    python -m pytest tests/test_deepseek.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentic_flow import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekAuthError,
    DeepSeekRateLimitError,
    DeepSeekNetworkError,
    DeepSeekServerError,
    create_deepseek_agent,
)
from agentic_flow.deepseek import _map_openai_error


# --------------------------------------------------------------------------- #
# Helpers: build real OpenAI SDK exceptions without any HTTP traffic           #
# --------------------------------------------------------------------------- #

def _status_error(cls, status: int):
    """Build an APIStatusError subclass (auth / ratelimit / server) instance."""
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    return cls("mock error", response=response, body={})


def _make_exc(kind: str):
    """Return a real OpenAI SDK exception instance of the requested kind."""
    from openai import (
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OpenAIError,
    )

    if kind == "auth":
        return _status_error(AuthenticationError, 401)
    if kind == "ratelimit":
        return _status_error(RateLimitError, 429)
    if kind == "conn":
        return APIConnectionError(message="boom", request=MagicMock())
    if kind == "timeout":
        return APITimeoutError(request=MagicMock())
    if kind == "server":
        return _status_error(InternalServerError, 500)
    if kind == "generic":
        return OpenAIError("weird")
    raise ValueError(f"unknown kind: {kind}")


# --------------------------------------------------------------------------- #
# _map_openai_error -- the core mapping logic                                  #
# --------------------------------------------------------------------------- #

class TestMapOpenAIError:
    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("auth", DeepSeekAuthError),
            ("ratelimit", DeepSeekRateLimitError),
            ("conn", DeepSeekNetworkError),
            ("timeout", DeepSeekNetworkError),
            ("server", DeepSeekServerError),
            ("generic", DeepSeekError),
        ],
    )
    def test_maps_to_typed_error(self, kind, expected):
        mapped = _map_openai_error(_make_exc(kind))
        assert isinstance(mapped, expected)

    def test_generic_maps_to_base_not_specialized(self):
        mapped = _map_openai_error(_make_exc("generic"))
        assert isinstance(mapped, DeepSeekError)
        assert not isinstance(
            mapped,
            (
                DeepSeekAuthError,
                DeepSeekRateLimitError,
                DeepSeekNetworkError,
                DeepSeekServerError,
            ),
        )


# --------------------------------------------------------------------------- #
# Client construction                                                          #
# --------------------------------------------------------------------------- #

class TestClientConstruction:
    def test_missing_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(DeepSeekAuthError):
            DeepSeekClient()

    def test_explicit_key_is_accepted(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        client = DeepSeekClient(api_key="sk-fake-key")
        assert client.api_key == "sk-fake-key"
        assert client.model  # default model resolved from env/default
        assert client.base_url  # default base url resolved


# --------------------------------------------------------------------------- #
# chat / chat_text -- error mapping + happy path (mocked LLMClient)            #
# --------------------------------------------------------------------------- #

class TestChatErrorMapping:
    @staticmethod
    def _client_with_mock_llm(monkeypatch, *, raises=None, returns=None):
        """Build a DeepSeekClient whose underlying LLMClient is a MagicMock."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        client = DeepSeekClient(api_key="sk-fake-key")
        mock_llm = MagicMock()
        if raises is not None:
            mock_llm.chat.side_effect = raises
        else:
            mock_llm.chat.return_value = returns
        # `llm` is a lazy property that returns self._llm once set.
        client._llm = mock_llm
        return client

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("auth", DeepSeekAuthError),
            ("ratelimit", DeepSeekRateLimitError),
            ("conn", DeepSeekNetworkError),
            ("timeout", DeepSeekNetworkError),
            ("server", DeepSeekServerError),
        ],
    )
    def test_chat_maps_openai_error(self, monkeypatch, kind, expected):
        client = self._client_with_mock_llm(monkeypatch, raises=_make_exc(kind))
        with pytest.raises(expected):
            client.chat([{"role": "user", "content": "hi"}])

    def test_chat_returns_content_dict(self, monkeypatch):
        client = self._client_with_mock_llm(
            monkeypatch,
            returns={"content": "hello", "tool_calls": [], "usage": {}},
        )
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out["content"] == "hello"

    def test_chat_text_returns_text(self, monkeypatch):
        client = self._client_with_mock_llm(monkeypatch, returns={"content": "OK"})
        assert client.chat_text("ping") == "OK"


# --------------------------------------------------------------------------- #
# verify_connection                                                            #
# --------------------------------------------------------------------------- #

class TestVerifyConnection:
    def test_success_returns_ok_payload(self, monkeypatch):
        client = TestChatErrorMapping._client_with_mock_llm(
            monkeypatch, returns={"content": "OK"}
        )
        result = client.verify_connection()
        assert result["ok"] is True
        assert result["model"] == client.model
        assert result["base_url"] == client.base_url
        assert result["response"] == "OK"

    def test_propagates_auth_error(self, monkeypatch):
        client = TestChatErrorMapping._client_with_mock_llm(
            monkeypatch, raises=_make_exc("auth")
        )
        with pytest.raises(DeepSeekAuthError):
            client.verify_connection()


# --------------------------------------------------------------------------- #
# create_deepseek_agent factory                                                #
# --------------------------------------------------------------------------- #

class TestCreateAgent:
    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(DeepSeekAuthError):
            create_deepseek_agent()

    def test_with_key_builds_agent_and_pins_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        agent = create_deepseek_agent(
            api_key="sk-fake-key", name="helper", instructions="Be helpful."
        )
        assert agent.name == "helper"
        assert agent.api_key == "sk-fake-key"  # pinned, no OPENAI_API_KEY fallback
        assert agent.model  # resolved from default
