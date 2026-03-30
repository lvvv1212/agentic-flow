"""Memory subsystem for agentic-flow.

Provides short-term (conversation) memory and long-term (vector) memory
backed by NumPy cosine similarity — no heavy dependencies required.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Short-term memory  (conversation / message history)
# ---------------------------------------------------------------------------

@dataclass
class ShortTermMemory:
    """Rolling conversation history with optional max-length trimming.

    Parameters
    ----------
    max_messages : int | None
        When set, the oldest messages beyond this count are dropped (the
        system message, if present, is always preserved).
    """

    max_messages: int | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, content: str, **extra: Any) -> None:
        msg: dict[str, Any] = {"role": role, "content": content, **extra}
        self.messages.append(msg)
        self._trim()

    def add_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._trim()

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        if self.max_messages is None or len(self.messages) <= self.max_messages:
            return
        # Preserve the system message at index 0 if present
        if self.messages and self.messages[0].get("role") == "system":
            system = self.messages[0]
            self.messages = [system] + self.messages[-(self.max_messages - 1) :]
        else:
            self.messages = self.messages[-self.max_messages :]


# ---------------------------------------------------------------------------
# Long-term memory  (vector store with NumPy cosine similarity)
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> Any:
    """Embed a batch of texts via the OpenAI embeddings API.

    Returns a NumPy array of shape ``(len(texts), dim)``.
    """
    import numpy as np

    try:
        import openai

        client = openai.OpenAI()
        response = client.embeddings.create(input=texts, model=model)
        return np.array([d.embedding for d in response.data], dtype=np.float32)
    except Exception:
        # Fallback: simple hash-based pseudo-embeddings for offline/test use
        dim = 256
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = np.frombuffer(h * (dim // len(h) + 1), dtype=np.uint8)[:dim].astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


@dataclass
class _MemoryEntry:
    text: str
    metadata: dict[str, Any]
    timestamp: float


@dataclass
class LongTermMemory:
    """Simple vector store using NumPy cosine similarity.

    Stores text entries with metadata and supports semantic retrieval.
    Falls back to hash-based pseudo-embeddings when no embedding API is
    available, so the framework works fully offline for development.

    Parameters
    ----------
    embedding_model : str
        OpenAI embedding model name.
    """

    embedding_model: str = "text-embedding-3-small"
    _entries: list[_MemoryEntry] = field(default_factory=list, repr=False)
    _vectors: Any = field(default=None, init=False, repr=False)  # np.ndarray | None

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a text entry to memory."""
        entry = _MemoryEntry(text=text, metadata=metadata or {}, timestamp=time.time())
        self._entries.append(entry)
        # Recompute vectors lazily on next search
        self._vectors = None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant entries for *query*."""
        import numpy as np

        if not self._entries:
            return []

        # Build / refresh vector index
        if self._vectors is None:
            texts = [e.text for e in self._entries]
            self._vectors = _embed_texts(texts, model=self.embedding_model)

        query_vec = _embed_texts([query], model=self.embedding_model)
        # Cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9
        normed = self._vectors / norms
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = (normed @ q_norm.T).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for idx in top_indices:
            e = self._entries[idx]
            results.append({
                "text": e.text,
                "score": float(scores[idx]),
                "metadata": e.metadata,
            })
        return results

    def clear(self) -> None:
        self._entries.clear()
        self._vectors = None


# ---------------------------------------------------------------------------
# Memory manager  (unified short + long-term)
# ---------------------------------------------------------------------------

@dataclass
class MemoryManager:
    """Combines short-term and long-term memory for an agent.

    On every user/assistant turn the text is also persisted to long-term
    memory so that relevant context can be retrieved later.
    """

    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = field(default_factory=LongTermMemory)

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        self.short_term.add(role, content, **extra)
        if role in ("user", "assistant") and content:
            self.long_term.add(content, metadata={"role": role})

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.long_term.search(query, top_k=top_k)

    def get_messages(self) -> list[dict[str, Any]]:
        return self.short_term.get_messages()

    def clear(self) -> None:
        self.short_term.clear()
        self.long_term.clear()
