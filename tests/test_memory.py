"""Tests for the memory subsystem."""

from agentic_flow.memory import ShortTermMemory, LongTermMemory, MemoryManager


class TestShortTermMemory:
    def test_add_and_get(self) -> None:
        mem = ShortTermMemory()
        mem.add("user", "hello")
        mem.add("assistant", "hi there")
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "hi there"

    def test_max_messages_trimming(self) -> None:
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add("user", f"msg {i}")
        assert len(mem.get_messages()) == 3
        assert mem.get_messages()[0]["content"] == "msg 2"

    def test_system_message_preserved(self) -> None:
        mem = ShortTermMemory(max_messages=3)
        mem.add("system", "You are helpful.")
        for i in range(5):
            mem.add("user", f"msg {i}")
        messages = mem.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "system"

    def test_clear(self) -> None:
        mem = ShortTermMemory()
        mem.add("user", "hello")
        mem.clear()
        assert len(mem.get_messages()) == 0


class TestLongTermMemory:
    def test_add_and_search(self) -> None:
        mem = LongTermMemory()
        mem.add("The capital of France is Paris.")
        mem.add("Python is a programming language.")
        mem.add("The sky is blue on a clear day.")
        results = mem.search("What is the capital of France?", top_k=2)
        assert len(results) == 2
        assert all("text" in r and "score" in r for r in results)

    def test_empty_search(self) -> None:
        mem = LongTermMemory()
        results = mem.search("anything")
        assert results == []

    def test_clear(self) -> None:
        mem = LongTermMemory()
        mem.add("test entry")
        mem.clear()
        assert mem.search("test") == []


class TestMemoryManager:
    def test_combined_memory(self) -> None:
        mgr = MemoryManager()
        mgr.add_message("user", "Tell me about Python.")
        mgr.add_message("assistant", "Python is a versatile programming language.")

        messages = mgr.get_messages()
        assert len(messages) == 2

        results = mgr.retrieve_relevant("programming language")
        assert len(results) > 0
