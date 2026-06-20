"""Tests for memory_manager.py — per-agent message isolation."""

import pytest

from liteagent.core.memory_manager import MemoryManager


class TestMemoryManager:
    @pytest.fixture
    def mm(self):
        return MemoryManager()

    def test_add_and_get_messages(self, mm):
        mm.add_message("agent_1", "system", "You are helpful.")
        mm.add_message("agent_1", "user", "Hello")
        msgs = mm.get_messages("agent_1")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_agents_are_isolated(self, mm):
        mm.add_message("agent_1", "system", "System A")
        mm.add_message("agent_2", "system", "System B")

        assert mm.get_messages("agent_1")[0]["content"] == "System A"
        assert mm.get_messages("agent_2")[0]["content"] == "System B"

    def test_unknown_agent_returns_empty(self, mm):
        assert mm.get_messages("nonexistent") == []

    def test_clear_single_agent(self, mm):
        mm.add_message("agent_1", "user", "Hello")
        mm.add_message("agent_2", "user", "World")
        mm.clear("agent_1")
        assert mm.get_messages("agent_1") == []
        assert len(mm.get_messages("agent_2")) == 1

    def test_clear_all(self, mm):
        mm.add_message("agent_1", "user", "Hello")
        mm.add_message("agent_2", "user", "World")
        mm.clear()
        assert mm.get_messages("agent_1") == []
        assert mm.get_messages("agent_2") == []

    def test_set_messages(self, mm):
        mm.add_message("agent_1", "user", "old")
        mm.set_messages("agent_1", [{"role": "system", "content": "new"}])
        assert mm.get_messages("agent_1") == [{"role": "system", "content": "new"}]

    def test_add_message_with_extra_fields(self, mm):
        mm.add_message("agent_1", "tool", "result", tool_call_id="call_1", name="search")
        msg = mm.get_messages("agent_1")[0]
        assert msg["tool_call_id"] == "call_1"
        assert msg["name"] == "search"

    def test_checkpoint_and_restore(self, mm):
        mm.add_message("agent_1", "system", "A")
        mm.add_message("agent_2", "user", "B")

        snapshot = mm.checkpoint()

        mm.add_message("agent_1", "user", "C")
        assert len(mm.get_messages("agent_1")) == 2

        mm.restore(snapshot)
        assert len(mm.get_messages("agent_1")) == 1
        assert mm.get_messages("agent_1")[0]["content"] == "A"
        assert mm.get_messages("agent_2")[0]["content"] == "B"
