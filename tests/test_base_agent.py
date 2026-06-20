"""Tests for base_agent.py — agent invoke loop and output format."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from liteagent.core.base_agent import AgentOutput, BaseAgent
from liteagent.core.llm_interface import ChatResponse, ToolCall


class FakeLLM:
    """A controllable fake LLM for testing agent behavior."""

    def __init__(self, responses: list[ChatResponse]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[tuple] = []

    def chat_completion(self, messages, tools=None, tool_choice="auto"):
        self.call_count += 1
        self.calls.append((messages, tools))
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return ChatResponse(content="Fallback response", token_usage={})


class FakeSkillRegistry:
    def get_schema(self, name):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Fake {name}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def get_skill(self, name):
        class FakeSkill:
            def execute(self, **kwargs):
                return {"status": "ok", "result": f"Executed {name}"}

        return FakeSkill()


class _TestAgent(BaseAgent):
    def _build_user_message(self, local_input: dict) -> str:
        name = local_input.get("name", "World")
        return f"Hello, {name}!"


SYSTEM_PROMPT = "You are a test agent.\n\nAlways be concise."


@pytest.fixture
def system_prompt_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(SYSTEM_PROMPT)
        return f.name


@pytest.fixture
def fake_registry():
    return FakeSkillRegistry()


class TestAgentOutput:
    def test_defaults(self):
        output = AgentOutput()
        assert output.business_result == ""
        assert output.token_usage == {}
        assert output.error is None

    def test_full_construction(self):
        output = AgentOutput(
            business_result="Approved",
            token_usage={"total_tokens": 100},
            error=None,
        )
        assert output.business_result == "Approved"
        assert output.token_usage["total_tokens"] == 100


class TestBaseAgent:
    def test_simple_invoke_no_tools(self, system_prompt_file, fake_registry):
        llm = FakeLLM([ChatResponse(content="Hi there!", token_usage={"total_tokens": 10})])
        agent = _TestAgent(
            name="test_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=[],
            skill_registry=fake_registry,
        )

        output = agent.invoke({"name": "Alice"})
        assert output.business_result == "Hi there!"
        assert output.token_usage["total_tokens"] == 10
        assert output.error is None
        assert llm.call_count == 1

    def test_tool_call_loop(self, system_prompt_file, fake_registry):
        responses = [
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="search", arguments={"q": "test"})
                ],
                token_usage={"total_tokens": 20},
            ),
            ChatResponse(content="Final answer", token_usage={"total_tokens": 15}),
        ]
        llm = FakeLLM(responses)
        agent = _TestAgent(
            name="test_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=["search"],
            skill_registry=fake_registry,
        )

        output = agent.invoke({"name": "Bob"})
        assert output.business_result == "Final answer"
        assert output.token_usage["total_tokens"] == 35
        assert llm.call_count == 2

    def test_invoke_with_error(self, system_prompt_file, fake_registry):
        class ErrorLLM:
            def chat_completion(self, messages, tools=None, tool_choice="auto"):
                raise RuntimeError("LLM connection failed")

        agent = _TestAgent(
            name="test_agent",
            system_prompt_path=system_prompt_file,
            llm=ErrorLLM(),
            skill_names=[],
            skill_registry=fake_registry,
        )

        output = agent.invoke({"name": "Charlie"})
        assert output.error is not None
        assert "LLM connection failed" in output.error

    def test_private_memory_is_reset_per_invoke(self, system_prompt_file, fake_registry):
        llm = FakeLLM([ChatResponse(content="Response 1", token_usage={})])
        agent = _TestAgent(
            name="test_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=[],
            skill_registry=fake_registry,
        )

        agent.invoke({"name": "Alice"})
        first_call_count = llm.call_count

        llm = FakeLLM([ChatResponse(content="Response 2", token_usage={})])
        agent._llm = llm
        agent.invoke({"name": "Bob"})
        assert llm.call_count == 1
