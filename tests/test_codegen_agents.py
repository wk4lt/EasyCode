"""Tests for codegen agents — LearnAgent and GenerateAgent."""

import tempfile
from unittest.mock import MagicMock

import pytest

from liteagent.core.base_agent import AgentOutput
from liteagent.core.llm_interface import ChatResponse


SYSTEM_PROMPT_LEARN = "You are a learn agent.\n\nAlways be concise."


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def chat_completion(self, messages, tools=None, tool_choice="auto"):
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return ChatResponse(content="Done", token_usage={"total_tokens": 5})


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


@pytest.fixture
def system_prompt_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(SYSTEM_PROMPT_LEARN)
        return f.name


@pytest.fixture
def fake_registry():
    return FakeSkillRegistry()


class TestLearnAgent:
    def test_builds_user_message_with_dirs(self, system_prompt_file, fake_registry):
        from examples.domain_agents.learn_agent import LearnAgent

        llm = FakeLLM()
        agent = LearnAgent(
            name="learn_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=["file_reader", "code_embedder"],
            skill_registry=fake_registry,
        )
        msg = agent._build_user_message({
            "example_dirs": ["/path/to/examples"],
            "pair_count": 5,
        })
        assert "/path/to/examples" in msg
        assert "5" in msg

    def test_builds_user_message_with_clarification(self, system_prompt_file, fake_registry):
        from examples.domain_agents.learn_agent import LearnAgent

        llm = FakeLLM()
        agent = LearnAgent(
            name="learn_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=["file_reader", "code_embedder"],
            skill_registry=fake_registry,
        )
        msg = agent._build_user_message({
            "example_dirs": ["/path"],
            "pair_count": 2,
            "clarification_response": "Use approach A.",
        })
        assert "USER RESPONSE" in msg
        assert "Use approach A" in msg

    def test_invoke_returns_agent_output(self, system_prompt_file, fake_registry):
        from examples.domain_agents.learn_agent import LearnAgent

        llm = FakeLLM([ChatResponse(
            content="**Learning Summary:**\nTotal pairs: 2\nSuccessfully indexed: 2",
            token_usage={"total_tokens": 30},
        )])
        agent = LearnAgent(
            name="learn_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=[],
            skill_registry=fake_registry,
        )
        output = agent.invoke({"example_dirs": ["/tmp"], "pair_count": 2})
        assert output.business_result
        assert "Successfully indexed" in output.business_result


class TestGenerateAgent:
    def test_builds_user_message_new_generation(self, system_prompt_file, fake_registry):
        from examples.domain_agents.gen_agent import GenerateAgent

        llm = FakeLLM()
        agent = GenerateAgent(
            name="gen_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=["file_reader", "code_retriever", "code_generator", "code_tester"],
            skill_registry=fake_registry,
        )
        msg = agent._build_user_message({
            "design_doc_path": "/path/design.md",
            "test_file_path": "/path/test.py",
            "output_path": "/path/impl.py",
            "attempt": 0,
        })
        assert "design.md" in msg
        assert "test.py" in msg
        assert "impl.py" in msg

    def test_builds_user_message_fix_attempt(self, system_prompt_file, fake_registry):
        from examples.domain_agents.gen_agent import GenerateAgent

        llm = FakeLLM()
        agent = GenerateAgent(
            name="gen_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=["file_reader", "code_retriever", "code_generator", "code_tester"],
            skill_registry=fake_registry,
        )
        msg = agent._build_user_message({
            "design_doc_path": "/path/design.md",
            "test_file_path": "/path/test.py",
            "output_path": "/path/impl.py",
            "attempt": 2,
            "previous_code": "def broken(): pass",
            "test_error": "NameError",
        })
        assert "Fix attempt #2" in msg
        assert "NameError" in msg

    def test_builds_user_message_with_clarification(self, system_prompt_file, fake_registry):
        from examples.domain_agents.gen_agent import GenerateAgent

        llm = FakeLLM()
        agent = GenerateAgent(
            name="gen_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=["file_reader", "code_retriever", "code_generator", "code_tester"],
            skill_registry=fake_registry,
        )
        msg = agent._build_user_message({
            "design_doc_path": "/path/design.md",
            "test_file_path": "/path/test.py",
            "output_path": "/path/impl.py",
            "clarification_response": "Use option B.",
        })
        assert "USER RESPONSE" in msg
        assert "Use option B" in msg

    def test_invoke_returns_agent_output(self, system_prompt_file, fake_registry):
        from examples.domain_agents.gen_agent import GenerateAgent

        llm = FakeLLM([ChatResponse(
            content="**Generation Summary:**\n- Output file: impl.py\n- Test result: PASSED",
            token_usage={"total_tokens": 100},
        )])
        agent = GenerateAgent(
            name="gen_agent",
            system_prompt_path=system_prompt_file,
            llm=llm,
            skill_names=[],
            skill_registry=fake_registry,
        )
        output = agent.invoke({
            "design_doc_path": "design.md",
            "test_file_path": "test.py",
            "output_path": "impl.py",
            "attempt": 0,
        })
        assert "PASSED" in output.business_result
