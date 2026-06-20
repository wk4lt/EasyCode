"""Tests for codegen workflows — IndexWorkflow and CodeGenWorkflow."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from liteagent.core.base_agent import AgentOutput
from liteagent.core.llm_interface import ChatResponse
from examples.workflows.states import IndexState, CodeGenState


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def chat_completion(self, messages, tools=None, tool_choice="auto"):
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return ChatResponse(content="Done", token_usage={"total_tokens": 5})


class _MockAgent:
    """Mock agent that returns a controlled AgentOutput."""

    def __init__(self, output: AgentOutput):
        self._output = output
        self.name = "mock_agent"

    def invoke(self, local_input):
        return self._output


class TestIndexWorkflow:
    @pytest.fixture
    def mock_rag(self):
        from liteagent.core.rag_store import InMemoryRAGStore
        return InMemoryRAGStore()

    @pytest.fixture
    def mock_config(self):
        from liteagent.core.config import AppConfig, LLMConfig, RAGConfig

        config = AppConfig(
            llm=LLMConfig(api_key="sk-test", model="gpt-4o"),
            rag=RAGConfig(chroma_path="/tmp/test_rag"),
        )
        return config

    def test_discover_node_finds_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my_tool"
            skill_dir.mkdir()
            (skill_dir / "SKILL_tool.md").write_text("# Skill: tool\n## Description\nTest.\n## Parameters\n")
            (skill_dir / "tool_impl.py").write_text("class ToolImpl:\n    pass\n")

            from examples.workflows.index_workflow import discover_node

            state = IndexState(example_dirs=[str(tmpdir)])
            result = discover_node(state)
            assert len(result["design_doc_paths"]) == 1
            assert len(result["impl_paths"]) == 1
            assert result["status"] == "processing"

    def test_discover_node_empty_dir(self):
        from examples.workflows.index_workflow import discover_node

        state = IndexState(example_dirs=["/nonexistent/path"])
        result = discover_node(state)
        assert result["design_doc_paths"] == []
        assert result["impl_paths"] == []

    def test_learn_reducer_detects_clarification(self):
        from examples.workflows.index_workflow import learn_reducer

        state = IndexState()
        output = AgentOutput(
            business_result=(
                "[NEED_CLARIFICATION]\n"
                "Question: Which pattern should I use?\n"
                "Context: Found two different conventions.\n"
                "Options: A) Use async, B) Use sync\n"
            ),
        )
        result = learn_reducer(state, output)
        assert result["needs_clarification"] is True
        assert "Which pattern" in result["clarification_question"]
        assert result["status"] == "needs_user"

    def test_learn_reducer_parses_success(self):
        from examples.workflows.index_workflow import learn_reducer

        state = IndexState()
        output = AgentOutput(
            business_result="**Learning Summary:**\nTotal pairs: 3\nSuccessfully indexed: 3",
        )
        result = learn_reducer(state, output)
        assert result["indexed_count"] == 3
        assert result["status"] == "done"

    def test_needs_user_check_routes(self):
        from examples.workflows.index_workflow import needs_user_check

        assert needs_user_check(IndexState(needs_clarification=True)) == "needs_user"
        assert needs_user_check(IndexState(needs_clarification=False)) == "done"

    @patch("examples.workflows.index_workflow.create_llm")
    def test_build_does_not_crash(self, mock_create_llm, mock_config, mock_rag):
        mock_llm = FakeLLM()
        mock_create_llm.return_value = mock_llm

        from examples.workflows.index_workflow import IndexWorkflow

        wf = IndexWorkflow(state_model=IndexState)
        wf.build(config=mock_config, rag_store=mock_rag)
        compiled = wf.compile()
        assert compiled is not None


class TestCodeGenWorkflow:
    @pytest.fixture
    def mock_rag(self):
        from liteagent.core.rag_store import InMemoryRAGStore
        store = InMemoryRAGStore()
        store.add_example("Design: web search tool", "class SearchTool:\n    pass")
        store.add_example("Design: risk checker", "class RiskCheck:\n    pass")
        store.add_example("Design: data parser", "def parse_data(x): return x")
        return store

    @pytest.fixture
    def mock_config(self):
        from liteagent.core.config import AppConfig, LLMConfig, RAGConfig

        config = AppConfig(
            llm=LLMConfig(api_key="sk-test", model="gpt-4o"),
            rag=RAGConfig(chroma_path="/tmp/test_rag"),
        )
        return config

    def test_read_inputs_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            design_path = Path(tmpdir) / "design.md"
            design_path.write_text("# Design\nTest module.")

            test_path = Path(tmpdir) / "test.py"
            test_path.write_text("def test(): pass")

            from examples.workflows.codegen_workflow import read_inputs

            state = CodeGenState(
                design_doc_path=str(design_path),
                test_file_path=str(test_path),
            )
            result = read_inputs(state)
            assert "Test module" in result["design_content"]
            assert "def test(): pass" in result["test_content"]
            assert result["final_status"] == "generating"

    def test_gen_condition_routing(self):
        from examples.workflows.codegen_workflow import gen_condition

        assert gen_condition(CodeGenState(tests_passed=True)) == "passed"
        assert gen_condition(CodeGenState(tests_passed=False, attempt=5, max_attempts=5)) == "failed"
        assert gen_condition(CodeGenState(tests_passed=False, attempt=2, max_attempts=5)) == "retry"
        assert gen_condition(CodeGenState(needs_clarification=True)) == "needs_user"

    def test_write_output_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.py"

            from examples.workflows.codegen_workflow import write_output

            state = CodeGenState(
                output_path=str(output_path),
                generated_code="def generated():\n    return True",
            )
            result = write_output(state)
            assert result["final_status"] == "passed"
            assert output_path.exists()
            assert "def generated()" in output_path.read_text()

    def test_retrieve_reducer_detects_clarification(self):
        from examples.workflows.codegen_workflow import retrieve_reducer

        state = CodeGenState(attempt=1)
        output = AgentOutput(
            business_result=(
                "[NEED_CLARIFICATION]\n"
                "Question: Should this be synchronous or async?\n"
                "Context: Design doesn't specify.\n"
                "Options: A) sync, B) async\n"
            ),
        )
        result = retrieve_reducer(state, output)
        assert result["needs_clarification"] is True
        assert result["final_status"] == "needs_user"

    @patch("examples.workflows.codegen_workflow.create_llm")
    def test_build_does_not_crash(self, mock_create_llm, mock_config, mock_rag):
        mock_llm = FakeLLM([ChatResponse(content="PASSED", token_usage={})])
        mock_create_llm.return_value = mock_llm

        from examples.workflows.codegen_workflow import CodeGenWorkflow

        wf = CodeGenWorkflow(state_model=CodeGenState)
        wf.build(config=mock_config, rag_store=mock_rag)
        compiled = wf.compile()
        assert compiled is not None
