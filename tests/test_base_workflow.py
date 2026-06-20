"""Tests for base_workflow.py — graph construction and state transitions."""

import pytest
from pydantic import BaseModel, Field

from liteagent.core.base_agent import AgentOutput, BaseAgent
from liteagent.core.base_workflow import BaseWorkflow


class _TestState(BaseModel):
    """Minimal global state for workflow testing."""

    input_value: str = ""
    agent_result: str = ""
    routing_decision: str = "unknown"


class MockAgent(BaseAgent):
    """Agent that appends a suffix and returns it."""

    def __init__(self, name: str, suffix: str = "_processed"):
        self.suffix = suffix
        self.call_count = 0
        self.received_input: dict = {}

    @property
    def name(self):
        return "mock_agent"

    def invoke(self, local_input: dict) -> AgentOutput:
        self.call_count += 1
        self.received_input = local_input
        text = local_input.get("text", "")
        return AgentOutput(
            business_result=f"{text}{self.suffix}",
            token_usage={"total_tokens": 5},
        )

    def _build_user_message(self, local_input: dict) -> str:
        return str(local_input)


class MockWorkflow(BaseWorkflow):
    """Concrete workflow for testing."""

    def build(self, agent, input_mapper, reducer):
        self.add_agent_node("process", agent, input_mapper, reducer)
        self.set_entry_point("process")

        def decision(state):
            return {"routing_decision": "done"}

        self._graph.add_node("done_node", decision)
        self.add_edge("process", "done_node")
        self.set_finish_point("done_node")


class TestBaseWorkflow:
    @pytest.fixture
    def agent(self):
        return MockAgent("test")

    def test_input_mapper_receives_state_not_dict(self, agent):
        state = _TestState(input_value="hello")

        def mapper(s: _TestState) -> dict:
            assert isinstance(s, _TestState)
            return {"text": s.input_value}

        def reducer(s: _TestState, ao: AgentOutput) -> dict:
            return {"agent_result": ao.business_result}

        wf = MockWorkflow(state_model=_TestState)
        wf.build(agent, mapper, reducer)

        result = wf.invoke(state)
        assert result.agent_result == "hello_processed"

    def test_reducer_merges_output(self, agent):
        def mapper(s):
            return {"text": s.input_value}

        def reducer(s, ao):
            return {"agent_result": ao.business_result.upper()}

        wf = MockWorkflow(state_model=_TestState)
        wf.build(agent, mapper, reducer)

        result = wf.invoke(_TestState(input_value="test"))
        assert result.agent_result == "TEST_PROCESSED"

    def test_agent_receives_only_mapped_fields(self, agent):
        extra_field_not_passed = None

        def mapper(s: _TestState) -> dict:
            return {"text": s.input_value}

        def reducer(s, ao):
            nonlocal extra_field_not_passed
            extra_field_not_passed = agent.received_input.get("secret_field", None)
            return {"agent_result": ao.business_result}

        wf = MockWorkflow(state_model=_TestState)
        wf.build(agent, mapper, reducer)

        result = wf.invoke(_TestState(input_value="secret_data"))
        assert result.agent_result == "secret_data_processed"
        assert extra_field_not_passed is None

    def test_conditional_routing(self, agent):
        def mapper(s):
            return {"text": str(s.input_value)}

        def reducer(s, ao):
            return {"agent_result": ao.business_result}

        def condition(s: _TestState) -> str:
            return "approved" if s.agent_result else "rejected"

        class BranchWorkflow(BaseWorkflow):
            def build(self):
                self.add_agent_node("process", agent, mapper, reducer)
                self.set_entry_point("process")

                def approved_node(s):
                    return {"routing_decision": "approved"}

                def rejected_node(s):
                    return {"routing_decision": "rejected"}

                self._graph.add_node("approved", approved_node)
                self._graph.add_node("rejected", rejected_node)

                self.add_conditional_edges(
                    "process",
                    condition,
                    {"approved": "approved", "rejected": "rejected"},
                )
                self.set_finish_point("approved")
                self.set_finish_point("rejected")

        wf = BranchWorkflow(state_model=_TestState)
        wf.build()

        result = wf.invoke(_TestState(input_value="yes"))
        assert result.routing_decision == "approved"

    def test_compile_and_reuse(self, agent):
        def mapper(s):
            return {"text": s.input_value}

        def reducer(s, ao):
            return {"agent_result": ao.business_result}

        wf = MockWorkflow(state_model=_TestState)
        wf.build(agent, mapper, reducer)
        compiled = wf.compile()

        result1 = wf.invoke(_TestState(input_value="first"), compiled=compiled)
        result2 = wf.invoke(_TestState(input_value="second"), compiled=compiled)

        assert result1.agent_result == "first_processed"
        assert result2.agent_result == "second_processed"

    def test_state_requires_defaults(self):
        class BadState(BaseModel):
            required_field: str

        with pytest.raises(ValueError, match="must have a default"):
            wf = MockWorkflow(state_model=BadState)
            agent = MockAgent("test")
            wf.build(agent, lambda s: {}, lambda s, ao: {})
