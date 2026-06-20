"""Code Generation Workflow — generates code from design docs and tests.

Flow:
    read_inputs → retrieve → generate → test → [passed | retry_loop | needs_user]

Reads design doc + test file, queries RAG for similar examples, generates
code via LLM, runs tests, and fixes iteratively. Pauses for user
clarification when the agent encounters ambiguity.

Layer: Workflow layer (third layer).
"""

import logging
from pathlib import Path
from typing import Optional

from liteagent.core.base_agent import AgentOutput
from liteagent.core.base_workflow import BaseWorkflow
from liteagent.core.config import AppConfig, create_llm, load_config
from liteagent.core.rag_store import create_rag_store, BaseRAGStore
from examples.workflows.states import CodeGenState

_log = logging.getLogger(__name__)


def read_inputs(state: CodeGenState) -> dict:
    """Read the design document and test file from disk.

    Args:
        state: Current CodeGenState.

    Returns:
        Partial dict with design_content and test_content populated.
    """
    design_content = ""
    test_content = ""

    if state.design_doc_path:
        dp = Path(state.design_doc_path)
        if dp.exists():
            design_content = dp.read_text(encoding="utf-8")

    if state.test_file_path:
        tp = Path(state.test_file_path)
        if tp.exists():
            test_content = tp.read_text(encoding="utf-8")

    return {
        "design_content": design_content,
        "test_content": test_content,
        "final_status": "generating",
    }


def retrieve_input_mapper(state: CodeGenState) -> dict:
    """Extract fields for the code retriever operation.

    Note: This is passed to a simple processing agent.

    Args:
        state: Current CodeGenState.

    Returns:
        Local dict with the design document for RAG query.
    """
    return {
        "design_doc_path": state.design_doc_path,
        "test_file_path": state.test_file_path,
        "output_path": state.output_path,
        "attempt": state.attempt,
        "previous_code": state.generated_code,
        "test_error": state.test_result,
        "clarification_response": state.clarification_response,
    }


def retrieve_reducer(state: CodeGenState, agent_output: AgentOutput) -> dict:
    """Process RAG retrieval and code generation output.

    Detects [NEED_CLARIFICATION] for checkpoint pauses.

    Args:
        state: Current CodeGenState.
        agent_output: Output from the generation agent.

    Returns:
        Partial dict to update CodeGenState fields.
    """
    result = agent_output.business_result

    if agent_output.error and agent_output.error != "max_rounds_exceeded":
        return {"error": agent_output.error, "final_status": "error"}

    if "[NEED_CLARIFICATION]" in result:
        question = ""
        ctx = ""
        opts = ""
        for line in result.split("\n"):
            line_s = line.strip()
            if line_s.startswith("Question:"):
                question = line_s[len("Question:"):].strip()
            elif line_s.startswith("Context:"):
                ctx = line_s[len("Context:"):].strip()
            elif line_s.startswith("Options:"):
                opts = line_s[len("Options:"):].strip()

        full_q = question
        if ctx:
            full_q += f"\n\nContext: {ctx}"
        if opts:
            full_q += f"\n\nOptions: {opts}"

        return {
            "needs_clarification": True,
            "clarification_question": full_q,
            "final_status": "needs_user",
        }

    if "PASSED" in result or "passed" in result.lower():
        tests_passed = True
    elif "FAILED" in result:
        tests_passed = False
    else:
        tests_passed = state.tests_passed

    generated_code = result

    if "GENERATED CODE:\n" in result:
        code_start = result.index("GENERATED CODE:\n") + len("GENERATED CODE:\n")
        generated_code = result[code_start:].strip()
    elif "Output file:" in result:
        tests_passed = "PASSED" in result

    return {
        "generated_code": generated_code,
        "tests_passed": tests_passed,
        "attempt": state.attempt + 1,
    }


def gen_condition(state: CodeGenState) -> str:
    """Determine routing after each generation attempt.

    Args:
        state: Current CodeGenState.

    Returns:
        Routing key: 'passed', 'retry', 'needs_user', or 'failed'.
    """
    if state.needs_clarification:
        return "needs_user"

    if state.tests_passed:
        return "passed"

    if state.attempt >= state.max_attempts:
        return "failed"

    return "retry"


def write_output(state: CodeGenState) -> dict:
    """Write the generated code to the output file.

    Args:
        state: Current CodeGenState.

    Returns:
        Partial dict with final_status updated.
    """
    if state.output_path and state.generated_code:
        out_path = Path(state.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(state.generated_code, encoding="utf-8")
        return {"final_status": "passed"}
    return {"final_status": "failed", "error": "No output path or generated code."}


def fail_node(state: CodeGenState) -> dict:
    """Terminal node for max retries exceeded.

    Args:
        state: Current CodeGenState.

    Returns:
        Partial dict with final_status 'failed'.
    """
    return {"final_status": "failed", "error": f"Max attempts ({state.max_attempts}) exceeded."}


def wait_node(state: CodeGenState) -> dict:
    """Terminal node when waiting for user clarification.

    Args:
        state: Current CodeGenState.

    Returns:
        Empty dict (state already has needs_user status).
    """
    return {}


class CodeGenWorkflow(BaseWorkflow):
    """Code generation workflow: read → retrieve → generate → test → [passed | retry | needs_user].

    Takes a design document and test file as input, generates implementation
    code using RAG-retrieved examples, and validates with tests.
    """

    def build(
        self,
        config: Optional[AppConfig] = None,
        skills_path: str = "examples/domain_skills/",
        prompts_path: str = "examples/domain_agents/prompts/",
        rag_store: Optional[BaseRAGStore] = None,
    ) -> None:
        """Assemble the code generation workflow graph.

        Args:
            config: Optional pre-loaded AppConfig.
            skills_path: Path to the domain_skills directory.
            prompts_path: Path to the agent prompts directory.
            rag_store: Optional pre-created RAGStore instance.
        """
        from examples.domain_agents.gen_agent import GenerateAgent
        from examples.domain_skills.codegen_tools.code_retriever_impl import CodeRetrieverImpl
        from examples.domain_skills.codegen_tools.code_generator_impl import CodeGeneratorImpl
        from examples.domain_skills.codegen_tools.code_tester_impl import CodeTesterImpl
        from examples.domain_skills.codegen_tools.file_reader_impl import FileReaderImpl
        from liteagent.skill_registry import SkillRegistry

        if config is None:
            config = load_config()

        llm = create_llm(config.llm)

        if rag_store is None:
            rag_store = create_rag_store(
                persist_dir=config.rag.chroma_path,
                api_key=config.llm.api_key,
                embedding_model=config.rag.embedding_model,
                collection_name=config.rag.collection_name,
            )

        sr = SkillRegistry()

        file_reader = FileReaderImpl(
            contract_path=f"{skills_path}/codegen_tools/SKILL_file_reader.md",
        )
        sr.register_skill(file_reader)

        code_retriever = CodeRetrieverImpl(
            contract_path=f"{skills_path}/codegen_tools/SKILL_code_retriever.md",
        )
        code_retriever.set_rag_store(rag_store)
        sr.register_skill(code_retriever)

        code_generator = CodeGeneratorImpl(
            contract_path=f"{skills_path}/codegen_tools/SKILL_code_generator.md",
        )
        code_generator.set_llm(llm)
        sr.register_skill(code_generator)

        code_tester = CodeTesterImpl(
            contract_path=f"{skills_path}/codegen_tools/SKILL_code_tester.md",
        )
        sr.register_skill(code_tester)

        gen_agent = GenerateAgent(
            name="gen_agent",
            system_prompt_path=f"{prompts_path}/gen_agent.md",
            llm=llm,
            skill_names=["file_reader", "code_retriever", "code_generator", "code_tester"],
            skill_registry=sr,
        )

        self._graph.add_node("read_inputs", read_inputs)
        self.add_agent_node("gen_node", gen_agent, retrieve_input_mapper, retrieve_reducer)

        self._graph.add_node("write_node", write_output)
        self._graph.add_node("fail_node", fail_node)
        self._graph.add_node("wait_node", wait_node)

        self.set_entry_point("read_inputs")
        self.add_edge("read_inputs", "gen_node")

        self.add_conditional_edges(
            "gen_node",
            gen_condition,
            {
                "passed": "write_node",
                "retry": "gen_node",
                "needs_user": "wait_node",
                "failed": "fail_node",
            },
        )

        self.set_finish_point("write_node")
        self.set_finish_point("fail_node")
        self.set_finish_point("wait_node")
