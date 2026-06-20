"""Index Workflow — scans example directories and indexes code pairs into RAG.

Flow:
    discover_node → learn_node → [done | needs_user]

The discover node scans for design doc + implementation code pairs.
The learn agent processes each pair and stores it in the RAG store.
If the agent needs clarification, the workflow pauses for user input.

Layer: Workflow layer (third layer).
"""

import logging
import os
from pathlib import Path
from typing import Optional

from liteagent.core.base_agent import AgentOutput
from liteagent.core.base_workflow import BaseWorkflow
from liteagent.core.config import AppConfig, create_llm, load_config
from liteagent.core.rag_store import create_rag_store, BaseRAGStore
from examples.workflows.states import IndexState

_log = logging.getLogger(__name__)


def discover_design_impl_pairs(example_dirs: list[str]) -> list[dict]:
    """Scan directories for .md design docs paired with corresponding .py impls.

    A pair is identified when a .md file and a *_impl.py (or .py) file
    share the same base name in the same directory.

    Args:
        example_dirs: List of directory paths to scan.

    Returns:
        List of dicts with 'design_path' and 'impl_path' keys.
    """
    pairs = []
    for dir_path in example_dirs:
        root = Path(dir_path)
        if not root.exists():
            _log.warning("dir_not_found", extra={"layer": "workflow", "path": str(root)})
            continue

        for entry in root.rglob("*.md"):
            md_path = entry
            if not md_path.name.startswith("SKILL_"):
                continue

            stem = md_path.stem
            if stem.startswith("SKILL_"):
                skill_name = stem[len("SKILL_"):]
            else:
                skill_name = stem

            impl_candidates = []
            for pattern in [f"{skill_name}_impl.py", f"{skill_name}.py"]:
                cand = md_path.parent / pattern
                if cand.exists():
                    impl_candidates.append(cand)

            if impl_candidates:
                pairs.append({
                    "design_path": str(md_path),
                    "impl_path": str(impl_candidates[0]),
                })

    return pairs


def discover_node(state: IndexState) -> dict:
    """Scan directories and populate the list of design+impl pairs.

    Args:
        state: Current IndexState.

    Returns:
        Partial dict with design_doc_paths and impl_paths populated.
    """
    pairs = discover_design_impl_pairs(state.example_dirs)
    return {
        "design_doc_paths": [p["design_path"] for p in pairs],
        "impl_paths": [p["impl_path"] for p in pairs],
        "status": "processing",
    }


def learn_input_mapper(state: IndexState) -> dict:
    """Extract fields for the learn agent.

    Args:
        state: Current IndexState.

    Returns:
        Local dict with example directories, pair count, and
        clarification response if resuming from a checkpoint.
    """
    return {
        "example_dirs": state.example_dirs,
        "pair_count": len(state.design_doc_paths),
        "clarification_response": state.clarification_response,
    }


def learn_reducer(state: IndexState, agent_output: AgentOutput) -> dict:
    """Merge learn agent output back into IndexState.

    Detects [NEED_CLARIFICATION] markers and sets the checkpoint flag.

    Args:
        state: Current IndexState (unused).
        agent_output: Output from the learn agent.

    Returns:
        Partial dict to update IndexState fields.
    """
    result = agent_output.business_result

    if agent_output.error and agent_output.error != "max_rounds_exceeded":
        return {"error": agent_output.error, "status": "error"}

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
            "status": "needs_user",
        }

    indexed = 0
    if "Successfully indexed:" in result:
        import re
        m = re.search(r"Successfully indexed:\s*(\d+)", result)
        if m:
            indexed = int(m.group(1))

    return {
        "indexed_count": indexed,
        "status": "done",
        "error": "",
    }


def needs_user_check(state: IndexState) -> str:
    """Route based on whether the agent needs user clarification.

    Args:
        state: Current IndexState.

    Returns:
        'needs_user' or 'done'.
    """
    return "needs_user" if state.needs_clarification else "done"


def wait_for_user(state: IndexState) -> dict:
    """No-op node that acts as a terminal when waiting for user input.

    Args:
        state: Current IndexState.

    Returns:
        Empty dict (no state changes).
    """
    return {}


class IndexWorkflow(BaseWorkflow):
    """Example indexing workflow: discover → learn → [done | needs_user].

    Scans directories for SKILL_*.md + *_impl.py pairs, processes them
    through the learn agent, and stores them in RAG.
    """

    def build(
        self,
        config: Optional[AppConfig] = None,
        skills_path: str = "examples/domain_skills/",
        prompts_path: str = "examples/domain_agents/prompts/",
        rag_store: Optional[BaseRAGStore] = None,
    ) -> None:
        """Assemble the indexing workflow graph.

        Args:
            config: Optional pre-loaded AppConfig.
            skills_path: Path to the domain_skills directory.
            prompts_path: Path to the agent prompts directory.
            rag_store: Optional pre-created RAGStore instance.
        """
        from examples.domain_agents.learn_agent import LearnAgent
        from examples.domain_skills.codegen_tools.code_embedder_impl import CodeEmbedderImpl
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

        code_embedder = CodeEmbedderImpl(
            contract_path=f"{skills_path}/codegen_tools/SKILL_code_embedder.md",
        )
        code_embedder.set_rag_store(rag_store)
        sr.register_skill(code_embedder)

        learn_agent = LearnAgent(
            name="learn_agent",
            system_prompt_path=f"{prompts_path}/learn_agent.md",
            llm=llm,
            skill_names=["file_reader", "code_embedder"],
            skill_registry=sr,
        )

        self._graph.add_node("discover_node", discover_node)
        self.add_agent_node("learn_node", learn_agent, learn_input_mapper, learn_reducer)

        self._graph.add_node("waiting_node", wait_for_user)
        self.set_entry_point("discover_node")
        self.add_edge("discover_node", "learn_node")

        self.add_conditional_edges(
            "learn_node",
            needs_user_check,
            {"needs_user": "waiting_node", "done": "__end__"},
        )

        self.set_finish_point("waiting_node")
