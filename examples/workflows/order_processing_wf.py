"""Order Processing Workflow — a three-node example demonstrating the
Skill→Agent→Workflow architecture.

Flow:
    search_node → risk_node → decision_node → [approved | held | blocked]

The search agent gathers customer/product context, the risk agent evaluates
the order, and a conditional edge routes to the appropriate final action.

Layer: Workflow layer (third layer).
"""

from typing import Optional

from liteagent.core.base_agent import AgentOutput
from liteagent.core.base_workflow import BaseWorkflow
from liteagent.core.config import AppConfig, create_llm, load_config
from examples.workflows.states import OrderState


def search_input_mapper(state: OrderState) -> dict:
    """Extract fields from Global State for the search agent.

    Only passes the search_query and product — never the full state.

    Args:
        state: Full OrderState.

    Returns:
        Local dict with only search-relevant fields.
    """
    query = state.search_query or f"product information for {state.product}" if state.product else state.customer_name
    return {
        "search_task": f"Find relevant background information for order {state.order_id}",
        "query": query,
    }


def search_reducer(state: OrderState, agent_output: AgentOutput) -> dict:
    """Merge search agent output back into Global State.

    Args:
        state: Current OrderState (unused, kept for interface consistency).
        agent_output: Output from the search agent.

    Returns:
        Partial dict to update Global State fields.
    """
    if agent_output.error:
        return {"error": agent_output.error, "search_result": f"Search failed: {agent_output.error}"}
    return {"search_result": agent_output.business_result, "error": ""}


def risk_input_mapper(state: OrderState) -> dict:
    """Extract fields from Global State for the risk agent.

    Args:
        state: Full OrderState.

    Returns:
        Local dict with only risk-relevant fields.
    """
    return {
        "order_id": state.order_id,
        "amount": state.amount,
        "customer_tier": state.customer_tier,
        "region": state.region,
    }


def risk_reducer(state: OrderState, agent_output: AgentOutput) -> dict:
    """Merge risk agent output back into Global State.

    Parses the agent's business_result for structured risk data.
    Falls back to default values if parsing fails.

    Args:
        state: Current OrderState (unused).
        agent_output: Output from the risk agent.

    Returns:
        Partial dict to update Global State fields.
    """
    if agent_output.error:
        return {
            "risk_score": 0,
            "risk_level": "error",
            "risk_flags": [f"Agent error: {agent_output.error}"],
            "error": agent_output.error,
        }

    result = _parse_risk_result(agent_output.business_result)
    return {**result, "error": ""}


def _parse_risk_result(text: str) -> dict:
    """Attempt to parse structured risk data from agent text output.

    Args:
        text: Raw text from the risk agent's business_result.

    Returns:
        Dict with risk_score, risk_level, risk_flags extracted.
    """
    import re

    score = 0
    level = "unknown"
    flags: list[str] = []

    score_match = re.search(r"Risk Score:\s*(\d+)", text)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            pass

    level_match = re.search(r"Risk Level:\s*(\w+)", text)
    if level_match:
        level = level_match.group(1).lower()

    flags_section = re.search(r"Flags?\s*:\s*(.+?)(?:\n|$)", text)
    if flags_section:
        flags_text = flags_section.group(1).strip()
        flags = [f.strip().lstrip("- ") for f in flags_text.split(",") if f.strip()]

    return {
        "risk_score": min(max(score, 0), 100),
        "risk_level": level,
        "risk_flags": flags,
    }


def decision_condition(state: OrderState) -> str:
    """Determine the next routing step based on risk assessment.

    This is a deterministic conditional edge: the routing decision is
    made by code, not by an agent.

    Args:
        state: Current OrderState.

    Returns:
        Routing key: 'approved', 'held', or 'blocked'.
    """
    score = state.risk_score

    if score >= 70:
        return "blocked"
    elif score >= 50:
        return "held"
    else:
        return "approved"


def make_decision(state: OrderState) -> dict:
    """Terminal node: set the final_decision and reason based on risk score.

    Args:
        state: Current OrderState.

    Returns:
        Partial dict with final_decision and decision_reason.
    """
    score = state.risk_score
    level = state.risk_level

    if score >= 70:
        return {
            "final_decision": "blocked",
            "decision_reason": f"High risk score ({score}/100, level: {level}). Order requires manual review.",
        }
    elif score >= 50:
        return {
            "final_decision": "held",
            "decision_reason": f"Elevated risk score ({score}/100, level: {level}). Additional verification needed.",
        }
    else:
        return {
            "final_decision": "approved",
            "decision_reason": f"Acceptable risk score ({score}/100, level: {level}). Order approved.",
        }


class OrderProcessingWorkflow(BaseWorkflow):
    """Order processing workflow: search → risk → decision.

    Demonstrates the three-layer architecture:
      - SearchAgent + RiskAgent (Agent layer)
      - web_search + risk_check (Skill layer)
      - This workflow (Workflow layer)
    """

    def build(
        self,
        config: Optional[AppConfig] = None,
        skills_path: str = "examples/domain_skills/",
        prompts_path: str = "examples/domain_agents/prompts/",
    ) -> None:
        """Assemble the workflow graph.

        Args:
            config: Optional pre-loaded AppConfig. Loads from config.yaml if None.
            skills_path: Path to the domain_skills directory.
            prompts_path: Path to the agent prompts directory.
        """
        from examples.domain_agents.search_agent import SearchAgent
        from examples.domain_agents.risk_agent import RiskAgent
        from liteagent.skill_registry import SkillRegistry

        if config is None:
            config = load_config()

        llm = create_llm(config.llm)

        sr = SkillRegistry()
        sr.discover(skills_path)

        s_prompt = f"{prompts_path}/search_agent.md"
        r_prompt = f"{prompts_path}/risk_agent.md"

        search_agent = SearchAgent(
            name="search_agent",
            system_prompt_path=s_prompt,
            llm=llm,
            skill_names=["web_search"],
            skill_registry=sr,
        )

        risk_agent = RiskAgent(
            name="risk_agent",
            system_prompt_path=r_prompt,
            llm=llm,
            skill_names=["risk_check"],
            skill_registry=sr,
        )

        self.add_agent_node("search_node", search_agent, search_input_mapper, search_reducer)
        self.add_agent_node("risk_node", risk_agent, risk_input_mapper, risk_reducer)

        self.set_entry_point("search_node")
        self.add_edge("search_node", "risk_node")

        self._graph.add_node("approved_node", make_decision)
        self._graph.add_node("held_node", make_decision)
        self._graph.add_node("blocked_node", make_decision)

        self.add_conditional_edges(
            "risk_node",
            decision_condition,
            {
                "approved": "approved_node",
                "held": "held_node",
                "blocked": "blocked_node",
            },
        )

        self.set_finish_point("approved_node")
        self.set_finish_point("held_node")
        self.set_finish_point("blocked_node")
