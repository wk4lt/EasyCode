"""Base Agent protocol for LiteAgent framework.

Agents are the middle layer of the Skill→Agent→Workflow architecture.
Each Agent is a domain-specific decision-maker with private memory,
bound to a set of Skills and a system prompt.

Agents must never:
  - Receive the full Global State (only filtered local_input via Mapper).
  - Directly call another Agent.
  - Mutate Global State (output goes through a Reducer).

Layer: Agent layer (second layer).
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from liteagent.core.llm_interface import LLMInterface
from liteagent.core.memory_manager import MemoryManager

_log = logging.getLogger(__name__)


class AgentOutput(BaseModel):
    """Standardized output from an Agent invocation.

    This struct is returned to the Workflow layer. The Workflow's Reducer
    merges it back into Global State — the Agent never touches Global State
    directly.

    Attributes:
        business_result: The final business conclusion produced by the agent.
        token_usage: Aggregated token usage across all LLM calls in this invocation.
        error: Error message if the agent failed, None otherwise.
    """

    business_result: str = Field(default="", description="Final business conclusion from the agent.")
    token_usage: dict = Field(default_factory=dict, description="Aggregated token usage.")
    error: Optional[str] = Field(default=None, description="Error message if the agent failed.")


class BaseAgent(ABC):
    """Abstract base class for all domain Agents.

    Each Agent instance maintains:
      - A private _messages list (its memory wall).
      - A system prompt loaded from a .md file.
      - A reference to an LLM provider.
      - A reference to the skill registry for tool execution.

    Subclasses must implement _build_user_message() to define how
    local_input is translated into the first user message.
    """

    def __init__(
        self,
        name: str,
        system_prompt_path: str,
        llm: LLMInterface,
        skill_names: list[str],
        skill_registry: Any,
        memory: Optional[MemoryManager] = None,
        max_tool_rounds: int = 10,
    ):
        """Initialize the Agent.

        Args:
            name: Unique name for this agent (used as agent_id in MemoryManager).
            system_prompt_path: Path to the .md prompt file.
            llm: An LLMInterface implementation for chat completions.
            skill_names: Names of skills this agent is allowed to use.
            skill_registry: A SkillRegistry instance for resolving skill lookups.
            memory: Optional MemoryManager instance (creates one if None).
            max_tool_rounds: Maximum number of tool-calling loop iterations.
        """
        self.name = name
        self._llm = llm
        self._skill_names = skill_names
        self._skill_registry = skill_registry
        self._max_tool_rounds = max_tool_rounds
        self._memory = memory or MemoryManager()

        self._system_prompt = self._load_system_prompt(system_prompt_path)

    def _load_system_prompt(self, path: str) -> str:
        """Load the system prompt from a .md file.

        Args:
            path: Path to the .md prompt file.

        Returns:
            The file content as a string.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @property
    def _messages(self) -> list[dict]:
        """Retrieve the private message history for this agent."""
        return self._memory.get_messages(self.name)

    def _get_tool_schemas(self) -> list[dict]:
        """Get function calling schemas for this agent's registered skills.

        Returns:
            List of tool schema dicts in OpenAI format.
        """
        schemas = []
        for skill_name in self._skill_names:
            schema = self._skill_registry.get_schema(skill_name)
            if schema:
                schemas.append(schema)
        return schemas

    @abstractmethod
    def _build_user_message(self, local_input: dict) -> str:
        """Build the first user message from the mapper-provided local input.

        Subclasses define how to translate structured local_input into a
        natural-language prompt for the LLM.

        Args:
            local_input: Filtered dict from the Workflow's input_mapper.

        Returns:
            A string to use as the user message content.
        """
        ...

    def invoke(self, local_input: dict) -> AgentOutput:
        """Execute the agent with the given local input.

        This is the only public entry point for the Workflow to call.

        The agent loop:
          1. Build messages: system prompt + formatted user input.
          2. Call LLM with available tool schemas.
          3. If tool calls are requested, execute them and feed results back.
          4. Repeat until the LLM produces a final text response or max rounds reached.

        Args:
            local_input: Filtered dict from the Workflow's input_mapper.

        Returns:
            An AgentOutput with the agent's business conclusion and token usage.
        """
        ctx = {"layer": "agent", "agent_name": self.name}
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            _log.info("invoke_start", extra={**ctx, "skills": self._skill_names})
            self._memory.clear(self.name)

            system_msg = {"role": "system", "content": self._system_prompt}
            self._memory.add_message(self.name, "system", self._system_prompt)

            user_content = self._build_user_message(local_input)
            self._memory.add_message(self.name, "user", user_content)

            tools = self._get_tool_schemas()

            for round_idx in range(self._max_tool_rounds):
                response = self._llm.chat_completion(
                    messages=self._memory.get_messages(self.name),
                    tools=tools if tools else None,
                )

                for key in total_token_usage:
                    total_token_usage[key] += response.token_usage.get(key, 0)

                if response.tool_calls:
                    tool_names = [tc.name for tc in response.tool_calls]
                    _log.info("tool_calls", extra={**ctx, "step": f"round_{round_idx}", "tools": tool_names})

                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                },
                            }
                            for tc in response.tool_calls
                        ],
                    }
                    self._memory.add_message(self.name, "assistant", response.content or "", **{
                        k: v for k, v in assistant_msg.items() if k != "role" and k != "content"
                    })

                    for tc in response.tool_calls:
                        tool_output = self._execute_skill(tc.name, tc.arguments)
                        if tool_output.get("status") == "error":
                            _log.warning("tool_error", extra={**ctx, "tool": tc.name, "error": tool_output.get("error", "")})
                        self._memory.add_message(
                            self.name,
                            "tool",
                            json.dumps(tool_output, ensure_ascii=False),
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                else:
                    _log.info("invoke_done", extra={**ctx, "rounds": round_idx + 1, "total_tokens": total_token_usage["total_tokens"]})
                    return AgentOutput(
                        business_result=response.content or "",
                        token_usage=total_token_usage,
                    )

            _log.warning("max_rounds_exceeded", extra={**ctx, "max_rounds": self._max_tool_rounds})
            return AgentOutput(
                business_result="Max tool rounds exceeded without final response.",
                token_usage=total_token_usage,
                error="max_rounds_exceeded",
            )

        except Exception as e:
            _log.error("invoke_failed", extra={**ctx, "error": str(e)}, exc_info=True)
            return AgentOutput(
                business_result="",
                token_usage=total_token_usage,
                error=str(e),
            )

    def _execute_skill(self, skill_name: str, arguments: dict) -> dict:
        """Resolve and execute a skill by name.

        Args:
            skill_name: Name of the skill to execute (matches contract name).
            arguments: Keyword arguments parsed from the tool call.

        Returns:
            A dict with the skill execution result or error.
        """
        skill = self._skill_registry.get_skill(skill_name)
        if skill is None:
            return {"status": "error", "error": f"Skill '{skill_name}' not found in registry."}

        try:
            result = skill.execute(**arguments)
            return result
        except Exception as e:
            return {"status": "error", "error": f"Skill '{skill_name}' execution failed: {e}"}
