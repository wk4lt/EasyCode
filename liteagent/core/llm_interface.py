"""Standardized LLM interface for LiteAgent framework.

Provides an abstract protocol and concrete adapters (OpenAI) to decouple
agent logic from specific LLM provider SDKs.

Layer: Core infrastructure.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """Represents a single tool/function call from the LLM."""

    id: str = Field(description="Unique identifier for this tool call.")
    name: str = Field(description="Name of the function to call.")
    arguments: dict = Field(default_factory=dict, description="Parsed keyword arguments for the function.")


class ChatResponse(BaseModel):
    """Standardized response from an LLM chat completion call."""

    content: Optional[str] = Field(default=None, description="Text content of the assistant message.")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Requested tool/function calls, if any.")
    token_usage: dict = Field(default_factory=dict, description="Token usage stats: {prompt_tokens, completion_tokens, total_tokens}.")


class LLMInterface(ABC):
    """Abstract protocol for LLM providers.

    All agent invocations go through this interface, never directly to a
    provider SDK. This enforces a clean seam for testing and provider swaps.
    """

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Send a chat completion request and return a standardized response.

        Args:
            messages: List of message dicts in OpenAI format.
            tools: Optional list of function/tool schemas in OpenAI format.
            tool_choice: How the model should choose tools ("auto", "none", "required").

        Returns:
            ChatResponse with content, tool_calls, and token_usage.

        Raises:
            RuntimeError: If the LLM call fails after retries.
        """
        ...


class OpenAIAdapter(LLMInterface):
    """OpenAI SDK adapter implementing the standardized LLM interface."""

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 4096, base_url: str = ""):
        """Initialize the OpenAI adapter.

        Args:
            api_key: OpenAI API key.
            model: Model identifier.
            temperature: Sampling temperature (0.0 for deterministic output).
            max_tokens: Maximum completion tokens.
            base_url: Optional custom base URL for OpenAI-compatible providers.
        """
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        """Lazy-import the OpenAI client to avoid import-time coupling."""
        from openai import OpenAI

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def chat_completion(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            _log.error("llm_call_failed", extra={"layer": "llm", "model": self._model, "error": str(e)})
            raise RuntimeError(f"OpenAI chat completion failed: {e}") from e

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            import json

            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        _log.debug("llm_call_done", extra={
            "layer": "llm",
            "model": self._model,
            "tokens": token_usage["total_tokens"],
            "tool_calls": len(tool_calls),
        })

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            token_usage=token_usage,
        )
