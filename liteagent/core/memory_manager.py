"""Memory Manager for LiteAgent framework.

Manages per-agent private message histories, enforcing the context isolation
principle: each Agent maintains its own private _messages array that no other
Agent can read or mutate.

Layer: Core infrastructure.
"""

from typing import Optional


class MemoryManager:
    """Manages isolated message histories for multiple agents.

    Each agent is identified by a unique agent_id. Messages are stored
    as lists of dicts in OpenAI chat message format:

        {"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}

    The MemoryManager supports checkpoint/snapshot and restore for
    serialization and resumption of workflow state.
    """

    def __init__(self):
        """Initialize an empty memory store."""
        self._sessions: dict[str, list[dict]] = {}

    def add_message(self, agent_id: str, role: str, content: str, **extra) -> None:
        """Append a message to an agent's private history.

        Args:
            agent_id: Unique identifier for the agent.
            role: Message role (system, user, assistant, tool).
            content: Message content string.
            **extra: Additional fields appended to the message dict (e.g., tool_call_id, name).
        """
        if agent_id not in self._sessions:
            self._sessions[agent_id] = []
        message = {"role": role, "content": content, **extra}
        self._sessions[agent_id].append(message)

    def get_messages(self, agent_id: str) -> list[dict]:
        """Retrieve the full message history for an agent.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            List of message dicts. Returns empty list if no history exists.
        """
        return self._sessions.get(agent_id, [])

    def set_messages(self, agent_id: str, messages: list[dict]) -> None:
        """Replace the entire message history for an agent.

        Args:
            agent_id: Unique identifier for the agent.
            messages: New message list.
        """
        self._sessions[agent_id] = messages

    def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear message history.

        Args:
            agent_id: If provided, clears only that agent's history.
                If None, clears all sessions.
        """
        if agent_id:
            self._sessions.pop(agent_id, None)
        else:
            self._sessions.clear()

    def checkpoint(self) -> dict:
        """Create a serializable snapshot of all sessions.

        Returns:
            A deep copy of the sessions dict suitable for serialization.
        """
        import copy

        return copy.deepcopy(self._sessions)

    def restore(self, snapshot: dict) -> None:
        """Restore sessions from a previously captured checkpoint.

        Args:
            snapshot: A sessions dict returned by checkpoint().
        """
        self._sessions = dict(snapshot)
