"""Search Agent for web information gathering.

Domain agent that uses the web_search skill to look up and summarize
information from the web.

Layer: Agent layer (second layer).
"""

from liteagent.core.base_agent import BaseAgent


class SearchAgent(BaseAgent):
    """Agent specialized in web information retrieval and summarization."""

    def _build_user_message(self, local_input: dict) -> str:
        """Format the search task as a user message.

        Args:
            local_input: Dict with 'search_task' key from the input_mapper.

        Returns:
            Formatted search request string.
        """
        search_task = local_input.get("search_task", "")
        query = local_input.get("query", search_task)

        if query:
            return f"Please search for information about: {query}\n\nSearch task context: {search_task}"
        else:
            return f"Please search for information about: {search_task}"
