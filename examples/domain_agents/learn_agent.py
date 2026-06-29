"""Learn Agent for processing code example pairs.

Reads design documents and implementation code pairs, extracts patterns,
and stores them in the RAG vector database.

Layer: Agent layer (second layer).
"""

from liteagent.core.base_agent import BaseAgent


class LearnAgent(BaseAgent):
    """Agent specialized in learning from design+code example pairs."""

    def _build_user_message(self, local_input: dict) -> str:
        """Format the learning task as a user message.

        Args:
            local_input: Dict with 'example_dirs', 'design_doc_paths',
                'impl_paths', 'pair_count', and optional
                'clarification_response' from the user.

        Returns:
            Formatted learning request string.
        """
        example_dirs = local_input.get("example_dirs", [])
        design_doc_paths = local_input.get("design_doc_paths", [])
        impl_paths = local_input.get("impl_paths", [])
        pair_count = local_input.get("pair_count", 0)
        clarification_response = local_input.get("clarification_response", "")

        if clarification_response:
            return (
                f"USER RESPONSE TO YOUR CLARIFICATION QUESTION:\n"
                f"{clarification_response}\n\n"
                f"Continue processing the remaining {pair_count} example pairs "
                f"in directories: {', '.join(example_dirs)}."
            )

        if not example_dirs:
            return (
                "No example directories provided. Ask the user to specify "
                "directories containing design doc + implementation code pairs."
            )

        pairs_text = ""
        if design_doc_paths and impl_paths:
            pairs_text = "\nFiles discovered:\n"
            for i, (design, impl) in enumerate(zip(design_doc_paths, impl_paths), 1):
                pairs_text += f"  {i}. Design: {design}\n     Impl:   {impl}\n"
            pairs_text += f"\n"
        else:
            pairs_text = f"\nFound {pair_count} design+code pairs to process.\n\n"

        return (
            f"Process and learn from code examples in the following directories:\n\n"
            + "\n".join(f"  - {d}" for d in example_dirs)
            + f"\n\n{pairs_text}"
            + "For each pair listed above:\n"
            "1. Read the .md design document using file_reader\n"
            "2. Read the corresponding .py implementation using file_reader\n"
            "3. If the design or code is incomplete, ambiguous, or mismatched, "
            "STOP and request clarification immediately.\n"
            "4. If the pair is valid, embed it into RAG using code_embedder\n"
            "5. Report progress as you process each pair"
        )
