"""Generate Agent for automated code generation.

Reads design documents and test cases, retrieves similar examples from RAG,
generates implementation code, runs tests, and fixes issues iteratively.

Layer: Agent layer (second layer).
"""

from liteagent.core.base_agent import BaseAgent


class GenerateAgent(BaseAgent):
    """Agent specialized in generating code from design docs and tests."""

    def _build_user_message(self, local_input: dict) -> str:
        """Format the generation task as a user message.

        Args:
            local_input: Dict with 'design_doc_path', 'test_file_path',
                'output_path', 'attempt', 'previous_code', 'test_error',
                and optional 'clarification_response'.

        Returns:
            Formatted generation request string.
        """
        design_path = local_input.get("design_doc_path", "")
        test_path = local_input.get("test_file_path", "")
        output_path = local_input.get("output_path", "")
        attempt = local_input.get("attempt", 0)
        previous_code = local_input.get("previous_code", "")
        test_error = local_input.get("test_error", "")
        clarification_response = local_input.get("clarification_response", "")

        if clarification_response:
            return (
                f"USER RESPONSE TO YOUR CLARIFICATION QUESTION:\n"
                f"{clarification_response}\n\n"
                f"Continue the code generation task.\n"
                f"  Design document: {design_path}\n"
                f"  Test file: {test_path}\n"
                f"  Output: {output_path}"
            )

        parts = [
            "Generate a complete Python implementation based on the following:\n",
            f"  Design document: {design_path}\n",
            f"  Test file: {test_path}\n",
            f"  Output file: {output_path}\n",
        ]

        if attempt > 0 and previous_code and test_error:
            parts.extend([
                f"\nFix attempt #{attempt}. The previous code failed tests:\n",
                f"  Error: {test_error[:500]}\n",
                "Please analyze the error, fix the code, and regenerate.",
            ])

        if attempt == 0:
            parts.extend([
                "\nSteps:\n",
                "1. Read the design document using file_reader\n",
                "2. Read the test file using file_reader\n",
                "3. Query code_retriever for similar examples\n",
                "4. IF the design is ambiguous or unclear, STOP and request clarification\n",
                "5. Generate code using code_generator with reference examples\n",
                "6. Run code_tester on the generated code\n",
                "7. If tests fail, analyze errors and retry (up to 3 fix attempts)",
            ])
        else:
            parts.extend([
                "\nSteps:\n",
                "1. Use code_generator to fix the code based on test errors\n",
                "2. Run code_tester again\n",
                "3. If still failing after 3 total attempts, request user guidance\n",
                "4. If the issue seems fundamental, request clarification",
            ])

        return "".join(parts)
