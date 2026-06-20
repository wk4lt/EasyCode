"""Code generator skill implementation.

Generates Python code using the LLM based on design documents, test cases,
and reference examples from the RAG store.

Layer: Skill layer (first layer).
"""

import logging
from typing import Optional

from liteagent.core.base_skill import BaseSkill
from liteagent.core.llm_interface import LLMInterface

_log = logging.getLogger(__name__)


class CodeGeneratorImpl(BaseSkill):
    """Generate Python code based on design docs and test cases."""

    def __init__(
        self,
        contract_path: Optional[str] = None,
        llm: Optional[LLMInterface] = None,
    ):
        """Initialize with an optional LLM reference.

        Args:
            contract_path: Path to the .md contract file.
            llm: An LLMInterface implementation for code generation.
        """
        super().__init__(contract_path)
        self._llm = llm

    def set_llm(self, llm: LLMInterface) -> None:
        """Inject an LLM instance.

        Args:
            llm: An LLMInterface implementation.
        """
        self._llm = llm

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a professional Python code generator. Generate complete, "
            "correct implementation code that satisfies a design specification "
            "and passes the provided tests.\n\n"
            "RULES:\n"
            "1. Output ONLY the Python code. No markdown code fences, no explanations.\n"
            "2. Include all necessary imports.\n"
            "3. The code must be Python 3.10+ compatible.\n"
            "4. Follow the style and patterns shown in reference examples when provided.\n"
            "5. Include Google-style docstrings on all public classes and functions.\n"
            "6. Handle edge cases and errors defensively.\n"
            "7. If the design spec is ambiguous, output [NEED_CLARIFICATION]: <your question> "
            "at the TOP of your response and STOP. Do not guess.\n"
        )

    def execute(
        self,
        design_content: str,
        test_content: str,
        reference_examples: str = "",
        previous_attempt: str = "",
        test_error: str = "",
    ) -> dict:
        """Generate Python code based on design docs and tests.

        Args:
            design_content: The design document specifying requirements.
            test_content: The test file content to pass.
            reference_examples: RAG-retrieved reference examples.
            previous_attempt: Previous code that failed (for fix iteration).
            test_error: Error output from previous test run.

        Returns:
            dict with 'status' and 'generated_code' or 'error'.
        """
        try:
            if self._llm is None:
                return {"status": "error", "error": "LLM not initialized. Call set_llm() first."}

            messages: list[dict] = [{"role": "system", "content": self._build_system_prompt()}]

            user_parts = [
                "## Design Document\n",
                design_content,
                "\n## Test Cases\n",
                test_content,
            ]

            if reference_examples:
                user_parts.extend([
                    "\n## Reference Examples (from similar code)\n",
                    reference_examples,
                ])

            if previous_attempt and test_error:
                user_parts.extend([
                    "\n## Previous Attempt (FAILED TESTS)\n",
                    "The following code failed the tests. Fix it.\n\n",
                    "### Code:\n```python\n",
                    previous_attempt,
                    "\n```\n\n### Test Error:\n",
                    test_error,
                ])
            elif previous_attempt:
                user_parts.extend([
                    "\n## Previous Attempt\n",
                    "Base your generation on this previous code:\n```python\n",
                    previous_attempt,
                    "\n```",
                ])
            else:
                user_parts.append(
                    "\nGenerate the complete implementation code now. "
                    "Output ONLY the code, no markdown fences."
                )

            user_message = "".join(user_parts)
            messages.append({"role": "user", "content": user_message[:80000]})

            response = self._llm.chat_completion(messages=messages)

            generated = response.content or ""

            if "```python" in generated:
                lines = generated.split("\n")
                code_lines = []
                in_code = False
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_code:
                            break
                        in_code = True
                        continue
                    if in_code:
                        code_lines.append(line)
                if code_lines:
                    generated = "\n".join(code_lines)
                else:
                    prefix = generated.index("```python") + len("```python")
                    suffix_marker = generated.find("```", prefix)
                    if suffix_marker > 0:
                        generated = generated[prefix:suffix_marker].strip()
                    else:
                        generated = generated[prefix:].strip()

            return {
                "status": "ok",
                "generated_code": generated,
                "token_usage": response.token_usage,
            }

        except Exception as e:
            return {"status": "error", "error": f"Code generation failed: {e}"}
