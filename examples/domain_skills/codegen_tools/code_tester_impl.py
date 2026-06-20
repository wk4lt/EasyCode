"""Code tester skill implementation.

Runs pytest against generated implementation code and test files,
returning the test results for validation.

Layer: Skill layer (first layer).
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from liteagent.core.base_skill import BaseSkill


class CodeTesterImpl(BaseSkill):
    """Run tests against generated code using pytest."""

    TIMEOUT = 30

    def execute(self, impl_file_path: str, test_file_path: str) -> dict:
        """Run pytest on the generated code with the provided tests.

        Args:
            impl_file_path: Path to the implementation file.
            test_file_path: Path to the test file.

        Returns:
            dict with 'status', 'passed', 'stdout', 'stderr', 'return_code'.
        """
        try:
            impl_path = Path(impl_file_path)
            test_path = Path(test_file_path)

            if not impl_path.exists():
                return {"status": "error", "error": f"Implementation file not found: {impl_file_path}"}

            if not test_path.exists():
                return {"status": "error", "error": f"Test file not found: {test_file_path}"}

            result = subprocess.run(
                ["pytest", str(test_path), "-v", "--tb=short", "--color=no"],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                cwd=str(impl_path.parent),
                env={
                    **__import__("os").environ,
                    "PYTHONPATH": str(impl_path.parent) + ":" + __import__("os").environ.get("PYTHONPATH", ""),
                },
            )

            passed = result.returncode == 0

            return {
                "status": "ok",
                "passed": passed,
                "return_code": result.returncode,
                "stdout": result.stdout[:10000],
                "stderr": result.stderr[:10000],
            }

        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Test execution timed out after {self.TIMEOUT} seconds."}
        except Exception as e:
            return {"status": "error", "error": f"Test execution failed: {e}"}
