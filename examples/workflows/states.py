"""Global State models for CodeGen workflows.

All fields have explicit defaults as required by the LiteAgent architecture.
Pydantic v2 validates state invariants on every state transition.

Layer: Workflow layer (third layer).
"""

from pydantic import BaseModel, Field


class IndexState(BaseModel):
    """Global state for the example indexing workflow.

    Scans directories for design doc + implementation code pairs
    and stores them in the RAG vector database.
    """

    example_dirs: list[str] = Field(default_factory=list, description="Directories to scan for .md + .py pairs.")
    design_doc_paths: list[str] = Field(default_factory=list, description="Discovered design doc paths.")
    impl_paths: list[str] = Field(default_factory=list, description="Discovered implementation file paths.")
    indexed_count: int = Field(default=0, description="Number of successfully indexed examples.")
    skipped_count: int = Field(default=0, description="Number of skipped pairs.")
    needs_clarification: bool = Field(default=False, description="True when agent requests user input.")
    clarification_question: str = Field(default="", description="The question the agent is asking.")
    clarification_response: str = Field(default="", description="User's response to the clarification question.")
    status: str = Field(default="pending", description="Indexing status: pending, processing, done, needs_user.")
    error: str = Field(default="", description="Error message if something fails.")


class CodeGenState(BaseModel):
    """Global state for the code generation workflow.

    Takes a design document and test file, retrieves similar examples
    from RAG, generates implementation code, and validates with tests.
    """

    design_doc_path: str = Field(default="", description="Path to the new design document (.md).")
    test_file_path: str = Field(default="", description="Path to the test file (.py).")
    output_path: str = Field(default="", description="Where to write the generated code.")
    design_content: str = Field(default="", description="Parsed content of the design document.")
    test_content: str = Field(default="", description="Content of the test file.")
    retrieved_examples: str = Field(default="", description="RAG-retrieved reference examples.")
    generated_code: str = Field(default="", description="The generated implementation code.")
    test_result: str = Field(default="", description="Pytest output from running tests.")
    tests_passed: bool = Field(default=False, description="Whether the generated code passes tests.")
    attempt: int = Field(default=0, description="Number of generation+test fix attempts.")
    max_attempts: int = Field(default=5, description="Maximum number of fix iterations.")
    needs_clarification: bool = Field(default=False, description="True when agent requests user input.")
    clarification_question: str = Field(default="", description="The question the agent is asking.")
    clarification_response: str = Field(default="", description="User's response to the clarification question.")
    final_status: str = Field(default="pending", description="Generation status: pending, generating, passed, failed, needs_user.")
    error: str = Field(default="", description="Error message if something fails.")
