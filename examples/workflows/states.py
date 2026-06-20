"""Global State models for the example order processing workflow.

All fields have explicit defaults as required by the LiteAgent architecture.
Pydantic v2 validates state invariants on every state transition.

Layer: Workflow layer (third layer).
"""

from pydantic import BaseModel, Field


class OrderState(BaseModel):
    """Global state for the order processing workflow.

    This is the single source of truth pushed through the LangGraph
    state graph. Every field must have a default value.
    """

    order_id: str = Field(default="", description="Unique order identifier.")
    customer_name: str = Field(default="", description="Customer name for the order.")
    customer_tier: str = Field(default="new", description="Customer tier: new, bronze, silver, gold.")
    amount: float = Field(default=0.0, description="Order amount in USD.")
    region: str = Field(default="NA", description="Destination region code.")
    product: str = Field(default="", description="Product being ordered.")

    search_query: str = Field(default="", description="Formatted search query for the search agent.")
    search_result: str = Field(default="", description="Search agent output summary.")

    risk_score: int = Field(default=0, description="Risk score 0-100 from risk agent.")
    risk_level: str = Field(default="", description="Risk level: low, medium, high, critical.")
    risk_flags: list[str] = Field(default_factory=list, description="Risk flags from assessment.")

    final_decision: str = Field(default="pending", description="Final decision: pending, approved, held, blocked.")
    decision_reason: str = Field(default="", description="Reason for the final decision.")

    error: str = Field(default="", description="Error message if any node fails.")


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
