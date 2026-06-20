# Skill: code_generator

## Description
Generate Python implementation code based on a design document, test cases, and optional reference examples from the RAG store. Uses the LLM to produce complete, runnable code.

## Parameters
- design_content (string): The design document specifying requirements. Required.
- test_content (string): The test file content that the generated code must pass. Required.
- reference_examples (string): Reference examples retrieved from RAG to guide code generation. Optional, default="".
- previous_attempt (string): Previously generated code that failed tests, for iterative fixing. Optional, default="".
- test_error (string): Error output from running tests on the previous attempt. Optional, default="".

## Boundaries
- Max design_content length: 50000 characters
- Max test_content length: 50000 characters
- Generated code must be valid Python 3.10+
