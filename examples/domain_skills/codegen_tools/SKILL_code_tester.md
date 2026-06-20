# Skill: code_tester

## Description
Run pytest against a generated implementation file and a test file. Returns the test result output including passed/failed status and error details.

## Parameters
- impl_file_path (string): Path to the generated implementation file (.py). Required.
- test_file_path (string): Path to the test file (.py). Required.

## Boundaries
- Test execution timeout: 30 seconds
- Tests run in a subprocess using pytest
- Only supports pytest-based tests
