"""Tests for base_skill.py — Skill base class and function schema generation."""

import tempfile
from pathlib import Path

import pytest

from liteagent.core.base_skill import BaseSkill


class FakeSkill(BaseSkill):
    """A test skill that echos its input."""

    def execute(self, **kwargs) -> dict:
        return {"status": "ok", "echo": kwargs}


class TestBaseSkill:
    SIMPLE_MD = """# Skill: echo_tool

## Description
Echoes the input back for testing purposes.

## Parameters
- message (string): The message to echo. Required.
- repeat (integer): Number of times to repeat. Optional, default=1.

## Boundaries
- Max message length: 200 characters
"""

    @pytest.fixture
    def contract_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(self.SIMPLE_MD)
            return f.name

    def test_loads_contract_from_file(self, contract_file):
        skill = FakeSkill(contract_path=contract_file)
        assert skill.contract.name == "echo_tool"
        assert skill.contract_path == contract_file

    def test_get_function_schema(self, contract_file):
        skill = FakeSkill(contract_path=contract_file)
        schema = skill.get_function_schema()

        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "echo_tool"
        assert "Echoes the input" in func["description"]

        params = func["parameters"]
        assert params["type"] == "object"
        assert "message" in params["properties"]
        assert "repeat" in params["properties"]
        assert params["required"] == ["message"]

    def test_fails_when_contract_not_found(self):
        with pytest.raises(FileNotFoundError):
            FakeSkill(contract_path="/nonexistent/path/skill.md")

    def test_execute_returns_dict(self, contract_file):
        skill = FakeSkill(contract_path=contract_file)
        result = skill.execute(message="hello")
        assert result["status"] == "ok"
        assert result["echo"]["message"] == "hello"
