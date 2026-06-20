"""Tests for skill_registry.py — skill discovery and registration."""

import tempfile
from pathlib import Path

import pytest

from liteagent.core.base_skill import BaseSkill
from liteagent.skill_registry import SkillRegistry


class DummySkill(BaseSkill):
    def execute(self, **kwargs) -> dict:
        return {"status": "ok", "args": kwargs}


class TestSkillRegistry:
    @pytest.fixture
    def registry(self):
        return SkillRegistry()

    def test_register_and_get_skill(self, registry):
        md_content = """# Skill: my_tool

## Description
A test tool.

## Parameters
- x (string): Input. Required.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            contract_path = f.name

        skill = DummySkill(contract_path=contract_path)
        registry.register_skill(skill)
        assert registry.get_skill("my_tool") is skill

    def test_get_schema(self, registry):
        md_content = """# Skill: my_tool

## Description
A test tool.

## Parameters
- x (string): Input. Required.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            contract_path = f.name

        skill = DummySkill(contract_path=contract_path)
        registry.register_skill(skill)
        schema = registry.get_schema("my_tool")
        assert schema is not None
        assert schema["function"]["name"] == "my_tool"

    def test_get_skill_missing_returns_none(self, registry):
        assert registry.get_skill("nonexistent") is None

    def test_get_schema_missing_returns_none(self, registry):
        assert registry.get_schema("nonexistent") is None

    def test_get_schemas_filters_missing(self, registry):
        md_content = """# Skill: tool_a
## Description
A.
## Parameters
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            contract_path = f.name

        skill = DummySkill(contract_path=contract_path)
        registry.register_skill(skill)

        schemas = registry.get_schemas(["tool_a", "nonexistent"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "tool_a"

    def test_list_skills(self, registry):
        md_content = "# Skill: tool_b\n## Description\nB.\n## Parameters\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            contract_path = f.name

        skill = DummySkill(contract_path=contract_path)
        registry.register_skill(skill)
        assert "tool_b" in registry.list_skills()

    def test_discover_from_directory(self, registry):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool_dir = Path(tmpdir) / "my_tools"
            tool_dir.mkdir()

            md_path = tool_dir / "SKILL_discovered.md"
            md_content = """# Skill: discovered

## Description
Discovered tool.

## Parameters
- input (string): Input param. Required.
"""
            md_path.write_text(md_content)

            impl_path = tool_dir / "discovered_impl.py"
            impl_content = '''"""Discovered tool implementation."""

from liteagent.core.base_skill import BaseSkill


class DiscoveredImpl(BaseSkill):
    def execute(self, **kwargs) -> dict:
        return {"status": "ok"}
'''
            impl_path.write_text(impl_content)

            count = registry.discover(str(tmpdir))
            assert count == 1
            assert registry.get_skill("discovered") is not None
            assert registry.get_schema("discovered") is not None

    def test_discover_empty_directory(self, registry):
        with tempfile.TemporaryDirectory() as tmpdir:
            count = registry.discover(str(tmpdir))
            assert count == 0

    def test_discover_nonexistent_path(self, registry):
        count = registry.discover("/nonexistent/path")
        assert count == 0

    def test_clear(self, registry):
        md_content = "# Skill: tool_c\n## Description\nC.\n## Parameters\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            contract_path = f.name

        skill = DummySkill(contract_path=contract_path)
        registry.register_skill(skill)
        registry.clear()
        assert registry.list_skills() == []
        assert registry.get_skill("tool_c") is None
