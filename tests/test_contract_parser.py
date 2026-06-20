"""Tests for contract_parser.py — parsing .md skill contract files."""

import pytest

from liteagent.core.contract_parser import (
    ParamDef,
    SkillContract,
    _extract_boundaries,
    _extract_name,
    _extract_parameters,
    _extract_section,
    parse_skill_contract,
)


class TestExtractName:
    def test_extracts_skill_name(self):
        content = "# Skill: web_search\n\n## Description\nSearch the web."
        assert _extract_name(content) == "web_search"

    def test_returns_empty_for_missing_header(self):
        assert _extract_name("## Description\nNo skill header.") == ""

    def test_handles_extra_whitespace(self):
        content = "# Skill:   my_tool  \n\n## Description\nSomething."
        assert _extract_name(content) == "my_tool"


class TestExtractSection:
    def test_extracts_description(self):
        content = "## Description\nThis is a test skill.\n\n## Parameters"
        assert _extract_section(content, "## Description") == "This is a test skill."

    def test_extracts_multiline_section(self):
        content = "## Description\nLine one.\nLine two.\n\n## Parameters"
        assert _extract_section(content, "## Description") == "Line one.\nLine two."

    def test_returns_empty_for_missing_section(self):
        assert _extract_section("## Description\nFoo.", "## Boundaries") == ""


class TestExtractParameters:
    SIMPLE_PARAMS = """## Parameters
- query (string): The search query. Required.
- limit (integer): Max results. Optional, default=10."""

    def test_extracts_parameters(self):
        params = _extract_parameters(self.SIMPLE_PARAMS)
        assert len(params) == 2

    def test_extracts_required_param(self):
        params = _extract_parameters(self.SIMPLE_PARAMS)
        query = params[0]
        assert query.name == "query"
        assert query.type == "string"
        assert query.required is True
        assert "search query" in query.description

    def test_extracts_optional_with_default(self):
        params = _extract_parameters(self.SIMPLE_PARAMS)
        limit = params[1]
        assert limit.name == "limit"
        assert limit.type == "integer"
        assert limit.required is False
        assert limit.default == "10"

    def test_returns_empty_for_no_params(self):
        assert _extract_parameters("## Parameters\n") == []

    def test_returns_empty_for_missing_section(self):
        assert _extract_parameters("## Description\nFoo.") == []


class TestExtractBoundaries:
    def test_extracts_key_value_pairs(self):
        content = "## Boundaries\n- Max length: 1000\n- Rate limit: 50/min"
        boundaries = _extract_boundaries(content)
        assert boundaries == {"Max length": "1000", "Rate limit": "50/min"}

    def test_returns_empty_for_missing_section(self):
        assert _extract_boundaries("## Parameters\n- x") == {}


class TestParseSkillContract:
    FULL_CONTRACT = """# Skill: web_search

## Description
Search the web for information and return relevant results.

## Parameters
- query (string): The search query string. Required.
- max_results (integer): Maximum number of results, 1-10. Optional, default=5.

## Boundaries
- Max query length: 500 characters
- Rate limit: 50 calls per minute
"""

    def test_parses_full_contract(self):
        contract = parse_skill_contract(self.FULL_CONTRACT)
        assert contract.name == "web_search"
        assert "Search the web" in contract.description
        assert len(contract.parameters) == 2
        assert contract.parameters[0].name == "query"
        assert contract.parameters[1].name == "max_results"
        assert contract.parameters[1].default == "5"
        assert "Max query length" in contract.boundaries

    def test_raises_for_missing_skill_header(self):
        with pytest.raises(ValueError, match="Skill contract must contain"):
            parse_skill_contract("## Description\nNo skill header.")
