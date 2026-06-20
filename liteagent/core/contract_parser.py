"""Parses .md skill contract files into structured schema definitions.

Every Skill in LiteAgent has a corresponding .md contract file that defines
its name, description, parameters, and boundaries. This module parses those
files into machine-readable SkillContract objects that are used to generate
Function Calling schemas at startup.

Contract file format:

    # Skill: skill_name

    ## Description
    A brief description of what the skill does.

    ## Parameters
    - param_name (type): Description of the parameter. Required.
    - another_param (type): Description. Optional, default=value.

    ## Boundaries
    - Max input length: 1000 characters
    - Rate limit: 100 calls per minute

Layer: Core infrastructure.
"""

import re
from typing import Optional

from pydantic import BaseModel, Field


class ParamDef(BaseModel):
    """Definition of a single skill parameter."""

    name: str = Field(description="Parameter name.")
    type: str = Field(default="string", description="Data type (string, integer, number, boolean, array, object).")
    description: str = Field(default="", description="Human-readable description.")
    required: bool = Field(default=False, description="Whether this parameter is required.")
    default: Optional[str] = Field(default=None, description="Default value as a string, if any.")


class SkillContract(BaseModel):
    """Structured representation of a parsed skill .md contract."""

    name: str = Field(description="Skill name (from # Skill: header).")
    description: str = Field(default="", description="Skill description (from ## Description section).")
    parameters: list[ParamDef] = Field(default_factory=list, description="List of parameters.")
    boundaries: dict = Field(default_factory=dict, description="Key-value boundary constraints.")


def parse_skill_contract(md_content: str) -> SkillContract:
    """Parse a skill .md contract string into a SkillContract.

    Args:
        md_content: Raw markdown content from the .md file.

    Returns:
        A SkillContract with name, description, parameters, and boundaries populated.

    Raises:
        ValueError: If the contract has no 'Skill:' header.
    """
    name = _extract_name(md_content)
    if not name:
        raise ValueError("Skill contract must contain a '# Skill: <name>' header.")

    description = _extract_section(md_content, "## Description")
    parameters = _extract_parameters(md_content)
    boundaries = _extract_boundaries(md_content)

    return SkillContract(
        name=name.strip(),
        description=description.strip(),
        parameters=parameters,
        boundaries=boundaries,
    )


def _extract_name(content: str) -> str:
    """Extract the skill name from the '# Skill: <name>' header."""
    match = re.search(r"^#\s*Skill:\s*(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_section(content: str, header: str) -> str:
    """Extract the text under a markdown section header until the next header.

    Args:
        content: Full markdown text.
        header: Section header to find (e.g. '## Description').

    Returns:
        The section body text, or empty string if not found.
    """
    pattern = rf"^{re.escape(header)}\s*\n(.*?)(?=^#|\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_parameters(content: str) -> list[ParamDef]:
    """Extract parameter definitions from the '## Parameters' section.

    Recognizes lines of the form:
        - name (type): description. Required.
        - name (type): description. Optional, default=value.
    """
    section = _extract_section(content, "## Parameters")
    if not section:
        return []

    params = []
    for line in section.strip().split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        line = line[2:]

        match = re.match(r"(\w+)\s*\((\w+)\)\s*:\s*(.*)", line)
        if not match:
            continue

        name = match.group(1)
        ptype = match.group(2)
        rest = match.group(3).strip()

        required = "required" in rest.lower() or "Required." in rest
        default = None

        default_match = re.search(r"[Dd]efault\s*[:=]\s*(\S+)", rest)
        if default_match:
            default = default_match.group(1).rstrip(".,")

        desc = rest
        for suffix in ["Required.", "Optional.", "Required", "Optional"]:
            desc = desc.replace(suffix, "")
        desc = desc.rstrip("., ")

        params.append(
            ParamDef(
                name=name,
                type=ptype,
                description=desc.strip(),
                required=required,
                default=default,
            )
        )

    return params


def _extract_boundaries(content: str) -> dict:
    """Extract boundary rules from the '## Boundaries' section.

    Returns a dict mapping constraint keys to values.
    """
    section = _extract_section(content, "## Boundaries")
    if not section:
        return {}

    boundaries = {}
    for line in section.strip().split("\n"):
        line = line.strip().lstrip("- ")
        if ":" in line:
            key, _, value = line.partition(":")
            boundaries[key.strip()] = value.strip()
        else:
            boundaries[line.strip()] = ""

    return boundaries
