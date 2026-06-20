"""Base Skill protocol for LiteAgent framework.

Skills are the bottom layer of the Skill→Agent→Workflow architecture.
They are stateless pure functions that perform domain-specific operations
(database queries, API calls, file operations, etc.).

Each Skill is defined by two files:
  1. A .md contract file describing the skill's interface.
  2. A .py implementation file containing the concrete logic.

Layer: Skill layer (first layer).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from liteagent.core.contract_parser import SkillContract, parse_skill_contract


class BaseSkill(ABC):
    """Abstract base class for all Skills.

    Skills are stateless: they must not store any context or conversation
    history between invocations. Each call to execute() is independent.

    Attributes:
        contract: Parsed SkillContract from the .md contract file.
        contract_path: Path to the .md contract file.
    """

    def __init__(self, contract_path: Optional[str] = None):
        """Initialize the skill, loading its .md contract.

        Args:
            contract_path: Path to the .md contract file. If None, the
                default contract path is derived from the class module.
        """
        self._contract_path = contract_path or self._default_contract_path()
        self.contract = self._load_contract()

    def _default_contract_path(self) -> str:
        """Derive the default .md contract path from the class module.

        Returns:
            Path string to the expected .md contract file.

        Raises:
            FileNotFoundError: If the contract file does not exist.
        """
        import inspect
        module_file = Path(inspect.getfile(self.__class__))
        skill_dir = module_file.parent
        skill_name = self.__class__.__name__.replace("Impl", "").lower()
        candidates = [
            skill_dir / f"SKILL_{skill_name}.md",
            skill_dir / f"{skill_name}.md",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        raise FileNotFoundError(
            f"No .md contract found for skill {self.__class__.__name__} "
            f"in {skill_dir}. Expected one of: {candidates}"
        )

    def _load_contract(self) -> SkillContract:
        """Load and parse the .md contract file.

        Returns:
            A SkillContract parsed from the .md file.
        """
        with open(self._contract_path, "r", encoding="utf-8") as f:
            content = f.read()
        return parse_skill_contract(content)

    @property
    def contract_path(self) -> str:
        """Path to the .md contract file."""
        return self._contract_path

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """Execute the skill's core logic.

        All implementations must be defensive: catch exceptions internally
        and return a standardized error dict instead of propagating failures.

        Args:
            **kwargs: Parameters matching the contract definition.

        Returns:
            A dict with at least a 'status' key ('ok' or 'error') and a
            'result' or 'error' key.
        """
        ...

    def get_function_schema(self) -> dict:
        """Generate an OpenAI-compatible function calling schema from the contract.

        Returns:
            A dict suitable for the 'tools' parameter in OpenAI chat completions.
        """
        properties = {}
        required_params = []

        for param in self.contract.parameters:
            type_mapping = {
                "string": "string",
                "str": "string",
                "integer": "integer",
                "int": "integer",
                "number": "number",
                "float": "number",
                "boolean": "boolean",
                "bool": "boolean",
                "array": "array",
                "list": "array",
                "object": "object",
                "dict": "object",
            }
            json_type = type_mapping.get(param.type, "string")

            prop_def: dict = {
                "type": json_type,
                "description": param.description,
            }
            if param.default is not None:
                prop_def["default"] = param.default

            properties[param.name] = prop_def

            if param.required:
                required_params.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.contract.name,
                "description": self.contract.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }
