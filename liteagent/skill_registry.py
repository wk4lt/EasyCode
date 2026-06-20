"""Skill Registry for LiteAgent framework.

The SkillRegistry is the central lookup for all Skills available to the
system. At startup it scans domain_skills/ directories, parses .md contract
files, discovers matching implementation modules, and makes skills
available by name to agents.

Layer: Core infrastructure (cross-cutting).
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from liteagent.core.base_skill import BaseSkill
from liteagent.core.contract_parser import parse_skill_contract

_log = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for skill discovery, registration, and lookup.

    Skills are registered by name (from the contract). The registry
    can auto-discover skills by scanning directories for .md contract
    files and their matching *_impl.py implementation modules.

    Usage::

        registry = SkillRegistry()
        registry.discover("examples/domain_skills/")
        schema = registry.get_schema("google_search")
        skill = registry.get_skill("google_search")
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._skills: dict[str, BaseSkill] = {}
        self._schemas: dict[str, dict] = {}

    def register_skill(self, skill: BaseSkill) -> None:
        """Register a skill instance.

        Args:
            skill: An instantiated BaseSkill subclass.
        """
        name = skill.contract.name
        self._skills[name] = skill
        self._schemas[name] = skill.get_function_schema()

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Look up a registered skill by name.

        Args:
            name: Skill name (matches contract name).

        Returns:
            The BaseSkill instance or None if not found.
        """
        return self._skills.get(name)

    def get_schema(self, name: str) -> Optional[dict]:
        """Get the function calling schema for a registered skill.

        Args:
            name: Skill name (matches contract name).

        Returns:
            OpenAI-compatible function schema dict or None if not found.
        """
        return self._schemas.get(name)

    def get_schemas(self, skill_names: list[str]) -> list[dict]:
        """Get schemas for multiple skills by name.

        Args:
            skill_names: List of skill names.

        Returns:
            List of function schema dicts. Missing skills are silently omitted.
        """
        return [s for name in skill_names if (s := self._schemas.get(name)) is not None]

    def list_skills(self) -> list[str]:
        """List all registered skill names.

        Returns:
            Sorted list of skill names.
        """
        return sorted(self._skills.keys())

    def discover(self, base_path: str) -> int:
        """Auto-discover skills by scanning a directory tree.

        Walks the directory tree looking for .md contract files (SKILL_*.md).
        For each contract found, looks for a matching implementation file
        (*_impl.py) in the same directory, then imports and instantiates the
        skill class.

        Args:
            base_path: Root directory to scan for skills.

        Returns:
            Number of newly discovered skills.
        """
        count = 0
        base = Path(base_path)

        if not base.exists():
            _log.warning("discover_path_not_found", extra={"layer": "core", "path": base_path})
            return 0

        _log.info("discover_start", extra={"layer": "core", "path": base_path})

        for root, _dirs, files in os.walk(base):
            root_path = Path(root)
            md_files = [f for f in files if f.startswith("SKILL_") and f.endswith(".md")]

            if not md_files:
                md_files = [f for f in files if f.endswith(".md")]

            for md_file in md_files:
                md_path = root_path / md_file

                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    contract = parse_skill_contract(content)
                except Exception:
                    continue

                skill_name = contract.name

                if skill_name in self._skills:
                    continue

                impl_module = self._find_impl_module(root_path, md_file, skill_name)
                if impl_module is None:
                    continue

                skill_class = self._find_skill_class(impl_module)
                if skill_class is None:
                    continue

                try:
                    skill_instance = skill_class(contract_path=str(md_path))
                    self.register_skill(skill_instance)
                    _log.info("skill_registered", extra={"layer": "core", "skill": skill_name, "path": str(md_path)})
                    count += 1
                except Exception:
                    continue

        _log.info("discover_done", extra={"layer": "core", "count": count})
        return count

    def _find_impl_module(self, directory: Path, md_filename: str, skill_name: str) -> Optional[object]:
        """Find and import the implementation module for a skill.

        Looks for a *_impl.py file in the same directory. Tries parsing
        the md contract name as a hint for the module name.

        Args:
            directory: Directory containing the .md file.
            md_filename: The .md contract filename.
            skill_name: Parsed skill name from the contract.

        Returns:
            Imported module object, or None if not found.
        """
        impl_candidates = []

        for entry in directory.iterdir():
            if entry.suffix == ".py" and entry.name.endswith("_impl.py"):
                impl_candidates.append(entry)

        if not impl_candidates:
            return None

        impl_path = impl_candidates[0]
        module_name = impl_path.stem

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(impl_path))
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            return None

    def _find_skill_class(self, module: object) -> Optional[type]:
        """Find the first BaseSkill subclass in an imported module.

        Args:
            module: An imported Python module object.

        Returns:
            A BaseSkill subclass, or None if not found.
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseSkill)
                and attr is not BaseSkill
            ):
                return attr
        return None

    def clear(self) -> None:
        """Remove all registered skills."""
        self._skills.clear()
        self._schemas.clear()
