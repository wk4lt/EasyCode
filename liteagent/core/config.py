"""Configuration loader for LiteAgent.

Reads config.yaml at startup, resolves ${ENV_VAR} placeholders, and
provides typed Pydantic models for all settings. Also includes a
factory for creating LLM adapters from config.

Usage::

    from liteagent.core.config import load_config, create_llm

    config = load_config("config.yaml")
    llm = create_llm(config.llm)

Layer: Core infrastructure.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from liteagent.core.llm_interface import LLMInterface, OpenAIAdapter


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default="openai", description="Provider name: openai.")
    model: str = Field(default="gpt-4o", description="Model identifier.")
    api_key: str = Field(default="", description="API key for the provider.")
    base_url: str = Field(default="", description="Optional custom base URL.")
    temperature: float = Field(default=0.0, description="Sampling temperature (0=deterministic).")
    max_tokens: int = Field(default=4096, description="Maximum completion tokens.")


class RAGConfig(BaseModel):
    """RAG / vector store configuration."""

    chroma_path: str = Field(default="./chroma_data/", description="Directory for ChromaDB persistent storage.")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model name.")
    collection_name: str = Field(default="code_examples", description="ChromaDB collection name.")


class AppConfig(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    skills: dict = Field(default_factory=dict)
    agents: dict = Field(default_factory=dict)
    rag: RAGConfig = Field(default_factory=RAGConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and parse a YAML configuration file.

    Resolves ${ENV_VAR} placeholders in string values against
    OS environment variables.

    Args:
        path: Path to the config.yaml file.

    Returns:
        An AppConfig instance with all settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is malformed or required keys are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    resolved = _resolve_env_vars(raw)
    return AppConfig(**resolved)


def create_llm(config: LLMConfig) -> LLMInterface:
    """Create an LLM adapter from configuration.

    Args:
        config: LLMConfig instance with provider settings.

    Returns:
        An LLMInterface implementation.

    Raises:
        ValueError: If the provider is unknown or API key is missing.
    """
    provider = config.provider.lower()

    if provider == "openai":
        if not config.api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in config.yaml or via env var.")
        return OpenAIAdapter(
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: '{config.provider}'. Supported: openai")


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in a nested structure.

    Args:
        obj: A dict, list, str, or scalar value from YAML.

    Returns:
        The same structure with all ${...} patterns replaced.
    """
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    elif isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), ""),
            obj,
        )
    return obj
