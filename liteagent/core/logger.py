"""Structured logging for the LiteAgent framework.

Thin wrapper around Python's standard `logging` module. Every log line
carries structured context (layer, agent name, node name) via `extra=`.

Usage::

    import logging
    _log = logging.getLogger(__name__)

    _log.info("agent_invoked", extra={"layer": "agent", "agent_name": self.name})
    _log.debug("llm_call", extra={"layer": "llm", "agent_name": agent.name, "model": "gpt-4o"})

Output::

    [INFO ] agent    | search_agent    : agent_invoked  model=gpt-4o tokens=150
    [DEBUG] llm      | gpt-4o          : llm_call tokens=150

Layer: Core infrastructure.
"""

import logging
import sys
from typing import Any, Optional

LAYER_WIDTH = 8
NAME_WIDTH = 18


class LiteAgentFormatter(logging.Formatter):
    """Human-readable structured formatter for LiteAgent logs."""

    def format(self, record: logging.LogRecord) -> str:
        layer = getattr(record, "layer", "core")
        agent_name = getattr(record, "agent_name", "")
        node_name = getattr(record, "node_name", "")
        model = getattr(record, "model", "")
        step = getattr(record, "step", "")

        if agent_name:
            display = agent_name
        elif node_name:
            display = node_name
        elif model:
            display = model
        else:
            display = record.name.rsplit(".", 1)[-1]

        layer_str = layer[:LAYER_WIDTH].ljust(LAYER_WIDTH)
        name_str = display[:NAME_WIDTH].ljust(NAME_WIDTH)
        level_str = record.levelname.ljust(8)

        msg = record.getMessage()
        extra_fields = self._format_extra(record)
        suffix = f"  {extra_fields}" if extra_fields else ""

        line = f"[{level_str}] {layer_str} | {name_str} : {msg}{suffix}"

        if record.exc_info and record.exc_info != (None, None, None):
            import traceback
            tb = "".join(traceback.format_exception(*record.exc_info)).rstrip()
            line += "\n" + tb

        return line

    @staticmethod
    def _format_extra(record: logging.LogRecord) -> str:
        known = {
            "layer", "agent_name", "node_name", "model", "step",
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "taskName",
        }
        pairs = []
        for key, value in sorted(vars(record).items()):
            if key not in known and not key.startswith("_"):
                pairs.append(f"{key}={value}")
        return "  ".join(pairs)


def configure_root_logger(
    level: int = logging.INFO,
    stream: Optional[Any] = None,
) -> None:
    """Configure the liteagent root logger once at application startup.

    Args:
        level: Logging level for all liteagent loggers.
        stream: Output stream (defaults to sys.stderr).
    """
    root = logging.getLogger("liteagent")
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(LiteAgentFormatter())
        root.addHandler(handler)

    for name in ("openai", "httpcore", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
