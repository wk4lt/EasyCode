"""LiteAgent interactive REPL — CodeGen workflows.

Usage:
    python examples/main.py
    python examples/main.py --config path/to/config.yaml

Commands (type at the liteagent> prompt):
    learn <dir> [...]      Index design doc + implementation pairs into RAG
    generate <d> <t> <o>   Generate code from design doc + test file
    config                 Show current LLM, path, and RAG configuration
    raginfo                Show RAG store statistics
    verbose [on|off]       Toggle verbose (DEBUG) logging on/off
    help                   Show this help
    quit                   Exit the REPL
"""

import logging
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liteagent.core.config import load_config
from liteagent.core.logger import configure_root_logger

BANNER = r"""
╔══════════════════════════════════════════════╗
║           LiteAgent CodeGen REPL             ║
║     Skill → Agent → Workflow Architecture    ║
║         Learn → Generate → Test              ║
╚══════════════════════════════════════════════╝"""

HELP_TEXT = """
Commands:
  learn <dir> [...]      Index SKILL_*.md + *_impl.py pairs into RAG
  generate <d> <t> <o>   Generate code from design doc + test file
  config                 Show current configuration
  raginfo                Show RAG store statistics
  verbose [on|off]       Toggle verbose (DEBUG) logging
  help                   Show this help
  quit, exit             Exit the REPL

CodeGen workflow: read → retrieve → generate → test
Index workflow:  discover → learn → [done | needs_user]
"""


def _make_rag_store(config):
    from liteagent.core.rag_store import create_rag_store

    return create_rag_store(
        persist_dir=config.rag.chroma_path,
        embedding_provider=config.rag.embedding_provider,
        embedding_model=config.rag.embedding_model,
        collection_name=config.rag.collection_name,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )


def cmd_config(config_path: str):
    """Display the current configuration."""
    import yaml

    if not os.path.exists(config_path):
        print(f"\n  Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    print(f"\n  Config: {config_path}")
    print(f"  {'─'*30}")

    if raw:
        llm = raw.get("llm", {})
        print(f"  Provider:     {llm.get('provider', '?')}")
        print(f"  Model:        {llm.get('model', '?')}")
        print(f"  Temperature:  {llm.get('temperature', '?')}")
        print(f"  Max tokens:   {llm.get('max_tokens', '?')}")
        api_key = llm.get("api_key", "")
        key_display = "***" if api_key and api_key != "${OPENAI_API_KEY}" else api_key
        print(f"  API key:      {key_display}")

    skills = (raw or {}).get("skills", {})
    agents = (raw or {}).get("agents", {})
    print(f"  Skills path:  {skills.get('base_path', '?')}")
    print(f"  Prompts path: {agents.get('prompts_path', '?')}")

    rag = (raw or {}).get("rag", {})
    if rag:
        print(f"  RAG path:     {rag.get('chroma_path', '?')}")
        print(f"  RAG model:    {rag.get('embedding_model', '?')}")
        print(f"  RAG coll:     {rag.get('collection_name', '?')}")
    print()


def cmd_learn(config, args):
    """Index design doc + implementation code pairs from directories into RAG.

    Usage: learn <dir1> [dir2 ...]
    """
    if not args:
        print("  Usage: learn <dir1> [dir2 ...]")
        print("  Example: learn examples/domain_skills/codegen_tools/")
        return

    from examples.workflows.index_workflow import IndexWorkflow
    from examples.workflows.states import IndexState

    print(f"\n  Scanning directories for SKILL_*.md + *_impl.py pairs...")

    rag_store = _make_rag_store(config)

    try:
        wf = IndexWorkflow(state_model=IndexState)
        wf.build(config=config, rag_store=rag_store)

        initial_state = IndexState(example_dirs=list(args))
        result = wf.invoke(initial_state)
    except Exception as e:
        print(f"\n  ERROR: Indexing failed: {e}")
        return

    while result.status == "needs_user" and result.clarification_question:
        print(f"\n  {'─'*40}")
        print(f"  Agent needs clarification:")
        print(f"  {result.clarification_question}")
        print(f"  {'─'*40}")
        response = input("\n  Your response (or 'skip' to abort): ").strip()
        if response.lower() == "skip":
            print("  Aborted.")
            return
        result.clarification_response = response
        result.needs_clarification = False
        try:
            result = wf.invoke(result)
        except Exception as e:
            print(f"\n  ERROR: Workflow failed after clarification: {e}")
            return

    print(f"\n  {'─'*40}")
    print(f"  Status:    {result.status}")
    print(f"  Indexed:   {result.indexed_count}")
    if result.error:
        print(f"  Error:     {result.error}")
    print(f"  RAG store: {rag_store.count()} total documents")
    print(f"  {'─'*40}\n")


def cmd_generate(config, args):
    """Generate code from design doc + test file.

    Usage: generate <design.md> <test.py> <output.py>
    """
    if len(args) < 3:
        print("  Usage: generate <design.md> <test.py> <output.py>")
        return

    from pathlib import Path
    from examples.workflows.codegen_workflow import CodeGenWorkflow
    from examples.workflows.states import CodeGenState

    design_path, test_path, output_path = args[0], args[1], args[2]

    if not Path(design_path).exists():
        print(f"  Design doc not found: {design_path}")
        return
    if not Path(test_path).exists():
        print(f"  Test file not found: {test_path}")
        return

    print(f"\n  Generating code...")
    print(f"    Design:  {design_path}")
    print(f"    Test:    {test_path}")
    print(f"    Output:  {output_path}")

    rag_store = _make_rag_store(config)
    print(f"    RAG:     {rag_store.count()} examples available")

    try:
        wf = CodeGenWorkflow(state_model=CodeGenState)
        wf.build(config=config, rag_store=rag_store)

        initial_state = CodeGenState(
            design_doc_path=design_path,
            test_file_path=test_path,
            output_path=output_path,
        )
        result = wf.invoke(initial_state)
    except Exception as e:
        print(f"\n  ERROR: Code generation failed: {e}")
        return

    while result and result.final_status == "needs_user" and result.clarification_question:
        print(f"\n  {'─'*40}")
        print(f"  Agent needs clarification:")
        print(f"  {result.clarification_question}")
        print(f"  {'─'*40}")
        response = input("\n  Your response (or 'skip' to abort): ").strip()
        if response.lower() == "skip":
            print("  Aborted.")
            return
        result.clarification_response = response
        result.needs_clarification = False
        try:
            result = wf.invoke(result)
        except Exception as e:
            print(f"\n  ERROR: Workflow failed after clarification: {e}")
            return

    if result is None:
        return

    print(f"\n  {'─'*40}")
    print(f"  Status:    {result.final_status.upper()}")
    print(f"  Attempts:  {result.attempt}")
    if result.tests_passed:
        print(f"  Tests:     PASSED")
        print(f"  Output:    {result.output_path}")
    elif result.error:
        print(f"  Error:     {result.error}")
    else:
        print(f"  Tests:     FAILED")
        print(f"  Test output (last 500 chars):")
        print(f"  {result.test_result[-500:]}")
    print(f"  {'─'*40}\n")


def cmd_raginfo(config):
    """Show RAG store statistics."""
    print(f"\n  RAG Store Info")
    print(f"  {'─'*30}")
    rag_store = _make_rag_store(config)
    print(f"  Path:        {config.rag.chroma_path}")
    print(f"  Collection:  {config.rag.collection_name}")
    print(f"  Model:       {config.rag.embedding_model}")
    print(f"  Documents:   {rag_store.count()}")
    ids = rag_store.list_ids()
    if ids:
        print(f"  IDs ({len(ids)}):")
        for doc_id in ids[:20]:
            print(f"    - {doc_id}")
        if len(ids) > 20:
            print(f"    ... and {len(ids) - 20} more")
    print()


def _toggle_verbose(verbose: bool, args: list[str]) -> bool:
    """Toggle verbose logging on or off."""
    if args and args[0].lower() == "on":
        verbose = True
    elif args and args[0].lower() == "off":
        verbose = False
    else:
        verbose = not verbose

    level = logging.DEBUG if verbose else logging.INFO
    configure_root_logger(level=level)
    print(f"  Verbose logging: {'ON' if verbose else 'OFF'}")
    return verbose


def main():
    """Entry point — start the REPL loop."""
    config_path = "config.yaml"
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--config":
        config_path = argv[1]

    verbose = [False]

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"\n  Config file not found: {config_path}")
        print("  Create one with: cp examples/config.yaml config.yaml")
        print("  Then set your OPENAI_API_KEY in config.yaml or as an env var.")
        config = None
    except Exception as e:
        print(f"\n  Failed to load config: {e}")
        config = None

    configure_root_logger(level=logging.INFO)
    print(BANNER)

    if config:
        print(f"  Model: {config.llm.model}  |  Log: {'DEBUG' if verbose[0] else 'INFO'}")
    else:
        print("  WARNING: Config not loaded. Commands will be disabled.")
    print(f'  Type "help" for commands, "quit" to exit.\n')

    commands = {
        ("quit", "exit", "q"): lambda _args: "QUIT",
        "help": lambda _args: print(HELP_TEXT),
        "config": lambda _args: cmd_config(config_path),
        "verbose": lambda args: verbose.__setitem__(0, _toggle_verbose(verbose[0], args)),
        "raginfo": lambda _args: _require_config(cmd_raginfo, config, _args),
        "learn": lambda args: _require_config(cmd_learn, config, args),
        "generate": lambda args: _require_config(cmd_generate, config, args),
        "gen": lambda args: _require_config(cmd_generate, config, args),
    }

    while True:
        try:
            raw = input("liteagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = shlex.split(raw)
        cmd = parts[0].lower()
        args = parts[1:]

        handler = None
        for key, fn in commands.items():
            if isinstance(key, tuple):
                if cmd in key:
                    handler = fn
                    break
            elif key == cmd:
                handler = fn
                break

        if handler is None:
            print(f"  Unknown command: {cmd}. Type 'help' for available commands.")
            continue

        result = handler(args)
        if result == "QUIT":
            break

    print("  Goodbye.\n")


def _require_config(fn, config, args):
    """Only invoke fn if config is loaded."""
    if config is None:
        print("  Command unavailable: no config loaded. Check config.yaml exists.")
        return
    fn(config, args)


if __name__ == "__main__":
    main()
