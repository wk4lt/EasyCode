"""LiteAgent interactive REPL — order processing + code generation.

Usage:
    python examples/main.py
    python examples/main.py --config path/to/config.yaml

Commands (type at the liteagent> prompt):
    process             Enter a new order interactively and run the workflow
    demo                Run batch demo with 4 sample orders
    learn <dir> [...]   Index design doc + implementation pairs into RAG
    generate <d> <t> <o>  Generate code from design doc + test file
    config              Show current LLM, path, and RAG configuration
    verbose [on|off]    Toggle verbose logging on/off
    help                Show this help
    quit                Exit the REPL
"""

import logging
import os
import shlex
import sys

from liteagent.core.config import create_llm, load_config
from liteagent.core.logger import configure_root_logger


BANNER = r"""
╔══════════════════════════════════════════════╗
║          LiteAgent Order Processing          ║
║     Skill → Agent → Workflow Architecture    ║
║         + CodeGen: Learn → Generate          ║
╚══════════════════════════════════════════════╝"""

HELP_TEXT = """
Commands:
  process             Enter a new order interactively and run the workflow
  demo                Run batch demo with 4 sample orders
  learn <dir> [...]   Index design doc + implementation pairs into RAG
  generate <d> <t> <o>  Generate code from design doc + test file
  config              Show current configuration
  raginfo             Show RAG store statistics
  verbose [on|off]    Toggle verbose (DEBUG) logging
  help                Show this help
  quit, exit          Exit the REPL

Order workflow: search → risk → decision
CodeGen workflow: read → retrieve → generate → test
"""

TIER_CHOICES = {"new", "bronze", "silver", "gold"}
REGION_CHOICES = {"NA", "EU", "AS", "AF", "SA", "OC"}


def _prompt(prompt_text: str, default: str = "", choices: set[str] | None = None) -> str:
    """Prompt the user for input with optional validation.

    Args:
        prompt_text: Display text for the prompt.
        default: Default value if user enters empty string.
        choices: If set, only accept values in this set (case-insensitive).

    Returns:
        The validated input string.
    """
    while True:
        if default:
            value = input(f"  {prompt_text} [{default}]: ").strip()
            if not value:
                return default
        else:
            value = input(f"  {prompt_text}: ").strip()

        if choices and value.lower() not in choices:
            valid = ", ".join(sorted(choices))
            print(f"  Invalid choice. Must be one of: {valid}")
            continue

        return value


def cmd_process(wf_cls, state_cls, config):
    """Interactive order entry — prompt for fields, then run the workflow."""
    print("\n  Enter order details (press Enter for defaults):\n")

    order_id = _prompt("Order ID", default="ORD-001")
    customer_name = _prompt("Customer Name", default="Alice Johnson")
    customer_tier = _prompt("Customer Tier", default="new", choices=TIER_CHOICES)
    raw_amount = _prompt("Amount (USD)", default="1000.00")

    try:
        amount = float(raw_amount)
    except ValueError:
        print(f"  Invalid amount '{raw_amount}', using 1000.00")
        amount = 1000.00

    region = _prompt("Region", default="NA", choices=REGION_CHOICES)
    product = _prompt("Product", default="Electronics")

    initial_state = state_cls(
        order_id=order_id,
        customer_name=customer_name,
        customer_tier=customer_tier.lower(),
        amount=amount,
        region=region.upper(),
        product=product,
        search_query=f"background information for {customer_name} and {product}",
    )

    print(f"\n  Running workflow: search_node → risk_node → decision_node ...")

    wf = wf_cls(state_model=state_cls)
    wf.build(config=config)

    try:
        result = wf.invoke(initial_state)
    except Exception as e:
        print(f"\n  ERROR: Workflow execution failed: {e}")
        return

    print(f"\n  {'─'*40}")
    print(f"  Decision:    {result.final_decision.upper()}")
    print(f"  Reason:      {result.decision_reason}")
    print(f"  Risk Score:  {result.risk_score}/100 ({result.risk_level})")
    flags = ", ".join(result.risk_flags) if result.risk_flags else "none"
    print(f"  Risk Flags:  {flags}")
    if result.search_result:
        print(f"\n  --- Search Result (first 300 chars) ---")
        print(f"  {result.search_result[:300]}")
    if result.error:
        print(f"\n  --- Error ---")
        print(f"  {result.error}")
    print(f"  {'─'*40}\n")


def cmd_demo(wf_cls, state_cls, config):
    """Run the 4 sample orders and show a summary table."""
    orders = [
        {"id": "ORD-001", "name": "Alice Johnson", "tier": "gold", "amount": 1500.00, "region": "NA", "product": "Laptop Computer"},
        {"id": "ORD-002", "name": "Bob Smith", "tier": "new", "amount": 15000.00, "region": "AF", "product": "Industrial Equipment"},
        {"id": "ORD-003", "name": "Charlie Brown", "tier": "silver", "amount": 6000.00, "region": "AS", "product": "Smartphone"},
        {"id": "ORD-004", "name": "Diana Prince", "tier": "bronze", "amount": 300.00, "region": "EU", "product": "Books"},
    ]

    wf = wf_cls(state_model=state_cls)
    wf.build(config=config)

    rows = []
    print(f"\n  {'ID':<10} {'Customer':<18} {'Amount':>10} {'Region':<8} {'Decision':<10} {'Risk'}")
    print(f"  {'─'*10} {'─'*18} {'─'*10} {'─'*8} {'─'*10} {'─'*6}")

    for o in orders:
        state = state_cls(
            order_id=o["id"],
            customer_name=o["name"],
            customer_tier=o["tier"],
            amount=o["amount"],
            region=o["region"],
            product=o["product"],
            search_query=f"background information for {o['name']} and {o['product']}",
        )
        try:
            result = wf.invoke(state)
            decision = result.final_decision.upper()
            risk = f"{result.risk_score}/100"
        except Exception as e:
            decision = "ERROR"
            risk = str(e)[:20]

        print(f"  {o['id']:<10} {o['name']:<18} ${o['amount']:>9,.2f} {o['region']:<8} {decision:<10} {risk}")

    print()


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
        model = llm.get("model", "?")
        provider = llm.get("provider", "?")
        temp = llm.get("temperature", "?")
        max_tok = llm.get("max_tokens", "?")
        api_key = llm.get("api_key", "")
        key_display = "***" if api_key and api_key != "${OPENAI_API_KEY}" else api_key

        print(f"  Provider:     {provider}")
        print(f"  Model:        {model}")
        print(f"  Temperature:  {temp}")
        print(f"  Max tokens:   {max_tok}")
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
        print("  Example: learn examples/domain_skills/search_tools/")
        return

    from liteagent.core.rag_store import create_rag_store
    from examples.workflows.index_workflow import IndexWorkflow
    from examples.workflows.states import IndexState

    print(f"\n  Scanning directories for SKILL_*.md + *_impl.py pairs...")

    rag_store = create_rag_store(
        persist_dir=config.rag.chroma_path,
        api_key=config.llm.api_key,
        embedding_model=config.rag.embedding_model,
        collection_name=config.rag.collection_name,
    )

    wf = IndexWorkflow(state_model=IndexState)

    initial_state = IndexState(example_dirs=list(args))

    wf.build(config=config, rag_store=rag_store)

    try:
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
    print(f"  Skipped:   {result.skipped_count}")
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

    design_path, test_path, output_path = args[0], args[1], args[2]

    from pathlib import Path
    from liteagent.core.rag_store import create_rag_store
    from examples.workflows.codegen_workflow import CodeGenWorkflow
    from examples.workflows.states import CodeGenState

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

    rag_store = create_rag_store(
        persist_dir=config.rag.chroma_path,
        api_key=config.llm.api_key,
        embedding_model=config.rag.embedding_model,
        collection_name=config.rag.collection_name,
    )

    print(f"    RAG:     {rag_store.count()} examples available")

    wf = CodeGenWorkflow(state_model=CodeGenState)

    initial_state = CodeGenState(
        design_doc_path=design_path,
        test_file_path=test_path,
        output_path=output_path,
    )

    wf.build(config=config, rag_store=rag_store)

    result = None
    try:
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
    from liteagent.core.rag_store import create_rag_store

    rag_store = create_rag_store(
        persist_dir=config.rag.chroma_path,
        api_key=config.llm.api_key,
        embedding_model=config.rag.embedding_model,
        collection_name=config.rag.collection_name,
    )

    print(f"\n  RAG Store Info")
    print(f"  {'─'*30}")
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


def main():
    """Entry point — build the REPL loop."""

    config_path = "config.yaml"
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--config":
        config_path = argv[1]

    verbose = False

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

    from examples.workflows.order_processing_wf import OrderProcessingWorkflow
    from examples.workflows.states import OrderState

    print(BANNER)

    if config:
        print(f"  Model: {config.llm.model}  |  Log: {'DEBUG' if verbose else 'INFO'}")
    else:
        print("  WARNING: Config not loaded. 'process' and 'demo' are disabled.")
    print(f'  Type "help" for commands, "quit" to exit.\n')

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

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "config":
            cmd_config(config_path)
        elif cmd == "verbose":
            verbose = _toggle_verbose(verbose, args)
        elif cmd == "process":
            if config is None:
                print("  Cannot process: no config loaded. Check config.yaml exists.")
                continue
            cmd_process(OrderProcessingWorkflow, OrderState, config)
        elif cmd == "demo":
            if config is None:
                print("  Cannot run demo: no config loaded. Check config.yaml exists.")
                continue
            cmd_demo(OrderProcessingWorkflow, OrderState, config)
        elif cmd == "learn":
            if config is None:
                print("  Cannot learn: no config loaded. Check config.yaml exists.")
                continue
            cmd_learn(config, args)
        elif cmd == "generate" or cmd == "gen":
            if config is None:
                print("  Cannot generate: no config loaded. Check config.yaml exists.")
                continue
            cmd_generate(config, args)
        elif cmd == "raginfo":
            if config is None:
                print("  Cannot show RAG info: no config loaded.")
                continue
            cmd_raginfo(config)
        else:
            print(f"  Unknown command: {cmd}. Type 'help' for available commands.")

    print("  Goodbye.\n")


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


if __name__ == "__main__":
    main()
