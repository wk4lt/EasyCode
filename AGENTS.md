# AGENTS.md

## Architecture spec

`Agent.md` is the canonical architecture document. Read it first. Every module in this
repo implements part of the Skill->Agent->Workflow three-layer architecture described
there.

## Tech stack (non-negotiable)

- Python 3.10+
- Pydantic v2 for all state validation
- LangGraph as the workflow/graph engine
- OpenAI SDK as the LLM adapter (wrapped behind a standard interface)
- `.md` files for skill contracts and agent system prompts

## Layer boundaries (enforced)

- **Skills** (`domain_skills/`): Stateless pure functions. Each Skill = one `.py` impl
  + one `.md` contract file. The `.md` is parsed at startup to generate the Function
  Calling schema. Never save context in a Skill.
- **Agents** (`domain_agents/`): Domain decision-makers. Private `_messages` list per
  instance (backed by `MemoryManager`, keyed by `agent_id`). Input is *only* what the
  Workflow Mapper passes -- never the full Global State. Output is an `AgentOutput`
  struct (no direct state mutation).
- **Workflows** (`workflows/`): Deterministic graph engine via LangGraph StateGraph.
  Global State is a Pydantic BaseModel with explicit defaults on every field. Every
  agent node needs an explicit `input_mapper` and `reducer`. Conditional routing must
  use `if/else` on deterministic fields of Global State.

## Hard rules for code generation

1. **No Agent-to-Agent calls.** Agent A must never invoke Agent B. All cross-domain
   coordination happens in the Workflow graph.
2. **No state passthrough.** Never pass the full Global State to an Agent. Extract only
   needed fields via an input_mapper.
3. **Reducer always.** Agent output must be merged back into Global State via a
   reducer function, not by direct assignment.
4. **Defensive Skill layer.** Every Skill impl must catch exceptions and return
   standardized error dicts -- never let an unhandled exception crash the graph.
5. **Pydantic defaults.** Every field on every Global State model must have a default
   value.
6. **Google-style docstrings** on all public classes and interfaces.
7. **Prompts are `.md` files** under `domain_agents/prompts/`, loaded at runtime.

## Directory structure

```
liteagent/                  # Framework package
├── core/                   # Infrastructure
│   ├── base_skill.py
│   ├── base_agent.py
│   ├── base_workflow.py
│   ├── memory_manager.py
│   ├── llm_interface.py
│   ├── contract_parser.py
│   ├── config.py
│   └── logger.py
└── skill_registry.py

examples/                   # Demonstration project
├── domain_skills/          # Skill implementations (.py + .md pairs)
│   └── codegen_tools/      # CodeGen skills (file_reader, embedder, retriever, generator, tester)
├── domain_agents/          # Agent definitions + prompts/
│   ├── prompts/
│   │   ├── learn_agent.md  # CodeGen learn agent prompt (with [NEED_CLARIFICATION] checkpoint)
│   │   └── gen_agent.md    # CodeGen generate agent prompt (with [NEED_CLARIFICATION] checkpoint)
│   ├── learn_agent.py
│   └── gen_agent.py
├── workflows/              # Workflow graphs + states.py
│   ├── states.py           # Global State models (IndexState, CodeGenState)
│   ├── index_workflow.py   # IndexWorkflow: discover → learn → [done|needs_user]
│   └── codegen_workflow.py # CodeGenWorkflow: read → retrieve → generate → test → [passed|retry|needs_user]
├── config.yaml
└── main.py
```

## Commands

```bash
# Install dependencies (including dev, optional rag extras)
pip install -e ".[dev]"

# Optional: install ChromaDB for persistent RAG (heavy deps: onnxruntime, kubernetes, etc.)
pip install -e ".[rag]"

# Run all tests (no env vars or services needed; LLM calls are mocked)
pytest tests/ -v

# Run a single test file
pytest tests/test_base_skill.py -v

# Run the example REPL (requires OPENAI_API_KEY env var or in config.yaml)
python examples/main.py
```

## CodeGen workflows

Two new workflows for learning from examples and generating code:

**IndexWorkflow** (`index_workflow.py`): Scans directories for `SKILL_*.md` + `*_impl.py` pairs and indexes them into RAG.
- Flow: `discover_node → learn_node → [done | needs_user]`
- REPL: `learn <dir1> [dir2 ...]`

**CodeGenWorkflow** (`codegen_workflow.py`): Takes a design doc (.md) + test file (.py), retrieves similar examples from RAG, generates implementation code, runs tests, and fixes failures iteratively (up to 5 retries).
- Flow: `read_inputs → gen_node → [passed | retry | needs_user | failed]`
- REPL: `generate <design.md> <test.py> <output.py>`

**Checkpoint / Human-in-the-loop**: Both agents are instructed to output `[NEED_CLARIFICATION]` markers when ambiguous. The workflow reducers detect this and set `needs_clarification=True` + `final_status="needs_user"`. The REPL prompts the user, then re-invokes the workflow with `clarification_response` filled. Agents receive the user's response via `local_input["clarification_response"]`.

**RAG backends**: `liteagent/core/rag_store.py` provides two backends:
- `ChromaRAGStore`: Persistent, requires `chromadb` package. Uses OpenAI embeddings.
- `InMemoryRAGStore`: Ephemeral, always available. Uses TF-IDF + cosine similarity (falls back to word overlap).
- Factory: `create_rag_store()` auto-selects ChromaDB if installed, otherwise InMemory.

## Conventions and gotchas

- **`config.yaml` uses `${ENV_VAR}` substitution.** API keys should use `${OPENAI_API_KEY}` rather than plaintext. Both root `config.yaml` and `examples/config.yaml` are identical.
- **Skill auto-discovery prefers `SKILL_*.md` filenames** but falls back to any `.md` file. Implementation files must be `*_impl.py` in the same directory.
- **Every `Workflow` subclass must implement a `build()` method** that calls `add_agent_node`, `set_entry_point`, etc. The `CodeGenWorkflow` in `examples/workflows/codegen_workflow.py` is the reference pattern.
- **`examples/main.py` is an interactive REPL** (type `help` at the prompt). Use `--config` flag to point at a different config file.
- **Tests mock all LLM calls.** They do not require any API keys or network access.
- **Logger names matter.** `configure_root_logger()` sets the `liteagent` logger level and silences `openai`, `httpcore`, `httpx`, `urllib3` at WARNING. Always use `logging.getLogger(__name__)` with `extra={"layer": "...", "agent_name": "..."}` for structured output.
- **`AgentOutput` is a Pydantic model** with fields: `business_result: str`, `token_usage: dict`, `error: Optional[str]`. Reducer functions always receive `(state: StateType, agent_output: AgentOutput) -> dict`.
- **`MemoryManager` isolates agent contexts.** Each agent has a private message list keyed by `agent_id`. `BaseAgent.invoke()` clears history at the start of each call (`self._memory.clear(self.name)`).
