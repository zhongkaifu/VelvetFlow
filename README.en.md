## English Version

> 中文版本: [README.md](README.md)

VelvetFlow is a reusable LLM-driven workflow planning and execution demo. It ships hybrid retrieval, agent tool-driven structure and parameter completion, static validation with self-repair, a binding DSL, simulated executors, and DAG visualization. The planner now relies on the **OpenAI Agent SDK** (`agents.Agent`/`Runner`/`function_tool`) so one set of tool definitions works for both cloud agents and local debugging. The goal is to turn natural-language requirements into executable workflows with built-in security auditing, defensive fixes, and rollback paths to shorten PoC cycles and increase delivery certainty.

### Documentation & Navigation
- [docs/quickstart.en.md](docs/quickstart.en.md): Install dependencies, build the index, and run the demo quickly.
- [docs/core_concepts.en.md](docs/core_concepts.en.md): Data models and reference rules for Workflow/Node/Edge/Binding.
- [docs/workflow_dsl_schema.en.md](docs/workflow_dsl_schema.en.md): Field descriptions, node types, and a complete DSL example.
- [docs/advanced_guide.en.md](docs/advanced_guide.en.md): Retrieval tuning, model swaps, async node recovery, and other advanced topics.
- [docs/internal_design.en.md](docs/internal_design.en.md): Execution engine/state-machine diagrams, design patterns, and module layering.
- [docs/troubleshooting.en.md](docs/troubleshooting.en.md): Common errors, self-check steps, and log locations.
- [docs/openai_token_security.en.md](docs/openai_token_security.en.md): Secure storage for OpenAI credentials, integration-test toggles, and least-privilege tips.

### Architecture Overview (Execution Engine & State Transitions)
Hybrid retrieval feeds planning, validation/repair, and dynamic execution. The executor cycles through **ready → running → suspended → resume → completed**, supporting multiple async suspensions. Core state fields—`visited`, `reachable`, `blocked`, and binding context snapshots—live inside `ExecutionCheckpoint` for serialization and restore.

### Project Structure
```
VelvetFlow (repo root)
├── build_action_index.py        # Offline action retrieval index builder
├── simulation_data.json         # Simulated action return templates
├── velvetflow/
│   ├── action_registry.py       # Loads tools/business_actions/ with security metadata
│   ├── aggregation.py           # Aggregation/filtering helpers
│   ├── bindings.py              # Parameter binding DSL parsing/validation
│   ├── reference_utils.py       # Reference path normalization/replacement
│   ├── workflow_parser.py       # Workflow JSON loading and normalization
│   ├── metrics.py               # RunManager and metrics collection
│   ├── config.py                # Default OpenAI model config
│   ├── executor/                # Dynamic executor with condition/loop/async suspension
│   ├── logging_utils.py         # Console-friendly and event logs
│   ├── loop_dsl.py              # Loop node export schema helpers
│   ├── models.py                # Strongly typed Workflow/Node/Edge models and validation
│   ├── planner/                 # Planning, parameter fill, updates, and repair on the Agent SDK
│   │   ├── agent_runtime.py     # Agent SDK compatibility exports Agent/Runner/function_tool
│   │   ├── workflow_builder.py  # Mutable skeleton builder with implicit edges/depends_on
│   │   ├── structure.py         # Structure planning agent (requirement analysis + node build + param fill)
│   │   ├── params_tools.py      # Tool schemas for parameter completion (Jinja-only params)
│   │   ├── repair.py            # Self-repair agent with patching/naming tools
│   │   ├── orchestrator.py      # Pipeline orchestration and multi-round repair
│   │   ├── action_guard.py / approval.py / relations.py / tools.py and helpers
│   ├── verification/            # Shared static validation for planning/update/execution
│   ├── search.py                # Online hybrid retrieval using the offline index
│   ├── search_index.py          # Offline keyword/vector index build and persistence
│   ├── workflow_parser.py       # DSL parsing and incremental validation
│   └── visualization.py         # Render workflow DAGs as JPEG
├── tools/
│   ├── business_actions/        # Sample action library (HR/OPS/CRM namespaces)
│   ├── builtin.py               # Built-in retrieval/web scraping and other tools
│   └── action_index.json        # Default retrieval index cache
├── webapp/                      # FastAPI + Canvas visualization/interactive planner & executor
├── build_workflow.py            # End-to-end generation + visualization entrypoint
├── execute_workflow.py          # Execute a workflow JSON
└── LICENSE
```

Key highlights:
- **Agent-SDK-driven planner**: `planner/structure.py` handles structure planning and parameter fill using requirement analysis output and tools such as `update_node_params`. `planner/agent_runtime.py` centralizes `Agent`/`Runner`/`function_tool` exports for cloud or local execution.
- **Action registry**: `action_registry.py` loads `tools/business_actions/`, adds `requires_approval`/`allowed_roles`, and exposes `get_action_by_id`.
- **Offline index + online hybrid retrieval**: `search_index.py` builds keyword/vector indexes with `text-embedding-3-large`; `search.py` blends keyword scoring and Faiss cosine similarity while embedding only the query. `HybridActionSearchService` drives planner-time retrieval.
- **Workflow planning orchestrator**: `planner/orchestrator.py` implements `plan_workflow_with_two_pass` and updates, chaining Action Guard, Jinja normalization/local repair, and static validation with LLM repair.
- **DSL models and validation**: `models.py` defines strong types for nodes/edges/workflows; edges are derived from bindings/conditions and normalized before visualization/execution. Validation covers node types, implicit edges, and loop schemas, reporting via `ValidationError`.
- **Binding DSL**: Only **Jinja expressions** (e.g., `"{{ result_of.node.field }}"`) are accepted; validation and type inference reside in `bindings.py` and `verification/`.
- **Executor**: `executor/DynamicActionExecutor` validates action IDs, runs topological execution, supports condition (`params.expression`) and loop nodes (`body_subgraph` + `params.exports`), and can use `simulation_data.json` for simulated results. Logging uses `logging_utils.py`.
- **Visualization**: `visualization.py` renders workflows to JPEG with Unicode font fallback.
- **Jinja expression support**: `jinja_utils.py` builds a strict Jinja environment, folding literal templates into constants before validation/execution. Static checks prevent syntax errors; runtime evaluation powers conditions/aggregation.【F:velvetflow/jinja_utils.py†L8-L114】【F:velvetflow/verification/jinja_validation.py†L50-L189】【F:velvetflow/executor/conditions.py†L14-L114】

### Business Value & Roadmap
- **ROI**: Automates requirement analysis, action matching, parameter grounding, and validation to avoid manual handoffs and rollbacks—ideal for quickly validating HR/OPS/CRM automations.
- **Reliability**: Multi-round static checks, local auto-repair, and LLM repair return valid workflows even with imperfect model output, reducing launch risk.【F:velvetflow/planner/orchestrator.py†L361-L940】
- **Roadmap**: Potential additions include live enterprise data in the Action Registry, permission/cost signals in retrieval, or swapping the simulator for real backend calls to validate end-to-end quality.

### Usage
1. **Install dependencies (pip or uv)**
   - Using pip + venv:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     pip install agents
     ```
   - Using [uv](https://github.com/astral-sh/uv):
     ```bash
     uv venv --python 3.10
     source .venv/bin/activate
     uv sync
     uv add agents
     ```
2. **Set OpenAI credentials**: `export OPENAI_API_KEY="<your_api_key>"`
3. **(Optional) Rebuild the action index** when `tools/business_actions/` changes:
   ```bash
   python build_action_index.py --output tools/action_index.json --model text-embedding-3-large
   ```
4. **Run end-to-end generation**:
   ```bash
   python build_workflow.py
   ```
   Outputs: `workflow_output.json` (validated DSL) and `workflow_dag.jpg` (DAG image).
5. **Execute or visualize an existing workflow**:
   ```bash
   python execute_workflow.py --workflow-json workflow_output.json
   python render_workflow_image.py --workflow-json workflow_output.json --output workflow_dag.jpg
   ```
6. **Incrementally update an existing workflow**:
   ```bash
   python update_workflow.py path/to/workflow.json --requirement "add an approval step" --output workflow_updated.json
   ```
