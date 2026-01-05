## Internal Design (English)

> 中文版本: [internal_design.md](internal_design.md)

The internal design explains how planning, validation, and execution cooperate.

### Execution Engine and State Machine
- The executor transitions through **ready → running → suspended → resume → completed**, allowing multiple async suspensions. State snapshots (`ExecutionCheckpoint`) keep `visited`, `reachable`, `blocked`, and binding contexts for persistence and recovery.

### Orchestrator and Repair Flow
- `planner/orchestrator.py` runs `plan_workflow_with_two_pass` and incremental updates, chaining Action Guard, Jinja normalization/local repair, static validation, and LLM repair. Loop body references are pre-checked before parameter filling, and Action Guard whitelists `action_id` with retrieval-based replacement.

### Module Layers
- **Planner**: Requirement analysis, structure building, parameter completion, and repair on the Agent SDK (`planner/structure.py`, `params_tools.py`, `repair.py`).
- **Verification**: Shared validators across planning and execution (`verification/`).
- **Executor**: Dynamic execution with condition/loop support and suspension/resume (`executor/`).
- **Visualization & Tooling**: DAG rendering, action registry, search index, and logging utilities support end-to-end workflows.
