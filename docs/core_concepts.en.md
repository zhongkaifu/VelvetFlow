## Core Concepts (English)

> 中文版本: [core_concepts.md](core_concepts.md)

This section outlines the data models and reference rules used by VelvetFlow.

### Workflow / Node / Edge
- **Workflow**: A DAG composed of nodes and edges describing end-to-end automation. Supports metadata, bindings, and validation for execution.
- **Node**: Typed units such as `action`, `condition`, and `loop` with parameters, bindings, and optional `body_subgraph` for loops. Nodes declare exports for downstream bindings.
- **Edge**: Derived from bindings or explicit `depends_on`. Edges are normalized before visualization or execution to ensure reachability and ordering.

### Bindings and References
- Bindings must use **Jinja expressions** like `"{{ result_of.node.field }}"`; legacy structured bindings are rejected.
- Reference rules ensure the binding path targets exported fields, respects loop scoping, and avoids blocked branches. Normalization aligns aliases and consistent paths.

### Validation Notes
- Static validation covers node types, implicit edges, loop subgraph schemas, and binding paths. Errors surface through `ValidationError` with precise locations.
- The planner and executor share the same validation logic to guarantee consistency between planned workflows and runtime execution.
