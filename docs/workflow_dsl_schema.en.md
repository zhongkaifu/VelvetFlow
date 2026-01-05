## Workflow DSL Schema (English)

> 中文版本: [workflow_dsl_schema.md](workflow_dsl_schema.md)

- **Workflow fields**: Metadata, nodes list, and optional edges (often derived from bindings). Ensures DAG semantics and reachability before execution.
- **Node types**:
  - `action`: Executes a registered action (`action_id`) with parameters/bindings and declares exports.
  - `condition`: Evaluates a Jinja `params.expression` to decide branch traversal.
  - `loop`: Iterates over a collection, executing `body_subgraph` and collecting `params.exports` each round.
- **Bindings**: Must use Jinja expressions, referencing `result_of.<node>.<field>` or loop exports. Normalization resolves aliases and ensures consistent paths.
- **Example**: A full workflow JSON example (see Chinese section above) shows nodes, bindings, exports, and edges as produced by the planner and validated before execution.
