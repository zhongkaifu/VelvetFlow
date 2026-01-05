## Troubleshooting (English)

> 中文版本: [troubleshooting.md](troubleshooting.md)

- **Missing OpenAI credentials**: Ensure `OPENAI_API_KEY` is set; planners and retrieval embedding calls require it.
- **Index not found**: Rebuild the action index with `build_action_index.py` or confirm `tools/action_index.json` exists.
- **Validation failures**: Review `ValidationError` locations for invalid bindings, missing exports, or unsupported node types; rerun validation after fixing DSL fields.
- **Execution stalls**: Check for async nodes returning `async_pending` and resume with saved `ExecutionCheckpoint` once external work finishes.
- **Logging**: Use console logs and event logs (`logging_utils.py`) to locate planner or executor issues.
