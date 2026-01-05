## Advanced Guide (English)

> 中文版本: [advanced_guide.md](advanced_guide.md)

- **Retrieval tuning**: Adjust the offline index build parameters (model choice, chunking) and rerank strategies in `search.py` to improve tool recall quality.
- **Model replacement**: Swap LLM or embedding models via `velvetflow/config.py` while keeping tool schemas stable; ensure validation and orchestrator expectations still hold.
- **Async node recovery**: Use the executor's suspension/resume flow to persist checkpoints, restore `ExecutionCheckpoint`, and continue once external async work completes.
- **Debugging planners**: Run planners locally with the Agent SDK (`planner/agent_runtime.py`) to inspect tool inputs/outputs and iterate on prompt/tool definitions.
- **Customization**: Extend `tools/business_actions/` with namespaced actions and rebuild the index; adjust Action Guard policies to enforce role/approval constraints.
