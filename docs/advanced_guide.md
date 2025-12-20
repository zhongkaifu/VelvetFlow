# 进阶指南

## Agent SDK 与规划工具
- 结构规划、参数补全与修复均基于 OpenAI Agent SDK 的 `Agent`/`Runner`/`function_tool` 运行，`planner/agent_runtime.py` 统一导出依赖，便于在云端 Agent 与本地兼容层之间切换。【F:velvetflow/planner/agent_runtime.py†L4-L26】【F:velvetflow/planner/structure.py†L1134-L1163】
- 结构规划工具在 `plan_workflow_structure_with_llm` 内联定义，借助闭包维护 `WorkflowBuilder`、动作候选集与覆盖度反馈，并在 finalize 时自动补齐 edges/depends_on；节点 params 会按节点类型或 action schema 过滤无关字段，降低 Agent 生成的噪声。【F:velvetflow/planner/structure.py†L603-L829】【F:velvetflow/planner/workflow_builder.py†L20-L160】【F:velvetflow/planner/workflow_builder.py†L179-L222】
- 参数补全过程按拓扑逐节点调用 `submit_node_params`/`validate_node_params`，在校验通过前不会写入结果，且会携带跨节点的绑定记忆以保持实体 ID/引用一致性。【F:velvetflow/planner/params.py†L195-L233】【F:velvetflow/planner/params.py†L430-L491】
- 修复阶段 Agent 可选择命名修复工具或提交补丁文本，未提交最终版本时会使用工作副本兜底，确保始终返回可序列化的 workflow。【F:velvetflow/planner/repair.py†L616-L756】

## 检索与模型调优
- 调整混合检索权重：`velvetflow/search.py` 的 `FeatureRanker` 允许通过 `feature_weights`、`keyword_weight`、`embedding_weight` 微调关键信号。
- 更换向量模型：`build_action_index.py` 与 `velvetflow/search_index.py` 默认使用 `text-embedding-3-large`，可以替换成自建 embedding 服务并更新索引。
- 过滤命名空间/域：构建索引时可在业务动作中设置 `domain/tags`，`HybridActionSearchService` 会在检索时尊重过滤条件。

## 异步节点与恢复
- 动作若返回 `{"status": "async_pending", "request_id": ...}` 会触发 `WorkflowSuspension`，序列化的 `ExecutionCheckpoint` 保存绑定上下文、可达节点集与剩余拓扑。
- 外部回调到达后，可将结果写回文件并执行：
  ```bash
  python execute_workflow.py --workflow-json <wf.json> \
    --resume-from <suspension.json> --tool-result-file <result.json>
  ```
- 也可在 Python 中通过 `DynamicActionExecutor.resume_from_suspension` 继续执行，执行器会自动补齐上次节点的 `params` 并恢复拓扑顺序。

## 自定义执行行为
- 替换模拟结果：`simulation_data.json` 定义默认的动作返回，可以通过 `--simulation-file` 指向自定义 JSON。
- Jinja 严格模式：`jinja_utils.py` 会在校验和执行前折叠纯字面量模板并阻断语法错误，可在条件或聚合参数中直接嵌入表达式。
- 权限校验：业务动作在 `action_registry.py` 中可声明 `allowed_roles`，执行时若用户角色不符会返回 `forbidden` 状态而不会中断整个工作流。
- parallel 占位：`parallel` 节点目前主要用于 UI 分组与可视化，不会触发真正的并发调度；如需控制实际执行顺序，请使用 `depends_on` 与绑定推导。

## 可观测性与日志
- 规划阶段：`planner/structure.py` 会记录每次 tool-calling、覆盖度缺失与 LLM 返回，便于复现失败案例。
- 执行阶段：`velvetflow/logging_utils.py` 提供结构化事件日志，配合 run_manager 可以收集指标（节点数、结果数、异步请求 ID）。
