# 核心概念

## Workflow 与节点模型
- **Workflow**：包含 `nodes`、可选 `edges` 与元数据；edges 可由参数绑定或条件/循环出口自动推导，确保渲染与执行前结构闭合。
- **节点类型**：
  - `action` 节点绑定业务动作与参数 Schema。
  - `condition`/`switch` 节点按布尔或多分支结果选择下游，未命中的分支会被标记为阻断以避免重复执行。
  - `loop` 节点使用 `iter` 定义循环集合，`body_subgraph` 执行子图，`exports.items/aggregates` 收集逐轮与聚合结果。
- `velvetflow.models` 使用 Pydantic 强类型校验节点字段、loop 子图完整性与引用合法性，失败时抛出统一的 `ValidationError`。

## 绑定与上下文
- **BindingContext** 会在执行时携带 `results` 字典与 loop 局部上下文，供后续节点通过 `__from__` 引用上游输出。
- 聚合表达式 `__agg__` 支持 `identity`、`count`、`format_join`、`filter_map`、`pipeline` 等，引用路径会与动作/循环 schema 做兼容性检查。
- 绑定解析与 Jinja 模板折叠由 `bindings.py` 和 `jinja_utils.py` 完成，确保在执行前暴露类型不一致或模板语法错误。

## 规划与校验
- **结构规划 + 覆盖度检查**：`planner/structure.py` 通过 tool-calling 逐步构建骨架，`coverage.py` 将缺失点反馈给 LLM 直至覆盖需求。
- **Action Guard**：`planner/orchestrator.py` 在补参前后检查缺失或未注册的 `action_id`，必要时基于混合检索自动替换并提示模型修复。
- **静态校验与修复循环**：`verification/validation.py`、`planner/repair_tools.py` 联合提供本地修复（填充默认值、删除未知字段、类型矫正）与按需的 LLM 修复，保证输出的 Workflow 可执行。
