# 核心概念

> English version: [core_concepts.en.md](core_concepts.en.md)


## Workflow 与节点模型
- **Workflow**：包含 `nodes`、可选 `edges` 与元数据；edges 可由参数绑定或条件/循环/多分支出口自动推导，Planner 会在输出中附带只读 edges/depends_on 方便可视化与下游校验。
- **节点类型**：
  - `action` 节点绑定业务动作与参数 Schema。
  - `condition`/`switch` 节点按布尔或多分支结果选择下游，未命中的分支会被标记为阻断以避免重复执行。
- `loop` 节点使用 `source` 定义循环集合，`item_alias` 指定循环体内引用当前元素的名称，`body_subgraph` 执行子图，`exports` 使用 `{key: Jinja 表达式}` 收集逐轮结果列表，表达式必须引用 body_subgraph 节点字段（例如 `{{ result_of.node.field }}`），最终通过 `result_of.<loop_id>.exports.<key>` 读取。
  - 循环体可使用 `loop.item`/`loop.index`/`loop.size`/`loop.accumulator`，或 `item_alias` 指定的别名。
  - `parallel/end` 仍可作为结构辅助节点存在（`parallel` 目前只用于 UI 分组，执行器会将其视为无副作用节点）。
- `velvetflow.models` 内置 `model_validate` 强类型校验节点字段、loop 子图完整性与引用合法性，失败时抛出统一的 `ValidationError`（接口形态与 Pydantic 类似但不依赖其运行时）。

## 绑定与上下文
- **BindingContext** 会在执行时携带 `results` 字典与 loop 局部上下文，供后续节点通过 Jinja 模板引用上游输出（例如 `"{{ result_of.node.field }}"`）。
- 绑定阶段仅接受 Jinja 表达式或字面量，聚合需求应在 Action 内或模板中自行实现。
- 绑定解析与 Jinja 模板折叠由 `bindings.py` 和 `jinja_utils.py` 完成，确保在执行前暴露类型不一致或模板语法错误。
- 绑定解析与 Jinja 模板折叠由 `bindings.py` 和 `jinja_utils.py` 完成，确保在执行前暴露类型不一致或模板语法错误。

## 规划与校验
- **基于 Agent SDK 的规划与修复**：`planner/structure.py` 使用 OpenAI Agent SDK 的 `Agent`/`Runner`/`function_tool` 复用需求拆解产物构建节点并补全 params；增量更新与自修复沿用相同的工具协议，通过 `orchestrator.update_workflow_with_two_pass` 驱动，核心依赖由 `planner/agent_runtime.py` 集中导出，便于在云端 Agent 或本地兼容层之间切换。
- **需求拆解**：`requirement_analysis.analyze_user_requirement` 将自然语言拆解为结构化清单，为后续的节点规划与校验提供上下文，规划阶段直接复用该清单。
- **Action Guard**：`planner/orchestrator.py` 在结构规划后检查缺失或未注册的 `action_id`，必要时基于混合检索自动替换并提示模型修复。
- **静态校验与修复循环**：`workflow_parser.py`、`verification/validation.py`、`planner/repair_tools.py` 联合提供语法/语义校验与本地修复（填充默认值、删除未知字段、类型矫正），在必要时调用 Agent 修复，保证输出的 Workflow 可执行。
