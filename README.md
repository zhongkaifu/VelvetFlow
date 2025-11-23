# VelvetFlow

VelvetFlow 是一个可复用的 LLM 驱动工作流规划与执行演示项目。它通过混合检索、两阶段 LLM 规划、静态校验与可模拟执行器，展示了如何从自然语言需求自动构建可运行的业务工作流。本文档覆盖项目背景、需求分析、总体/详细设计、核心流程、技术要点、示例、环境搭建与测试运行方法，旨在让读者仅凭本文件即可重新实现全部功能。

## 背景与需求
- **目标**：从自然语言描述生成可执行的业务工作流，包括节点规划、连线、参数补全、校验与执行。适用于通用业务（HR/OPS/CRM 等）动作库示例。
- **核心需求**
  - 提供业务动作注册表，可供检索与执行模拟使用。
  - 支持基于文本/向量的混合检索，帮助 LLM 选择合适的业务动作。
  - 通过两阶段 LLM 规划：先生成结构，再补全参数，并在必要时自动修复。
  - 支持结果绑定 DSL，允许节点参数引用上游输出并进行聚合/过滤/格式化。
  - 提供可模拟的执行器，按 DAG 顺序解析参数、执行动作并打印上下文。

## 模块总体设计
```
VelvetFlow (repo root)
├── velvetflow/
│   ├── action_registry.py   # 业务动作定义与查询
│   ├── search.py            # 混合检索 (BM25 + 向量) 及嵌入
│   ├── planner.py           # 两阶段 LLM 规划、覆盖度检查、自修复、校验
│   ├── executor.py          # 动态执行器与参数绑定 DSL 解析
│   ├── config.py            # OpenAI 模型配置
│   └── simulation_data.json # 动作模拟返回模板
└── workflow_fc_hybrid_actions_demo.py # 端到端示例入口
```

## 详细设计与技术要点
### 动作注册表（Action Registry）
- 定义在 `velvetflow/action_registry.py`，包含 HR/OPS/CRM 等示例动作及它们的参数/输出 JSON Schema。支持通过 `get_action_by_id` 查询，供规划/校验/执行阶段使用。【F:velvetflow/action_registry.py†L1-L212】【F:velvetflow/action_registry.py†L214-L236】

### 混合检索与本地向量库
- `velvetflow/search.py` 提供简化版 BM25 (`FakeElasticsearch`) 与向量库 (`VectorClient`)，并使用词袋 embedding (`embed_text_local`) 构建全局词表。【F:velvetflow/search.py†L1-L119】【F:velvetflow/search.py†L121-L160】
- `HybridActionSearchService` 将 BM25 与向量相似度按权重融合，返回按得分排序的候选动作，用于 LLM tool-calling 期间的动作选择。【F:velvetflow/search.py†L162-L253】

### 两阶段 LLM 规划流程
- **结构规划**：`plan_workflow_structure_with_llm` 使用 OpenAI function-calling 工具（搜索动作、设置元数据、增添节点/边、最终确认）迭代生成工作流骨架；若缺少边则调用二次 LLM 接线或线性串联补全，并校验动作合法性与连通性。【F:velvetflow/planner.py†L113-L255】【F:velvetflow/planner.py†L256-L344】
- **覆盖度检查与结构改进**：规划后通过 LLM 检查需求覆盖度，若存在缺失则调用增量改进接口补充节点/边，循环直至覆盖或达到最大轮次。【F:velvetflow/planner.py†L346-L417】
- **参数补全**：`fill_params_with_llm` 根据动作的 `arg_schema`、上下游关系与绑定 DSL 例子补全必填参数，允许引用上游输出并生成结构化条件节点参数。【F:velvetflow/planner.py†L765-L964】
- **静态校验与自修复**：`validate_completed_workflow`（同文件后半部分）检查必填字段、路径合法性与 Schema 对齐；`repair_workflow_with_llm` 在尽量不改结构的前提下修复参数/路径/字段名；`plan_workflow_with_two_pass` 负责 orchestrate 两阶段规划 + 校验 + 多轮自修复并返回最终可执行 DSL。【F:velvetflow/planner.py†L965-L1291】【F:velvetflow/planner.py†L1293-L1420】

### 参数绑定 DSL 与执行器
- 执行器 `DynamicActionExecutor` 支持从上游上下文读取 `result_of.<node_id>[.<field>...]`，解析聚合（`identity`、`count_if`、`filter_map`->`pipeline`、`pipeline`），并将解析后的参数传入动作模拟结果生成器 `_simulate_action`。条件节点支持 `any_greater_than`、`equals` 等结构化 params。【F:velvetflow/executor.py†L1-L210】【F:velvetflow/executor.py†L212-L420】
- 运行时会按 DAG 顺序遍历节点、解析参数、打印执行信息并将结果写入上下文；模拟数据来源于 `simulation_data.json`，可通过模板占位符 `{{field}}` 与默认值渲染输出。【F:velvetflow/executor.py†L422-L499】【F:velvetflow/simulation_data.json†L1-L23】

### 配置与模型
- `velvetflow/config.py` 用于集中管理 OpenAI 模型名称（默认 `gpt-4.1-mini`），可按需修改。【F:velvetflow/config.py†L1-L4】

### 端到端示例入口
- `workflow_fc_hybrid_actions_demo.py` 构建混合检索服务、读取用户需求（或默认需求）、调用 `plan_workflow_with_two_pass` 生成/修复/补参工作流，再使用 `DynamicActionExecutor` 结合 `simulation_data.json` 进行模拟运行。需提前设置环境变量 `OPENAI_API_KEY` 并安装依赖（`openai`、`numpy`）。【F:workflow_fc_hybrid_actions_demo.py†L1-L73】【F:workflow_fc_hybrid_actions_demo.py†L75-L96】

## 核心流程图（文字版）
1. **输入需求** → 用户自然语言描述。
2. **结构规划** → LLM tool-calling 搜索动作、添加节点/边 → 无边时自动接线 → 连通性补全 + 动作合法性检查。
3. **覆盖度校验** → LLM 判断需求覆盖度，若缺失则调用结构改进 LLM。
4. **参数补全** → LLM 根据 Schema/上下游关系生成必填参数与绑定 DSL。
5. **静态校验 & 自修复** → 本地规则校验；如有错误，LLM 按最小改动策略修复。
6. **执行** → 加载模拟数据 → 解析参数/条件 → 按 DAG 运行节点并输出上下文。

## DSL 速查
- **工作流**：`{ "workflow_name": str, "description": str, "nodes": [ ... ], "edges": [ {"from": str, "to": str, "condition": "true"|"false"|null} ] }`。
- **节点**：`type` 可为 `start/end/action/condition/loop/parallel`；`action` 节点需 `action_id` 与 `params`；`condition` 节点 `params` 需 `kind` 与 `source` 等字段。
- **参数绑定**：`{"__from__": "result_of.node.field", "__agg__": "identity"}`；`count_if` 支持 `field/op/value`；`filter_map` 会被自动转换为 `pipeline`；`pipeline` 步骤支持 `filter`、`map`、`format_join`。

## 环境搭建
1. **准备依赖**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade openai numpy
   ```
2. **设置凭证**
   ```bash
   export OPENAI_API_KEY="<your_api_key>"
   ```
3. **可选：检查代码语法**
   ```bash
   python -m compileall velvetflow
   ```

## 运行示例
```bash
python workflow_fc_hybrid_actions_demo.py
```
- 按提示输入自然语言需求，或直接回车使用默认案例；程序将打印生成的工作流 DSL 与模拟执行日志。

## 开发与测试样例
- **替换动作库**：编辑 `velvetflow/action_registry.py`，添加/禁用动作并提供对应 `arg_schema`/`output_schema`，即可让检索与规划使用新的动作集合。【F:velvetflow/action_registry.py†L1-L212】
- **调整检索策略**：在 `workflow_fc_hybrid_actions_demo.py` 的 `build_default_search_service` 中修改融合系数 `alpha` 或改写 `embed_text_local` 以适配自定义向量模型。【F:workflow_fc_hybrid_actions_demo.py†L17-L39】
- **自定义模型**：修改 `velvetflow/config.py` 中的 `OPENAI_MODEL` 即可切换模型。【F:velvetflow/config.py†L1-L4】
- **模拟数据扩展**：在 `velvetflow/simulation_data.json` 增加动作 ID、默认值及模板，执行器会自动渲染并返回结构化结果。【F:velvetflow/simulation_data.json†L1-L23】

## 重新实现指引
- 按上述模块职责实现：
  1) 定义动作注册表与 Schema；
  2) 提供 BM25/向量混合检索接口；
  3) 构建两阶段 LLM 规划（结构→参数）并加入覆盖度检查、自修复与静态校验；
  4) 实现参数绑定 DSL（含聚合/管道）与条件节点求值；
  5) 通过模拟数据或真实 API 执行动作并维护上下文。
- 遵循 DSL 结构与函数接口，即可从自然语言需求到可执行工作流的完整链路。

