# Workflow DSL Schema 参考

本文汇总 VelvetFlow 的 workflow DSL 结构、节点类型与示例片段，便于规划/校验阶段快速对照。所有示例均可直接通过 `Workflow.model_validate` 校验并在执行器中运行。

## 顶层结构：Workflow Graph
- **字段**：
  - `workflow_name`：可选字符串，默认 `unnamed_workflow`。
  - `description`：可选描述文本。
  - `nodes`：必填数组，包含任意数量的节点对象；边由参数引用或显式分支字段自动推导，不要求手写。
- **隐式 edges**：运行时通过参数绑定和条件/循环/多分支出口推导，`Workflow.edges` 会返回标准化的 `from_node`/`to_node`/`condition` 结构，便于可视化或校验；`workflow_parser.py` 在校验时若发现缺失 edges 会给出可恢复的提示。

### 最小可运行示例
```json
{
  "workflow_name": "send_newsletter",
  "description": "向 CRM 客户发送新品资讯并记录审批",
  "nodes": [
    {"id": "start", "type": "start"},
    {
      "id": "search_users",
      "type": "action",
      "action_id": "crm.search_customers",
      "params": {"segment": "recent_buyers"}
    },
    {
      "id": "approve",
      "type": "condition",
      "params": {"expression": "{{ result_of.search_users.count >= 10 }}"},
      "true_to_node": "send_email",
      "false_to_node": "end"
    },
    {
      "id": "send_email",
      "type": "action",
      "action_id": "crm.send_newsletter",
      "params": {
        "audience": "{{ result_of.search_users.items }}",
        "template_id": "spring_launch"
      }
    },
    {"id": "end", "type": "end"}
  ]
}
```
上述 workflow 的 edges 会自动推导为 `start → search_users → approve → send_email/end`，无需显式维护。

## 通用节点字段
- `id`：必填字符串且在同一 graph 内唯一。
- `type`：必填枚举，支持 `start`、`end`、`action`、`condition`、`switch`、`loop`、`parallel`。
- `display_name`：可选友好名称，便于可视化。
- `depends_on`：可选显式依赖数组，用于覆盖自动推导或串联未被绑定引用捕获的顺序约束。
- `params`：节点专属参数，结构取决于节点类型。

### 参数绑定 DSL 速览
- **Jinja 模板（唯一支持）**：`"{{ result_of.<node_id>.<field> }}"` 直接引用上游输出，规划/执行阶段均假定 params 仅包含字符串模板或字面量。
- **聚合**：若需要聚合数组，建议在 Action 内部或自定义模板中过滤/映射，legacy 的 `__agg__` 语法已移除。
- **模板语法校验**：params 字符串支持 Jinja 表达式，校验/执行时会折叠常量并报出语法错误。

## 节点类型与示例
### 1. `start` / `end`
结构化入口或终点，通常只需 `id` 与 `type`：
```json
{"id": "start", "type": "start"}
{"id": "end", "type": "end"}
```

### 2. `action`
- **字段**：`action_id` 指向注册表中的工具，`params` 对应工具的 `arg_schema`，`out_params_schema` 可选（覆盖/补充动作的输出 Schema）。
- **示例**：
```json
{
  "id": "summarize",
  "type": "action",
  "action_id": "search.web_summary",
  "params": {
    "query": "最新 AI 法规",
    "__invoke_mode": "async"
  },
  "out_params_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}
}
```

### 3. `condition`
- **字段**：`params.expression` 为布尔 Jinja 表达式；`true_to_node`、`false_to_node` 指定分支去向，未命中分支自动阻断。
- **示例**：
```json
{
  "id": "check_budget",
  "type": "condition",
  "params": {"expression": "{{ result_of.estimate.budget > 10000 }}"},
  "true_to_node": "request_approval",
  "false_to_node": "auto_purchase"
}
```

### 4. `switch`
- **字段**：`params.source` 定位被匹配的对象（可用绑定或 `result_of.*` 路径）；`params.field` 可选，用于从对象中取子字段；`cases` 为数组，支持 `match`/`value`、`field`（进一步取子字段）、`to_node`；`default_to_node` 为兜底分支。
- **示例**：
```json
{
  "id": "route_channel",
  "type": "switch",
  "params": {"source": "{{ result_of.detect_user.preferences }}", "field": "channel"},
  "cases": [
    {"match": ["email", "newsletter"], "to_node": "send_email"},
    {"match": "sms", "to_node": "send_sms"}
  ],
  "default_to_node": "send_push"
}
```

### 5. `loop`
- **字段**：
  - `loop_kind`：必填，支持 `for_each`/`foreach`/`while`。
  - `source`：数组/序列的引用路径（如 `{{ result_of.search.items }}`），仅支持字符串/Jinja 模板。
  - `item_alias`：为每轮迭代命名，便于子图中引用当前项（例如 `loop.<item_alias>` 或直接使用别名）。
  - `condition`：仅 `while` 循环需要的布尔表达式（Jinja）。
  - `body_subgraph`：必填子图，包含至少一个 `action` 节点。可选 `entry`/`exit` 字段用于指定起止节点。
  - `exports`：指定输出：
    - 使用 `{key: Jinja 表达式}` 形式暴露循环体字段，表达式必须引用 body_subgraph 节点字段（如 `{{ result_of.node.field }}`）；执行时每个 key 收集逐轮结果形成列表，外部通过 `result_of.<loop_id>.exports.<key>` 访问。
    - `exports` 必须位于 `params.exports`，不允许放在 `body_subgraph.exports`。
- **示例**：
```json
{
  "id": "batch_payments",
  "type": "loop",
  "params": {
    "loop_kind": "for_each",
    "source": "{{ result_of.list_invoices.items }}",
    "item_alias": "invoice",
    "body_subgraph": {
      "nodes": [
        {
          "id": "pay_once",
          "type": "action",
          "action_id": "finance.pay_invoice",
          "params": {
            "invoice_id": "{{ loop.invoice.id }}",
            "amount": "{{ loop.invoice.amount }}"
          }
        }
      ]
    },
    "exports": {
      "payments": "{{ result_of.pay_once }}",
      "payment_amounts": "{{ result_of.pay_once.amount }}"
    }
  }
}
```

### 6. `parallel`
- **字段**：`params.branches` 为非空数组，每个分支包含 `id`、`entry_node` 与 `sub_graph_nodes` 列表；该结构当前主要用于可视化与前端分组，执行器不会真正并发调度 `branches` 中的节点（运行时请将需要执行的节点放在顶层 `nodes` 并通过 `depends_on`/绑定推导控制顺序）。
- **示例**：
```json
{
  "id": "multi_channel",
  "type": "parallel",
  "params": {
    "branches": [
      {
        "id": "email_path",
        "entry_node": "send_email",
        "sub_graph_nodes": [
          {"id": "send_email", "type": "action", "action_id": "crm.send_newsletter", "params": {"template": "spring"}}
        ]
      },
      {
        "id": "sms_path",
        "entry_node": "send_sms",
        "sub_graph_nodes": [
          {"id": "send_sms", "type": "action", "action_id": "crm.send_sms", "params": {"text": "新品上线"}}
        ]
      }
    ]
  }
}
```

## 组合示例：含条件与循环的 DAG
```json
{
  "workflow_name": "async_order_pipeline",
  "nodes": [
    {"id": "start", "type": "start"},
    {"id": "search_orders", "type": "action", "action_id": "crm.list_orders", "params": {"days": 7}},
    {
      "id": "if_has_orders",
      "type": "condition",
      "params": {"expression": "{{ result_of.search_orders.count > 0 }}"},
      "true_to_node": "for_each_order",
      "false_to_node": "end"
    },
    {
      "id": "for_each_order",
      "type": "loop",
      "params": {
        "loop_kind": "for_each",
        "source": "{{ result_of.search_orders.items }}",
        "item_alias": "order",
        "body_subgraph": {
          "nodes": [
            {
              "id": "fetch_detail",
              "type": "action",
              "action_id": "crm.get_order_detail",
              "params": {"order_id": "{{ loop.order.id }}"}
            }
          ]
        },
        "exports": {"orders": "{{ result_of.fetch_detail.id }}"}
      }
    },
    {
      "id": "ops_summary",
      "type": "action",
      "depends_on": ["for_each_order"],
      "action_id": "ops.send_report",
      "params": {"orders": "{{ result_of.for_each_order.exports.orders }}"}
    },
    {
      "id": "finance_summary",
      "type": "action",
      "depends_on": ["for_each_order"],
      "action_id": "finance.sync_erp",
      "params": {"orders": "{{ result_of.for_each_order.exports.orders }}"}
    },
    {"id": "end", "type": "end"}
  ]
}
```
该 DAG 由 `nodes` 中的绑定与条件字段推导出完整拓扑，可直接用于可视化与执行。
