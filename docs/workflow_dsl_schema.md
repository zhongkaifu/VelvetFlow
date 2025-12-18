# Workflow DSL Schema 参考

本文汇总 VelvetFlow 的 workflow DSL 结构、节点类型与示例片段，便于规划/校验阶段快速对照。所有示例均可直接通过 `Workflow.model_validate` 校验并在执行器中运行。

## 顶层结构：Workflow Graph
- **字段**：
  - `workflow_name`：可选字符串，默认 `unnamed_workflow`。
  - `description`：可选描述文本。
  - `nodes`：必填数组，包含任意数量的节点对象；边由参数引用或显式分支字段自动推导，不要求手写。
- **隐式 edges**：运行时通过参数绑定和条件/循环/多分支出口推导，`Workflow.edges` 会返回标准化的 `from_node`/`to_node`/`condition` 结构，便于可视化或校验。

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
        "audience": {"__from__": "result_of.search_users.items"},
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
- `{"__from__": "result_of.<node_id>.<field>"}`：从上游节点输出取值。
- 聚合：`"__agg__": "identity|count|count_if|join|filter_map|format_join|pipeline"`，可配合 `field`、`op`、`value`、`map_field` 等字段完成筛选/映射。
- 模板：params 字符串支持 Jinja 表达式，校验/执行时会折叠常量并报出语法错误。

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
  "params": {"source": "result_of.detect_user.preferences", "field": "channel"},
  "cases": [
    {"match": ["email", "newsletter"], "to_node": "send_email"},
    {"match": "sms", "to_node": "send_sms"}
  ],
  "default_to_node": "send_push"
}
```

### 5. `loop`
- **字段**：
  - `loop_kind`：可选标签，描述循环用途。
  - `source`：数组/序列的引用路径（如 `result_of.search.items`）。
  - `item_alias`：为每轮迭代命名，便于子图中引用当前项。
  - `condition`：可选布尔表达式，控制循环提前终止。
  - `body_subgraph`：必填子图，包含至少一个 `action` 节点。
  - `exports`：指定输出：
    - `items.fields`：逐轮收集的字段列表。
    - `aggregates`：聚合计算数组，`{ "name": "total", "kind": "sum", "expr": {"source": "result_of.step.amount", "field": "value"} }`。
- **示例**：
```json
{
  "id": "batch_payments",
  "type": "loop",
  "params": {
    "loop_kind": "for_each_invoice",
    "source": "result_of.list_invoices.items",
    "item_alias": "invoice",
    "body_subgraph": {
      "nodes": [
        {
          "id": "pay_once",
          "type": "action",
          "action_id": "finance.pay_invoice",
          "params": {
            "invoice_id": {"__from__": "loop.invoice.id"},
            "amount": {"__from__": "loop.invoice.amount"}
          }
        }
      ]
    },
    "exports": {
      "items": {"fields": ["id", "status", "amount"]},
      "aggregates": [
        {
          "name": "paid_total",
          "kind": "sum",
          "expr": {"source": "result_of.pay_once", "field": "amount"}
        },
        {
          "name": "paid_count",
          "kind": "count_if",
          "expr": {"source": "result_of.pay_once", "field": "status", "op": "equals", "value": "success"}
        }
      ]
    }
  }
}
```

### 6. `parallel`
- **字段**：`params.branches` 为非空数组，每个分支包含 `id`、`entry_node` 与 `sub_graph_nodes` 列表；执行器会并发触发各分支并等待全部完成后继续。
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

## 组合示例：含条件、循环与并行的 DAG
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
        "source": "result_of.search_orders.items",
        "item_alias": "order",
        "body_subgraph": {
          "nodes": [
            {
              "id": "fetch_detail",
              "type": "action",
              "action_id": "crm.get_order_detail",
              "params": {"order_id": {"__from__": "loop.order.id"}}
            }
          ]
        },
        "exports": {"items": {"fields": ["id", "status"]}}
      }
    },
    {
      "id": "notify_parallel",
      "type": "parallel",
      "depends_on": ["for_each_order"],
      "params": {
        "branches": [
          {
            "id": "notify_ops",
            "entry_node": "ops_summary",
            "sub_graph_nodes": [
              {
                "id": "ops_summary",
                "type": "action",
                "action_id": "ops.send_report",
                "params": {"orders": {"__from__": "result_of.for_each_order.items"}}
              }
            ]
          },
          {
            "id": "notify_finance",
            "entry_node": "finance_summary",
            "sub_graph_nodes": [
              {
                "id": "finance_summary",
                "type": "action",
                "action_id": "finance.sync_erp",
                "params": {"orders": {"__from__": "result_of.for_each_order.items"}}
              }
            ]
          }
        ]
      }
    },
    {"id": "end", "type": "end"}
  ]
}
```
该 DAG 由 `nodes` 中的绑定与条件字段推导出完整拓扑，可直接用于可视化与执行。
