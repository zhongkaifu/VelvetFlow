# Workflow DSL 语法规范（基于编译原理）

本文将 VelvetFlow 的 JSON 工作流描述抽象为一套可读、可验证的 DSL。设计目标是：

- 使用上下文无关文法（EBNF）定义 DSL 形态，便于生成/格式化/补全。
- 明确词法单元，避免模板式字符串带来的歧义。
- 把静态语义（类型、幂等性、资源约束）建模为可组合的分析阶段，为构建和执行提供早期反馈。

## 词法定义

- **identifier**：`[A-Za-z_][A-Za-z0-9_-]*`，用于 `node_id`、`action_id`、字段名。
- **string**：`"…"` 的 JSON 字符串。
- **number**：JSON number。
- **bool/null**：JSON `true`/`false`/`null`。
- **punctuation**：`{ } [ ] : , .`。

绑定路径 `result_of.<node>.<field>[.<field>...]` 仅允许 `identifier` 与点号组合，不接受模板占位（如 `{{…}}`）。

## Workflow 顶层文法（EBNF）

```ebnf
workflow      ::= "{" workflow_body "}"
workflow_body ::= "workflow_name" ":" string ","
                  "description" ":" string ","
                  "nodes" ":" node_list
                  ["," "edges" ":" edge_list]

node_list     ::= "[" node {"," node} "]"
node          ::= "{" "id" ":" identifier "," "type" ":" node_type
                   ["," "action_id" ":" identifier]
                   ["," "display_name" ":" string]
                   ["," "params" ":" param_obj]
                   ["," "true_to_node" ":" identifier]
                   ["," "false_to_node" ":" identifier]
                   "}"
node_type     ::= "start" | "end" | "action" | "condition" | "loop"

edge_list     ::= "[" edge {"," edge} "]"
edge          ::= "{" "from" ":" identifier "," "to" ":" identifier
                   ["," "condition" ":" ("true"|"false")] "}"

param_obj     ::= "{" {param_field} "}"
param_field   ::= identifier ":" param_value
param_value   ::= string | number | bool | null
                 | binding_expr | param_obj | param_array
param_array   ::= "[" {param_value ","} param_value? "]"
```

## 参数绑定子文法

```ebnf
binding_expr ::= "{" "__from__" ":" binding_source
                  ["," "__agg__" ":" agg_op
                    ["," agg_args]
                  ] "}"

binding_source ::= "result_of" "." identifier {"." identifier}
                 | "params" "." identifier {"." identifier}

agg_op       ::= "identity" | "count" | "count_if"
               | "filter_map" | "format_join" | "pipeline"
agg_args     ::= "field" ":" identifier
               | "sep" ":" string
               | "steps" ":" param_array
```

## 静态语义与类型系统

- **节点签名**：来自 Action Registry 的 `arg_schema`/`output_schema`，补充 `loop` 虚拟输出（`status`、`iterations`、`aggregates` 等）。
- **类型匹配**：`__from__` 绑定的源字段类型必须与目标参数期望类型兼容（如 `integer`→`integer`，`array`→`count` 聚合）。
- **聚合约束**：`count/count_if/filter_map/pipeline` 仅接受 `array` 源，`format_join` 仅接受 `array|string` 源。
- **权限/幂等性标记**：Action Registry 可扩展 `requires` / `idempotent` 元数据，语义分析阶段据此阻止资源冲突或提醒非幂等操作。
- **作用域**：`loop` 子图节点只能通过 `exports` 暴露给外部引用；条件分支引用必须落在同一 DAG 作用域内。

## 分析流程

1. **词法/语法**：使用上文 EBNF 对 `__from__` 路径做正则化 token 化，拒绝非法标识符或模板字符串。
2. **抽象语法树**：将 JSON 解析后的对象映射为 `Workflow`/`Node`/`ParamBinding` 数据类，保留参数路径信息，便于错误定位。
3. **静态语义**：基于节点签名进行类型检查、聚合合法性检查、loop 导出检查；生成 `ValidationError` 供 CLI/IDE 定位问题。

此规范既可作为人类可读的 DSL 参考，也可驱动 IDE/CLI 的格式化、补全与静态校验。 
