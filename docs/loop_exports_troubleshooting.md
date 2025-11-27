# Loop exports 常见错误排查

当 loop 节点声明 `exports.items` 或 `exports.aggregates` 时，`fields` 必须对应 `from_node` 输出 schema 中存在的字段，否则会触发 `SCHEMA_MISMATCH`。下面结合实际示例说明原因与修复思路。

## 示例：`loop_check_temp` 出现未知字段错误

在以下工作流中，`loop_check_temp` 的 `exports.items` 配置为：

```json
"exports": {
  "items": {
    "from_node": "collect_warning_list",
    "fields": [
      "employee_id"
    ],
    "mode": "collect"
  }
}
```

执行时出现错误：

```
[SCHEMA_MISMATCH] node=loop_check_temp field=exports.items.fields message=loop 节点 'loop_check_temp' 的 exports.items.fields 包含未知字段 'employee_id'
```

### 原因

- `fields` 必须从 `from_node` 的输出 schema 中选择字段；如果 action `collect_warning_list` 的输出不含 `employee_id`，或字段位于嵌套结构（例如 `data.employee_id`），直接填入 `employee_id` 会被视为未知字段。
- 根据校验规则，loop 只能通过显式 exports 暴露 body 子图数据，下游引用时必须与 exports 定义保持一致，不能直接引用 body 内部节点。相关约束见 planner 参数校验说明【F:velvetflow/planner/params.py†L167-L172】与修复提示【F:velvetflow/planner/repair.py†L235-L248】。

### 修复建议

1. 确认 `collect_warning_list` 的输出 schema。若返回的字段名不同（例如 `data`、`record`），将 `fields` 改为匹配的字段或嵌套路径（如 `data.employee_id`）。
2. 如果需要汇总 loop 结果，确保 exports 使用正确 `from_node` 与字段路径，然后在下游通过 `result_of.loop_check_temp.items` 引用。
3. 当 `fields` 只是包装字段时，请通过 `<包装字段>.<子字段>` 访问内部属性，避免使用不存在的扁平字段名。

这样调整后，loop 节点即可通过 schema 校验，并让下游节点安全引用循环结果。
