<!-- Language toggle tabs -->
<style>
.lang-tabs { border: 1px solid #d0d7de; border-radius: 6px; padding: 0.75rem; }
.lang-tabs input[type="radio"] { display: none; }
.lang-tabs label { padding: 0.35rem 0.75rem; border: 1px solid #d0d7de; border-bottom: none; border-radius: 6px 6px 0 0; margin-right: 0.25rem; cursor: pointer; background: #f6f8fa; font-weight: 600; }
.lang-tabs input[type="radio"]:checked + label { background: #fff; border-bottom: 1px solid #fff; }
.lang-tabs .tabs-body { border-top: 1px solid #d0d7de; padding-top: 0.75rem; }
.lang-tabs .tab-content { display: none; }
#troubleshooting-lang-zh:checked ~ .tabs-body #troubleshooting-tab-zh,
#troubleshooting-lang-en:checked ~ .tabs-body #troubleshooting-tab-en { display: block; }
</style>
<div class="lang-tabs">
<input type="radio" id="troubleshooting-lang-zh" name="troubleshooting-lang" checked>
<label for="troubleshooting-lang-zh">中文</label>
<input type="radio" id="troubleshooting-lang-en" name="troubleshooting-lang">
<label for="troubleshooting-lang-en">English</label>
<div class="tabs-body">
<div class="tab-content" id="troubleshooting-tab-zh">
# 故障排除

## 常见错误
- **缺少 OpenAI 凭证**：确保已设置 `OPENAI_API_KEY` 环境变量。
- **缺少 Agent SDK**：如果报 `No module named 'agents'` 或 Planner 无法导入 Agent/Runner，请额外执行 `pip install agents`（或 `uv add agents`），因为结构/补参/修复阶段依赖 OpenAI Agent SDK。
- **未注册的 action_id**：规划或执行阶段报错时，检查 `tools/business_actions/` 是否存在该动作并重新构建索引；执行器启动时会提前失败以避免运行期出错。
- **绑定路径不合法**：`ValidationError` 提示找不到引用路径时，确认上游节点的输出 Schema 是否包含该字段或在 loop.exports 中声明，并将参数写成 Jinja 模板（如 `{{ result_of.node.field }}`）。
- **Jinja 表达式错误**：`INVALID_JINJA_EXPRESSION` 往往意味着 params 未使用合法的 `{{ ... }}` 模板；建议改为 Jinja 表达式或在规划阶段让 Agent 修复。
- **DSL 语法/语义错误**：`SYNTAX_ERROR`/`GRAMMAR_VIOLATION` 表示 JSON 语法或 DSL 结构不满足 `workflow_parser.py` 的要求（例如缺少 `nodes`），可先按报错位置修正，再执行校验。
- **异步节点无法恢复**：如果 `resume_from_suspension` 失败，确认 `suspension.json` 与回调结果的 `request_id` 一致，并确保结果文件包含 `status` 或业务返回体。
- **loop 被跳过**：运行日志提示 `source_not_list` 时，检查 loop 的 `params.source` 是否指向数组/序列，必要时使用 Jinja 模板引用上游数组（如 `{{ result_of.search.items }}`）。
- **parallel 分支未执行**：`parallel` 节点当前仅用于前端分组与可视化，执行器不会调度 `branches` 内的节点；请将实际节点放在顶层 `nodes` 并使用 `depends_on`/绑定推导控制顺序。

## 定位手段
- **打印归一化 DSL**：
  ```bash
  python validate_workflow.py path/to/workflow.json --print-normalized
  ```
- **查看规划日志**：`build_workflow.py` 会输出需求拆解、Agent 工具调用与返回结果，便于定位 LLM 生成阶段的问题。
- **执行期事件**：`execute_workflow.py` 默认打印节点起止、条件结果、loop 聚合等日志，异步挂起时会输出 `WorkflowSuspension` 细节。

## 进一步支持
- 如果需要替换真实后端服务或扩展动作 Schema，可先在 `simulation_data.json` 模拟期望返回，再将工具映射加入 `tools/business_actions/` 并重新构建索引。

</div>
<div class="tab-content" id="troubleshooting-tab-en">
## Troubleshooting (English)
- **Missing OpenAI credentials**: Ensure `OPENAI_API_KEY` is set; planners and retrieval embedding calls require it.
- **Index not found**: Rebuild the action index with `build_action_index.py` or confirm `tools/action_index.json` exists.
- **Validation failures**: Review `ValidationError` locations for invalid bindings, missing exports, or unsupported node types; rerun validation after fixing DSL fields.
- **Execution stalls**: Check for async nodes returning `async_pending` and resume with saved `ExecutionCheckpoint` once external work finishes.
- **Logging**: Use console logs and event logs (`logging_utils.py`) to locate planner or executor issues.

</div>
</div>
</div>
