# 故障排除

## 常见错误
- **缺少 OpenAI 凭证**：确保已设置 `OPENAI_API_KEY` 环境变量。
- **未注册的 action_id**：规划或执行阶段报错时，检查 `tools/business_actions/` 是否存在该动作并重新构建索引；执行器启动时会提前失败以避免运行期出错。
- **绑定路径不合法**：`ValidationError` 提示找不到 `__from__` 路径时，确认上游节点的输出 Schema 是否包含该字段或在 loop.exports 中声明。
- **异步节点无法恢复**：如果 `resume_from_suspension` 失败，确认 `suspension.json` 与回调结果的 `request_id` 一致，并确保结果文件包含 `status` 或业务返回体。

## 定位手段
- **打印归一化 DSL**：
  ```bash
  python validate_workflow.py path/to/workflow.json --print-normalized
  ```
- **查看规划日志**：`build_workflow.py` 会输出覆盖度缺失和 tool-calling 结果，便于定位 LLM 生成阶段的问题。
- **执行期事件**：`execute_workflow.py` 默认打印节点起止、条件结果、loop 聚合等日志，异步挂起时会输出 `WorkflowSuspension` 细节。

## 进一步支持
- 如果需要替换真实后端服务或扩展动作 Schema，可先在 `simulation_data.json` 模拟期望返回，再将工具映射加入 `tools/business_actions/` 并重新构建索引。
