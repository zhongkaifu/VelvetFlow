# 异步工具挂起/恢复演示

本示例展示如何在调用异步业务工具时挂起 workflow，并在获取工具结果后恢复执行。

## 文件说明
- `workflow_async_health.json`：包含一个异步节点 `record_event_async`，使用 `__invoke_mode: "async"` 调用 `hr.record_health_event.v1`，并在下游节点引用其输出。
- `tool_result_completed.json`：模拟外部异步工具完成后的回调结果，可直接用于恢复演示。

## 运行步骤
1. **首次运行，生成挂起文件**：
   ```bash
   python execute_workflow.py \
     --workflow-json examples/async_suspend_resume/workflow_async_health.json \
     --suspension-file examples/async_suspend_resume/suspension.json
   ```
   - 执行会在异步节点处停止，返回 `WorkflowSuspension` 并在同目录生成 `suspension.json`，其中包含 request_id、node_id 以及完整的上下文快照。

2. **提供工具结果并恢复执行**：
   ```bash
   python execute_workflow.py \
     --workflow-json examples/async_suspend_resume/workflow_async_health.json \
     --resume-from examples/async_suspend_resume/suspension.json \
     --tool-result-file examples/async_suspend_resume/tool_result_completed.json
   ```
   - 执行器会载入挂起快照，将工具结果合并为 `async_resolved`，然后继续执行下游的 `create_incident` 节点。

> 若需要自定义模拟输出，可复制 `tool_result_completed.json` 并替换内容，再次运行恢复命令。`--simulation-file` 参数可用来自定义模拟数据路径。
