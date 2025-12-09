# Author: OpenAI
# License: BSD 3-Clause License

from tools import Tool
from tools.registry import register_tool

from velvetflow.executor import DynamicActionExecutor, WorkflowSuspension
from velvetflow.executor.async_runtime import AsyncToolHandle, GLOBAL_ASYNC_RESULT_STORE
from velvetflow.models import Workflow


def _async_stub_tool(payload: str):
    return AsyncToolHandle(
        request_id="req-async-001", tool_name="async_stub_tool", params={"payload": payload}
    )


# Register the async stub tool once for the test module.
register_tool(
    Tool(
        name="async_stub_tool",
        description="Stub async tool that returns a handle.",
        function=_async_stub_tool,
    )
)


def setup_function(_func):
    # Ensure a clean async store for each test run.
    GLOBAL_ASYNC_RESULT_STORE._pending.clear()  # type: ignore[attr-defined]
    GLOBAL_ASYNC_RESULT_STORE._results.clear()  # type: ignore[attr-defined]


def test_executor_suspends_and_resumes_after_async_tool():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "async_resume_demo",
            "nodes": [
                {
                    "id": "enqueue_async",
                    "type": "action",
                    "action_id": "demo.async_stub.v1",
                    "params": {"payload": "hello"},
                },
                {
                    "id": "record_followup",
                    "type": "action",
                    "action_id": "hr.record_health_event.v1",
                    "params": {
                        "event_type": "result_of.enqueue_async.value",
                        "date": "2024-06-01",
                        "abnormal_count": 1,
                    },
                },
            ],
            "edges": [{"from": "enqueue_async", "to": "record_followup"}],
        }
    )

    executor = DynamicActionExecutor(workflow)
    suspension = executor.run()

    assert isinstance(suspension, WorkflowSuspension)
    assert suspension.node_id == "enqueue_async"
    snapshot = suspension.checkpoint.binding_snapshot["results"]["enqueue_async"]
    assert snapshot["status"] == "async_pending"

    GLOBAL_ASYNC_RESULT_STORE.complete(
        suspension.request_id, {"status": "ok", "value": "async-finished"}
    )

    results = executor.resume_from_suspension(suspension)

    assert isinstance(results, dict)
    assert results["enqueue_async"]["value"] == "async-finished"
    assert results["record_followup"]["event_type"] == "async-finished"
