# Author: OpenAI
# License: BSD 3-Clause License

import itertools

from tools import Tool
from tools.registry import register_tool

from velvetflow.executor import DynamicActionExecutor, WorkflowSuspension
from velvetflow.executor.async_runtime import AsyncToolHandle, GLOBAL_ASYNC_RESULT_STORE
from velvetflow.models import Workflow


_REQUEST_COUNTER = itertools.count(1)


def _async_stub_tool(payload: str):
    return AsyncToolHandle(
        request_id=f"req-async-{next(_REQUEST_COUNTER):03d}",
        tool_name="async_stub_tool",
        params={"payload": payload},
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


def test_resume_can_return_new_suspension_for_second_async_tool():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "double_async_resume_demo",
            "nodes": [
                {
                    "id": "enqueue_async",
                    "type": "action",
                    "action_id": "demo.async_stub.v1",
                    "params": {"payload": "first"},
                },
                {
                    "id": "second_async",
                    "type": "action",
                    "action_id": "demo.async_stub.v1",
                    "params": {"payload": "result_of.enqueue_async.value"},
                },
            ],
            "edges": [{"from": "enqueue_async", "to": "second_async"}],
        }
    )

    executor = DynamicActionExecutor(workflow)
    first_suspension = executor.run()

    assert isinstance(first_suspension, WorkflowSuspension)
    assert first_suspension.node_id == "enqueue_async"

    resumed = executor.resume_from_suspension(
        first_suspension, tool_result={"status": "ok", "value": "first-done"}
    )

    assert isinstance(resumed, WorkflowSuspension)
    assert resumed.node_id == "second_async"


def test_resume_requires_async_result():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "missing_async_result_demo",
            "nodes": [
                {
                    "id": "enqueue_async",
                    "type": "action",
                    "action_id": "demo.async_stub.v1",
                    "params": {"payload": "no_result"},
                }
            ],
            "edges": [],
        }
    )

    executor = DynamicActionExecutor(workflow)
    suspension = executor.run()

    assert isinstance(suspension, WorkflowSuspension)

    try:
        executor.resume_from_suspension(suspension)
    except ValueError as exc:
        assert "未找到异步请求" in str(exc)
    else:
        raise AssertionError("Expected resume_from_suspension to raise without async result")
