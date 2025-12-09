# Author: OpenAI
# License: BSD 3-Clause License

from pathlib import Path

from tools import Tool
from tools.registry import register_tool

from velvetflow.bindings import BindingContext
from velvetflow.executor import DynamicActionExecutor, WorkflowSuspension
from velvetflow.executor.async_runtime import (
    AsyncToolHandle,
    ExecutionCheckpoint,
    GLOBAL_ASYNC_RESULT_STORE,
)
from velvetflow.models import Workflow


def _async_stub_tool(payload: str):
    return AsyncToolHandle(
        request_id="persist-req-001", tool_name="persist_async_stub", params={"payload": payload}
    )


register_tool(
    Tool(
        name="persist_async_stub",
        description="Async tool used for persistence testing.",
        function=_async_stub_tool,
    )
)


def setup_function(_func):
    GLOBAL_ASYNC_RESULT_STORE._pending.clear()  # type: ignore[attr-defined]
    GLOBAL_ASYNC_RESULT_STORE._results.clear()  # type: ignore[attr-defined]


def test_suspension_and_snapshot_round_trip(tmp_path: Path):
    workflow = Workflow.model_validate(
        {
            "workflow_name": "async_persistence_demo",
            "nodes": [
                {
                    "id": "enqueue_async",
                    "type": "action",
                    "action_id": "demo.persist_async_stub.v1",
                    "params": {"payload": "hello"},
                },
                {
                    "id": "followup",
                    "type": "action",
                    "action_id": "hr.record_health_event.v1",
                    "params": {
                        "event_type": "result_of.enqueue_async.value",
                        "date": "2024-07-03",
                        "abnormal_count": 0,
                    },
                },
            ],
            "edges": [{"from": "enqueue_async", "to": "followup"}],
        }
    )

    executor = DynamicActionExecutor(workflow)
    suspension = executor.run()

    assert isinstance(suspension, WorkflowSuspension)

    suspension_path = tmp_path / "suspension.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    snapshot_path = tmp_path / "snapshot.json"

    suspension.save_to_file(suspension_path)
    suspension.checkpoint.save_to_file(checkpoint_path)
    BindingContext.save_snapshot_to_file(
        suspension.checkpoint.binding_snapshot, snapshot_path
    )

    reloaded_suspension = WorkflowSuspension.load_from_file(suspension_path)
    reloaded_checkpoint = ExecutionCheckpoint.load_from_file(checkpoint_path)
    restored_ctx = BindingContext.load_snapshot_from_file(workflow, snapshot_path)

    assert reloaded_suspension.node_id == suspension.node_id
    assert reloaded_suspension.request_id == suspension.request_id
    assert reloaded_checkpoint.binding_snapshot == suspension.checkpoint.binding_snapshot
    assert restored_ctx.results["enqueue_async"]["status"] == "async_pending"

    GLOBAL_ASYNC_RESULT_STORE.complete(
        reloaded_suspension.request_id, {"status": "ok", "value": "persisted"}
    )

    results = executor.resume_from_suspension(reloaded_suspension)

    assert isinstance(results, dict)
    assert results["enqueue_async"]["value"] == "persisted"
    assert results["followup"]["event_type"] == "persisted"
