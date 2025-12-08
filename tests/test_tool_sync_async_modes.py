from velvetflow.executor import DynamicActionExecutor, WorkflowSuspension
from velvetflow.executor.async_runtime import GLOBAL_ASYNC_RESULT_STORE
from velvetflow.models import Workflow


def setup_function(_func):
    GLOBAL_ASYNC_RESULT_STORE._pending.clear()  # type: ignore[attr-defined]
    GLOBAL_ASYNC_RESULT_STORE._results.clear()  # type: ignore[attr-defined]


def test_tool_supports_sync_mode_by_default():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "sync_tool_call",
            "nodes": [
                {
                    "id": "record_health",
                    "type": "action",
                    "action_id": "hr.record_health_event.v1",
                    "params": {
                        "event_type": "fever",
                        "date": "2024-07-01",
                        "abnormal_count": 1,
                    },
                }
            ],
            "edges": [],
        }
    )

    executor = DynamicActionExecutor(workflow)
    results = executor.run()

    assert isinstance(results, dict)
    payload = results["record_health"]
    assert payload["status"] == "recorded"
    assert payload["event_type"] == "fever"


def test_tool_can_be_forced_into_async_mode():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "async_tool_call",
            "nodes": [
                {
                    "id": "record_health_async",
                    "type": "action",
                    "action_id": "hr.record_health_event.v1",
                    "params": {
                        "__invoke_mode": "async",
                        "event_type": "headache",
                        "date": "2024-07-02",
                        "abnormal_count": 0,
                    },
                }
            ],
            "edges": [],
        }
    )

    executor = DynamicActionExecutor(workflow)
    suspension = executor.run()

    assert isinstance(suspension, WorkflowSuspension)
    assert suspension.node_id == "record_health_async"

    results = executor.resume_from_suspension(suspension)

    assert isinstance(results, dict)
    payload = results["record_health_async"]
    assert payload["status"] == "async_resolved"
    assert payload.get("event_type") == "headache"
