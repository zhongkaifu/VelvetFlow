import json
import sys
from pathlib import Path

from execute_workflow import main as run_cli
from velvetflow.executor.async_runtime import (
    GLOBAL_ASYNC_RESULT_STORE,
    WorkflowSuspension,
)


def setup_function(_func):
    GLOBAL_ASYNC_RESULT_STORE._pending.clear()  # type: ignore[attr-defined]
    GLOBAL_ASYNC_RESULT_STORE._results.clear()  # type: ignore[attr-defined]


def _write_workflow(tmp_path: Path) -> Path:
    workflow = {
        "workflow_name": "cli_async_workflow",
        "nodes": [
            {
                "id": "record_health_async",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "__invoke_mode": "async",
                    "event_type": "from_cli",
                },
            }
        ],
        "edges": [],
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_executor_persists_suspension_file(tmp_path, monkeypatch):
    workflow_path = _write_workflow(tmp_path)
    suspension_path = tmp_path / "suspension.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_workflow.py",
            "--workflow-json",
            str(workflow_path),
            "--suspension-file",
            str(suspension_path),
        ],
    )

    result = run_cli()

    assert isinstance(result, WorkflowSuspension)
    assert suspension_path.exists()
    persisted = WorkflowSuspension.load_from_file(suspension_path)
    assert persisted.node_id == "record_health_async"
    assert persisted.request_id == result.request_id


def test_executor_resumes_from_saved_suspension(tmp_path, monkeypatch):
    workflow_path = _write_workflow(tmp_path)
    suspension_path = tmp_path / "suspension.json"

    # First run to create a suspension file
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_workflow.py",
            "--workflow-json",
            str(workflow_path),
            "--suspension-file",
            str(suspension_path),
        ],
    )
    _ = run_cli()

    tool_result_file = tmp_path / "tool_result.json"
    tool_result_file.write_text(
        json.dumps({"status": "completed", "note": "from file"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_workflow.py",
            "--workflow-json",
            str(workflow_path),
            "--resume-from",
            str(suspension_path),
            "--tool-result-file",
            str(tool_result_file),
        ],
    )

    resumed = run_cli()

    assert isinstance(resumed, dict)
    payload = resumed["record_health_async"]
    assert payload["status"] == "async_resolved"
    assert payload.get("tool_status") == "completed"
