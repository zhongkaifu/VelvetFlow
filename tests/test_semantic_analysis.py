import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import validate_workflow
from execute_workflow import load_workflow_from_file, validate_workflow_for_execution
from validate_workflow import validate_workflow_data
from velvetflow.verification.validation import validate_completed_workflow

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def _workflow_with_binding(binding):
    return {
        "workflow_name": "semantic_checks",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "fetch",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-06-18"},
            },
            {
                "id": "record",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "demo",
                    "abnormal_count": binding,
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "fetch"},
            {"from": "fetch", "to": "record"},
            {"from": "record", "to": "end"},
        ],
    }


def test_detects_type_mismatch_between_binding_and_target():
    binding = {"__from__": "result_of.fetch.date"}  # string -> expected integer
    errors = validate_workflow_data(_workflow_with_binding(binding), ACTION_REGISTRY)

    assert any("不兼容" in err.message for err in errors)


def test_detects_invalid_aggregation_source_type():
    binding = {"__from__": "result_of.fetch.date", "__agg__": "count"}
    errors = validate_workflow_data(_workflow_with_binding(binding), ACTION_REGISTRY)

    assert any("仅支持数组类型来源" in err.message for err in errors)


def test_validate_completed_workflow_runs_semantic_checks():
    binding = {"__from__": "result_of.fetch.date"}  # string -> expected integer
    errors = validate_completed_workflow(_workflow_with_binding(binding), ACTION_REGISTRY)

    assert any("不兼容" in err.message for err in errors)


def test_execution_entrypoint_validates_semantics(tmp_path):
    binding = {"__from__": "result_of.fetch.date"}
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(_workflow_with_binding(binding)), encoding="utf-8")

    workflow = load_workflow_from_file(str(workflow_path))
    errors = validate_workflow_for_execution(workflow)

    assert any("不兼容" in err.message for err in errors)
