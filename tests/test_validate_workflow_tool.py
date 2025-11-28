import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.models import ValidationError, Workflow
from velvetflow.planner.repair_tools import _apply_local_repairs_for_unknown_params

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def _basic_workflow(params: dict) -> dict:
    return {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "notify", "type": "action", "action_id": "hr.notify_human.v1", "params": params},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "notify"},
            {"from": "notify", "to": "end"},
        ],
    }


def test_validate_workflow_success():
    workflow = _basic_workflow({"message": "hello"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_validate_workflow_missing_required_param():
    workflow = _basic_workflow({})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation to fail due to missing required param"
    assert any(e.code == "MISSING_REQUIRED_PARAM" for e in errors)


def test_validate_workflow_unknown_param():
    workflow = _basic_workflow({"message": "hello", "date_filter": "today"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation to fail due to unknown param"
    assert any(e.code == "UNKNOWN_PARAM" and e.field == "date_filter" for e in errors)


def test_local_repair_removes_unknown_param():
    workflow = Workflow.model_validate(
        _basic_workflow({"message": "hello", "date_filter": "today"})
    )
    validation_errors = [
        ValidationError(
            code="UNKNOWN_PARAM",
            node_id="notify",
            field="date_filter",
            message="unexpected param",
        )
    ]

    repaired = _apply_local_repairs_for_unknown_params(
        current_workflow=workflow,
        validation_errors=validation_errors,
        action_registry=ACTION_REGISTRY,
    )

    assert repaired is not None
    repaired_params = next(
        n for n in repaired.model_dump(by_alias=True)["nodes"] if n["id"] == "notify"
    )["params"]
    assert "date_filter" not in repaired_params


def test_invalid_condition_reference_detected():
    workflow = {
        "workflow_name": "health_alert",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "condition_any_high_temp",
                "type": "condition",
                "params": {"kind": "list_not_empty", "source": []},
            },
            {
                "id": "action_generate_alert_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "High Temperature Alert",
                    "abnormal_count": {
                        "__from__": "result_of.condition_any_high_temp.exports.items",
                        "__agg__": "count",
                    },
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "condition_any_high_temp"},
            {"from": "condition_any_high_temp", "to": "action_generate_alert_report"},
            {"from": "action_generate_alert_report", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        err.code == "SCHEMA_MISMATCH" and "exports.items" in err.message for err in errors
    )
