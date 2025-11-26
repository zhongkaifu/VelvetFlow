import json
from pathlib import Path

from velvetflow.validate_workflow import validate_workflow_data

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
