import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

validate_workflow = importlib.import_module("validate_workflow")
validate_workflow_data = validate_workflow.validate_workflow_data


def test_type_validation_surfaces_param_mismatch():
    action_registry = [
        {
            "action_id": "demo.echo",
            "name": "Echo",
            "arg_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
            "output_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ]

    workflow = {
        "workflow_name": "type-check-demo",
        "nodes": [
            {
                "id": "echo1",
                "type": "action",
                "action_id": "demo.echo",
                "params": {"text": 123},
            }
        ],
        "edges": [],
    }

    errors = validate_workflow_data(workflow, action_registry)

    assert any(e.code == "SCHEMA_MISMATCH" for e in errors)
    mismatch = next(e for e in errors if e.code == "SCHEMA_MISMATCH")
    assert mismatch.node_id == "echo1"
    assert mismatch.field == "params.text"
    assert "expects string" in mismatch.message or "string" in mismatch.message
