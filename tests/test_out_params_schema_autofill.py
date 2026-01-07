import copy

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.verification.validation import validate_completed_workflow


def test_action_out_params_autofilled_from_registry():
    workflow = {
        "workflow_name": "autofill_out_params_schema",
        "description": "",
        "nodes": [
            {
                "id": "demo_action",
                "type": "action",
                "action_id": BUSINESS_ACTIONS[0]["action_id"],
                "display_name": "Demo action",
                "params": {
                    "prompt": "hello",
                    "expected_format": '{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}',
                },
                "out_params_schema": None,
                "depends_on": [],
            }
        ],
    }

    action_def = next(a for a in BUSINESS_ACTIONS if a["action_id"] == workflow["nodes"][0]["action_id"])
    expected_schema = copy.deepcopy(action_def.get("output_schema"))
    expected_schema["properties"]["results"] = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    errors = validate_completed_workflow(workflow, action_registry=BUSINESS_ACTIONS)

    assert not errors
    assert workflow["nodes"][0]["out_params_schema"] == expected_schema

