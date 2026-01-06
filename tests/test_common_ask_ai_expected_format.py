import json

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.models import Workflow
from velvetflow.planner.orchestrator import _attach_out_params_schema
from velvetflow.verification.validation import validate_completed_workflow


def _actions_by_id():
    return {action["action_id"]: action for action in BUSINESS_ACTIONS}


def test_ask_ai_requires_expected_format_param():
    workflow = {
        "workflow_name": "ask_ai_missing_expected_format",
        "nodes": [
            {
                "id": "ask_ai",
                "type": "action",
                "action_id": "common.ask_ai.v1",
                "display_name": "Ask AI",
                "params": {"prompt": "hello"},
                "depends_on": [],
            }
        ],
    }

    errors = validate_completed_workflow(workflow, action_registry=BUSINESS_ACTIONS)

    assert any(
        err.code == "MISSING_REQUIRED_PARAM" and err.field == "expected_format"
        for err in errors
    )


def test_expected_format_updates_results_schema():
    expected_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer"],
    }

    workflow = Workflow.model_validate(
        {
            "workflow_name": "ask_ai_expected_format_schema",
            "nodes": [
                {
                    "id": "ask_ai",
                    "type": "action",
                    "action_id": "common.ask_ai.v1",
                    "display_name": "Ask AI",
                    "params": {
                        "prompt": "hello",
                        "expected_format": json.dumps(expected_schema),
                    },
                    "out_params_schema": None,
                    "depends_on": [],
                }
            ],
        }
    )

    updated = _attach_out_params_schema(workflow, _actions_by_id())
    node = updated.model_dump(by_alias=True)["nodes"][0]
    results_schema = node["out_params_schema"]["properties"]["results"]

    assert node["out_params_schema"]["required"] and "results" in node["out_params_schema"]["required"]
    assert results_schema == expected_schema
