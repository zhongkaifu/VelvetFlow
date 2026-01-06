from velvetflow.models import ValidationError, Workflow
from velvetflow.planner.orchestrator import _collect_missing_required_param_errors


def test_collect_missing_required_params_flags_empty_strings():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "id": "writer",
                    "type": "action",
                    "action_id": "text.generate",
                    "params": {"prompt": "", "tone": "casual"},
                }
            ]
        }
    )

    action_registry = [
        {
            "action_id": "text.generate",
            "arg_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["prompt", "tone"],
            },
        }
    ]

    errors = _collect_missing_required_param_errors(workflow, action_registry)

    assert any(isinstance(err, ValidationError) and err.field == "prompt" for err in errors)
    assert all(err.node_id == "writer" for err in errors)
