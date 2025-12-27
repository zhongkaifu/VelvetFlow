from velvetflow.verification.validation import validate_completed_workflow


def test_action_node_requires_action_id():
    workflow = {
        "workflow_name": "missing_action_id",
        "nodes": [
            {
                "id": "action_missing",
                "type": "action",
                "action_id": None,
                "params": {"prompt": "hello"},
            }
        ],
    }

    errors = validate_completed_workflow(workflow, action_registry=[])

    assert any(err.code == "MISSING_ACTION_ID" for err in errors)
