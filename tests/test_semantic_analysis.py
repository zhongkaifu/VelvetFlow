import json
from pathlib import Path

from validate_workflow import validate_workflow_data


ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def _base_workflow():
    return {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "end", "type": "end"},
        ],
        "edges": [],
    }


def test_duplicate_node_symbol_detected():
    workflow = _base_workflow()
    workflow["nodes"].append({"id": "start", "type": "action", "action_id": "productivity.compose_outlook_email.v1"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(err.code == "DUPLICATE_SYMBOL" for err in errors)


def test_undefined_edge_reference_reported():
    workflow = _base_workflow()
    workflow["edges"].append({"from": "start", "to": "missing"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(err.code == "UNDEFINED_REFERENCE" and "missing" in err.message for err in errors)


def test_contract_violation_for_incompatible_binding():
    workflow = {
        "workflow_name": "contract_mismatch",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "compose_email",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "hello"},
            },
            {
                "id": "record_event",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "alert",
                    "abnormal_count": {"__from__": "result_of.compose_email.message"},
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "compose_email"},
            {"from": "compose_email", "to": "record_event"},
            {"from": "record_event", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(err.code == "CONTRACT_VIOLATION" for err in errors)

