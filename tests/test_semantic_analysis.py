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


def test_undefined_reference_from_binding_reported():
    workflow = _base_workflow()
    workflow["nodes"].append(
        {
            "id": "send",
            "type": "action",
            "action_id": "productivity.compose_outlook_email.v1",
            "params": {"email_content": {"__from__": "result_of.missing.message"}},
        }
    )

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


def test_contract_violation_ignored_when_aggregator_transforms_format():
    workflow = {
        "workflow_name": "contract_with_transform",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "get_temperature",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-01-01"},
            },
            {
                "id": "record_event",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": {
                        "__from__": "result_of.get_temperature.data",
                        "__agg__": "format_join",
                        "format": "{city}:{temperature}",
                        "sep": ";",
                    }
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "get_temperature"},
            {"from": "get_temperature", "to": "record_event"},
            {"from": "record_event", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert not any(err.code == "CONTRACT_VIOLATION" for err in errors)


def test_contract_violation_ignored_when_filter_map_formats_array():
    workflow = {
        "workflow_name": "contract_with_filter_map",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "get_temperature",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-01-02"},
            },
            {
                "id": "record_event",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": {
                        "__from__": "result_of.get_temperature.data",
                        "__agg__": "filter_map",
                        "filter_field": "temperature",
                        "filter_op": ">",
                        "filter_value": 37,
                        "map_field": "city",
                        "sep": ", ",
                    }
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "get_temperature"},
            {"from": "get_temperature", "to": "record_event"},
            {"from": "record_event", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert not any(err.code == "CONTRACT_VIOLATION" for err in errors)

