import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data

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


def test_validate_workflow_rejects_template_params():
    workflow = {
        "workflow_name": "template-demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "summarize_nvidia",
                "type": "action",
                "action_id": "common.summarize.v1",
                "params": {"text": "{{nvidia_news.snippet}}"},
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "summarize_nvidia"},
            {"from": "summarize_nvidia", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(err.code == "INVALID_TEMPLATE_BINDING" for err in errors)


def test_validate_workflow_rejects_loop_body_reference_from_outside():
    """Bindings must not directly reference loop body nodes from outside the loop."""

    workflow = {
        "workflow_name": "employee_temp_checks",
        "description": "",
        "nodes": [
            {
                "id": "fetch_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-06-07"},
            },
            {
                "id": "loop_check_temp",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {
                        "__from__": "result_of.fetch_temperatures.data",
                        "__agg__": "identity",
                    },
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "check_temp_condition",
                                "type": "condition",
                                "params": {
                                    "kind": "any_greater_than",
                                    "source": "employee",
                                    "field": "temperature",
                                    "threshold": 38,
                                },
                            },
                            {
                                "id": "append_high_temp_id",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": {
                                        "__from__": "employee.employee_id",
                                        "__agg__": "identity",
                                    }
                                },
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [
                            {
                                "from": "check_temp_condition",
                                "to": "append_high_temp_id",
                                "condition": "true",
                            },
                            {
                                "from": "append_high_temp_id",
                                "to": "exit",
                                "condition": None,
                            },
                        ],
                        "entry": "check_temp_condition",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "__from__": "result_of.loop_check_temp.items",
                            "__agg__": "identity",
                        },
                        "aggregates": [
                            {
                                "name": "high_temp_employee_ids",
                                "__from__": "result_of.append_high_temp_id.employee_id",
                                "__agg__": "collect",
                            }
                        ],
                    },
                },
            },
        ],
        "edges": [
            {"from": "fetch_temperatures", "to": "loop_check_temp"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        err.code == "STATIC_RULES_SUMMARY"
        and "循环外不允许直接引用子图节点 'append_high_temp_id'" in err.message
        for err in errors
    )
