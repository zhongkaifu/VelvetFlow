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


def test_loop_alias_source_does_not_crash_array_field_validation():
    """Ensure loop body conditions using item_alias strings won't trigger index errors."""

    workflow = {
        "workflow_name": "temperature_loop_alias",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start", "params": {}},
            {
                "id": "get_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": {"__from__": "result_of.start", "__agg__": "identity"}},
            },
            {
                "id": "loop_check_temp",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.get_temperatures.data", "__agg__": "identity"},
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
                                "id": "record_high_temp",
                                "type": "action",
                                "action_id": "hr.record_health_event.v1",
                                "params": {},
                            },
                        ],
                        "edges": [
                            {"from": "check_temp_condition", "to": "record_high_temp", "condition": "true"}
                        ],
                        "entry": "check_temp_condition",
                        "exit": "record_high_temp",
                    },
                    "exports": {"items": {"from_node": "record_high_temp", "fields": ["event_id"]}},
                },
            },
        ],
        "edges": [
            {"from": "start", "to": "get_temperatures"},
            {"from": "get_temperatures", "to": "loop_check_temp"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "MISSING_REQUIRED_PARAM" and e.node_id == "record_high_temp" for e in errors
    ), "Expected validation to surface missing params without crashing"


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


def test_loop_exports_allow_aggregates_from_loop_self():
    """Aggregates may use the loop id as from_node to avoid referencing body nodes externally."""

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
                            {"id": "start", "type": "start"},
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
                            {"from": "start", "to": "append_high_temp_id", "condition": None},
                            {"from": "append_high_temp_id", "to": "exit", "condition": None},
                        ],
                        "entry": "start",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "append_high_temp_id",
                            "fields": ["employee_id"],
                            "mode": "collect",
                        },
                        "aggregates": [
                            {
                                "name": "high_temp_employee_ids",
                                "kind": "collect",
                                "from_node": "loop_check_temp",
                                "source": "result_of.loop_check_temp.items",
                                "expr": {"field": "employee_id"},
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

    assert errors == []


def test_loop_exports_accept_body_node_bindings_within_same_loop():
    """Exports inside the loop can legally reference body nodes without tripping the static rule."""

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
                            {"id": "start", "type": "start"},
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
                            {"from": "start", "to": "append_high_temp_id", "condition": None},
                            {"from": "append_high_temp_id", "to": "exit", "condition": None},
                        ],
                        "entry": "start",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "append_high_temp_id",
                            "fields": ["employee_id"],
                            "mode": "collect",
                        },
                        "aggregates": [
                            {
                                "name": "high_temp_employee_ids",
                                "kind": "collect",
                                "from_node": "append_high_temp_id",
                                "source": "result_of.loop_check_temp.items",
                                "expr": {"field": "employee_id"},
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

    assert errors == []
