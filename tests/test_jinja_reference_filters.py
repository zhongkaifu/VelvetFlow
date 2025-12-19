import pytest

from velvetflow.loop_dsl import index_loop_body_nodes
from velvetflow.verification.binding_checks import _index_actions_by_id
from velvetflow.verification.node_rules import _validate_nodes_recursive


def test_filter_expression_paths_validated_on_base_reference():
    workflow = {
        "nodes": [
            {
                "id": "get_today_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "{{ system.date }}"},
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "employee_id": {"type": "string"},
                                    "temperature": {"type": "number"},
                                    "status": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
            {
                "id": "loop_employees",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ result_of.get_today_temperatures.data }}",
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "collect_high_temp_employee",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "{{ employee.employee_id }}",
                                    "status": "High Temperature",
                                },
                                "out_params_schema": {
                                    "type": "object",
                                    "properties": {
                                        "employee_id": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                },
                            }
                        ]
                    },
                    "exports": {"statuses": "{{ result_of.collect_high_temp_employee.status }}"},
                },
                "depends_on": ["get_today_temperatures"],
            },
            {
                "id": "generate_warning_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "abnormal_count": "{{ result_of.loop_employees.exports.statuses | select('equalto', '异常') | list | length }}",
                },
                "depends_on": ["loop_employees"],
            },
        ]
    }

    nodes = workflow["nodes"]
    nodes_by_id = {n["id"]: n for n in nodes}
    loop_body_parents = index_loop_body_nodes(workflow)
    actions_by_id = _index_actions_by_id(
        [
            {"action_id": "hr.update_employee_health_profile.v1", "output_schema": nodes[1]["params"]["body_subgraph"]["nodes"][0]["out_params_schema"]},
            {"action_id": "hr.record_health_event.v1", "output_schema": {"type": "object", "properties": {"event_id": {"type": "string"}}}},
            {"action_id": "hr.get_today_temperatures.v1", "output_schema": nodes[0]["out_params_schema"]},
        ]
    )

    errors = []
    _validate_nodes_recursive(nodes, nodes_by_id, actions_by_id, loop_body_parents, errors)

    assert errors == []
