import pytest

from velvetflow.models import Workflow


def _workflow_with_scalar_loop_source():
    return {
        "workflow_name": "test",
        "description": "",
        "nodes": [
            {
                "id": "fetch",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "display_name": "fetch",
                "params": {"date": "{{system.date}}"},
                "depends_on": [],
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "employee_id": {"type": "string"},
                                    "temperature": {"type": "number"},
                                },
                                "required": ["employee_id", "temperature"],
                            },
                        },
                    },
                    "required": ["date", "data"],
                },
            },
            {
                "id": "loop_invalid",
                "type": "loop",
                "display_name": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ result_of.fetch.date }}",
                    "item_alias": "row",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "noop",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "display_name": "noop",
                                "params": {"employee_id": "{{ row }}"},
                                "out_params_schema": {
                                    "type": "object",
                                    "properties": {"employee_id": {"type": "string"}},
                                    "required": ["employee_id"],
                                },
                                "parent_node_id": "loop_invalid",
                                "depends_on": [],
                            }
                        ]
                    },
                    "exports": {"items": "{{ result_of.noop }}"},
                },
                "depends_on": ["fetch"],
            },
        ],
    }


def test_loop_source_must_be_array_reports_actual_type():
    workflow = _workflow_with_scalar_loop_source()

    with pytest.raises(Exception) as excinfo:
        Workflow.model_validate(workflow)

    message = str(excinfo.value)
    assert "source 应该引用数组/序列" in message
    assert "类型为 string" in message or "类型为 未知" in message
