from velvetflow.models import Workflow


def test_constant_string_templates_render_before_validation():
    workflow = {
        "workflow_name": "{{ 'literal_workflow' }}",
        "description": "{{ 'desc' }}",
        "nodes": [
            {
                "id": "{{ 'loop_employees' }}",
                "type": "{{ 'loop' }}",
                "display_name": "{{ 'loop employees' }}",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ [] }}",
                    "item_alias": "{{ 'employee' }}",
                    "exports": {"items": ["employee_id"]},
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "{{ 'check_temperature' }}",
                                "type": "{{ 'condition' }}",
                                "display_name": "{{ 'check temp' }}",
                                "params": {"expression": "{{ 1 == 1 }}"},
                                "true_to_node": "collect_warning",
                                "false_to_node": None,
                                "parent_node_id": "{{ 'loop_employees' }}",
                                "depends_on": [],
                            },
                            {
                                "id": "{{ 'collect_warning' }}",
                                "type": "{{ 'action' }}",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "display_name": "{{ 'collect warning' }}",
                                "params": {"status": "{{ 'ok' }}"},
                                "parent_node_id": "{{ 'loop_employees' }}",
                                "depends_on": ["check_temperature"],
                            },
                        ]
                    },
                },
                "depends_on": [],
            }
        ],
    }

    validated = Workflow.model_validate(workflow)

    loop = validated.nodes[0]
    assert loop.id == "loop_employees"
    assert loop.type == "loop"
    assert loop.params["item_alias"] == "employee"
    assert validated.workflow_name == "literal_workflow"

    body_nodes = loop.params["body_subgraph"]["nodes"]
    assert any(node.get("type") == "action" for node in body_nodes)
    assert body_nodes[0]["id"] == "check_temperature"
