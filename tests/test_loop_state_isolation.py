# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow


def build_loop_workflow(action_id: str) -> Workflow:
    workflow_dict = {
        "workflow_name": "loop_isolation",
        "nodes": [
            {
                "id": "loop",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": [1, 2],
                    "item_alias": "value",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "loop_action",
                                "type": "action",
                                "action_id": action_id,
                                "params": {"email_content": "{{ loop.item }}"},
                            }
                        ],
                        "edges": [],
                    },
                    "exports": {"items": "{{ result_of.loop_action.email_content }}"},
                },
            }
        ],
        "edges": [],
    }

    return Workflow.model_validate(workflow_dict)


def test_loop_body_results_are_cleared_between_iterations():
    action_id = "productivity.compose_outlook_email.v1"
    workflow = build_loop_workflow(action_id)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            action_id: {"result": {"status": "simulated", "email_content": "{{email_content}}"}}
        },
    )

    results = executor.run()

    assert "loop_action" not in results
    assert results["loop"]["exports"]["items"] == ["1", "2"]


def test_nested_loop_results_do_not_leak_across_outer_iterations():
    workflow_dict = {
        "workflow_name": "nested_loop_isolation",
        "nodes": [
            {
                "id": "outer_loop",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": [[1, 2], [3]],
                    "item_alias": "group",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "inner_loop",
                                "type": "loop",
                                "params": {
                                    "loop_kind": "for_each",
                                    "source": "{{ loop.group }}",
                                    "item_alias": "item",
                                    "body_subgraph": {
                                        "nodes": [
                                            {
                                                "id": "record_item",
                                                "type": "action",
                                                "action_id": "productivity.compose_outlook_email.v1",
                                                "params": {"email_content": "{{ loop.item }}"},
                                            }
                                        ]
                                    },
                                    "exports": {"items": "{{ result_of.record_item.email_content }}"},
                                },
                            }
                        ]
                    },
                    "exports": {"items": "{{ result_of.inner_loop.exports.items }}"},
                },
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {
                "result": {"email_content": "{{email_content}}"}
            }
        },
    )

    results = executor.run()

    assert "record_item" not in results
    assert "inner_loop" not in results
    assert results["outer_loop"]["exports"]["items"] == ["1", "2", "3"]


def test_condition_branch_inside_loop_body_uses_true_target():
    workflow_dict = {
        "workflow_name": "loop_condition_branch",
        "nodes": [
            {
                "id": "loop",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": [True],
                    "item_alias": "flag",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "check_flag",
                                "type": "condition",
                                "params": {"expression": "{{ loop.item == true }}"},
                                "true_to_node": "t_branch",
                                "false_to_node": None,
                            },
                            {
                                "id": "t_branch",
                                "type": "action",
                                "action_id": "productivity.compose_outlook_email.v1",
                                "params": {"email_content": "flag true"},
                            },
                        ],
                    },
                    "exports": {"items": "{{ result_of.t_branch.branch }}"},
                },
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"branch": "true_taken"}}
        },
    )

    results = executor.run()

    assert results["loop"]["exports"]["items"] == ["true_taken"]


def test_condition_false_branch_inside_loop_body_skips_true_target():
    workflow_dict = {
        "workflow_name": "loop_condition_branch_false",
        "nodes": [
            {
                "id": "loop",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": [
                        {"employee_id": "001", "temperature": 36.5},
                    ],
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "check_temperature_condition",
                                "type": "condition",
                                "params": {
                                    "expression": "{{ loop.employee.temperature > 38 }}",
                                },
                                "true_to_node": "add_to_warning_list",
                                "false_to_node": None,
                            },
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "{{ loop.employee.employee_id }}",
                                    "last_temperature": "{{ loop.employee.temperature }}",
                                    "status": "fever",
                                },
                            },
                        ],
                    },
                    "exports": {"items": "{{ result_of.add_to_warning_list.employee_id }}"},
                },
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow, simulations={"hr.update_employee_health_profile.v1": {"result": {"employee_id": "{{employee_id}}"}}}
    )

    results = executor.run()

    assert results["loop"]["exports"]["items"] == []
