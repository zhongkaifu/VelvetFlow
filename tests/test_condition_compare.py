# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow


def _build_workflow(condition_params, true_node_id="t_branch", false_node_id="f_branch"):
    return Workflow.model_validate(
        {
            "workflow_name": "condition_kind_extensions",
            "description": "",
            "nodes": [
                {"id": "start", "type": "start"},
                {
                    "id": "check",
                    "type": "condition",
                    "params": condition_params,
                    "true_to_node": true_node_id,
                    "false_to_node": false_node_id,
                },
                {
                    "id": true_node_id,
                    "type": "action",
                    "action_id": "productivity.compose_outlook_email.v1",
                    "params": {"email_content": "true branch"},
                },
                {
                    "id": false_node_id,
                    "type": "action",
                    "action_id": "productivity.compose_outlook_email.v1",
                    "params": {"email_content": "false branch"},
                },
            ],
        }
    )


def test_condition_compare_executes_true_branch():
    workflow = _build_workflow(
        {
            "kind": "compare",
            "source": {"score": 8},
            "field": "score",
            "op": ">=",
            "value": 5,
        }
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    results = executor.run()

    assert "t_branch" in results
    assert "f_branch" not in results


def test_condition_compare_supports_custom_operator():
    workflow = _build_workflow(
        {
            "kind": "compare",
            "source": {"score": 2},
            "field": "score",
            "op": ">=",
            "value": 5,
        }
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    results = executor.run()

    assert "f_branch" in results
    assert "t_branch" not in results


def test_condition_greater_than_executes_true_branch():
    workflow = _build_workflow(
        {
            "kind": "greater_than",
            "source": {"temperature": 39},
            "field": "temperature",
            "threshold": 38,
        },
        true_node_id="add_to_warning_list",
        false_node_id="log_normal",
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    results = executor.run()

    assert "add_to_warning_list" in results
    assert "log_normal" not in results


def test_condition_greater_than_executes_false_branch_when_below_threshold():
    workflow = _build_workflow(
        {
            "kind": "greater_than",
            "source": {"temperature": 36.5},
            "field": "temperature",
            "threshold": 38,
        },
        true_node_id="add_to_warning_list",
        false_node_id="log_normal",
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    results = executor.run()

    assert "add_to_warning_list" not in results
    assert "log_normal" in results


def test_condition_greater_than_accepts_value_field():
    workflow = _build_workflow(
        {
            "kind": "greater_than",
            "source": {"temperature": 39},
            "field": "temperature",
            "value": 38,
        },
        true_node_id="add_to_warning_list",
        false_node_id="log_normal",
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    results = executor.run()

    assert "add_to_warning_list" in results
    assert "log_normal" not in results


def test_condition_compare_supports_expression_operator_and_target():
    workflow = _build_workflow(
        {
            "kind": "compare",
            "source": {"employee": {"temperature": 37}},
            "field": "employee.temperature",
            "expression": "value > 38",
        }
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    results = executor.run()

    assert "f_branch" in results
    assert "t_branch" not in results
