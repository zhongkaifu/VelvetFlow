# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.executor import DynamicActionExecutor
from velvetflow.executor.conditions import ConditionEvaluationMixin
from velvetflow.models import Workflow
from velvetflow.bindings import BindingContext


def _build_workflow(expression, true_node_id="t_branch", false_node_id="f_branch"):
    return Workflow.model_validate(
        {
            "workflow_name": "condition_expression_eval",
            "description": "",
            "nodes": [
                {
                    "id": "check",
                    "type": "condition",
                    "params": {"expression": expression},
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


class _ConditionHarness(ConditionEvaluationMixin):
    """Lightweight harness to call condition evaluation helpers directly."""


def test_condition_alias_is_exposed_at_top_level():
    workflow = Workflow.model_validate({"workflow_name": "alias_ctx", "nodes": []})
    ctx = BindingContext(
        workflow,
        results={},
        loop_ctx={
            "item": {"temperature": 39},
            "employee": {"temperature": 39},
            "index": 0,
        },
        loop_id="loop_check_temperature",
    )
    harness = _ConditionHarness()
    node = {"params": {"expression": "{{ employee.temperature > 38 }}"}}

    result, debug = harness._eval_condition(node, ctx, include_debug=True)

    assert result is True
    assert debug["resolved_value"] is True


def test_condition_compare_executes_true_branch():
    workflow = _build_workflow("{{ 8 >= 5 }}")

    executor = DynamicActionExecutor(
        workflow,
        simulations={"productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}},
    )

    results = executor.run()

    assert "t_branch" in results
    assert "f_branch" not in results


def test_condition_greater_than_executes_true_branch():
    workflow = _build_workflow("{{ 39 > 38 }}", true_node_id="add_to_warning_list", false_node_id="log_normal")

    executor = DynamicActionExecutor(
        workflow,
        simulations={"productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}},
    )

    results = executor.run()

    assert "add_to_warning_list" in results
    assert "log_normal" not in results


def test_condition_greater_than_executes_false_branch_when_below_threshold():
    workflow = _build_workflow("{{ 36.5 > 38 }}", true_node_id="add_to_warning_list", false_node_id="log_normal")

    executor = DynamicActionExecutor(
        workflow,
        simulations={"productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}},
    )

    results = executor.run()

    assert "add_to_warning_list" not in results
    assert "log_normal" in results


