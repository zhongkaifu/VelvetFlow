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
                    "action_id": "hr.notify_human.v1",
                    "params": {},
                },
                {
                    "id": false_node_id,
                    "type": "action",
                    "action_id": "hr.notify_human.v1",
                    "params": {},
                },
            ],
        }
    )


def test_condition_expression_executes_true_branch():
    workflow = _build_workflow(
        {
            "kind": "expression",
            "source": [1, 2, 3, 4],
            "expression": "len(value) >= 4",
        }
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={"hr.notify_human.v1": {"result": {"status": "simulated"}}},
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
        simulations={"hr.notify_human.v1": {"result": {"status": "simulated"}}},
    )

    results = executor.run()

    assert "f_branch" in results
    assert "t_branch" not in results
