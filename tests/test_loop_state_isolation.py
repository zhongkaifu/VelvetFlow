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
                                "params": {"value": {"__from__": "loop.item"}},
                            }
                        ],
                        "edges": [],
                    },
                    "exports": {
                        "items": {
                            "from_node": "loop_action",
                            "fields": ["value"],
                        }
                    },
                },
            }
        ],
        "edges": [],
    }

    return Workflow.model_validate(workflow_dict)


def test_loop_body_results_are_cleared_between_iterations():
    action_id = "hr.notify_human.v1"
    workflow = build_loop_workflow(action_id)
    executor = DynamicActionExecutor(
        workflow, simulations={action_id: {"result": {"value": "{{value}}"}}}
    )

    results = executor.run()

    assert "loop_action" not in results
    assert results["loop"]["items"] == [{"value": "1"}, {"value": "2"}]


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
                                "params": {"kind": "equals", "source": {"__from__": "loop.item"}, "value": True},
                                "true_to_node": "t_branch",
                                "false_to_node": None,
                            },
                            {
                                "id": "t_branch",
                                "type": "action",
                                "action_id": "hr.notify_human.v1",
                                "params": {},
                            },
                        ],
                    },
                    "exports": {
                        "items": {
                            "from_node": "t_branch",
                            "fields": ["branch"],
                        }
                    },
                },
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow, simulations={"hr.notify_human.v1": {"result": {"branch": "true_taken"}}}
    )

    results = executor.run()

    assert results["loop"]["items"] == [{"branch": "true_taken"}]
