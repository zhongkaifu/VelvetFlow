import pytest

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow
from velvetflow.planner.workflow_builder import attach_condition_branches


def _find_node(nodes, node_id):
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def test_attach_condition_branches_populates_meta():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "check", "type": "condition", "params": {}},
            {"id": "yes", "type": "action", "action_id": "hr.notify_human.v1", "params": {}},
            {"id": "no", "type": "action", "action_id": "hr.notify_human.v1", "params": {}},
        ],
        "edges": [
            {"from": "check", "to": "yes", "condition": True},
            {"from": "check", "to": "no", "condition": False},
        ],
    }

    updated = attach_condition_branches(workflow)
    check_node = _find_node(updated["nodes"], "check")

    assert check_node["meta"].get("next_on_true") == "yes"
    assert check_node["meta"].get("next_on_false") == "no"


def test_executor_respects_condition_branch_meta():
    workflow_dict = {
        "workflow_name": "branching_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"kind": "equals", "source": True, "value": True},
                "meta": {"next_on_true": "notify_yes", "next_on_false": "notify_no"},
            },
            {
                "id": "notify_yes",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {"from_condition": {"__from__": "result_of.check.condition_result"}},
            },
            {
                "id": "notify_no",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {"from_condition": {"__from__": "result_of.check.condition_result"}},
            },
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "hr.notify_human.v1": {"result": {"status": "simulated", "branch": "notified"}}
        },
    )

    results = executor.run()

    assert "notify_yes" in results
    assert "notify_no" not in results
