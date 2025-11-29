import pytest

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow
from velvetflow.verification.validation import validate_completed_workflow
from velvetflow.planner.workflow_builder import WorkflowBuilder, attach_condition_branches


def _find_node(nodes, node_id):
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def test_attach_condition_branches_populates_branch_targets():
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

    assert check_node.get("true_to_node") == "yes"
    assert check_node.get("false_to_node") == "no"


def test_executor_respects_condition_branch_targets():
    workflow_dict = {
        "workflow_name": "branching_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"kind": "equals", "source": True, "value": True},
                "true_to_node": "notify_yes",
                "false_to_node": "notify_no",
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


def test_builder_preserves_explicit_condition_targets():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="check",
        node_type="condition",
        action_id=None,
        display_name=None,
        params={"kind": "equals", "source": True, "value": True},
        true_to_node="yes",
        false_to_node="no",
    )
    builder.add_node(
        node_id="yes",
        node_type="action",
        action_id="demo.yes",
        display_name=None,
        params={},
    )
    builder.add_node(
        node_id="no",
        node_type="action",
        action_id="demo.no",
        display_name=None,
        params={},
    )

    wf = builder.to_workflow()
    check_node = _find_node(wf["nodes"], "check")

    assert check_node["true_to_node"] == "yes"
    assert check_node["false_to_node"] == "no"


def test_condition_branch_allows_null_exit():
    workflow_dict = {
        "workflow_name": "null_branch_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"kind": "equals", "source": True, "value": True},
                "true_to_node": None,
                "false_to_node": "end",
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "check"},
            {"from": "check", "to": "end", "condition": False},
        ],
    }

    errors = validate_completed_workflow(workflow_dict, action_registry=[])

    assert errors == []

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(workflow, simulations={})

    results = executor.run()

    assert "check" in results
    assert "end" not in results


def test_executor_prefers_explicit_branch_over_edges_for_null_target():
    workflow_dict = {
        "workflow_name": "explicit_null_branch_stops",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"kind": "equals", "source": True, "value": True},
                "true_to_node": None,
                "false_to_node": "f_branch",
            },
            {
                "id": "t_branch",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {},
            },
            {
                "id": "f_branch",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {},
            },
        ],
        "edges": [
            {"from": "start", "to": "check"},
            {"from": "check", "to": "t_branch", "condition": True},
            {"from": "check", "to": "f_branch", "condition": False},
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

    assert "check" in results
    assert "t_branch" not in results
    assert "f_branch" not in results


def test_string_null_target_stops_branch():
    workflow_dict = {
        "workflow_name": "string_null_branch_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"kind": "equals", "source": True, "value": True},
                "true_to_node": "null",
                "false_to_node": "notify",
            },
            {
                "id": "notify",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {},
            },
        ],
        "edges": [
            {"from": "start", "to": "check"},
            {"from": "check", "to": "null", "condition": True},
            {"from": "check", "to": "notify", "condition": False},
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "hr.notify_human.v1": {
                "result": {"status": "simulated", "branch": "notified"}
            }
        },
    )

    results = executor.run()

    assert "check" in results
    assert "notify" not in results
