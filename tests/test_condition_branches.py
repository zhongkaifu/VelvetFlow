# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import pytest

from velvetflow.executor import DynamicActionExecutor
from velvetflow.bindings import BindingContext
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
            {
                "id": "yes",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "yes branch"},
            },
            {
                "id": "no",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "no branch"},
            },
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
                "params": {"expression": "{{ true }}"},
                "true_to_node": "notify_yes",
                "false_to_node": "notify_no",
            },
            {
                "id": "notify_yes",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "from_condition": {"__from__": "result_of.check.condition_result"},
                    "email_content": "condition true",
                },
            },
            {
                "id": "notify_no",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "from_condition": {"__from__": "result_of.check.condition_result"},
                    "email_content": "condition false",
                },
            },
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated", "branch": "notified"}}
        },
    )

    results = executor.run()

    assert "notify_yes" in results
    assert "notify_no" not in results


def test_condition_blocks_inactive_branch_even_if_preseeded():
    workflow_dict = {
        "workflow_name": "branching_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"expression": "{{ true }}"},
                "true_to_node": "notify_yes",
                "false_to_node": "notify_no",
            },
            {
                "id": "notify_yes",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "from_condition": {"__from__": "result_of.check.condition_result"},
                    "email_content": "condition true",
                },
            },
            {
                "id": "notify_no",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "from_condition": {"__from__": "result_of.check.condition_result"},
                    "email_content": "condition false",
                },
            },
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated"}}
        },
    )

    all_nodes = [n.id for n in workflow.nodes]
    binding_ctx = BindingContext(workflow, {})
    results = executor._execute_graph(
        executor.workflow_dict, binding_ctx, start_nodes=all_nodes
    )

    assert "notify_yes" in results
    assert "notify_no" not in results


def test_builder_preserves_explicit_condition_targets():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="check",
        node_type="condition",
        action_id=None,
        display_name=None,
        params={"expression": "{{ true }}"},
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
                "params": {"expression": "{{ true }}"},
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


def test_condition_requires_branch_targets():
    workflow = {
        "workflow_name": "missing_condition_branches",
        "description": "",
        "nodes": [
            {
                "id": "start",
                "type": "start",
            },
            {
                "id": "check",
                "type": "condition",
                "params": {"expression": "{{ true }}"},
            },
        ],
    }

    errors = validate_completed_workflow(workflow, action_registry=[])

    assert errors
    assert errors[0].code == "MISSING_REQUIRED_PARAM"
    assert "true_to_node" in errors[0].message
    assert "LLM" in errors[0].message


def test_executor_prefers_explicit_branch_over_edges_for_null_target():
    workflow_dict = {
        "workflow_name": "explicit_null_branch_stops",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "check",
                "type": "condition",
                "params": {"expression": "{{ true }}"},
                "true_to_node": None,
                "false_to_node": "f_branch",
            },
            {
                "id": "t_branch",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "true branch"},
            },
            {
                "id": "f_branch",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "false branch"},
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
            "productivity.compose_outlook_email.v1": {"result": {"status": "simulated", "branch": "notified"}}
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
                "params": {"expression": "{{ true }}"},
                "true_to_node": "null",
                "false_to_node": "notify",
            },
            {
                "id": "notify",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "notify"},
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
            "productivity.compose_outlook_email.v1": {
                "result": {"status": "simulated", "branch": "notified"}
            }
        },
    )

    results = executor.run()

    assert "check" in results
    assert "notify" not in results
