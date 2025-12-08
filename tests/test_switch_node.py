# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import PydanticValidationError, Workflow, infer_edges_from_bindings


def test_switch_executes_matching_case():
    workflow_dict = {
        "workflow_name": "switch_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "route",
                "type": "switch",
                "params": {"source": {"color": "blue"}, "field": "color"},
                "cases": [
                    {"match": "blue", "to_node": "blue_action"},
                    {"match": "red", "to_node": "red_action"},
                ],
                "default_to_node": "fallback",
            },
            {
                "id": "blue_action",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {},
            },
            {
                "id": "red_action",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {},
            },
            {
                "id": "fallback",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {},
            },
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {
                "result": {"status": "simulated", "branch": "composed"}
            }
        },
    )

    results = executor.run()

    assert results["route"]["matched_case"] == "blue"
    assert "blue_action" in results
    assert "red_action" not in results
    assert "fallback" not in results


def test_switch_infers_branch_edges():
    nodes = [
        {
            "id": "route",
            "type": "switch",
            "params": {},
            "cases": [
                {"match": 1, "to_node": "one"},
                {"match": [2, 3], "to_node": "two_or_three"},
            ],
            "default_to_node": "other",
        },
        {"id": "one", "type": "action", "action_id": "demo.one", "params": {}},
        {"id": "two_or_three", "type": "action", "action_id": "demo.two", "params": {}},
        {"id": "other", "type": "action", "action_id": "demo.other", "params": {}},
    ]

    edges = infer_edges_from_bindings(nodes)

    assert {"from": "route", "to": "one", "condition": "1"} in edges
    assert {"from": "route", "to": "two_or_three", "condition": "[2, 3]"} in edges
    assert {"from": "route", "to": "other", "condition": "default"} in edges


def test_switch_validates_case_targets():
    with pytest.raises(PydanticValidationError):
        Workflow.model_validate(
            {
                "workflow_name": "invalid",
                "nodes": [
                    {
                        "id": "route",
                        "type": "switch",
                        "params": {},
                        "cases": [{"match": "x", "to_node": 123}],
                    }
                ],
            }
        )
