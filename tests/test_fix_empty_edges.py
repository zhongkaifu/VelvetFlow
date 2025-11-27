import json
from pathlib import Path

from velvetflow.planner.repair_tools import fix_empty_edges


def _workflow_with_blank_edges():
    return {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "action", "type": "action", "action_id": "noop"},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {},
            {"from": "", "to": "end"},
        ],
    }


def test_fix_empty_edges_autoconnects_and_drops_blank_entries(tmp_path: Path):
    workflow = _workflow_with_blank_edges()

    fixed, summary = fix_empty_edges(workflow)

    assert summary["applied"] is True
    assert summary["removed_edges"] == 2
    assert fixed["edges"], "edges 应该被自动补全"
    assert all(e.get("from") and e.get("to") for e in fixed["edges"])



def test_fix_empty_edges_noop_when_edges_valid():
    workflow = _workflow_with_blank_edges()
    workflow["edges"] = [
        {"from": "start", "to": "action", "condition": None},
        {"from": "action", "to": "end", "condition": None},
    ]

    fixed, summary = fix_empty_edges(workflow)

    assert summary.get("applied") is False
    assert fixed["edges"] == workflow["edges"]
