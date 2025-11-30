import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def _base_workflow():
    return {
        "workflow_name": "source_list",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "classifier",
                "type": "action",
                "action_id": "common.classify_text.v1",
                "params": {"text": "placeholder", "labels": ["news"]},
            },
        ],
        "edges": [{"from": "search_news", "to": "classifier"}],
    }


def test_param_binding_accepts_source_list():
    workflow = _base_workflow()
    workflow["nodes"][1]["params"]["text"] = {
        "__from__": ["result_of.search_news.results", "result_of.search_news.results"],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert not errors


def test_param_binding_source_list_reports_missing_paths():
    workflow = _base_workflow()
    workflow["nodes"][1]["params"]["text"] = {
        "__from__": ["result_of.search_news.results", "result_of.missing.output"],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors
    assert any(
        err.field == "text" and "节点 'missing'" in err.message for err in errors
    )
