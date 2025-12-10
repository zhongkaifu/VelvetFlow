# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.action_registry import BUSINESS_ACTIONS

ACTION_REGISTRY = BUSINESS_ACTIONS


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
        "__agg__": "format_join",
        "format": "{title}",
        "sep": ", ",
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert not errors


def test_param_binding_source_list_requires_type_compatibility():
    workflow = _base_workflow()
    workflow["nodes"][1]["params"]["text"] = {
        "__from__": ["result_of.search_news.results", "result_of.search_news.results"],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(err.code == "CONTRACT_VIOLATION" for err in errors)


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
