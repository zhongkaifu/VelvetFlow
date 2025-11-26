import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def _basic_workflow(params: dict) -> dict:
    return {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "notify", "type": "action", "action_id": "hr.notify_human.v1", "params": params},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "notify"},
            {"from": "notify", "to": "end"},
        ],
    }


def test_validate_workflow_success():
    workflow = _basic_workflow({"message": "hello"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_validate_workflow_missing_required_param():
    workflow = _basic_workflow({})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation to fail due to missing required param"
    assert any(e.code == "MISSING_REQUIRED_PARAM" for e in errors)


def test_validate_workflow_rejects_template_params():
    workflow = {
        "workflow_name": "template-demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "summarize_nvidia",
                "type": "action",
                "action_id": "common.summarize.v1",
                "params": {"text": "{{nvidia_news.snippet}}"},
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "summarize_nvidia"},
            {"from": "summarize_nvidia", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(err.code == "INVALID_TEMPLATE_BINDING" for err in errors)
