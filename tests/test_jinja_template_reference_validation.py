# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.verification.validation import validate_completed_workflow

ACTION_REGISTRY = BUSINESS_ACTIONS


def _workflow_with_expression_param(expression: str):
    return {
        "workflow_name": "jinja_expression_param",
        "description": "",
        "nodes": [
            {
                "id": "fetch_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-01-01"},
            },
            {
                "id": "record_event",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "temperature",
                    "date": "{{ result_of.fetch_temperatures.date }}",
                    "abnormal_count": expression,
                },
            },
        ],
        "edges": [
            {"from": "fetch_temperatures", "to": "record_event"},
        ],
    }


def test_template_expression_with_missing_field_is_reported():
    workflow = _workflow_with_expression_param(
        "{{ (result_of.fetch_temperatures.data | length) > 0 and result_of.fetch_temperatures.data[0].missing }}"
    )
    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert any(
        err.code == "SCHEMA_MISMATCH" and err.field == "abnormal_count"
        for err in errors
    )


def test_template_expression_with_valid_field_passes():
    workflow = _workflow_with_expression_param(
        "{{ (result_of.fetch_temperatures.data | length) > 0 and result_of.fetch_temperatures.data[0].temperature }}"
    )
    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert errors == []
