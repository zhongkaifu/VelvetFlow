# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.verification.jinja_validation import (
    normalize_condition_params_to_jinja,
    normalize_params_to_jinja,
)


BASE_WORKFLOW = {
    "workflow_name": "normalize_expr",
    "description": "",
    "nodes": [
        {
            "id": "fetch",
            "type": "action",
            "action_id": "hr.get_today_temperatures.v1",
            "params": {"date": "2024-01-01"},
        },
        {
            "id": "record",
            "type": "action",
            "action_id": "hr.record_health_event.v1",
            "params": {
                "event_type": "temperature",
                "date": "result_of.fetch.date",
                "abnormal_count": "result_of.fetch.data | length > 0",
            },
        },
        {
            "id": "check",
            "type": "condition",
            "params": {
                "expression": "result_of.fetch.data | length > 0",
            },
            "true_to_node": None,
            "false_to_node": None,
        },
    ],
    "edges": [
        {"from": "fetch", "to": "record"},
        {"from": "record", "to": "check"},
    ],
}


def test_normalize_params_wraps_unwrapped_jinja_expression():
    normalized, summary, errors = normalize_params_to_jinja(BASE_WORKFLOW)

    assert errors == []
    assert summary.get("applied") is True

    record_params = normalized["nodes"][1]["params"]
    assert record_params["date"] == "{{ result_of.fetch.date }}"
    assert record_params["abnormal_count"] == "{{ result_of.fetch.data | length > 0 }}"


def test_normalize_condition_params_wraps_unwrapped_expression():
    normalized, summary, errors = normalize_condition_params_to_jinja(BASE_WORKFLOW)

    assert errors == []
    assert summary.get("applied") is True

    condition_params = normalized["nodes"][2]["params"]
    assert condition_params["expression"] == "{{ result_of.fetch.data | length > 0 }}"
