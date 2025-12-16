import pytest

from velvetflow.verification.jinja_validation import normalize_params_to_jinja
from velvetflow.verification.jinja_validation import normalize_condition_params_to_jinja


def _workflow(params):
    return {"workflow_name": "demo", "nodes": [{"id": "n1", "type": "action", "params": params}]}


def test_params_binding_converted_to_jinja():
    workflow = _workflow({"target": {"__from__": "result_of.source.value"}})

    normalized, summary, errors = normalize_params_to_jinja(workflow)

    assert errors == []
    assert summary["applied"] is True
    assert normalized["nodes"][0]["params"]["target"] == "{{ result_of.source.value }}"


def test_aggregator_dsl_is_flagged():
    workflow = _workflow({"target": {"__from__": "result_of.source.items", "__agg__": {"op": "count"}}})

    _, summary, errors = normalize_params_to_jinja(workflow)

    assert summary.get("forbidden_paths")
    assert errors and errors[0].code == "INVALID_JINJA_EXPRESSION"
    assert errors[0].node_id == "n1"


def test_condition_field_literal_wrapped_as_jinja():
    workflow = {
        "workflow_name": "demo",
        "nodes": [
            {
                "id": "c1",
                "type": "condition",
                "params": {
                    "kind": "list_not_empty",
                    "source": {"__from__": "result_of.some_loop.exports.items"},
                    "field": "employee_id",
                },
            }
        ],
    }

    normalized, summary, errors = normalize_condition_params_to_jinja(workflow)

    assert errors == []
    assert summary["applied"] is True
    assert normalized["nodes"][0]["params"]["field"] == "{{ 'employee_id' }}"
