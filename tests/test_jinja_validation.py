import pytest

from velvetflow.verification.jinja_validation import normalize_params_to_jinja


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
    assert summary["applied"] is False
    assert errors and errors[0].code == "INVALID_JINJA_EXPRESSION"
    assert errors[0].node_id == "n1"
