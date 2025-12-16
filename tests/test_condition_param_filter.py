import pytest

from velvetflow.verification.node_rules import (
    CONDITION_PARAM_FIELDS,
    _filter_params_by_supported_fields,
)


def test_condition_params_allow_expression_only():
    node = {
        "id": "cond1",
        "type": "condition",
        "params": {"expression": "{{ 1 == 1 }}"},
    }

    removed = _filter_params_by_supported_fields(node=node, actions_by_id={})

    assert CONDITION_PARAM_FIELDS == {"expression"}
    assert removed == []
    assert node["params"] == {"expression": "{{ 1 == 1 }}"}
