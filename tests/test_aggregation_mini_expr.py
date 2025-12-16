import pytest

from velvetflow.bindings import BindingContext
from velvetflow.models import Workflow


def _workflow_with_node(node_id: str = "source") -> Workflow:
    return Workflow.model_validate({"nodes": [{"id": node_id, "type": "start", "params": {}}]})


def test_count_if_with_jinja_expression_and_defaults():
    workflow = _workflow_with_node()
    ctx = BindingContext(
        workflow,
        {"source": {"items": [{"score": 0.9}, {"score": 0.3}, {"score": 0.75}]}},
    )

    binding = {
        "__from__": "result_of.source.items",
        "__agg__": {
            "op": "count_if",
            "condition": "item.score > 0.5",
            "input_type": "array",
            "output_type": "integer",
            "on_empty": {"default": 0},
        },
    }

    assert ctx.resolve_binding(binding) == 2


def test_filter_map_uses_on_empty_and_type_guard():
    workflow = _workflow_with_node()
    ctx = BindingContext(workflow, {"source": {"items": [{"name": "a"}, {"name": "b"}]}})

    binding = {
        "__from__": "result_of.source.items",
        "__agg__": {
            "op": "filter_map",
            "condition": "item.name == 'z'",
            "input_type": "array",
            "output_type": "string",
            "on_empty": {"default": "no-match"},
        },
        "format": "{{ name }}",
    }

    assert ctx.resolve_binding(binding) == "no-match"

    binding["__agg__"]["output_type"] = "integer"
    with pytest.raises(ValueError):
        ctx.resolve_binding(binding)
