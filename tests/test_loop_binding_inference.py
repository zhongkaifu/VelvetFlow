# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from velvetflow.bindings import BindingContext
from velvetflow.models import Workflow


def _build_loop_ctx(exports, results):
    workflow = Workflow.model_validate(
        {
            "workflow_name": "loop-binding-inference",
            "nodes": [
                {
                    "id": "loop",
                    "type": "loop",
                    "params": {
                        "loop_kind": "for_each",
                        "source": [],
                        "item_alias": "item",
                        "exports": exports,
                        "body_subgraph": {
                            "nodes": [{"id": "noop", "type": "action"}],
                            "edges": [],
                        },
                    },
                }
            ],
            "edges": [],
        }
    )
    return BindingContext(workflow, results)


def test_missing_items_segment_is_inferred():
    ctx = _build_loop_ctx(
        exports={"items": {"fields": ["value"]}},
        results={
            "loop": {
                "items": [{"value": 1}, {"value": 2}],
                "exports": {"items": [{"value": 1}, {"value": 2}]},
            }
        },
    )

    assert ctx.resolve_binding({"__from__": "result_of.loop.value"}) == [1, 2]


def test_ambiguous_loop_reference_raises_error():
    ctx = _build_loop_ctx(
        exports={
            "items": {"fields": ["score"]},
            "aggregates": [{"name": "score"}],
        },
        results={
            "loop": {
                "items": [{"score": 10}],
                "aggregates": {"score": 99},
                "exports": {
                    "items": [{"score": 10}],
                    "aggregates": {"score": 99},
                },
            }
        },
    )

    with pytest.raises(ValueError, match=r"存在歧义"):
        ctx.resolve_binding({"__from__": "result_of.loop.score"})
