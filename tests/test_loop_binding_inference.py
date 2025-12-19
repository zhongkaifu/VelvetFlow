# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

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


def test_missing_exports_segment_is_inferred():
    ctx = _build_loop_ctx(
        exports={"values": "{{ result_of.noop.value }}"},
        results={
            "loop": {
                "exports": {"values": [1, 2]},
            }
        },
    )

    assert ctx.resolve_binding({"__from__": "result_of.loop.values"}) == [1, 2]
