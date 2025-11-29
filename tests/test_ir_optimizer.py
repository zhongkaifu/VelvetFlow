import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.ir import build_ir_from_workflow, generate_target_specs, materialize_workflow_from_ir, optimize_ir

PURE_ACTIONS = [
    {
        "action_id": "literal.int",
        "description": "constant integer",
        "pure": True,
        "arg_schema": {"type": "object", "properties": {"value": {"type": "integer"}}},
        "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}}},
    },
    {
        "action_id": "literal.str",
        "description": "constant string",
        "pure": True,
        "arg_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        "output_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
    },
    {
        "action_id": "math.add",
        "description": "add numbers",
        "pure": True,
        "arg_schema": {
            "type": "object",
            "properties": {
                "left": {"type": "number"},
                "right": {"type": "number"},
            },
        },
        "output_schema": {"type": "object", "properties": {"value": {"type": "number"}}},
    },
]


def _demo_workflow():
    return {
        "workflow_name": "ir_demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "const_a", "type": "action", "action_id": "literal.int", "params": {"value": 1}},
            {"id": "const_b", "type": "action", "action_id": "literal.int", "params": {"value": 1}},
            {"id": "unused_const", "type": "action", "action_id": "literal.str", "params": {"value": "drop"}},
            {
                "id": "combine",
                "type": "action",
                "action_id": "math.add",
                "params": {
                    "left": {"__from__": "result_of.const_a.value"},
                    "right": {"__from__": "result_of.const_b.value"},
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "const_a"},
            {"from": "start", "to": "const_b"},
            {"from": "start", "to": "unused_const"},
            {"from": "const_a", "to": "combine"},
            {"from": "const_b", "to": "combine"},
            {"from": "combine", "to": "end"},
        ],
    }


def test_ir_optimizations_inline_and_prune():
    ir = build_ir_from_workflow(_demo_workflow(), action_registry=PURE_ACTIONS)
    optimized = optimize_ir(ir)

    assert "unused_const" not in optimized.nodes  # dead, pure, unused
    assert optimized.metadata.get("cse_redirects", {}).get("const_b") == "const_a"

    materialized = materialize_workflow_from_ir(optimized)
    combine = next(n for n in materialized["nodes"] if n["id"] == "combine")
    assert combine["params"]["left"] == 1
    assert combine["params"]["right"] == 1

    layers = optimized.metadata.get("parallel_layers")
    assert layers and any({"const_a", "const_b"} & set(layer) for layer in layers)


def test_generate_targets_from_optimized_ir():
    workflow_json = json.dumps(_demo_workflow())
    ir = build_ir_from_workflow(workflow_json, action_registry=PURE_ACTIONS)
    optimized = optimize_ir(ir)
    targets = generate_target_specs(optimized)

    assert set(targets.keys()) == {"argo", "airflow", "k8s", "native"}
    assert any(t.get("name") == "combine" for t in targets["argo"]["templates"])
    assert targets["native"].get("pruned_nodes") is not None
