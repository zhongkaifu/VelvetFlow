from velvetflow.planner.repair_tools import (
    fill_loop_exports_defaults,
    normalize_binding_paths,
)


def test_fill_loop_exports_defaults_adds_missing_items_fields():
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "loop1",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.start"},
                    "item_alias": "item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "inner_action",
                                "type": "action",
                                "action_id": "search",
                                "params": {},
                            }
                        ],
                        "entry": "inner_action",
                        "exit": "inner_action",
                        "edges": [{"from": "inner_action", "to": "inner_action"}],
                    },
                },
            },
        ],
        "edges": [{"from": "start", "to": "loop1"}],
    }

    action_registry = [
        {
            "action_id": "search",
            "arg_schema": {},
            "output_schema": {"properties": {"title": {}, "url": {}, "score": {}}},
        }
    ]

    patched, summary = fill_loop_exports_defaults(workflow, action_registry=action_registry)

    assert summary["applied"] is True
    loop_node = next(n for n in patched["nodes"] if n.get("id") == "loop1")
    exports = loop_node["params"].get("exports")
    assert exports["items"]["from_node"] == "inner_action"
    assert exports["items"]["fields"]


def test_normalize_binding_paths_unwraps_templates():
    workflow = {
        "nodes": [
            {
                "id": "a1",
                "type": "action",
                "action_id": "stub",
                "params": {"path": {"__from__": "{{ result_of.previous.output }}"}},
            }
        ],
        "edges": [],
    }

    patched, summary = normalize_binding_paths(workflow)

    assert summary["applied"] is True
    node = patched["nodes"][0]
    assert node["params"]["path"]["__from__"] == "result_of.previous.output"
