# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.models import ValidationError
from velvetflow.planner.repair_tools import (
    fill_loop_exports_defaults,
    fix_missing_loop_exports_items,
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


def test_fix_missing_loop_exports_items_inserts_segment():
    workflow = {
        "nodes": [
            {
                "id": "get_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {},
            },
            {
                "id": "loop_employees",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.get_temperatures.data",
                    "item_alias": "employee",
                    "body_subgraph": {"nodes": [], "edges": []},
                    "exports": {
                        "items": {
                            "from_node": "add_to_warning_list",
                            "fields": ["employee_id"],
                        }
                    },
                },
            },
            {
                "id": "check_warning_list_empty",
                "type": "condition",
                "params": {
                    "source": "result_of.loop_employees.exports.employee_id",
                    "field": "employee_id",
                    "kind": "list_not_empty",
                },
            },
        ],
        "edges": [],
    }

    errors = [
        ValidationError(
            code="SCHEMA_MISMATCH",
            node_id="check_warning_list_empty",
            field="field",
            message=(
                "condition 节点 'check_warning_list_empty' 的引用 "
                "'result_of.loop_employees.exports.employee_id' 无法在 schema 中找到或缺少类型信息。"
            ),
        )
    ]

    patched, summary = fix_missing_loop_exports_items(workflow, errors)

    assert summary["applied"] is True
    condition = next(n for n in patched["nodes"] if n.get("id") == "check_warning_list_empty")
    assert (
        condition["params"]["source"]
        == "result_of.loop_employees.exports.items.employee_id"
    )
