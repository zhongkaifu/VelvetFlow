# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.models import ValidationError
from velvetflow.planner.repair import _summarize_validation_errors_for_llm
from velvetflow.planner.repair_tools import (
    fill_loop_exports_defaults,
    fix_missing_loop_exports_items,
)


def test_fill_loop_exports_defaults_adds_missing_exports():
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "loop1",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ result_of.start }}",
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
    assert exports["items"] == "{{ result_of.inner_action.title }}"


def test_summarize_validation_errors_adds_loop_item_guidance():
    err = ValidationError(
        code="SCHEMA_MISMATCH",
        node_id="condition_high_temp",
        field="field",
        message=(
            "condition 节点 'condition_high_temp' 的引用 "
            "'loop_employees.item.temperature' 无法在 schema 中找到或缺少类型信息。"
        ),
    )

    summary = _summarize_validation_errors_for_llm([err])

    assert "loop item 引用修复" in summary
    assert "result_of.loop_employees.exports.<key>" in summary
