from velvetflow.models import ValidationError
from velvetflow.planner.repair import apply_rule_based_repairs, _summarize_validation_errors_for_llm


def test_error_summary_includes_error_specific_prompt():
    summary = _summarize_validation_errors_for_llm(
        [
            ValidationError(code="INVALID_EDGE", node_id="b", field="from", message="missing"),
            ValidationError(code="EMPTY_PARAMS", node_id="c", field="params", message="empty"),
        ],
        workflow={},
        action_registry=[],
    )

    assert "错误特定提示" in summary
    assert "INVALID_EDGE" in summary and "EMPTY_PARAMS" in summary
    assert "连接无入度" in summary or "修复边" in summary


def test_error_summary_includes_previous_attempts():
    summary = _summarize_validation_errors_for_llm(
        [
            ValidationError(
                code="INVALID_EDGE", node_id="b", field="from", message="still missing"
            )
        ],
        workflow={},
        action_registry=[],
        previous_attempts={"INVALID_EDGE:b:from": ["轮次 1 使用 LLM 修复 未修复：missing"]},
    )

    assert "历史修复尝试" in summary
    assert "轮次 1 使用 LLM 修复 未修复" in summary


def test_error_summary_surfaces_missing_required_param_context():
    errors = [
        ValidationError(
            code="MISSING_REQUIRED_PARAM",
            node_id="writer",
            field="prompt",
            message="missing required",
        )
    ]
    workflow = {
        "nodes": [
            {
                "id": "writer",
                "type": "action",
                "action_id": "text.generate",
                "params": {"prompt": "", "tone": ""},
            }
        ]
    }
    action_registry = [
        {
            "action_id": "text.generate",
            "arg_schema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["prompt", "tone"],
            },
        }
    ]

    summary = _summarize_validation_errors_for_llm(
        errors, workflow=workflow, action_registry=action_registry
    )

    assert "text.generate" in summary
    assert "required" in summary and "tone" in summary


def test_rule_repairs_surface_missing_loop_body():
    workflow = {"nodes": [{"id": "loop1", "type": "loop", "params": {"exports": {}}}]}

    patched, remaining_errors = apply_rule_based_repairs(workflow, [], [])

    assert patched["nodes"][0]["id"] == "loop1"
    assert any(
        err.code == "INVALID_LOOP_BODY" and err.field == "body_subgraph"
        for err in remaining_errors
    )


def test_rule_repairs_drop_invalid_switch_defaults():
    workflow = {
        "nodes": [
            {
                "id": "route_by_label",
                "type": "switch",
                "params": {"source": "{{ result_of.start.label }}", "field": "label"},
                "cases": [],
                "default_to_node": "null",
            },
            {"id": "start", "type": "action", "action_id": "demo.start", "params": {}},
        ],
        "edges": [],
    }
    errors = [
        ValidationError(
            code="UNDEFINED_REFERENCE",
            node_id="route_by_label",
            field="default_to_node",
            message="switch default branch default_to_node points to nonexistent node 'null'",
        )
    ]

    patched, _remaining_errors = apply_rule_based_repairs(workflow, [], errors)

    route = next(node for node in patched["nodes"] if node.get("id") == "route_by_label")
    assert route.get("default_to_node") is None
