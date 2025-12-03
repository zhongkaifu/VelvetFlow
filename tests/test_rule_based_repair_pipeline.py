from velvetflow.models import ValidationError
from velvetflow.planner.repair import apply_rule_based_repairs, _summarize_validation_errors_for_llm


def test_rule_repairs_run_before_llm_and_clear_missing_param():
    workflow = {
        "nodes": [
            {"id": "producer", "type": "action", "action_id": "emit", "params": {}},
            {"id": "consumer", "type": "action", "action_id": "use", "params": {}},
        ],
        "edges": [],
    }
    actions = [
        {
            "action_id": "emit",
            "arg_schema": {"type": "object", "properties": {"seed": {"type": "integer"}}},
            "output_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        {
            "action_id": "use",
            "arg_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "output_schema": {},
        },
    ]
    errors = [
        ValidationError(
            code="MISSING_REQUIRED_PARAM",
            node_id="consumer",
            field="text",
            message="params is empty",
        )
    ]

    workflow["nodes"][0]["params"] = {"seed": 1}
    patched, remaining_errors = apply_rule_based_repairs(workflow, actions, errors)

    assert remaining_errors == []
    consumer = next(n for n in patched["nodes"] if n.get("id") == "consumer")
    assert consumer["params"]["text"]["__from__"] == "result_of.producer.text"
    assert patched.get("edges", []) == []


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
