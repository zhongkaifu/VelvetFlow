# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.models import ValidationError
from velvetflow.verification.repair_suggestions import generate_repair_suggestions


def test_ast_template_fills_missing_binding():
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
            "arg_schema": {},
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
            message="missing",
        )
    ]

    patched, suggestions = generate_repair_suggestions(workflow, actions, errors=errors)

    consumer = next(n for n in patched["nodes"] if n.get("id") == "consumer")
    assert consumer["params"]["text"]["__from__"] == "result_of.producer.text"
    assert any(s.strategy == "ast_template" for s in suggestions)


def test_constraint_solver_assigns_multiple_fields():
    workflow = {
        "nodes": [
            {
                "id": "text_source",
                "type": "action",
                "action_id": "emit_text",
                "params": {},
            },
            {
                "id": "count_source",
                "type": "action",
                "action_id": "emit_count",
                "params": {},
            },
            {"id": "consumer", "type": "action", "action_id": "use", "params": {}},
        ],
        "edges": [],
    }
    actions = [
        {
            "action_id": "emit_text",
            "arg_schema": {},
            "output_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
        {
            "action_id": "emit_count",
            "arg_schema": {},
            "output_schema": {"type": "object", "properties": {"total": {"type": "integer"}}},
        },
        {
            "action_id": "use",
            "arg_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "num": {"type": "integer"}},
            },
            "output_schema": {},
        },
    ]

    patched, suggestions = generate_repair_suggestions(workflow, actions)

    consumer = next(n for n in patched["nodes"] if n.get("id") == "consumer")
    assert consumer["params"]["text"]["__from__"] == "result_of.text_source.title"
    assert consumer["params"]["num"]["__from__"] == "result_of.count_source.total"
    assert any(s.strategy == "constraint_solver" for s in suggestions)


def test_statistical_fill_prefers_defaults():
    workflow = {
        "nodes": [
            {"id": "a1", "type": "action", "action_id": "emit", "params": {"mode": "fast"}},
            {"id": "a2", "type": "action", "action_id": "emit", "params": {}},
        ],
        "edges": [],
    }
    actions = [
        {
            "action_id": "emit",
            "arg_schema": {
                "type": "object",
                "properties": {"mode": {"type": "string", "default": "slow", "enum": ["fast", "slow"]}},
            },
            "output_schema": {"type": "object", "properties": {"out": {"type": "string"}}},
        }
    ]

    patched, suggestions = generate_repair_suggestions(workflow, actions)

    filled_node = next(n for n in patched["nodes"] if n.get("id") == "a2")
    mode_val = filled_node["params"]["mode"]
    if isinstance(mode_val, dict):
        assert mode_val.get("__from__") == "result_of.a1.out"
    else:
        assert mode_val in {"fast", "slow"}
    assert any(
        s.strategy in {"statistical_fill", "constraint_solver"} for s in suggestions
    )


def test_disconnected_nodes_receive_edge_suggestion():
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "lonely", "type": "action", "action_id": "emit", "params": {}},
        ],
        "edges": [],
    }
    actions = [
        {
            "action_id": "emit",
            "arg_schema": {},
            "output_schema": {"type": "object", "properties": {"out": {"type": "string"}}},
        }
    ]

    patched, suggestions = generate_repair_suggestions(workflow, actions)

    assert any(e.get("to") == "lonely" for e in patched.get("edges", []))
    assert any("连接未入度节点" in s.description for s in suggestions)
