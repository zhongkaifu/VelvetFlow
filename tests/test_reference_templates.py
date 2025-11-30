import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.models import Node, Workflow, infer_edges_from_bindings
from velvetflow.verification.validation import _schema_path_error

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def test_binding_context_supports_templated_references():
    workflow = Workflow.model_validate({"nodes": [{"id": "start", "type": "start"}], "edges": []})
    ctx = BindingContext(workflow, {"start": {"status": "ok"}})

    binding_value = ctx.resolve_binding({"__from__": "{{ result_of.start.status }}"})
    direct_value = ctx.get_value("${{result_of.start.status}}")

    node = Node(id="notify", type="action", params={"message": "${{result_of.start.status}}"})
    params = eval_node_params(node, ctx)

    assert binding_value == "ok"
    assert direct_value == "ok"
    assert params["message"] == "ok"


def test_validation_accepts_templated_result_reference():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "notify",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "email_content": {"__from__": "${{result_of.start.status}}"}
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [{"from": "start", "to": "notify"}, {"from": "notify", "to": "end"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_eval_params_parses_json_string_bindings():
    workflow = Workflow.model_validate({"nodes": [{"id": "start", "type": "start"}], "edges": []})
    ctx = BindingContext(
        workflow,
        {
            "start": {
                "items": [
                    {"title": "新闻一"},
                    {"title": "新闻二"},
                ]
            }
        },
    )

    binding_str = (
        '{"__from__":"result_of.start.items","__agg__":"format_join","format":"{title}","sep":"\\n"}'
    )
    node = Node(id="aggregate", type="action", params={"text": binding_str})

    params = eval_node_params(node, ctx)

    assert params["text"] == "新闻一\n新闻二"


def test_eval_params_resolves_nested_binding_in_dict():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {"id": "process_recipes", "type": "action", "params": {}},
                {"id": "notify", "type": "action", "params": {}},
            ],
            "edges": [],
        }
    )
    ctx = BindingContext(
        workflow,
        {
            "process_recipes": {
                "results": {
                    "monday": ["早餐", "午餐", "晚餐"],
                }
            }
        },
    )

    node = Node(
        id="notify",
        type="action",
        params={
            "context": {
                "weekly_menu": {"__from__": "result_of.process_recipes.results"}
            }
        },
    )

    params = eval_node_params(node, ctx)

    assert params["context"]["weekly_menu"] == {"monday": ["早餐", "午餐", "晚餐"]}


def test_format_join_prefers_named_fields_without_value_placeholder():
    workflow = Workflow.model_validate({"nodes": [{"id": "start", "type": "start"}], "edges": []})
    ctx = BindingContext(
        workflow,
        {
            "start": {
                "items": [
                    {"name": "Alice", "score": 95},
                    {"name": "Bob", "score": 88},
                ]
            }
        },
    )

    binding = {
        "__from__": "result_of.start.items",
        "__agg__": "format_join",
        "field": "name",
        "sep": ",",
    }
    node = Node(id="aggregate", type="action", params={"text": binding})

    params = eval_node_params(node, ctx)

    assert params["text"] == "Alice,Bob"


def test_join_binding_concatenates_strings():
    workflow = Workflow.model_validate(
        {"nodes": [{"id": "start", "type": "start"}, {"id": "aggregate", "type": "action"}], "edges": []}
    )

    ctx = BindingContext(workflow, {"start": {"items": ["a", "b", "c"]}})

    node = Node(
        id="aggregate",
        type="action",
        params={
            "text": {
                "__from__": "result_of.start.items",
                "__agg__": "join",
                "separator": "|",
            }
        },
    )

    params = eval_node_params(node, ctx)

    assert params["text"] == "a|b|c"


def test_infer_edges_from_embedded_result_refs():
    nodes = [
        Node(id="loop_nvidia_news", type="loop", params={}),
        Node(
            id="combine_summaries",
            type="action",
            params={
                "text": "结合Nvidia新闻总结：{{result_of.loop_nvidia_news.exports.items.summary}}，Google新闻总结：{{result_of.loop_google_news.exports.items.summary}}",
            },
        ),
        Node(id="loop_google_news", type="loop", params={}),
    ]

    edges = infer_edges_from_bindings(nodes)

    assert {"from": "loop_nvidia_news", "to": "combine_summaries", "condition": None} in edges
    assert {"from": "loop_google_news", "to": "combine_summaries", "condition": None} in edges


def test_eval_params_renders_interpolated_templates():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {"id": "loop_nvidia_news", "type": "loop"},
                {"id": "loop_google_news", "type": "loop"},
                {"id": "combine", "type": "action"},
            ],
            "edges": [],
        }
    )

    ctx = BindingContext(
        workflow,
        {
            "loop_nvidia_news": {"items": {"summary": "N1"}},
            "loop_google_news": {"items": {"summary": "G1"}},
        },
    )

    node = Node(
        id="combine",
        type="action",
        params={
            "text": "结合Nvidia新闻总结：{{result_of.loop_nvidia_news.exports.items.summary}}，Google新闻总结：{{result_of.loop_google_news.exports.items.summary}}",
            "subject": "demo",
        },
    )

    params = eval_node_params(node, ctx)

    assert params["text"] == "结合Nvidia新闻总结：N1，Google新闻总结：G1"
    assert params["subject"] == "demo"


def test_get_value_supports_list_index_fields():
    workflow = Workflow.model_validate(
        {"nodes": [{"id": "search_recipes", "type": "action"}], "edges": []}
    )
    ctx = BindingContext(
        workflow,
        {"search_recipes": {"results": [{"snippet": "first"}, {"snippet": "second"}]}},
    )

    assert ctx.get_value("result_of.search_recipes.results[0].snippet") == "first"


def test_loop_context_allows_fully_qualified_paths():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "id": "loop_check_each_employee",
                    "type": "loop",
                    "params": {"item_alias": "current_item"},
                }
            ],
            "edges": [],
        }
    )

    loop_ctx = {"current_item": {"temperature": 39.2}, "index": 0, "size": 1}
    ctx = BindingContext(
        workflow,
        {},
        loop_ctx=loop_ctx,
        loop_id="loop_check_each_employee",
    )

    node = Node(
        id="check",
        type="condition",
        params={
            "value": "loop_check_each_employee.current_item.temperature",
            "message": "体温：{{loop_check_each_employee.current_item.temperature}}",
        },
    )

    params = eval_node_params(node, ctx)

    assert params["value"] == 39.2
    assert params["message"] == "体温：39.2"


def test_schema_validation_supports_index_paths():
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "object", "properties": {"snippet": {"type": "string"}}},
            }
        },
    }

    assert _schema_path_error(schema, ["results", 0, "snippet"]) is None
