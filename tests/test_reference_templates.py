import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.models import Node, Workflow, infer_edges_from_bindings

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
                "action_id": "hr.notify_human.v1",
                "params": {
                    "message": {"__from__": "${{result_of.start.status}}"},
                    "subject": "demo",
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
