# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.models import Workflow


def _build_ctx(results):
    workflow = Workflow.model_validate(
        {
            "workflow_name": "binding-errors",
            "nodes": [
                {"id": "a", "type": "action", "params": {}},
            ],
            "edges": [],
        }
    )
    return BindingContext(workflow, results)


def test_missing_result_path_raises_descriptive_error():
    ctx = _build_ctx(results={})

    with pytest.raises(KeyError, match=r"result_of\.a: 上游节点未执行或没有结果"):
        ctx.resolve_binding({"__from__": "result_of.a.output"})


def test_missing_field_on_dict_gives_precise_path():
    ctx = _build_ctx(results={"a": {"output": {"value": 1}}})

    with pytest.raises(KeyError, match=r"result_of\.a\.output\.missing: 字段不存在"):
        ctx.resolve_binding({"__from__": "result_of.a.output.missing"})


def test_type_mismatch_reports_actual_type():
    ctx = _build_ctx(results={"a": {"output": 3}})

    with pytest.raises(TypeError, match=r"值类型为 int，不支持字段 'id' 访问"):
        ctx.resolve_binding({"__from__": "result_of.a.output.id"})


def test_missing_action_output_field_falls_back_to_schema():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "news-email",
            "nodes": [
                {
                    "id": "analyze_top_movers",
                    "type": "action",
                    "action_id": "common.search_news.v1",
                    "out_params_schema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                    },
                                    "required": ["title", "url", "snippet"],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                },
                {
                    "id": "send_email",
                    "type": "action",
                    "action_id": "productivity.compose_outlook_email.v1",
                    "params": {
                        "email_content": "{'__from__': 'result_of.analyze_top_movers.results', '__agg__': 'format_join', 'format': 'Title: {title}\\nSummary: {snippet}\\nURL: {url}'}",
                        "emailTo": "user@example.com",
                    },
                },
            ],
            "edges": [],
        }
    )

    ctx = BindingContext(
        workflow,
        {
            "analyze_top_movers": {
                "status": "tool_error",
                "tool_name": "search_news",
                "action_id": "common.search_news.v1",
                "error": "network failure",
                "params": {"query": "NVDA"},
            }
        },
    )

    email_node = next(n for n in workflow.nodes if n.id == "send_email")
    resolved = eval_node_params(email_node, ctx)

    assert resolved["email_content"] == ""


def test_format_join_missing_fields_render_empty_strings():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "news-email",
            "nodes": [
                {
                    "id": "analyze_top_movers",
                    "type": "action",
                    "action_id": "common.search_news.v1",
                    "out_params_schema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"},
                                    },
                                },
                            }
                        },
                    },
                },
                {
                    "id": "send_email",
                    "type": "action",
                    "action_id": "productivity.compose_outlook_email.v1",
                    "params": {
                        "email_content": "{'__from__': 'result_of.analyze_top_movers.results', '__agg__': 'format_join', 'format': 'Title: {title}\\nSummary: {snippet}\\nURL: {url}'}",
                        "emailTo": "user@example.com",
                    },
                },
            ],
            "edges": [],
        }
    )

    ctx = BindingContext(
        workflow,
        {
            "analyze_top_movers": {
                "results": [
                    {"title": "Stock A", "snippet": "Up 5%"},
                    {"url": "https://example.com/b"},
                ]
            }
        },
    )

    email_node = next(n for n in workflow.nodes if n.id == "send_email")
    resolved = eval_node_params(email_node, ctx)

    assert (
        resolved["email_content"]
        == "Title: Stock A\nSummary: Up 5%\nURL: \nTitle: \nSummary: \nURL: https://example.com/b"
    )
