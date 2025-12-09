# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

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

MIN_LOOP_BODY = {"nodes": [{"id": "noop", "type": "action"}], "edges": []}

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "tools" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def test_binding_context_supports_templated_references():
    workflow = Workflow.model_validate({"nodes": [{"id": "start", "type": "start"}], "edges": []})
    ctx = BindingContext(workflow, {"start": {"status": "ok"}})

    binding_value = ctx.resolve_binding({"__from__": "{{ result_of.start.status }}"})
    direct_value = ctx.get_value("${{result_of.start.status}}")
    direct_value_dollar = ctx.get_value("${ result_of.start.status }")

    node = Node(id="notify", type="action", params={"message": "${{result_of.start.status}}"})
    node_with_dollar = Node(id="notify2", type="action", params={"message": "${result_of.start.status}"})
    params = eval_node_params(node, ctx)
    params_with_dollar = eval_node_params(node_with_dollar, ctx)

    assert binding_value == "ok"
    assert direct_value == "ok"
    assert direct_value_dollar == "ok"
    assert params["message"] == "ok"
    assert params_with_dollar["message"] == "ok"


def test_binding_context_resolves_length_field():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "id": "loop_check_temperature",
                    "type": "action",
                    "action_id": "hr.check_temperature",
                    "params": {},
                }
            ],
            "edges": [],
        }
    )
    ctx = BindingContext(
        workflow,
        {"loop_check_temperature": {"exports": {"items": [{"v": 1}, {"v": 2}, {"v": 3}]}}},
    )

    assert (
        ctx.get_value("result_of.loop_check_temperature.exports.items.length")
        == 3
    )

    node = Node(
        id="generate_warning_report",
        type="action",
        action_id="hr.record_health_event.v1",
        params={
            "abnormal_count": "${result_of.loop_check_temperature.exports.items.length}",
        },
    )

    params = eval_node_params(node, ctx)

    assert params["abnormal_count"] == 3


def test_binding_context_resolves_count_field():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "id": "loop_check_temperature",
                    "type": "action",
                    "action_id": "hr.check_temperature",
                    "params": {},
                }
            ],
            "edges": [],
        }
    )
    ctx = BindingContext(
        workflow,
        {"loop_check_temperature": {"aggregates": {"high": 2, "normal": 5}}},
    )

    assert ctx.get_value("result_of.loop_check_temperature.aggregates.count") == 2

    node = Node(
        id="generate_warning_report",
        type="action",
        action_id="hr.record_health_event.v1",
        params={
            "abnormal_count": "${result_of.loop_check_temperature.aggregates.count}",
        },
    )

    params = eval_node_params(node, ctx)

    assert params["abnormal_count"] == 2


def test_eval_params_canonicalizes_unresolved_placeholders():
    workflow = Workflow.model_validate({"nodes": [{"id": "start", "type": "start"}], "edges": []})
    ctx = BindingContext(workflow, {"start": {"status": "ok"}})

    node = Node(
        id="notify",
        type="action",
        params={"message": "等待补全：${{missing.path}}"},
    )

    params = eval_node_params(node, ctx)

    assert params["message"] == "等待补全：{{missing.path}}"


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

    assert len(errors) == 1
    assert errors[0].code == "SCHEMA_MISMATCH"
    assert errors[0].node_id == "notify"
    assert errors[0].field == "email_content.__from__"
    assert "start" in errors[0].message
    assert "status" in errors[0].message


def test_validation_rejects_invalid_template_reference():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
                {
                    "id": "notify",
                    "type": "action",
                    "action_id": "hr.update_employee_health_profile.v1",
                    "params": {
                        "employee_id": "E001",
                        "status": "无法找到节点：{{result_of.missing.status}}",
                    },
                },
            {"id": "end", "type": "end"},
        ],
        "edges": [{"from": "start", "to": "notify"}, {"from": "notify", "to": "end"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert len(errors) == 1
    assert errors[0].node_id == "notify"
    assert errors[0].field == "status"
    assert "模板" in errors[0].message
    assert "missing" in errors[0].message


def test_validation_rejects_invalid_dollar_template_reference():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
                {
                    "id": "notify",
                    "type": "action",
                    "action_id": "hr.update_employee_health_profile.v1",
                    "params": {
                        "employee_id": "E001",
                        "status": "无法找到节点：${result_of.missing.status}",
                    },
                },
            {"id": "end", "type": "end"},
        ],
        "edges": [{"from": "start", "to": "notify"}, {"from": "notify", "to": "end"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert len(errors) == 1
    assert errors[0].node_id == "notify"
    assert errors[0].field == "status"
    assert "模板" in errors[0].message
    assert "missing" in errors[0].message


def test_validation_rejects_invalid_double_dollar_template_reference():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
                {
                    "id": "notify",
                    "type": "action",
                    "action_id": "hr.update_employee_health_profile.v1",
                    "params": {
                        "employee_id": "E001",
                        "status": "无法找到节点：${{result_of.missing.status}}",
                    },
                },
            {"id": "end", "type": "end"},
        ],
        "edges": [{"from": "start", "to": "notify"}, {"from": "notify", "to": "end"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert len(errors) == 1
    assert errors[0].node_id == "notify"
    assert errors[0].field == "status"
    assert "模板" in errors[0].message
    assert "missing" in errors[0].message


def test_validation_rejects_template_reference_with_unknown_field():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "search",
                "type": "action",
                "action_id": "common.search_web.v1",
                "params": {"query": "news"},
            },
            {
                "id": "notify",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "email_content": "Invalid reference: {{result_of.search.results.missing_field}}"
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "search"},
            {"from": "search", "to": "notify"},
            {"from": "notify", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert len(errors) == 1
    assert errors[0].node_id == "notify"
    assert errors[0].field == "email_content"
    assert "模板" in errors[0].message
    assert "missing_field" in errors[0].message


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


def test_eval_params_parses_single_quoted_binding_strings():
    workflow = Workflow.model_validate({"nodes": [{"id": "search_google_news", "type": "action"}], "edges": []})
    ctx = BindingContext(
        workflow,
        {
            "search_google_news": {
                "results": [
                    {"snippet": "First"},
                    {"snippet": "Second"},
                ]
            }
        },
    )

    binding_str = (
        "{'__from__': 'result_of.search_google_news.results', '__agg__': 'format_join', 'format': '{snippet}'}"
    )
    node = Node(id="aggregate", type="action", params={"text": binding_str})

    params = eval_node_params(node, ctx)

    assert params["text"] == "First\nSecond"


def test_template_wildcard_references_extract_list_fields():
    workflow = Workflow.model_validate({"nodes": [{"id": "start", "type": "start"}], "edges": []})
    ctx = BindingContext(
        workflow,
        {
            "start": {
                "results": [
                    {"snippet": "Alpha"},
                    {"snippet": "Beta"},
                ]
            }
        },
    )

    direct_value = ctx.get_value("result_of.start.results[*].snippet")
    node = Node(
        id="notify",
        type="action",
        params={"message": "Snippets: ${result_of.start.results[*].snippet}"},
    )

    params = eval_node_params(node, ctx)

    assert direct_value == ["Alpha", "Beta"]
    assert params["message"] == "Snippets: ['Alpha', 'Beta']"


def test_template_wildcard_without_result_of_prefix():
    workflow = Workflow.model_validate({"nodes": [{"id": "search_nvidia_news", "type": "action"}], "edges": []})
    ctx = BindingContext(
        workflow,
        {
            "search_nvidia_news": {
                "results": [
                    {"snippet": "NVDA hits record high"},
                    {"snippet": "New GPU architecture"},
                ]
            }
        },
    )

    node = Node(
        id="notify",
        type="action",
        params={"text": "${search_nvidia_news.results[*].snippet}"},
    )

    params = eval_node_params(node, ctx)

    assert params["text"] == ["NVDA hits record high", "New GPU architecture"]


def test_reference_allows_builtin_function_invocation():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "id": "loop_nvidia_news",
                    "type": "loop",
                    "params": {"body_subgraph": MIN_LOOP_BODY},
                }
            ],
            "edges": [],
        }
    )
    ctx = BindingContext(
        workflow,
        {
            "loop_nvidia_news": {"items": [{"summary": "N1"}, {"summary": "N2"}]}
        },
    )

    joined = ctx.get_value("result_of.loop_nvidia_news.exports.items[*].summary.join('\\n')")
    node = Node(
        id="notify",
        type="action",
        params={
            "email_content": "Nvidia新闻总结：\n${loop_nvidia_news.exports.items[*].summary.join('\\n')}",
        },
    )

    params = eval_node_params(node, ctx)

    assert joined == "N1\nN2"
    assert params["email_content"] == "Nvidia新闻总结：\nN1\nN2"


def test_eval_params_renders_embedded_json_binding_with_escapes():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {"id": "start", "type": "start"},
                {"id": "analyze_stock_performance", "type": "action"},
                {"id": "summarize_stock_info", "type": "action"},
            ],
            "edges": [],
        }
    )
    ctx = BindingContext(
        workflow,
        {
            "analyze_stock_performance": {
                "aggregate_summary": "市场总结 A +10% / B -5%",
            },
            "summarize_stock_info": {
                "results": [
                    {"title": "AAPL", "snippet": "上涨 3%"},
                    {"title": "TSLA", "snippet": "下跌 2%"},
                ]
            },
        },
    )

    binding_str = (
        "根据最新财经网站数据，以下是10只涨得最好和10只跌得最多的股票及其投资建议总结：\n\n"
        "涨幅最高的10只股票：{\"__from__\":\"result_of.analyze_stock_performance.aggregate_summary\",\"__agg__\":\"identity\"}\n"
        "跌幅最高的10只股票：{\"__from__\":\"result_of.analyze_stock_performance.aggregate_summary\",\"__agg__\":\"identity\"}\n"
        "投资建议：{\"__from__\":\"result_of.analyze_stock_performance.aggregate_summary\",\"__agg__\":\"identity\"}\n\n"
        "总结内容：{\"__from__\":\"result_of.summarize_stock_info.results\",\"__agg__\":\"format_join\",\"format\":\"{title}: {snippet}\",\"sep\":\"\\n\"}"
    )
    node = Node(id="aggregate", type="action", params={"text": binding_str})

    params = eval_node_params(node, ctx)

    assert (
        params["text"]
        == "根据最新财经网站数据，以下是10只涨得最好和10只跌得最多的股票及其投资建议总结：\n\n"
        "涨幅最高的10只股票：市场总结 A +10% / B -5%\n"
        "跌幅最高的10只股票：市场总结 A +10% / B -5%\n"
        "投资建议：市场总结 A +10% / B -5%\n\n"
        "总结内容：AAPL: 上涨 3%\nTSLA: 下跌 2%"
    )


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
                {
                    "id": "loop_nvidia_news",
                    "type": "loop",
                    "params": {"body_subgraph": MIN_LOOP_BODY},
                },
                {
                    "id": "loop_google_news",
                    "type": "loop",
                    "params": {"body_subgraph": MIN_LOOP_BODY},
                },
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
                    "params": {"item_alias": "current_item", "body_subgraph": MIN_LOOP_BODY},
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


def test_eval_params_renders_each_templates():
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {"id": "start", "type": "start"},
                {
                    "id": "loop_top10_products",
                    "type": "action",
                    "action_id": "demo.top_products",
                    "params": {},
                },
            ],
            "edges": [],
        }
    )

    ctx = BindingContext(
        workflow,
        {
            "loop_top10_products": {
                "exports": {
                    "items": [
                        {"title": "Laptop", "snippet": "Powerful"},
                        {"title": "Phone", "snippet": "Portable"},
                    ]
                }
            }
        },
    )

    node = Node(
        id="email",
        type="action",
        action_id="productivity.compose_outlook_email.v1",
        params={
            "email_content": (
                "以下是目前最热销的10款电子产品及其介绍：\n"
                "{{#each result_of.loop_top10_products.exports.items}}\n"
                "产品名称：{{this.title}}\n介绍：{{this.snippet}}\n\n{{/each}}"
            )
        },
    )

    params = eval_node_params(node, ctx)

    assert (
        params["email_content"]
        == "以下是目前最热销的10款电子产品及其介绍：\n\n产品名称：Laptop\n介绍：Powerful\n\n"
        "\n产品名称：Phone\n介绍：Portable\n\n"
    )


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
