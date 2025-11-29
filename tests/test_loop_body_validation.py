import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.models import PydanticValidationError, Workflow
from velvetflow.verification import precheck_loop_body_graphs

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def test_loop_body_exports_must_target_existing_nodes():
    """Exports referencing missing body nodes should surface a clear error."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "loop_summarize",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "placeholder"},
                            }
                        ],
                        "edges": [{"from": "summarize", "to": "exit"}],
                        "entry": "summarize",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "exit",
                            "fields": ["summary"],
                            "mode": "collect",
                        }
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_summarize"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation errors for invalid loop exports"
    assert any(e.field == "exports.items.from_node" for e in errors)


def test_exports_disallowed_outside_loop_body():
    """Non-loop nodes carrying exports should be rejected early."""

    workflow = {
        "workflow_name": "invalid_exports_location",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {
                    "query": "AI",
                    "exports": {"items": {"from_node": "search_news", "fields": ["results"]}},
                },
            }
        ],
        "edges": [],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation errors for misplaced exports"
    assert any(err.field == "exports" and err.code == "INVALID_SCHEMA" for err in errors)


def test_loop_body_nodes_cannot_be_empty():
    """Loop body graphs must contain at least one node during model validation."""

    workflow = {
        "workflow_name": "empty_loop_body",
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "loop_empty", "type": "loop", "params": {"body_subgraph": {"nodes": []}}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "loop_empty"},
            {"from": "loop_empty", "to": "end"},
        ],
    }

    with pytest.raises(PydanticValidationError) as excinfo:
        Workflow.model_validate(workflow)

    assert any(
        err.get("loc") == ("body_subgraph", "nodes")
        for err in excinfo.value.errors()
    )


def test_loop_body_allows_edge_free_body():
    """Loop body graphs can omit explicit edges when bindings define flow."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "loop_summarize",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "placeholder"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [],
                        "entry": "summarize",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "summarize",
                            "fields": ["summary"],
                            "mode": "collect",
                        }
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_summarize"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_loop_body_action_missing_required_param_is_caught():
    """Loop body action nodes should honor required params from Action Registry."""

    workflow = {
        "workflow_name": "Nvidia和Google财经新闻搜索总结通知",
        "nodes": [
            {
                "id": "search_news_nvidia_google",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "Nvidia AND Google", "limit": 5, "timeout": 8},
            },
            {
                "id": "loop_news",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news_nvidia_google.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize_news",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "summarize_news",
                            "fields": ["summary", "sentence_count"],
                            "mode": "collect",
                        }
                    },
                },
            },
            {
                "id": "notify_user",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {
                    "message": {
                        "__from__": "result_of.loop_news.items",
                        "__agg__": "format_join",
                        "format": "{summary}",
                        "sep": "\n\n",
                    }
                },
            },
        ],
        "edges": [
            {"from": "search_news_nvidia_google", "to": "loop_news"},
            {"from": "loop_news", "to": "notify_user"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "MISSING_REQUIRED_PARAM" and e.node_id == "summarize_news" and e.field == "text"
        for e in errors
    )


def test_loop_body_template_respects_item_schema():
    """模板引用 loop item 时应使用 source 的输出 schema 校验字段。"""

    workflow = {
        "workflow_name": "Nvidia 新闻总结",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "Nvidia"},
            },
            {
                "id": "loop_news",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize_news",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "{{news_item.content}}"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "summarize_news",
                            "fields": ["summary", "sentence_count"],
                            "mode": "collect",
                        }
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_news"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "SCHEMA_MISMATCH" and e.node_id == "summarize_news" and e.field == "text"
        for e in errors
    )


def test_stringified_binding_referencing_loop_body_is_flagged():
    """字符串化的 __from__ 绑定也应在 planner 校验阶段暴露错误。"""

    workflow = {
        "workflow_name": "Nvidia和Google财经新闻搜索总结通知",
        "nodes": [
            {
                "id": "search_news_nvidia_google",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "Nvidia AND Google"},
            },
            {
                "id": "loop_each_news",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news_nvidia_google.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize_news",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "placeholder"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {
                            "from_node": "summarize_news",
                            "fields": ["summary"],
                            "mode": "collect",
                        }
                    },
                },
            },
            {
                "id": "aggregate_summaries",
                "type": "action",
                "action_id": "common.summarize.v1",
                "params": {
                    "text": "{\"__from__\":\"result_of.loop_each_news.summarize_news.summary\",\"__agg__\":\"format_join\",\"sep\":\"\\n\"}",
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "search_news_nvidia_google", "to": "loop_each_news"},
            {"from": "loop_each_news", "to": "aggregate_summaries"},
            {"from": "aggregate_summaries", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "SCHEMA_MISMATCH" and e.node_id == "aggregate_summaries" and e.field == "text"
        for e in errors
    )


def test_precheck_is_available_for_planner_users():
    """Planner 层的 precheck 导出应可独立调用。"""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "loop_summarize",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "placeholder"},
                            }
                        ],
                        "entry": "summarize",
                        "exit": "missing",
                    },
                },
            }
        ],
    }

    errors = precheck_loop_body_graphs(workflow)

    assert errors == []


def test_loop_body_pydantic_errors_are_preserved():
    """Pydantic 校验错误需要原样向上传递，避免被包装成 ValueError。"""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "loop_summarize",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    # body_subgraph 缺少节点 id，会触发 Pydantic 校验错误
                    "body_subgraph": {"nodes": [{"type": "action"}], "edges": []},
                },
            }
        ],
        "edges": [],
    }

    with pytest.raises(PydanticValidationError) as exc_info:
        Workflow.model_validate(workflow)

    # 错误位置信息应该保留，并标注在 body_subgraph 下，方便修复逻辑使用
    error = exc_info.value.errors()[0]
    assert error.get("loc", ())[:3] == ("body_subgraph", "nodes", 0)


def test_loop_missing_item_alias_is_reported():
    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "loop_without_alias",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "exports": {
                        "items": {
                            "from_node": "exit",
                            "fields": ["summary"],
                        }
                    },
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "placeholder"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize", "to": "exit"}],
                        "entry": "summarize",
                        "exit": "exit",
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_without_alias"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "MISSING_REQUIRED_PARAM" and e.node_id == "loop_without_alias" and e.field == "item_alias"
        for e in errors
    )


def test_while_loop_requires_condition():
    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "loop_until_empty",
                "type": "loop",
                "params": {
                    "loop_kind": "while",
                    "source": "result_of.search_news.results",
                    "item_alias": "news_item",
                    "exports": {
                        "items": {
                            "from_node": "exit",
                            "fields": ["summary"],
                        }
                    },
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "placeholder"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize", "to": "exit"}],
                        "entry": "summarize",
                        "exit": "exit",
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_until_empty"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "MISSING_REQUIRED_PARAM" and e.node_id == "loop_until_empty" and e.field == "condition"
        for e in errors
    )

