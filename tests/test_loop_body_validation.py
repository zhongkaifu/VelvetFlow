# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.models import PydanticValidationError, Workflow
from velvetflow.loop_dsl import index_loop_body_nodes
from velvetflow.verification import precheck_loop_body_graphs

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "tools" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def test_loop_requires_body_subgraph():
    """Missing body_subgraph should be surfaced during workflow validation."""

    workflow = {
        "workflow_name": "missing_body_subgraph",
        "nodes": [
            {
                "id": "fetch_items",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "loop_over_items",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.fetch_items.results",
                    "item_alias": "item",
                },
            },
        ],
        "edges": [{"from": "fetch_items", "to": "loop_over_items"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation errors for missing body_subgraph"
    assert any(
        err.code == "INVALID_LOOP_BODY" and err.field == "body_subgraph" for err in errors
    )


def test_model_validation_rejects_missing_body_subgraph():
    """Workflow.model_validate should fail fast when loop body is absent."""

    workflow = {
        "workflow_name": "missing_body_subgraph_model",
        "nodes": [
            {
                "id": "loop_warning_summary",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.loop_check_temperature.exports.items",
                    "item_alias": "warning_employee",
                },
            }
        ],
    }

    with pytest.raises(PydanticValidationError):
        Workflow.model_validate(workflow)


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
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "email_content": {
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


def test_loop_exports_should_not_live_in_body_subgraph():
    """loop exports 应放在 params.exports，出现在 body_subgraph 时应报错。"""

    workflow = {
        "workflow_name": "重复 exports 示例",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "Nvidia"},
            },
            {
                "id": "loop_check_temperatures",
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
                                "params": {"text": "fixed"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                        "exports": {
                            "items": {"from_node": "summarize_news", "fields": ["summary"]},
                        },
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_check_temperatures"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "INVALID_SCHEMA"
        and e.node_id == "loop_check_temperatures"
        and e.field == "body_subgraph.exports"
        for e in errors
    )


def test_loop_body_nodes_should_not_reference_loop_exports():
    """Body nodes must rely on body_subgraph outputs instead of loop exports."""

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
                "id": "loop_summarize_news",
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
                                "params": {
                                    "text": {
                                        "__from__": "result_of.loop_summarize_news.exports.items"
                                    }
                                },
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {"from_node": "summarize_news", "fields": ["summary"]},
                    },
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_summarize_news"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "SCHEMA_MISMATCH"
        and e.node_id == "summarize_news"
        and "不可直接引用所属 loop" in e.message
        for e in errors
    )


def test_condition_cannot_reference_missing_loop_export_field():
    """条件节点引用 loop exports 下不存在的字段时应报错。"""

    workflow = {
        "workflow_name": "体温检测",
        "nodes": [
            {
                "id": "fetch_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "health"},
            },
            {
                "id": "loop_check_temperatures",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.fetch_news.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize_news",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "stable"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {
                        "items": {"from_node": "summarize_news", "fields": ["summary"]},
                    },
                },
            },
            {
                "id": "condition_any_warning",
                "type": "condition",
                "true_to_node": None,
                "false_to_node": None,
                "params": {
                    "kind": "any_greater_than",
                    "source": "result_of.loop_check_temperatures.exports",
                    "field": "status",
                    "threshold": 1,
                },
            },
        ],
        "edges": [
            {"from": "fetch_news", "to": "loop_check_temperatures"},
            {"from": "loop_check_temperatures", "to": "condition_any_warning"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        e.code == "SCHEMA_MISMATCH"
        and e.node_id == "condition_any_warning"
        and e.field == "field"
        for e in errors
    )


def test_condition_field_on_loop_item_alias_should_be_allowed():
    """Loop body aliases represent single objects and should allow field access."""

    workflow = {
        "workflow_name": "员工体温健康预警工作流",
        "nodes": [
            {
                "id": "get_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "today"},
            },
            {
                "id": "loop_check_each_employee",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.get_temperatures.data",
                    "item_alias": "employee_temp",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "condition_temp_high",
                                "type": "condition",
                                "display_name": "体温是否超过38度",
                                "params": {
                                    "kind": "any_greater_than",
                                    "source": "employee_temp",
                                    "field": "temperature",
                                    "threshold": 38,
                                },
                                "true_to_node": "add_to_warning_list",
                                "false_to_node": None,
                            },
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "display_name": "加入健康预警列表",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "employee_temp.employee_id",
                                    "last_temperature": "employee_temp.temperature",
                                    "status": "预警",
                                },
                            },
                        ],
                        "entry": "condition_temp_high",
                        "exit": "add_to_warning_list",
                    },
                },
            },
        ],
        "edges": [{"from": "get_temperatures", "to": "loop_check_each_employee"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert not any(
        e.code == "SCHEMA_MISMATCH"
        and e.node_id == "condition_temp_high"
        and e.field == "field"
        for e in errors
    )


def test_index_loop_body_nodes_includes_nested_loops():
    workflow = {
        "workflow_name": "nested_loop_mapping",
        "nodes": [
            {
                "id": "outer_loop",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": [],
                    "item_alias": "item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "inner_loop",
                                "type": "loop",
                                "params": {
                                    "loop_kind": "for_each",
                                    "source": [],
                                    "item_alias": "sub_item",
                                    "body_subgraph": {
                                        "nodes": [
                                            {
                                                "id": "inner_action",
                                                "type": "action",
                                                "action_id": "demo.action",
                                                "params": {},
                                            }
                                        ]
                                    },
                                },
                            }
                        ]
                    },
                },
            }
        ],
    }

    mapping = index_loop_body_nodes(workflow)

    assert mapping["inner_loop"] == "outer_loop"
    assert mapping["inner_action"] == "inner_loop"

