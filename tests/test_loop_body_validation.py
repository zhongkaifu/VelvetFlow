# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.loop_dsl import index_loop_body_nodes
from velvetflow.models import PydanticValidationError, Workflow
from velvetflow.verification import precheck_loop_body_graphs

ACTION_REGISTRY = BUSINESS_ACTIONS


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
                    "source": "result_of.loop_check_temperature.exports.employee_ids",
                    "item_alias": "warning_employee",
                },
            }
        ],
    }

    with pytest.raises(PydanticValidationError):
        Workflow.model_validate(workflow)

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
                    "exports": {"items": "{{ result_of.search_news.results }}"},
                },
            }
        ],
        "edges": [],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation errors for misplaced exports"
    assert any(err.field == "exports" and err.code == "INVALID_SCHEMA" for err in errors)


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


def test_loop_body_requires_action_node_for_planning():
    """Precheck should surface loop bodies that lack actionable steps."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "loop_summarize",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {"id": "start", "type": "start"},
                            {
                                "id": "guard_branch",
                                "type": "condition",
                                "params": {"expression": "{{ (loop.item or []) | length > 0 }}"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "entry": "guard_branch",
                        "exit": "exit",
                    },
                },
            }
        ],
    }

    errors = precheck_loop_body_graphs(workflow)

    assert any(
        err.code == "INVALID_LOOP_BODY" and err.field == "body_subgraph.nodes" for err in errors
    )


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
                                    "expression": "{{ employee_temp.temperature > 38 }}",
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


def test_loop_exports_allow_body_node_reference_without_field():
    """Loop exports may collect entire body nodes, not just their output fields."""

    workflow = {
        "workflow_name": "loop_body_node_exports",
        "nodes": [
            {
                "id": "search_news",
                "type": "action",
                "action_id": "common.search_news.v1",
                "params": {"query": "AI"},
            },
            {
                "id": "loop_collect_summaries",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize_news",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "{{ item.title }}"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "summarize_news", "to": "exit"}],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {"summaries": "{{ result_of.summarize_news }}"},
                },
            },
        ],
        "edges": [{"from": "search_news", "to": "loop_collect_summaries"}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_loop_source_allows_constant_template_list():
    """Loops may iterate over literal sequences provided via templated constants."""

    workflow = {
        "workflow_name": "loop_over_constant_contacts",
        "nodes": [
            {
                "id": "notify_contacts",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ [{'contact': 'user1@example.com', 'preference': 'dingding'}] }}",
                    "item_alias": "recipient",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "send_digest",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "params": {"text": "{{ recipient.contact }}"},
                            },
                            {"id": "exit", "type": "end"},
                        ],
                        "edges": [{"from": "send_digest", "to": "exit"}],
                        "entry": "send_digest",
                        "exit": "exit",
                    },
                },
            }
        ],
        "edges": [],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


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
