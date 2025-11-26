import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.verification import precheck_loop_body_graphs

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def test_loop_body_missing_exit_node_is_reported_before_pydantic():
    """Ensure invalid loop body edges/exit are caught with a readable error."""

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

    assert errors, "Expected validation errors for missing loop body nodes"
    assert all(e.code == "INVALID_LOOP_BODY" for e in errors)
    assert any("exit" in e.message for e in errors)


def test_loop_body_unknown_node_type_is_repairable():
    """Unknown node types inside loop bodies should surface as repairable errors."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {"id": "start", "type": "start", "params": {}},
            {
                "id": "loop_summarize",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.start",
                    "item_alias": "item",
                    "body_subgraph": {
                        "nodes": [
                            {"id": "summarize_news", "type": "action", "params": {}},
                            {"id": "exit", "type": "exit", "params": {}},
                        ],
                        "edges": [
                            {"from": "summarize_news", "to": "exit", "condition": None},
                        ],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                },
            },
        ],
        "edges": [{"from": "start", "to": "loop_summarize", "condition": None}],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(e.code == "INVALID_LOOP_BODY" for e in errors)
    # Should not crash during Workflow.model_validate; instead produce a repairable error list
    assert any("非法节点类型" in e.message for e in errors)


def test_loop_body_missing_nodes_and_edges_is_reported():
    """Loops with exports must provide a non-empty body_subgraph."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "loop_nvidia_news",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {
                        "__from__": "result_of.search_news_nvidia.results",
                        "__agg__": "identity",
                    },
                    "item_alias": "news_item",
                    "exports": {
                        "items": {
                            "from_node": "summarize_nvidia_news",
                            "fields": ["summary", "sentence_count"],
                            "mode": "collect",
                        }
                    },
                },
            }
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(e.code == "INVALID_LOOP_BODY" and e.field == "body_subgraph.nodes" for e in errors)
    assert any(e.code == "INVALID_LOOP_BODY" and e.field == "body_subgraph.edges" for e in errors)
    assert any(e.field == "exports.items.from_node" for e in errors)


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

    assert errors
    assert any(e.field == "body_subgraph.exit" for e in errors)


def test_precheck_reports_missing_body_targets():
    """Edges in loop body that point to missing nodes should be surfaced early."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "loop_summarize_each_news",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "item",
                    "body_subgraph": {
                        "nodes": [
                            {"id": "summarize_news", "type": "action", "action_id": "common.summarize.v1"},
                        ],
                        "edges": [
                            {"from": "summarize_news", "to": "exit", "condition": None},
                        ],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {"items": {"from_node": "summarize_news", "fields": ["summary"]}},
                },
            }
        ],
    }

    errors = precheck_loop_body_graphs(workflow)

    assert errors
    assert any(
        e.code == "INVALID_LOOP_BODY" and e.field == "body_subgraph.edges[0].to" for e in errors
    )


def test_precheck_rejects_illegal_body_node_types():
    """Invalid node types inside a loop body should not reach deep pydantic validation."""

    workflow = {
        "workflow_name": "news_summary",
        "nodes": [
            {
                "id": "loop_summarize_each_news",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "result_of.search_news.results",
                    "item_alias": "item",
                    "body_subgraph": {
                        "nodes": [
                            {"id": "summarize_news", "type": "action", "action_id": "common.summarize.v1"},
                            {"id": "exit", "type": "exit"},
                        ],
                        "edges": [
                            {"from": "summarize_news", "to": "exit", "condition": None},
                        ],
                        "entry": "summarize_news",
                        "exit": "exit",
                    },
                    "exports": {"items": {"from_node": "summarize_news", "fields": ["summary"]}},
                },
            }
        ],
    }

    errors = precheck_loop_body_graphs(workflow)

    assert errors
    assert any(
        e.code == "INVALID_LOOP_BODY"
        and e.field == "body_subgraph.nodes[1].type"
        and "非法节点类型" in e.message
        for e in errors
    )

