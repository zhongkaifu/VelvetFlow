import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.planner.validation import precheck_loop_body_graphs

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

