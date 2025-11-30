import json
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

crawl4ai = types.ModuleType("crawl4ai")
crawl4ai.AsyncWebCrawler = None
crawl4ai.BrowserConfig = None
crawl4ai.CacheMode = None
crawl4ai.CrawlerRunConfig = None
crawl4ai.LLMConfig = None
sys.modules.setdefault("crawl4ai", crawl4ai)
sys.modules.setdefault(
    "crawl4ai.extraction_strategy",
    types.ModuleType("crawl4ai.extraction_strategy"),
)
sys.modules["crawl4ai.extraction_strategy"].LLMExtractionStrategy = None

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow
from velvetflow.verification.validation import validate_completed_workflow


ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


def _build_workflow_with_list_source():
    return {
        "workflow_name": "list_source_condition",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "summarize_nvidia_news",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {"message": "nvidia news"},
            },
            {
                "id": "summarize_google_news",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {"message": "google news"},
            },
            {
                "id": "check_all_summarized",
                "type": "condition",
                "params": {
                    "kind": "list_not_empty",
                    "source": [
                        "result_of.summarize_nvidia_news.summary",
                        "result_of.summarize_google_news.summary",
                    ],
                },
                "true_to_node": "notify_summary",
                "false_to_node": None,
            },
            {
                "id": "notify_summary",
                "type": "action",
                "action_id": "hr.notify_human.v1",
                "params": {"message": "send summaries"},
            },
        ],
        "edges": [
            {"from": "start", "to": "summarize_nvidia_news"},
            {"from": "summarize_nvidia_news", "to": "summarize_google_news"},
            {"from": "summarize_google_news", "to": "check_all_summarized"},
            {"from": "check_all_summarized", "to": "notify_summary", "condition": True},
        ],
    }


def test_condition_validation_accepts_list_sources():
    workflow = _build_workflow_with_list_source()

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert errors == []


def test_executor_resolves_multiple_condition_sources():
    workflow = Workflow.model_validate(_build_workflow_with_list_source())
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "hr.notify_human.v1": {
                "result": {"summary": "summary content", "status": "simulated"}
            }
        },
    )

    results = executor.run()

    assert results["check_all_summarized"]["condition_result"] is True
    assert results["check_all_summarized"].get("resolved_value") == [
        "summary content",
        "summary content",
    ]
    assert "notify_summary" in results
