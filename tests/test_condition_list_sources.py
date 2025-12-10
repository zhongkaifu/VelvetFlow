# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

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

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.bindings import BindingContext
from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow
from velvetflow.verification.validation import validate_completed_workflow


ACTION_REGISTRY = BUSINESS_ACTIONS


def _build_workflow_with_list_source():
    return {
        "workflow_name": "list_source_condition",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "summarize_nvidia_news",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "nvidia news"},
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "status": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["summary", "status"],
                },
            },
            {
                "id": "summarize_google_news",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "google news"},
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "status": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["summary", "status"],
                },
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
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "send summaries"},
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "status": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["summary", "status"],
                },
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
            "productivity.compose_outlook_email.v1": {
                "result": {"summary": ["summary content"], "status": "simulated"}
            }
        },
    )

    results = executor.run()

    assert results["check_all_summarized"]["condition_result"] is True
    assert results["check_all_summarized"].get("resolved_value") == [
        ["summary content"],
        ["summary content"],
    ]
    assert "notify_summary" in results


def test_condition_source_allows_plain_node_id():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "plain_source_id",
            "description": "",
            "nodes": [
                {"id": "start", "type": "start"},
                {
                    "id": "summarize_nvidia_news",
                    "type": "action",
                    "action_id": "productivity.compose_outlook_email.v1",
                    "params": {"email_content": "nvidia news"},
                    "out_params_schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["status"],
                    },
                },
                {
                    "id": "cond_after_summarize_nvidia",
                    "type": "condition",
                    "params": {
                        "kind": "not_empty",
                        "source": "summarize_nvidia_news",
                    },
                    "true_to_node": None,
                    "false_to_node": None,
                },
            ],
            "edges": [
                {"from": "start", "to": "summarize_nvidia_news"},
                {"from": "summarize_nvidia_news", "to": "cond_after_summarize_nvidia"},
            ],
        }
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "productivity.compose_outlook_email.v1": {
                "result": {"summary": ["hello"], "status": "simulated"}
            }
        },
    )

    results = executor.run()

    assert results["cond_after_summarize_nvidia"].get("condition_result") is True


def test_condition_missing_schema_reports_error():
    workflow = {
        "workflow_name": "invalid_condition_source",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "scrape_page",
                "type": "action",
                "action_id": "common.scrape_web_page.v1",
                "display_name": "爬取财经新闻网页内容",
                "params": {
                    "urls": ["https://example.com/news"],
                    "user_request": "提取股票涨跌数据",
                    "llm_instruction": "分析网页内容，提取涨幅最大和跌幅最大的10只股票及其相关数据。",
                    "llm_provider": "openai/gpt-4o-mini",
                },
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "extracted_content": {"type": "string"},
                    },
                    "required": ["status", "extracted_content"],
                },
            },
            {
                "id": "condition_filter",
                "type": "condition",
                "display_name": "筛选涨跌幅最大股票",
                "params": {
                    "kind": "list_not_empty",
                    "source": "result_of.scrape_page.results",
                },
                "true_to_node": None,
                "false_to_node": None,
            },
        ],
        "edges": [
            {"from": "start", "to": "scrape_page"},
            {"from": "scrape_page", "to": "condition_filter"},
        ],
    }

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert any(
        e.code == "SCHEMA_MISMATCH"
        and e.node_id == "condition_filter"
        and "result_of.scrape_page.results" in e.message
        for e in errors
    )


def test_list_not_empty_handles_field_over_list_items():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "list_field_condition",
            "description": "",
            "nodes": [
                {
                    "id": "cond_list_field",
                    "type": "condition",
                    "params": {
                        "kind": "list_not_empty",
                        "source": [
                            {"employee_id": "a01", "value": 1},
                            {"employee_id": "a02", "value": 2},
                        ],
                        "field": "employee_id",
                    },
                    "true_to_node": None,
                    "false_to_node": None,
                }
            ],
        }
    )

    executor = DynamicActionExecutor(workflow)
    binding_ctx = BindingContext(workflow, results={})

    cond_result, debug = executor._eval_condition(
        workflow.nodes[0].model_dump(), binding_ctx, include_debug=True
    )

    assert cond_result is True
    assert debug.get("resolved_value") == ["a01", "a02"]
