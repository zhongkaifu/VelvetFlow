# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Provide stub modules for optional dependencies used by executor
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
from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow
from velvetflow.verification.validation import validate_completed_workflow


ACTION_REGISTRY = BUSINESS_ACTIONS


def _workflow_with_extracted_content_field():
    return {
        "workflow_name": "condition_field_type_check",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "scrape_news",
                "type": "action",
                "action_id": "common.scrape_web_page.v1",
                "params": {
                    "urls": ["https://example.com"],
                    "user_request": "news summary",
                },
            },
            {
                "id": "check_results",
                "type": "condition",
                "params": {
                    "kind": "not_empty",
                    "source": "result_of.scrape_news.extracted_content",
                },
                "true_to_node": None,
                "false_to_node": None,
            },
        ],
        "edges": [
            {"from": "start", "to": "scrape_news"},
            {"from": "scrape_news", "to": "check_results"},
        ],
    }


def test_condition_validation_accepts_extracted_content_field():
    workflow = _workflow_with_extracted_content_field()

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert errors == []


def test_condition_validation_detects_non_list_targets():
    workflow = {
        "workflow_name": "condition_field_not_list",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "log_event",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {"event_type": "login"},
            },
            {
                "id": "check_results",
                "type": "condition",
                "params": {
                    "kind": "list_not_empty",
                    "source": "result_of.log_event",
                    "field": "event_id",
                },
                "true_to_node": None,
                "false_to_node": None,
            },
        ],
        "edges": [
            {"from": "start", "to": "log_event"},
            {"from": "log_event", "to": "check_results"},
        ],
    }

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert any(
        err.code == "SCHEMA_MISMATCH" and err.field in {"field", "source"} for err in errors
    )


def test_condition_validation_accepts_loop_export_field_schema():
    workflow = {
        "workflow_name": "loop_export_items_schema",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "get_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2025-12-02"},
            },
            {
                "id": "loop_check_temperature",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "item_alias": "employee",
                    "source": "result_of.get_temperatures.data",
                    "body_subgraph": {
                        "entry": "cond_temp_above_38",
                        "exit": "add_to_warning_list",
                        "nodes": [
                            {
                                "id": "cond_temp_above_38",
                                "type": "condition",
                                "display_name": "体温是否超过38度",
                                "params": {
                                    "source": "employee",
                                    "field": "temperature",
                                    "threshold": 38,
                                    "kind": "greater_than",
                                },
                                "true_to_node": "add_to_warning_list",
                                "false_to_node": None,
                                "parent_node_id": "loop_check_temperature",
                            },
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "display_name": "将体温异常员工ID加入健康预警列表",
                                "params": {
                                    "employee_id": "{{employee.employee_id}}",
                                    "last_temperature": "{{employee.temperature}}",
                                    "status": "high_temperature",
                                },
                                "parent_node_id": "loop_check_temperature",
                            },
                        ],
                    },
                    "exports": {
                        "items": {
                            "from_node": "add_to_warning_list",
                            "fields": ["employee_id"],
                            "mode": "collect",
                        }
                    },
                },
            },
            {
                "id": "cond_warning_list_not_empty",
                "type": "condition",
                "params": {
                    "source": "result_of.loop_check_temperature.exports.items",
                    "field": "employee_id",
                    "kind": "list_not_empty",
                },
                "true_to_node": None,
                "false_to_node": None,
            },
        ],
        "edges": [
            {"from": "start", "to": "get_temperatures"},
            {"from": "get_temperatures", "to": "loop_check_temperature"},
            {"from": "loop_check_temperature", "to": "cond_warning_list_not_empty"},
        ],
    }

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert errors == []


def test_executor_uses_field_value_for_not_empty():
    workflow = Workflow.model_validate(_workflow_with_extracted_content_field())
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "common.scrape_web_page.v1": {
                "result": {"status": "ok", "extracted_content": "breaking news"}
            }
        },
    )

    results = executor.run()

    condition_result = results["check_results"].get("condition_result")
    resolved_value = results["check_results"].get("resolved_value")

    assert condition_result is True
    assert resolved_value == "breaking news"


def _workflow_with_numeric_field(field: str):
    return {
        "workflow_name": "condition_field_numeric_check",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-01-01"},
            },
            {
                "id": "check_temperature",
                "type": "condition",
                "params": {
                    "kind": "greater_than",
                    "source": "result_of.temperatures",
                    "field": field,
                    "threshold": 37,
                },
                "true_to_node": None,
                "false_to_node": None,
            },
        ],
        "edges": [
            {"from": "start", "to": "temperatures"},
            {"from": "temperatures", "to": "check_temperature"},
        ],
    }


def test_condition_numeric_validation_uses_field_schema():
    workflow = _workflow_with_numeric_field("data[*].temperature")

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert errors == []


def test_condition_numeric_validation_rejects_non_numeric_field():
    workflow = _workflow_with_numeric_field("data[*].employee_id")

    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)

    assert any(err.code == "SCHEMA_MISMATCH" and err.field == "field" for err in errors)
