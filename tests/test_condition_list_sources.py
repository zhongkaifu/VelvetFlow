# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow
from velvetflow.verification.validation import validate_completed_workflow

ACTION_REGISTRY = BUSINESS_ACTIONS


def _workflow_with_list_expression():
    return {
        "workflow_name": "list_source_condition",
        "description": "",
        "nodes": [
            {
                "id": "summarize_nvidia_news",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-01-01"},
            },
            {
                "id": "summarize_google_news",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-01-02"},
            },
            {
                "id": "check_all_summarized",
                "type": "condition",
                "params": {
                    "expression": (
                        "{{ (result_of.summarize_nvidia_news.data | length) > 0 and "
                        "(result_of.summarize_google_news.data | length) > 0 }}"
                    )
                },
                "true_to_node": "notify_summary",
                "false_to_node": None,
            },
            {
                "id": "notify_summary",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "send summaries"},
            },
        ],
        "edges": [
            {"from": "summarize_nvidia_news", "to": "summarize_google_news"},
            {"from": "summarize_google_news", "to": "check_all_summarized"},
            {"from": "check_all_summarized", "to": "notify_summary", "condition": True},
        ],
    }


def test_condition_validation_accepts_jinja_expression():
    workflow = _workflow_with_list_expression()
    errors = validate_completed_workflow(workflow, action_registry=ACTION_REGISTRY)
    assert errors == []


def test_executor_evaluates_expression_sources():
    workflow = Workflow.model_validate(_workflow_with_list_expression())
    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "hr.get_today_temperatures.v1": {
                "result": {"status": "ok", "data": [{"employee_id": "a01", "temperature": 36.5}]}
            }
        },
    )

    results = executor.run()

    assert results["check_all_summarized"]["condition_result"] is True
    assert "notify_summary" in results
