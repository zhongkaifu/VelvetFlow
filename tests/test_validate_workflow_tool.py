# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.models import ValidationError, Workflow
from velvetflow.planner.repair_tools import _apply_local_repairs_for_unknown_params
from velvetflow.verification.validation import validate_completed_workflow

ACTION_REGISTRY = BUSINESS_ACTIONS


def _basic_workflow(params: dict) -> dict:
    return {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "notify",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": params,
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "notify"},
            {"from": "notify", "to": "end"},
        ],
    }


def test_validate_workflow_success():
    workflow = _basic_workflow({"email_content": "hello"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_validate_workflow_with_optional_email_to():
    workflow = _basic_workflow({"email_content": "hello", "emailTo": "user@example.com"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_validate_workflow_missing_required_param():
    workflow = _basic_workflow({})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation to fail due to missing required param"
    assert any(e.code == "MISSING_REQUIRED_PARAM" for e in errors)
    assert any(
        e.code == "EMPTY_PARAMS" and e.field == "params" and e.node_id == "notify"
        for e in errors
    )


def test_empty_params_surface_llm_repair_hint():
    workflow = _basic_workflow({})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    empty_param_error = next(e for e in errors if e.code == "EMPTY_PARAMS")
    assert "LLM" in empty_param_error.message


def test_validate_workflow_empty_param_value_flagged():
    workflow = _basic_workflow({"email_content": "   "})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation to flag empty param value"
    assert any(
        e.code == "EMPTY_PARAM_VALUE" and e.field == "params.email_content"
        for e in errors
    )


def test_validate_workflow_unknown_param():
    workflow = _basic_workflow({"email_content": "hello", "date_filter": "today"})

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors, "Expected validation to fail due to unknown param"
    assert any(e.code == "UNKNOWN_PARAM" and e.field == "date_filter" for e in errors)


def test_invalid_agg_value_is_reported_with_context():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "prepare_email",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {"email_content": "hello"},
            },
            {
                "id": "notify",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "params": {
                    "email_content": {
                        "__from__": "result_of.prepare_email.message",
                        "__agg__": "sum",
                    }
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "prepare_email"},
            {"from": "prepare_email", "to": "notify"},
            {"from": "notify", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any("__agg__ 不支持值" in e.message for e in errors)
    assert any(e.node_id == "notify" for e in errors)


def test_self_reference_binding_is_flagged_for_llm_repair():
    workflow = _basic_workflow(
        {"email_content": {"__from__": "result_of.notify.message"}}
    )

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    self_ref_error = next(e for e in errors if e.code == "SELF_REFERENCE")
    assert self_ref_error.field == "params.email_content"
    assert "LLM" in self_ref_error.message and "工具" in self_ref_error.message


def test_local_repair_removes_unknown_param():
    workflow = Workflow.model_validate(
        _basic_workflow({"email_content": "hello", "date_filter": "today"})
    )
    validation_errors = [
        ValidationError(
            code="UNKNOWN_PARAM",
            node_id="notify",
            field="date_filter",
            message="unexpected param",
        )
    ]

    repaired = _apply_local_repairs_for_unknown_params(
        current_workflow=workflow,
        validation_errors=validation_errors,
        action_registry=ACTION_REGISTRY,
    )

    assert repaired is not None
    repaired_params = next(
        n for n in repaired.model_dump(by_alias=True)["nodes"] if n["id"] == "notify"
    )["params"]
    assert "date_filter" not in repaired_params


def test_invalid_condition_reference_detected():
    workflow = {
        "workflow_name": "health_alert",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "condition_any_high_temp",
                "type": "condition",
                "params": {"kind": "list_not_empty", "source": []},
            },
            {
                "id": "action_generate_alert_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "High Temperature Alert",
                    "abnormal_count": {
                        "__from__": "result_of.condition_any_high_temp.exports.items",
                        "__agg__": "count",
                    },
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "condition_any_high_temp"},
            {"from": "condition_any_high_temp", "to": "action_generate_alert_report"},
            {"from": "action_generate_alert_report", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert any(
        err.code == "SCHEMA_MISMATCH" and "exports.items" in err.message for err in errors
    )


def test_condition_array_binding_supports_builtin_length_field():
    workflow = {
        "workflow_name": "temperature_monitor",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "get_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "today"},
            },
            {
                "id": "loop_check_temperature",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.get_temperatures.data"},
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "employee.employee_id",
                                    "last_temperature": "employee.temperature",
                                    "status": "high_temperature",
                                },
                            }
                        ],
                        "entry": "add_to_warning_list",
                        "exit": "add_to_warning_list",
                    },
                    "exports": {
                        "items": {
                            "from_node": "add_to_warning_list",
                            "fields": ["employee_id", "last_temperature", "status"],
                            "mode": "collect",
                        }
                    },
                },
            },
            {
                "id": "condition_has_warning",
                "type": "condition",
                "params": {
                    "kind": "any_greater_than",
                    "source": {"__from__": "result_of.loop_check_temperature.items"},
                    "field": "length",
                    "threshold": 1,
                },
                "true_to_node": "end",
                "false_to_node": "end",
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "get_temperatures"},
            {"from": "get_temperatures", "to": "loop_check_temperature"},
            {"from": "loop_check_temperature", "to": "condition_has_warning"},
            {"from": "condition_has_warning", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_loop_export_length_reference_allowed_for_plain_string():
    workflow = {
        "workflow_name": "health_monitor",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "fetch_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "today"},
            },
            {
                "id": "loop_employees",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.fetch_temperatures.data"},
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "employee.employee_id",
                                    "last_temperature": "employee.temperature",
                                    "status": "high_temperature",
                                },
                            }
                        ],
                        "entry": "add_to_warning_list",
                        "exit": "add_to_warning_list",
                    },
                    "exports": {
                        "items": {
                            "from_node": "add_to_warning_list",
                            "fields": ["employee_id", "last_temperature", "status"],
                            "mode": "collect",
                        }
                    },
                },
            },
            {
                "id": "generate_warning_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "健康预警报告",
                    "date": {"__from__": "result_of.fetch_temperatures.date"},
                    "abnormal_count": "result_of.loop_employees.exports.items.length",
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "fetch_temperatures"},
            {"from": "fetch_temperatures", "to": "loop_employees"},
            {"from": "loop_employees", "to": "generate_warning_report"},
            {"from": "generate_warning_report", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_loop_aggregates_count_reference_allowed_for_plain_string():
    workflow = {
        "workflow_name": "health_monitor",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "fetch_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "today"},
            },
            {
                "id": "loop_employees",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.fetch_temperatures.data"},
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "record_health",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "employee.employee_id",
                                    "last_temperature": "employee.temperature",
                                    "last_check_date": "employee.date",
                                    "status": "high_temperature",
                                },
                            }
                        ],
                        "entry": "record_health",
                        "exit": "record_health",
                    },
                    "exports": {
                        "items": {
                            "from_node": "record_health",
                            "fields": ["employee_id", "last_temperature", "status"],
                            "mode": "collect",
                        },
                        "aggregates": [
                            {
                                "name": "record_total",
                                "kind": "count",
                                "from_node": "record_health",
                                "source": "result_of.loop_employees.items",
                                "expr": {"kind": "count"},
                            },
                            {
                                "name": "max_temperature",
                                "kind": "max",
                                "from_node": "record_health",
                                "source": "result_of.loop_employees.items",
                                "expr": {"kind": "max", "field": "last_temperature"},
                            },
                        ],
                    },
                },
            },
            {
                "id": "generate_warning_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "健康预警报告",
                    "date": {"__from__": "result_of.fetch_temperatures.date"},
                    "abnormal_count": "result_of.loop_employees.aggregates.count",
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "fetch_temperatures"},
            {"from": "fetch_temperatures", "to": "loop_employees"},
            {"from": "loop_employees", "to": "generate_warning_report"},
            {"from": "generate_warning_report", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []


def test_loop_invalid_accumulator_reference_is_repaired():
    workflow = {
        "workflow_name": "news_summary",
        "description": "",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "search_news_nvidia",
                "type": "action",
                "action_id": "common.search_news.v1",
                "display_name": "搜索Nvidia相关新闻",
                "params": {"query": "Nvidia", "limit": 5, "timeout": 8},
            },
            {
                "id": "loop_summarize_nvidia",
                "type": "loop",
                "display_name": "循环总结Nvidia新闻",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.search_news_nvidia.results"},
                    "item_alias": "news_item",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize_nvidia",
                                "type": "action",
                                "action_id": "common.summarize.v1",
                                "display_name": "总结Nvidia新闻内容",
                                "params": {
                                    "text": "{{news_item.snippet}}",
                                    "max_sentences": 3,
                                },
                                "parent_node_id": "loop_summarize_nvidia",
                            }
                        ],
                        "entry": "summarize_nvidia",
                        "exit": "summarize_nvidia",
                    },
                    "exports": {
                        "items": {
                            "from_node": "summarize_nvidia",
                            "fields": ["summary"],
                        }
                    },
                },
            },
            {
                "id": "send_email",
                "type": "action",
                "action_id": "productivity.compose_outlook_email.v1",
                "display_name": "发送新闻总结邮件",
                "params": {
                    "email_content": {
                        "__from__": "result_of.loop_summarize_nvidia.accumulator",
                        "__agg__": "format_join",
                        "format": "Nvidia新闻总结：{summary}",
                    },
                    "emailTo": "user@example.com",
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "search_news_nvidia"},
            {"from": "search_news_nvidia", "to": "loop_summarize_nvidia"},
            {"from": "loop_summarize_nvidia", "to": "send_email"},
            {"from": "send_email", "to": "end"},
        ],
    }

    workflow_model = Workflow.model_validate(workflow)
    workflow_dict = workflow_model.model_dump(by_alias=True)
    errors = validate_completed_workflow(workflow_dict, ACTION_REGISTRY)

    assert any(
        err.code == "SCHEMA_MISMATCH"
        and "loop_summarize_nvidia" in (err.message or "")
        and "accumulator" in (err.message or "")
        for err in errors
    )

    send_email = next(
        node for node in workflow_dict.get("nodes", []) if node.get("id") == "send_email"
    )
    assert "email_content" not in (send_email.get("params") or {})


def test_loop_and_condition_without_action_ids_still_pass_type_validation():
    workflow = {
        "workflow_name": "员工体温健康预警工作流",
        "nodes": [
            {"id": "start", "type": "start"},
            {
                "id": "get_temperatures",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "params": {"date": "2024-06-01"},
            },
            {
                "id": "loop_employees",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": {"__from__": "result_of.get_temperatures.data"},
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "check_temperature",
                                "type": "condition",
                                "params": {
                                    "source": "employee",
                                    "field": "temperature",
                                    "threshold": 38,
                                    "kind": "greater_than",
                                },
                                "true_to_node": "add_to_warning_list",
                                "false_to_node": None,
                                "parent_node_id": "loop_employees",
                            },
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "placeholder",
                                    "last_check_date": "2024-06-01",
                                    "last_temperature": 0,
                                    "status": "warning",
                                },
                                "parent_node_id": "loop_employees",
                            },
                        ],
                        "entry": "check_temperature",
                        "exit": "add_to_warning_list",
                    },
                        "exports": {
                            "items": {
                                "from_node": "add_to_warning_list",
                                "fields": ["employee_id", "last_temperature", "status"],
                            }
                        },
                },
            },
            {
                "id": "check_warning_list_empty",
                "type": "condition",
                "params": {
                    "source": "result_of.loop_employees.exports.items",
                    "kind": "list_not_empty",
                },
                "true_to_node": "generate_warning_report",
                "false_to_node": "generate_normal_report",
            },
            {
                "id": "generate_warning_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "健康预警报告",
                    "date": {"__from__": "result_of.get_temperatures.date"},
                    "abnormal_count": 1,
                },
            },
            {
                "id": "generate_normal_report",
                "type": "action",
                "action_id": "hr.record_health_event.v1",
                "params": {
                    "event_type": "普通健康报告",
                    "date": {"__from__": "result_of.get_temperatures.date"},
                    "abnormal_count": 0,
                },
            },
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "get_temperatures"},
            {"from": "get_temperatures", "to": "loop_employees"},
            {"from": "loop_employees", "to": "check_warning_list_empty"},
            {"from": "check_warning_list_empty", "to": "generate_warning_report", "condition": True},
            {"from": "check_warning_list_empty", "to": "generate_normal_report", "condition": False},
            {"from": "generate_warning_report", "to": "end"},
            {"from": "generate_normal_report", "to": "end"},
        ],
    }

    errors = validate_workflow_data(workflow, ACTION_REGISTRY)

    assert errors == []
