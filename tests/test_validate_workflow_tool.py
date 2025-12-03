# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validate_workflow import validate_workflow_data
from velvetflow.models import ValidationError, Workflow
from velvetflow.planner.repair_tools import _apply_local_repairs_for_unknown_params
from velvetflow.verification.validation import validate_completed_workflow

ACTION_REGISTRY = json.loads(
    (Path(__file__).parent.parent / "velvetflow" / "business_actions.json").read_text(
        encoding="utf-8"
    )
)


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
                                    "employee_id": "{{employee.employee_id}}",
                                    "last_temperature": "{{employee.temperature}}",
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
                                    "employee_id": "{{employee.employee_id}}",
                                    "last_temperature": "{{employee.temperature}}",
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
                    "abnormal_count": "{{result_of.loop_employees.exports.items.length}}",
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
