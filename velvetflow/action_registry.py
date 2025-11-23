"""Action registry definitions for VelvetFlow."""
from typing import Any, Dict, List, Optional

BUSINESS_ACTIONS: List[Dict[str, Any]] = [
    {
        "action_id": "hr.notify_human.v1",
        "name": "Notify HR about an event",
        "domain": "hr",
        "description": "Send a notification to HR staff about an important event, via IM/email.",
        "tags": ["notify", "hr", "alert"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "message": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["message"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        },
        "enabled": True,
    },
    {
        "action_id": "hr.record_health_event.v1",
        "name": "Record a business event",
        "domain": "hr",
        "description": "Record a generic business event into the HR event stream.",
        "tags": ["event", "log", "hr"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string"},
                "date": {"type": "string"},
                "abnormal_count": {"type": "integer"},
            },
            "required": ["event_type"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "event_id": {"type": "string"},
            },
        },
        "enabled": True,
    },
    {
        "action_id": "hr.get_today_temperatures.v1",
        "name": "Get today's employee temperature readings",
        "domain": "hr",
        "description": "Fetch today's temperature readings for all employees.",
        "tags": ["health", "temperature", "fetch"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
            },
            "required": ["date"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "string"},
                            "temperature": {"type": "number"},
                        },
                        "required": ["employee_id", "temperature"],
                    },
                }
            },
            "required": ["data"],
        },
        "enabled": True,
    },
    {
        "action_id": "hr.update_employee_health_profile.v1",
        "name": "Update an employee health profile",
        "domain": "hr",
        "description": "Upsert employee health profile with the latest temperature data and status.",
        "tags": ["health", "profile", "update"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "last_check_date": {"type": "string"},
                "last_temperature": {"type": "number"},
                "status": {"type": "string"},
            },
            "required": ["employee_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "string"},
                            "last_temperature": {"type": "number"},
                            "status": {"type": "string"},
                        },
                        "required": ["employee_id", "last_temperature"],
                    },
                }
            },
            "required": ["data"],
        },
        "enabled": True,
    },
    {
        "action_id": "ops.create_incident.v1",
        "name": "Create incident ticket",
        "domain": "ops",
        "description": "Create an incident ticket in the incident management system.",
        "tags": ["incident", "ticket", "ops"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["title"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
                "status": {"type": "string"},
            },
        },
        "enabled": True,
    },
    {
        "action_id": "ops.notify_oncall.v1",
        "name": "Notify oncall engineer",
        "domain": "ops",
        "description": "Notify the oncall engineer about a critical incident.",
        "tags": ["notify", "oncall", "ops"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        },
        "enabled": True,
    },
    {
        "action_id": "crm.create_lead.v1",
        "name": "Create CRM lead record",
        "domain": "crm",
        "description": "Create a new lead record in CRM system.",
        "tags": ["crm", "lead"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["name"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
                "status": {"type": "string"},
            },
        },
        "enabled": True,
    },
    {
        "action_id": "crm.log_interaction.v1",
        "name": "Log customer interaction",
        "domain": "crm",
        "description": "Log a customer interaction note in CRM.",
        "tags": ["crm", "log", "interaction"],
        "arg_schema": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["lead_id", "note"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        },
        "enabled": True,
    },
]


def get_action_by_id(action_id: str) -> Optional[Dict[str, Any]]:
    for a in BUSINESS_ACTIONS:
        if a["action_id"] == action_id:
            return a
    return None

