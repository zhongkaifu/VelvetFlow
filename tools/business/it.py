"""IT operations and service management tools."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from tools.base import Tool
from tools.registry import register_tool


_DEFECT_CATALOG = {
    "login": "auth-service",
    "payment": "payment-gateway",
    "report": "reporting-api",
}


def open_it_ticket(title: str, description: str, severity: str = "medium") -> Dict[str, Any]:
    """Create a mock IT ticket and return tracking metadata."""

    component = next((svc for key, svc in _DEFECT_CATALOG.items() if key in title.lower()), "core-platform")
    ticket_id = f"IT-{abs(hash(title + description)) % 10000:04d}"
    return {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "severity": severity,
        "component": component,
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def get_service_status(service_name: str) -> Dict[str, Any]:
    """Return a status page style snapshot for a service."""

    return {
        "service": service_name,
        "status": "operational" if service_name != "payment-gateway" else "degraded",
        "uptime_30d": 99.87,
        "active_incidents": [] if service_name != "payment-gateway" else ["INC-4421"],
    }


def provision_test_environment(owner: str, purpose: str | None = None) -> Dict[str, Any]:
    """Provision a mock sandbox environment for testing."""

    env_id = f"sandbox-{hash(owner + (purpose or 'general')) % 10000:04d}"
    return {
        "environment_id": env_id,
        "owner": owner,
        "purpose": purpose or "integration-testing",
        "status": "provisioned",
        "url": f"https://{env_id}.example.internal",
    }


def register_it_tools() -> None:
    register_tool(
        Tool(
            name="it.open_it_ticket",
            description="Open an IT support ticket with title, description, and severity.",
            function=open_it_ticket,
            args_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string"},
                },
                "required": ["title", "description"],
            },
        )
    )

    register_tool(
        Tool(
            name="it.get_service_status",
            description="Retrieve uptime and incident summary for a named service.",
            function=get_service_status,
            args_schema={
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        )
    )

    register_tool(
        Tool(
            name="it.provision_test_environment",
            description="Provision a mock sandbox environment for QA or integration testing.",
            function=provision_test_environment,
            args_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["owner"],
            },
        )
    )


__all__ = [
    "open_it_ticket",
    "get_service_status",
    "provision_test_environment",
    "register_it_tools",
]
