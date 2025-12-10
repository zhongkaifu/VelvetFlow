"""Additional HR tools for recruiting and workforce reporting."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

from tools.base import Tool
from tools.registry import register_tool


def schedule_candidate_interview(candidate_name: str, role: str, panel: list[str], date_slot: str) -> Dict[str, Any]:
    """Schedule a candidate interview and confirm panel availability."""

    return {
        "candidate": candidate_name,
        "role": role,
        "panel": panel,
        "scheduled_for": date_slot,
        "status": "confirmed",
    }


def get_headcount_snapshot(department: str | None = None) -> Dict[str, Any]:
    """Return headcount, openings, and attrition for a department."""

    dept_name = department or "all"
    return {
        "department": dept_name,
        "as_of": date.today().isoformat(),
        "headcount": 128 if dept_name == "engineering" else 64,
        "open_roles": 6 if dept_name == "engineering" else 3,
        "attrition_12m_pct": 7.4,
    }


def register_hr_tools() -> None:
    register_tool(
        Tool(
            name="hr.schedule_candidate_interview",
            description="Schedule an interview for a candidate with a panel and time slot.",
            function=schedule_candidate_interview,
            args_schema={
                "type": "object",
                "properties": {
                    "candidate_name": {"type": "string"},
                    "role": {"type": "string"},
                    "panel": {"type": "array", "items": {"type": "string"}},
                    "date_slot": {"type": "string"},
                },
                "required": ["candidate_name", "role", "panel", "date_slot"],
            },
        )
    )

    register_tool(
        Tool(
            name="hr.get_headcount_snapshot",
            description="Get headcount, openings, and attrition metrics for a department.",
            function=get_headcount_snapshot,
            args_schema={
                "type": "object",
                "properties": {"department": {"type": "string"}},
            },
        )
    )


__all__ = [
    "schedule_candidate_interview",
    "get_headcount_snapshot",
    "register_hr_tools",
]
