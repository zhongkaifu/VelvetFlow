"""Finance and accounting business tools with simulated data."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from tools.base import Tool
from tools.registry import register_tool


_EXPENSE_CATEGORIES = ["travel", "meals", "software", "office"]


def submit_expense_report(employee_id: str, amount: float, category: str, description: str) -> Dict[str, Any]:
    """Submit an expense report and return approval routing info."""

    if category not in _EXPENSE_CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(_EXPENSE_CATEGORIES)}")

    report_id = f"ER-{abs(hash(employee_id + description)) % 10000:04d}"
    approver = "manager" if amount < 500 else "finance-controller"
    return {
        "report_id": report_id,
        "employee_id": employee_id,
        "amount": amount,
        "currency": "USD",
        "category": category,
        "description": description,
        "status": "submitted",
        "next_approver": approver,
    }


def get_budget_variance(department: str, month: str | None = None) -> Dict[str, Any]:
    """Return budget versus actuals for a department and month."""

    month_value = month or date.today().strftime("%Y-%m")
    budget = 120000
    actual = 114500
    return {
        "department": department,
        "month": month_value,
        "budget": budget,
        "actual": actual,
        "variance": round(actual - budget, 2),
        "variance_pct": round((actual - budget) / budget * 100, 2),
    }


def forecast_cash_flow(months_ahead: int = 3) -> Dict[str, Any]:
    """Produce a lightweight forward cash flow forecast."""

    base_month = date.today().replace(day=1)
    series: List[Dict[str, Any]] = []
    for i in range(months_ahead):
        series.append(
            {
                "month": f"{base_month.year}-{base_month.month + i:02d}",
                "inflows": 320000 + i * 5000,
                "outflows": 250000 + i * 4000,
                "net": 70000 + i * 1000,
            }
        )
    return {"as_of": base_month.isoformat(), "projection": series}


def register_finance_tools() -> None:
    register_tool(
        Tool(
            name="finance.submit_expense_report",
            description="Submit an expense report and return a tracking id with approver info.",
            function=submit_expense_report,
            args_schema={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["employee_id", "amount", "category", "description"],
            },
        )
    )

    register_tool(
        Tool(
            name="finance.get_budget_variance",
            description="Retrieve budget versus actual spend for a department and month.",
            function=get_budget_variance,
            args_schema={
                "type": "object",
                "properties": {
                    "department": {"type": "string"},
                    "month": {"type": "string"},
                },
                "required": ["department"],
            },
        )
    )

    register_tool(
        Tool(
            name="finance.forecast_cash_flow",
            description="Generate a forward-looking cash flow forecast for the next N months.",
            function=forecast_cash_flow,
            args_schema={
                "type": "object",
                "properties": {"months_ahead": {"type": "integer", "minimum": 1, "default": 3}},
            },
        )
    )


__all__ = [
    "submit_expense_report",
    "get_budget_variance",
    "forecast_cash_flow",
    "register_finance_tools",
]
