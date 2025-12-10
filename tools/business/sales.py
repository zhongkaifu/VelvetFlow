"""Sales-related business tools with mock data."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from tools.base import Tool
from tools.registry import register_tool


def _default_stage_breakdown() -> List[Dict[str, Any]]:
    return [
        {"stage": "prospecting", "count": 18, "expected_value": 92000},
        {"stage": "qualification", "count": 9, "expected_value": 136000},
        {"stage": "proposal", "count": 5, "expected_value": 104000},
        {"stage": "negotiation", "count": 3, "expected_value": 80000},
    ]


def list_sales_pipeline(region: str | None = None) -> Dict[str, Any]:
    """Return a snapshot of the sales pipeline with optional regional filtering."""

    pipeline_date = date.today().isoformat()
    return {
        "as_of": pipeline_date,
        "region": region or "global",
        "stage_breakdown": _default_stage_breakdown(),
        "top_deals": [
            {"deal_id": "DL-1042", "account": "Acme Manufacturing", "value": 75000, "stage": "negotiation"},
            {"deal_id": "DL-1036", "account": "Bright Retail", "value": 42000, "stage": "proposal"},
            {"deal_id": "DL-1031", "account": "Nimbus Health", "value": 36000, "stage": "qualification"},
        ],
    }


def create_sales_deal(account: str, contact: str, value: float, source: str | None = None) -> Dict[str, Any]:
    """Create a mock sales deal record with an auto-generated identifier."""

    return {
        "deal_id": f"DL-{hash(account + contact) % 10000:04d}",
        "account": account,
        "contact": contact,
        "value": value,
        "source": source or "inbound",
        "status": "created",
    }


def get_quote_summary(deal_id: str, discount: float | None = None) -> Dict[str, Any]:
    """Return a structured quote summary for a requested deal."""

    base_value = 48000
    applied_discount = discount or 0.0
    final_value = round(base_value * (1 - applied_discount), 2)
    return {
        "deal_id": deal_id,
        "base_value": base_value,
        "discount": applied_discount,
        "final_value": final_value,
        "currency": "USD",
        "valid_until": date.today().replace(day=min(date.today().day + 14, 28)).isoformat(),
    }


def register_sales_tools() -> None:
    register_tool(
        Tool(
            name="sales.list_sales_pipeline",
            description="List current sales pipeline by stage, with optional region filter.",
            function=list_sales_pipeline,
            args_schema={
                "type": "object",
                "properties": {"region": {"type": "string"}},
            },
        )
    )

    register_tool(
        Tool(
            name="sales.create_sales_deal",
            description="Create a sales deal with account, contact, value, and optional source.",
            function=create_sales_deal,
            args_schema={
                "type": "object",
                "properties": {
                    "account": {"type": "string"},
                    "contact": {"type": "string"},
                    "value": {"type": "number"},
                    "source": {"type": "string"},
                },
                "required": ["account", "contact", "value"],
            },
        )
    )

    register_tool(
        Tool(
            name="sales.get_quote_summary",
            description="Generate a quote summary for a sales deal with optional discount applied.",
            function=get_quote_summary,
            args_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string"},
                    "discount": {"type": "number"},
                },
                "required": ["deal_id"],
            },
        )
    )


__all__ = [
    "list_sales_pipeline",
    "create_sales_deal",
    "get_quote_summary",
    "register_sales_tools",
]
