"""Marketing automation and analytics tools."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

from tools.base import Tool
from tools.registry import register_tool


_DEFAULT_CHANNELS = ["email", "social", "paid_search", "events"]


def plan_campaign_calendar(campaign_name: str, channels: List[str] | None = None) -> Dict[str, Any]:
    """Create a simple campaign calendar across channels."""

    chosen_channels = channels or _DEFAULT_CHANNELS
    start_date = date.today()
    schedule = []
    for idx, channel in enumerate(chosen_channels):
        schedule.append(
            {
                "channel": channel,
                "launch_date": (start_date + timedelta(days=idx * 3)).isoformat(),
                "kpi": "CTR" if channel == "email" else "CPL",
            }
        )
    return {"campaign": campaign_name, "schedule": schedule}


def fetch_campaign_performance(campaign: str) -> Dict[str, Any]:
    """Return mock performance metrics for a campaign."""

    return {
        "campaign": campaign,
        "metrics": {
            "impressions": 180000,
            "clicks": 12400,
            "conversions": 860,
            "cost": 12500,
        },
        "attribution_model": "last-touch",
    }


def propose_ab_test(goal: str, variants: List[str]) -> Dict[str, Any]:
    """Generate a suggested A/B test plan with sample sizes."""

    base_size = 1200
    plan = [
        {"variant": variant, "sample_size": base_size + idx * 100, "primary_metric": goal}
        for idx, variant in enumerate(variants)
    ]
    return {"goal": goal, "plan": plan}


def register_marketing_tools() -> None:
    register_tool(
        Tool(
            name="marketing.plan_campaign_calendar",
            description="Build a campaign launch calendar across channels.",
            function=plan_campaign_calendar,
            args_schema={
                "type": "object",
                "properties": {
                    "campaign_name": {"type": "string"},
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["campaign_name"],
            },
        )
    )

    register_tool(
        Tool(
            name="marketing.fetch_campaign_performance",
            description="Retrieve performance metrics for a campaign.",
            function=fetch_campaign_performance,
            args_schema={
                "type": "object",
                "properties": {"campaign": {"type": "string"}},
                "required": ["campaign"],
            },
        )
    )

    register_tool(
        Tool(
            name="marketing.propose_ab_test",
            description="Generate an A/B test plan including sample sizes for each variant.",
            function=propose_ab_test,
            args_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "variants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["goal", "variants"],
            },
        )
    )


__all__ = [
    "plan_campaign_calendar",
    "fetch_campaign_performance",
    "propose_ab_test",
    "register_marketing_tools",
]
