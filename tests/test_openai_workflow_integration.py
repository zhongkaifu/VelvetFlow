"""
Integration test that exercises real OpenAI endpoints to build a workflow end-to-end.
Set `VELVETFLOW_RUN_OPENAI_E2E=1` and `OPENAI_API_KEY` to enable.
"""
import os
from typing import List

import pytest

from velvetflow.action_registry import validate_actions
from velvetflow.planner import plan_workflow_with_two_pass
from velvetflow.search import build_search_service_from_actions


@pytest.mark.openai_e2e
@pytest.mark.skipif(
    os.environ.get("VELVETFLOW_RUN_OPENAI_E2E") != "1",
    reason="Set VELVETFLOW_RUN_OPENAI_E2E=1 to run real OpenAI integration tests.",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for live OpenAI endpoint tests.",
)
def test_real_openai_plans_weather_notification_workflow():
    actions: List[dict] = validate_actions(
        [
            {
                "action_id": "fetch_city_weather",
                "name": "查询城市天气",
                "description": "获取指定城市的当天天气摘要。",
                "domain": "weather",
                "tags": ["weather", "daily"],
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称",
                        }
                    },
                    "required": ["city"],
                },
                "inputs": [
                    {"name": "city", "type": "string", "description": "需要查询的城市"}
                ],
                "outputs": [
                    {
                        "name": "weather_summary",
                        "type": "string",
                        "description": "简要天气描述",
                    }
                ],
            },
            {
                "action_id": "notify_daily_weather",
                "name": "推送天气提醒",
                "description": "将当天的天气摘要推送给指定的收件人。",
                "domain": "notification",
                "tags": ["notification", "alert"],
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": "接收提醒的联系人或群组",
                        },
                        "channel": {
                            "type": "string",
                            "description": "发送渠道，例如 email/slack/钉钉",
                        },
                        "weather_summary": {
                            "type": "string",
                            "description": "要推送的天气内容",
                        },
                    },
                    "required": ["recipient", "channel", "weather_summary"],
                },
                "inputs": [
                    {
                        "name": "recipient",
                        "type": "string",
                        "description": "收件人/群组标识",
                    },
                    {
                        "name": "channel",
                        "type": "string",
                        "description": "发送渠道",
                    },
                    {
                        "name": "weather_summary",
                        "type": "string",
                        "description": "天气摘要",
                    },
                ],
                "outputs": [
                    {
                        "name": "delivery_status",
                        "type": "string",
                        "description": "通知发送状态或消息 ID",
                    }
                ],
            },
        ]
    )

    search_service = build_search_service_from_actions(
        actions,
        embedding_model=os.environ.get("VELVETFLOW_EMBEDDING_MODEL", "text-embedding-3-small"),
        alpha=0.7,
    )

    requirement = (
        "每天早上 7 点为上海用户生成当日天气摘要，"
        "并通过钉钉或邮件发送提醒，让他们在出门前就能知道天气情况。"
    )

    workflow = plan_workflow_with_two_pass(
        nl_requirement=requirement,
        search_service=search_service,
        action_registry=actions,
        max_rounds=3,
        max_repair_rounds=1,
    )

    action_ids = [
        getattr(node, "action_id", None)
        for node in workflow.nodes
        if getattr(node, "type", None) == "action"
    ]
    assert action_ids, "workflow should contain action nodes"
    assert set(action_ids).issubset({a["action_id"] for a in actions})

    connections = {
        (edge.source, edge.target)
        for edge in workflow.edges
        if getattr(edge, "source", None) and getattr(edge, "target", None)
    }
    assert connections, "workflow should include data flow edges"
