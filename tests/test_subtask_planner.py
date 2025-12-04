# License: BSD 3-Clause License

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.planner.subtask_planner import plan_remaining_subtasks


def test_plan_remaining_subtasks_flags_missing_steps():
    requirement = "收集用户反馈，分析结果并发送日报"
    workflow = {
        "nodes": [
            {
                "id": "collect_feedback",
                "type": "action",
                "action_id": "survey.collect.v1",
                "display_name": "收集用户反馈",
            },
            {
                "id": "analyze",
                "type": "action",
                "action_id": "analytics.summarize.v1",
                "display_name": "分析结果",
            },
        ]
    }

    result = plan_remaining_subtasks(nl_requirement=requirement, workflow=workflow)

    assert result["status"] == "ok"
    assert "发送日报" in result["remaining_subtasks"], "Should suggest the notification step"
    assert any("收集用户反馈" in item for item in result["covered_subtasks"])


def test_plan_remaining_subtasks_handles_empty_workflow():
    requirement = "准备素材并发布公告"

    result = plan_remaining_subtasks(nl_requirement=requirement, workflow={"nodes": []})

    assert result["remaining_subtasks"] == ["准备素材", "发布公告"]
    assert result["covered_subtasks"] == []

