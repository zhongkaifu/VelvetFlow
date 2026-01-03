import asyncio
import json
import types

from velvetflow.planner import requirement_analysis


def test_analyze_user_requirement_wraps_prompt_in_list(monkeypatch):
    captured_prompt = {}

    def fake_run_sync(agent, prompt, max_turns):  # pragma: no cover - behavior asserted
        captured_prompt["value"] = prompt
        # Simulate the agent invoking the tool to populate parsed_requirement
        payload = {
            "requirements": [
                {
                    "description": "desc",
                    "intent": "intent",
                    "inputs": [],
                    "constraints": [],
                    "status": "未开始",
                    "mapped_node": [],
                }
            ],
            "assumptions": [],
        }
        tool = agent.tools[0]
        ctx = types.SimpleNamespace(run_input=[], message_history=[])
        asyncio.run(tool.on_invoke_tool(ctx, json.dumps({"payload": payload}, ensure_ascii=False)))

    monkeypatch.setattr(requirement_analysis.Runner, "run_sync", fake_run_sync)

    result = requirement_analysis.analyze_user_requirement("demo request", existing_workflow={})

    assert isinstance(captured_prompt.get("value"), list)
    first = captured_prompt["value"][0]
    assert first["role"] == "user"
    assert json.loads(first["content"]) == {
        "nl_requirement": "demo request",
        "existing_workflow": {},
    }
    assert result["requirements"][0]["status"] == "未开始"


def test_analyze_user_requirement_fallback_on_missing_tool_call(monkeypatch):
    def fake_run_sync(agent, prompt, max_turns):
        # Simulate an agent run that never invokes any tool
        return None

    monkeypatch.setattr(requirement_analysis.Runner, "run_sync", fake_run_sync)

    result = requirement_analysis.analyze_user_requirement(
        "check employees", existing_workflow={}
    )

    assert result["requirements"] == [
        {
            "description": "check employees",
            "intent": "",
            "inputs": [],
            "constraints": [],
            "status": "未开始",
            "mapped_node": [],
        }
    ]
    assert result["assumptions"] == []

