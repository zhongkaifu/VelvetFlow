import asyncio
import copy
import json

import pytest

from velvetflow.planner import structure
from velvetflow.planner.structure import plan_workflow_structure_with_llm


class _DummySearch:
    def search(self, query: str, top_k: int = 10):  # pragma: no cover - trivial
        return []


def _invoke(tool, payload):
    """Invoke an Agent FunctionTool with JSON input synchronously."""

    if getattr(tool, "name", "") == "plan_user_requirement":
        input_body = {"payload": payload}
    else:
        input_body = payload

    raw = asyncio.run(tool.on_invoke_tool(None, json.dumps(input_body)))
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "error", "message": raw}
    return raw


def _fake_agent_run(agent, prompt, max_turns=None):
    """Drive the user requirement tools directly without invoking the model."""

    structure.Runner.last_agent = agent  # type: ignore[attr-defined]
    agent.test_results = {}

    # Trigger validation failure when status is missing.
    plan_tool = next(t for t in agent.tools if getattr(t, "name", "") == "plan_user_requirement")
    review_tool = next(t for t in agent.tools if getattr(t, "name", "") == "review_user_requirement")

    missing_status_payload = {
        "requirements": [
            {
                "description": "demo task",
                "intent": "plan",
                "inputs": ["query"],
                "constraints": [],
            }
        ],
        "assumptions": [],
    }
    agent.test_results["missing_status"] = _invoke(plan_tool, missing_status_payload)

    ok_pending_payload = {
        "requirements": [
            {
                "description": "demo task",
                "intent": "plan",
                "inputs": ["query"],
                "constraints": [],
                "status": "进行中",
            }
        ],
        "assumptions": ["none"],
    }
    agent.test_results["ok_pending"] = _invoke(plan_tool, copy.deepcopy(ok_pending_payload))

    ok_done_payload = copy.deepcopy(ok_pending_payload)
    ok_done_payload["requirements"][0]["status"] = "已完成"
    agent.test_results["ok_done"] = _invoke(plan_tool, ok_done_payload)

    agent.test_results["review"] = _invoke(review_tool, {})


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_user_requirements_require_status_and_are_reviewable(monkeypatch):
    # Patch the agent runner to avoid real model calls and to exercise the tools directly.
    monkeypatch.setattr(structure.Runner, "run_sync", _fake_agent_run)

    dummy_search = _DummySearch()
    action_registry = []

    workflow = plan_workflow_structure_with_llm(
        nl_requirement="demo",
        search_service=dummy_search,
        action_registry=action_registry,
        max_rounds=1,
    )

    # The fake runner attaches results to the agent instance for assertions.
    # We only need to validate the tool responses, not the empty workflow skeleton.
    test_results = structure.Runner.last_agent.test_results  # type: ignore[attr-defined]

    assert test_results["missing_status"]["status"] == "error"
    assert "缺少字段 status" in test_results["missing_status"]["message"]

    ok_pending_requirement = test_results["ok_pending"]["requirement"]["requirements"][0]
    assert ok_pending_requirement["status"] == "进行中"

    review_payload = test_results["review"]["requirement"]["requirements"][0]
    assert review_payload["status"] == "已完成"

    assert workflow.get("workflow_name") == "unnamed_workflow"
    assert workflow.get("nodes") == []


def _fake_agent_run_without_completion(agent, prompt, max_turns=None):
    """Simulate a planning run that never marks requirements as completed."""

    structure.Runner.last_agent = agent  # type: ignore[attr-defined]

    plan_tool = next(t for t in agent.tools if getattr(t, "name", "") == "plan_user_requirement")

    incomplete_payload = {
        "requirements": [
            {
                "description": "demo task",
                "intent": "plan",
                "inputs": ["query"],
                "constraints": [],
                "status": "进行中",
            }
        ],
        "assumptions": [],
    }

    _invoke(plan_tool, incomplete_payload)


def test_planner_requires_completed_requirements(monkeypatch):
    monkeypatch.setattr(structure.Runner, "run_sync", _fake_agent_run_without_completion)

    dummy_search = _DummySearch()

    with pytest.raises(ValueError, match="需求状态未全部为已完成"):
        plan_workflow_structure_with_llm(
            nl_requirement="demo",
            search_service=dummy_search,
            action_registry=[],
            max_rounds=1,
        )
