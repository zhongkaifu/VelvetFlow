import asyncio
import json

import pytest

from velvetflow.planner import structure
from velvetflow.planner.structure import plan_workflow_structure_with_llm


class _DummySearch:
    def search(self, query: str, top_k: int = 10):  # pragma: no cover - trivial
        return []


def _invoke(tool, payload):
    """Invoke an Agent FunctionTool with JSON input synchronously."""

    raw = asyncio.run(tool.on_invoke_tool(None, json.dumps(payload)))
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "error", "message": raw}
    return raw


def _fake_agent_run(agent, prompt, max_turns=None):
    """Drive the review tool directly without invoking the model."""

    structure.Runner.last_agent = agent  # type: ignore[attr-defined]
    agent.test_results = {}

    review_tool = next(t for t in agent.tools if getattr(t, "name", "") == "review_user_requirement")
    update_tool = next(t for t in agent.tools if getattr(t, "name", "") == "update_user_requirement")

    agent.test_results["tool_names"] = [getattr(t, "name", "") for t in agent.tools]
    agent.test_results["update"] = _invoke(
        update_tool, {"index": 0, "status": "已完成", "mapped_node": []}
    )
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
        parsed_requirement={
            "requirements": [
                {
                    "description": "prefilled",
                    "intent": "review",
                    "inputs": [],
                    "constraints": [],
                    "status": "未开始",
                    "mapped_node": [],
                }
            ],
            "assumptions": [],
        },
        search_service=dummy_search,
        action_registry=action_registry,
        max_rounds=1,
    )

    # The fake runner attaches results to the agent instance for assertions.
    # We only need to validate the tool responses, not the empty workflow skeleton.
    test_results = structure.Runner.last_agent.test_results  # type: ignore[attr-defined]

    assert "plan_user_requirement" not in test_results["tool_names"]

    assert test_results["update"]["status"] == "ok"

    review_payload = test_results["review"]["requirement"]["requirements"][0]
    assert review_payload["status"] == "已完成"
    assert review_payload.get("mapped_node") == []

    assert workflow.get("workflow_name") == "unnamed_workflow"
    assert workflow.get("nodes") == []


def _fake_agent_run_without_completion(agent, prompt, max_turns=None):
    """Simulate a planning run that never marks requirements as completed."""

    structure.Runner.last_agent = agent  # type: ignore[attr-defined]


def test_planner_requires_completed_requirements(monkeypatch):
    monkeypatch.setattr(structure.Runner, "run_sync", _fake_agent_run_without_completion)

    dummy_search = _DummySearch()

    with pytest.raises(ValueError, match="需求状态未全部为已完成"):
        plan_workflow_structure_with_llm(
            parsed_requirement={
                "requirements": [
                    {
                        "description": "demo task",
                        "intent": "plan",
                        "inputs": ["query"],
                        "constraints": [],
                        "status": "进行中",
                        "mapped_node": [],
                    }
                ],
                "assumptions": [],
            },
            search_service=dummy_search,
            action_registry=[],
            max_rounds=1,
        )
