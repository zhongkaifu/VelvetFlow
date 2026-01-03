import pytest

from velvetflow.planner import structure
from velvetflow.planner.structure import plan_workflow_structure_with_llm


class _DummySearch:
    def search(self, query: str, top_k: int = 10):  # pragma: no cover - trivial
        return []
def _fake_agent_run(agent, prompt, max_turns=None):
    """Record tool exposure without invoking the model."""

    structure.Runner.last_agent = agent  # type: ignore[attr-defined]
    agent.test_results = {"tool_names": [getattr(t, "name", "") for t in agent.tools]}


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_user_requirements_require_status_and_are_reviewable(monkeypatch):
    # Patch the agent runner to avoid real model calls and to capture the tool list.
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

    # Legacy requirement review/update helpers are no longer exposed.
    assert "review_user_requirement" not in test_results["tool_names"]
    assert "update_user_requirement" not in test_results["tool_names"]

    assert workflow.get("workflow_name") == "unnamed_workflow"
    assert workflow.get("nodes") == []


def test_planner_allows_incomplete_requirements(monkeypatch):
    monkeypatch.setattr(structure.Runner, "run_sync", _fake_agent_run)

    dummy_search = _DummySearch()

    workflow = plan_workflow_structure_with_llm(
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

    assert workflow.get("workflow_name") == "unnamed_workflow"
