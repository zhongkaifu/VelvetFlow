import copy

from velvetflow.planner import orchestrator
from velvetflow.planner.orchestrator import update_workflow_with_two_pass
from velvetflow.models import Workflow


class _DummySearch:
    def search(self, query: str, top_k: int = 10):  # pragma: no cover - trivial
        return []


def test_update_workflow_uses_existing_graph(monkeypatch):
    base_workflow = {
        "workflow_name": "demo",
        "description": "existing dag",
        "nodes": [
            {
                "id": "start",
                "type": "action",
                "action_id": "demo.start",
                "display_name": "Start",
                "params": {"message": "hello"},
            }
        ],
    }

    captured_existing = {}

    def _fake_plan(
        *,
        nl_requirement,
        search_service,
        action_registry,
        max_rounds,
        progress_callback,
        existing_workflow,
    ):
        captured_existing["value"] = copy.deepcopy(existing_workflow)
        updated_nodes = list(existing_workflow.get("nodes", [])) + [
            {
                "id": "summarize",
                "type": "action",
                "action_id": "demo.summary",
                "params": {"source": "{{ result_of.start.message }}"},
                "depends_on": ["start"],
            }
        ]
        return {
            "workflow_name": existing_workflow.get("workflow_name", "unnamed_workflow"),
            "description": existing_workflow.get("description", ""),
            "nodes": updated_nodes,
        }

    monkeypatch.setattr(orchestrator, "plan_workflow_structure_with_llm", _fake_plan)
    monkeypatch.setattr(
        orchestrator,
        "_ensure_actions_registered_or_repair",
        lambda workflow, **_: Workflow.model_validate(workflow)
        if isinstance(workflow, dict)
        else workflow,
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_and_repair_workflow",
        lambda current_workflow, **__: current_workflow,
    )

    updated = update_workflow_with_two_pass(
        existing_workflow=base_workflow,
        requirement="添加总结步骤",
        search_service=_DummySearch(),
        action_registry=[{"action_id": "demo.start", "params_schema": {}}, {"action_id": "demo.summary", "params_schema": {}}],
        max_rounds=1,
        max_repair_rounds=0,
    )

    assert captured_existing["value"]["workflow_name"] == "demo"
    assert captured_existing["value"]["nodes"][0]["id"] == "start"

    result_nodes = updated.model_dump(by_alias=True)["nodes"]
    assert any(node.get("id") == "start" for node in result_nodes)
    summarize = next(node for node in result_nodes if node.get("id") == "summarize")
    assert summarize.get("depends_on") == ["start"]
    assert summarize.get("params", {}).get("source") == "{{ result_of.start.message }}"

