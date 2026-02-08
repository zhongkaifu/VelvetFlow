import asyncio
import json
import types

from velvetflow.planner import structure
from velvetflow.planner.structure import plan_workflow_structure_with_llm


class _DummySearch:
    def search(self, query: str, top_k: int = 10):  # pragma: no cover - trivial
        return []

    def get_action_by_id(self, action_id: str):
        if action_id == "demo.action":
            return {
                "action_id": "demo.action",
                "name": "Demo",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "foo": {"type": "integer"},
                        "nested": {
                            "type": "object",
                            "properties": {
                                "left": {"type": "string"},
                                "right": {"type": "string"},
                            },
                        },
                    },
                },
                "output_schema": {"type": "object"},
            }
        return None


def _find_tool(agent, name):
    for tool in agent.tools:
        if getattr(tool, "name", "") == name:
            return tool
    raise AssertionError(f"tool {name} not found")


def test_update_action_node_only_updates_specified_params(monkeypatch):
    def fake_run_sync(agent, prompt, max_turns):  # pragma: no cover - behavior asserted
        ctx = types.SimpleNamespace(run_input=[], message_history=[])
        update_tool = _find_tool(agent, "update_action_node")
        asyncio.run(
            update_tool.on_invoke_tool(
                ctx,
                json.dumps(
                    {
                        "id": "a1",
                        "params": {"nested": {"right": "new"}},
                    },
                    ensure_ascii=False,
                ),
            )
        )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(structure.Runner, "run_sync", fake_run_sync)

    existing_workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [
            {
                "id": "a1",
                "type": "action",
                "action_id": "demo.action",
                "display_name": "A1",
                "params": {"foo": 1, "nested": {"left": "keep", "right": "old"}},
            }
        ],
    }

    workflow = plan_workflow_structure_with_llm(
        parsed_requirement={"requirements": [{"description": "partial update", "intent": "update", "inputs": [], "constraints": [], "status": "进行中", "mapped_node": []}], "assumptions": []},
        search_service=_DummySearch(),
        action_registry=[
            {
                "action_id": "demo.action",
                "name": "Demo",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "foo": {"type": "integer"},
                        "nested": {
                            "type": "object",
                            "properties": {
                                "left": {"type": "string"},
                                "right": {"type": "string"},
                            },
                        },
                    },
                },
                "output_schema": {"type": "object"},
            }
        ],
        existing_workflow=existing_workflow,
        max_rounds=1,
    )

    updated = next(node for node in workflow["nodes"] if node["id"] == "a1")
    assert updated["params"]["foo"] == 1
    assert updated["params"]["nested"]["left"] == "keep"
    assert updated["params"]["nested"]["right"] == "new"
