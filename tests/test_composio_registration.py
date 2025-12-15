from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.composio import collect_composio_tool_specs, register_composio_tools
from tools.registry import GLOBAL_TOOL_REGISTRY, get_registered_tool
from velvetflow import action_registry


class _FakeComposioToolSet:
    def __init__(self, specs: Sequence[Mapping[str, Any]]):
        self._specs = list(specs)
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []

    def get_openai_tools(self, actions: Sequence[Any] | None = None) -> Iterable[Mapping[str, Any]]:
        if actions:
            names = {str(a) for a in actions}
            return [s for s in self._specs if s.get("function", {}).get("name") in names]
        return list(self._specs)

    def execute_action(self, action: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = {"action": action, "params": dict(params)}
        self.calls.append((action, dict(params)))
        return {"status": "ok", **payload}


def _sample_specs() -> list[Mapping[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message via chat integration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["channel", "text"],
                },
            },
            "action": "CHAT_SEND_MESSAGE",
        },
        {
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a task in the tracker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "assignee": {"type": "string", "nullable": True},
                    },
                    "required": ["title"],
                },
            },
            "action": "TASK_CREATE",
        },
    ]


def test_collect_composio_tool_specs_returns_specs():
    fake_toolset = _FakeComposioToolSet(_sample_specs())

    specs = collect_composio_tool_specs(toolset=fake_toolset)
    assert specs == _sample_specs()

    filtered = collect_composio_tool_specs(toolset=fake_toolset, selected_actions=["create_task"])
    assert [s["function"]["name"] for s in filtered] == ["create_task"]


def test_register_composio_tools_registers_tools_and_actions():
    original_actions = list(action_registry.BUSINESS_ACTIONS)
    original_tools = dict(GLOBAL_TOOL_REGISTRY._tools)

    fake_toolset = _FakeComposioToolSet(_sample_specs())

    try:
        registered = register_composio_tools(toolset=fake_toolset, namespace="composio_test")

        assert "composio_test.send_message" in registered
        assert "composio_test.create_task" in registered

        tool = get_registered_tool("composio_test.send_message")
        assert tool is not None

        output = tool(channel="#general", text="hello team")
        assert output["status"] == "ok"
        assert fake_toolset.calls[-1] == (
            "CHAT_SEND_MESSAGE",
            {"channel": "#general", "text": "hello team"},
        )

        action = action_registry.get_action_by_id("composio_test.send_message.v1")
        assert action is not None
        assert action["tool_name"] == "composio_test.send_message"
        assert action["arg_schema"]["properties"]["channel"]["type"] == "string"
    finally:
        GLOBAL_TOOL_REGISTRY._tools = original_tools
        action_registry.BUSINESS_ACTIONS[:] = original_actions


def test_register_composio_tools_can_filter_actions():
    original_actions = list(action_registry.BUSINESS_ACTIONS)
    original_tools = dict(GLOBAL_TOOL_REGISTRY._tools)

    fake_toolset = _FakeComposioToolSet(_sample_specs())

    try:
        registered = register_composio_tools(
            toolset=fake_toolset,
            namespace="composio_filtered",
            selected_actions=["create_task"],
        )

        assert registered == ["composio_filtered.create_task"]
        assert get_registered_tool("composio_filtered.send_message") is None

        action = action_registry.get_action_by_id("composio_filtered.create_task.v1")
        assert action is not None
        assert "assignee" in action["arg_schema"]["properties"]
    finally:
        GLOBAL_TOOL_REGISTRY._tools = original_tools
        action_registry.BUSINESS_ACTIONS[:] = original_actions


def test_fetch_composio_tools_cli_exports_payload(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "fetch_composio_tools.py"
    fake_module_dir = tmp_path / "fake_modules"
    fake_module_dir.mkdir()

    (fake_module_dir / "composio_openai.py").write_text(
        "\n".join(
            [
                f"SPECS = {_sample_specs()}",
                "",
                "class ComposioToolSet:",
                "    def __init__(self):",
                "        self.calls = []",
                "",
                "    def get_openai_tools(self, actions=None):",
                "        if actions:",
                "            names = {str(a) for a in actions}",
                "            return [s for s in SPECS if s['function']['name'] in names]",
                "        return list(SPECS)",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "payload.json"
    env = {
        **os.environ,
        "PYTHONPATH": f"{fake_module_dir}:{Path(__file__).resolve().parents[1]}",
    }

    result = subprocess.run(
        [sys.executable, str(script_path), "--actions", "create_task", "--output", str(output_path), "--pretty"],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["tool_count"] == 1
    assert payload["tools"][0]["function"]["name"] == "create_task"
    assert "Saved 1 tool definitions" in result.stdout
