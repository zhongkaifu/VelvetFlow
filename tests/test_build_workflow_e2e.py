import json
from pathlib import Path

import build_workflow
from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.models import Node, Workflow


def _make_stub_workflow(action_id: str) -> Workflow:
    node = Node.model_validate({
        "id": "stub_action",
        "type": "action",
        "action_id": action_id,
        "params": {"message": "hello"},
    })
    return Workflow(nodes=[node], workflow_name="stub_workflow", description="test stub")


def test_build_workflow_main_persists_outputs(monkeypatch, tmp_path: Path):
    # Ensure OpenAI checks are bypassed.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Pick a known action id from the registry so tool-gap guidance is skipped.
    first_action_id = next(action["action_id"] for action in BUSINESS_ACTIONS if action.get("action_id"))

    stub_workflow = _make_stub_workflow(first_action_id)

    def fake_plan_workflow(user_nl: str):
        return stub_workflow

    def fake_render(workflow, output_path: str):
        path = Path(output_path)
        path.write_text("stub image", encoding="utf-8")
        return str(path)

    # Avoid interactive prompts.
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    monkeypatch.setattr(build_workflow, "plan_workflow", fake_plan_workflow)
    monkeypatch.setattr(build_workflow, "render_workflow_dag", fake_render)

    # Run in an isolated directory so persisted artifacts don't leak between tests.
    monkeypatch.chdir(tmp_path)

    build_workflow.main()

    workflow_json = tmp_path / build_workflow.DEFAULT_WORKFLOW_JSON
    dag_image = tmp_path / "workflow_dag.jpg"

    assert workflow_json.exists(), "workflow_output.json should be created"
    assert dag_image.exists(), "workflow_dag.jpg should be created"

    payload = json.loads(workflow_json.read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "stub_workflow"
    assert payload["nodes"], "workflow nodes should be persisted"
