from velvetflow.models import ValidationError, Workflow
from velvetflow.planner import repair


def test_missing_loop_body_gets_stubbed_in_fallback(monkeypatch):
    """Fallback repair should synthesize a loop body when LLM fixing fails."""

    broken_workflow = {
        "workflow_name": "missing_body",
        "nodes": [
            {
                "id": "loop_demo",
                "type": "loop",
                "params": {"loop_kind": "for_each", "source": "upstream.items"},
            }
        ],
    }

    errors = [
        ValidationError(
            code="INVALID_LOOP_BODY",
            node_id="loop_demo",
            field="body_subgraph",
            message="loop 节点 'loop_demo' 必须提供 body_subgraph。",
        )
    ]

    def _raise_on_llm(**_kwargs):  # noqa: ANN001
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(repair, "repair_workflow_with_llm", _raise_on_llm)

    repaired = repair._repair_with_llm_and_fallback(  # noqa: SLF001
        broken_workflow=broken_workflow,
        validation_errors=errors,
        action_registry=[],
        search_service=None,
        reason="test",
    )

    assert isinstance(repaired, Workflow)
    loop_node = next(n for n in repaired.nodes if n.id == "loop_demo")
    body = loop_node.params.get("body_subgraph")

    assert body and body.get("nodes")
    node_ids = {n.get("id") for n in body.get("nodes")}
    assert body.get("entry") in node_ids
    assert body.get("exit") in node_ids
