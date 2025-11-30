from velvetflow.planner.structure import _normalize_inline_from_references
from velvetflow.planner.workflow_builder import WorkflowBuilder


def test_normalizes_legacy_from_reference_strings():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="a1",
        node_type="action",
        action_id="demo_action",
        params={
            "direct": "__from__.result_of.start.value",
            "nested": [{"ref": "__from__.loop.item.id"}],
        },
    )

    changed = _normalize_inline_from_references(builder)

    assert changed is True
    assert builder.nodes["a1"]["params"]["direct"] == "{{result_of.start.value}}"
    assert builder.nodes["a1"]["params"]["nested"][0]["ref"] == "{{loop.item.id}}"


def test_normalizes_returns_false_when_no_legacy_references():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="a1",
        node_type="action",
        action_id="demo_action",
        params={"direct": "result_of.start.value"},
    )

    changed = _normalize_inline_from_references(builder)

    assert changed is False
    assert builder.nodes["a1"]["params"]["direct"] == "result_of.start.value"
