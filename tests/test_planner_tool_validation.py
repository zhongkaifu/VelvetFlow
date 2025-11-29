import copy

from velvetflow.planner.structure import _validate_modified_nodes
from velvetflow.planner.workflow_builder import WorkflowBuilder


def test_validate_nodes_rejects_missing_reference():
    builder = WorkflowBuilder()
    builder.add_node("fetch", node_type="action", action_id="search")
    previous_node_ids = set(copy.deepcopy(builder.nodes).keys())

    builder.add_node(
        "notify",
        node_type="action",
        action_id="notify_user",
        params={"content": {"__from__": "result_of.unknown.summary"}},
    )

    errors = _validate_modified_nodes(
        builder=builder, modified_node_ids=["notify"], previous_node_ids=previous_node_ids
    )

    assert any(err["type"] == "missing_references" for err in errors)


def test_validate_nodes_rejects_isolated_additions():
    builder = WorkflowBuilder()
    builder.add_node("fetch", node_type="action", action_id="search")
    previous_node_ids = set(copy.deepcopy(builder.nodes).keys())

    builder.add_node("summary", node_type="action", action_id="summarize")

    errors = _validate_modified_nodes(
        builder=builder, modified_node_ids=["summary"], previous_node_ids=previous_node_ids
    )

    assert any(err["type"] == "missing_dependencies" for err in errors)


def test_validate_nodes_accepts_connected_updates():
    builder = WorkflowBuilder()
    builder.add_node("fetch", node_type="action", action_id="search")
    previous_node_ids = set(copy.deepcopy(builder.nodes).keys())

    builder.add_node(
        "notify",
        node_type="action",
        action_id="notify_user",
        params={"content": {"__from__": "result_of.fetch.items"}},
    )

    errors = _validate_modified_nodes(
        builder=builder, modified_node_ids=["notify"], previous_node_ids=previous_node_ids
    )

    assert errors == []
