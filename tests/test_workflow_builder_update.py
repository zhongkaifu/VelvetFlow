from velvetflow.planner.workflow_builder import WorkflowBuilder


def _find(nodes, node_id):
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            return node
    raise AssertionError(f"node {node_id} not found")


def test_builder_update_node_overrides_fields():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="notify",
        node_type="action",
        action_id="hr.notify.v1",
        display_name="Notify",
        params={"message": "old"},
    )

    builder.update_node(
        "notify",
        {
            "display_name": "Updated notify",
            "params": {"message": "new", "channel": "email"},
        },
    )

    wf = builder.to_workflow()
    notify = _find(wf["nodes"], "notify")

    assert notify["display_name"] == "Updated notify"
    assert notify["params"] == {"message": "new", "channel": "email"}


def test_builder_update_node_allows_branch_overrides():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="check",
        node_type="condition",
        action_id=None,
        display_name=None,
        params={},
        true_to_node="yes",
        false_to_node="no",
    )

    builder.update_node("check", {"true_to_node": None, "false_to_node": "end"})

    wf = builder.to_workflow()
    check = _find(wf["nodes"], "check")

    assert check["true_to_node"] is None
    assert check["false_to_node"] == "end"
