# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

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
        display_name="Updated notify",
        params={"message": "new", "channel": "email"},
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

    builder.update_node("check", true_to_node=None, false_to_node="end")

    wf = builder.to_workflow()
    check = _find(wf["nodes"], "check")

    assert check["true_to_node"] is None
    assert check["false_to_node"] == "end"


def test_builder_update_node_supports_null_override():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="notify",
        node_type="action",
        action_id="hr.notify.v1",
        display_name="Notify",
        params={"message": "old"},
    )

    builder.update_node("notify", display_name=None, params=None)

    wf = builder.to_workflow()
    notify = _find(wf["nodes"], "notify")

    assert notify.get("display_name") is None
    assert notify.get("params") is None


def test_builder_tracks_parent_node_id():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="notify",
        node_type="action",
        action_id="hr.notify.v1",
        display_name="Notify",
        params={},
        parent_node_id="loop_1",
    )

    builder.update_node("notify", parent_node_id="loop_2")

    wf = builder.to_workflow()
    notify = _find(wf["nodes"], "notify")

    assert notify["parent_node_id"] is None


def test_loop_parent_places_node_into_body_subgraph():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="loop_1",
        node_type="loop",
        action_id=None,
        display_name="Loop",
        params={},
    )
    builder.add_node(
        node_id="child",
        node_type="action",
        action_id="demo.child",
        display_name=None,
        params={},
        parent_node_id="loop_1",
    )

    workflow = builder.to_workflow()
    loop_node = _find(workflow["nodes"], "loop_1")
    body_nodes = (loop_node.get("params") or {}).get("body_subgraph", {}).get("nodes", [])

    assert any(n.get("id") == "child" for n in body_nodes)
    assert not any(n.get("id") == "child" for n in workflow["nodes"])


def test_updating_parent_moves_node_into_loop_body():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="loop_1",
        node_type="loop",
        action_id=None,
        display_name=None,
        params={},
    )
    builder.add_node(
        node_id="task",
        node_type="action",
        action_id="demo.task",
        display_name=None,
        params={},
    )

    builder.update_node("task", parent_node_id="loop_1")

    workflow = builder.to_workflow()
    loop_node = _find(workflow["nodes"], "loop_1")
    body_nodes = (loop_node.get("params") or {}).get("body_subgraph", {}).get("nodes", [])

    assert any(n.get("id") == "task" for n in body_nodes)
    assert all(
        n.get("parent_node_id") is None
        for n in workflow["nodes"]
        if n.get("type") in {"action", "condition", "loop"}
    )


def test_non_loop_parent_normalizes_to_null():
    builder = WorkflowBuilder()
    builder.add_node(
        node_id="task",
        node_type="action",
        action_id="demo.task",
        display_name=None,
        params={},
        parent_node_id="not_a_loop",
    )

    workflow = builder.to_workflow()
    task_node = _find(workflow["nodes"], "task")

    assert task_node.get("parent_node_id") is None


