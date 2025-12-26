from velvetflow.planner.structure import _hydrate_builder_from_workflow
from velvetflow.planner.workflow_builder import WorkflowBuilder


def _find(nodes, node_id):
    return next(node for node in nodes if node.get("id") == node_id)


def _collect_all_nodes(workflow):
    found = {}

    def _walk(nodes):
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            nid = node.get("id")
            if isinstance(nid, str):
                found[nid] = node

            if node.get("type") == "loop":
                body_nodes = (node.get("params") or {}).get("body_subgraph", {}).get("nodes") or []
                _walk(body_nodes)

            if node.get("type") == "parallel":
                branches = (node.get("params") or {}).get("branches") or []
                for branch in branches:
                    _walk(branch.get("sub_graph_nodes"))

    _walk(workflow.get("nodes") or [])
    return found


def test_hydrate_builder_reconstructs_existing_workflow():
    existing_workflow = {
        "workflow_name": "demo",
        "description": "existing dag",
        "nodes": [
            {
                "id": "start",
                "type": "action",
                "action_id": "demo.start",
                "params": {"message": "hello"},
            },
            {
                "id": "loop_1",
                "type": "loop",
                "display_name": "Iterate",
                "params": {
                    "loop_kind": "sequential",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "child",
                                "type": "action",
                                "action_id": "demo.child",
                                "params": {"value": "{{ result_of.start.message }}"},
                                "depends_on": ["start"],
                            }
                        ],
                        "entry": "child",
                    },
                    "exports": {"items": "{{ result_of.child.status }}"},
                },
            },
        ],
    }

    builder = WorkflowBuilder()
    _hydrate_builder_from_workflow(builder=builder, workflow=existing_workflow)

    workflow = builder.to_workflow()

    assert workflow["workflow_name"] == "demo"
    assert workflow["description"] == "existing dag"

    loop_node = _find(workflow["nodes"], "loop_1")
    body_nodes = (loop_node.get("params") or {}).get("body_subgraph", {}).get("nodes") or []

    assert _find(workflow["nodes"], "start")["params"] == {"message": "hello"}
    assert any(node.get("id") == "child" for node in body_nodes)
    assert not any(node.get("id") == "child" for node in workflow["nodes"])

    child_node = _find(body_nodes, "child")
    assert child_node.get("depends_on") == ["start"]
    assert loop_node.get("params", {}).get("exports") == {"items": "{{ result_of.child.status }}"}


def test_hydrate_builder_preserves_branches_and_edges():
    existing_workflow = {
        "workflow_name": "branches",
        "description": "with parallel branches",
        "nodes": [
            {"id": "start", "type": "action", "action_id": "demo.start", "params": {"message": "hi"}},
            {
                "id": "check",
                "type": "condition",
                "params": {"expression": "{{ result_of.start.message != '' }}"},
            },
            {
                "id": "route",
                "type": "switch",
                "params": {"source": "{{ result_of.start }}", "field": "channel"},
                "cases": [{"case": "email", "to_node": "parallel_group"}],
                "default_to_node": "fallback",
            },
            {"id": "fallback", "type": "action", "action_id": "demo.fallback"},
            {
                "id": "parallel_group",
                "type": "parallel",
                "params": {
                    "branches": [
                        {
                            "id": "email",
                            "entry_node": "prepare_email",
                            "sub_graph_nodes": [
                                {
                                    "id": "prepare_email",
                                    "type": "action",
                                    "action_id": "demo.prepare_email",
                                    "params": {"subject": "{{ result_of.start.message }}"},
                                },
                                {
                                    "id": "loop_email",
                                    "type": "loop",
                                    "params": {
                                        "loop_kind": "for_each",
                                        "source": "{{ result_of.start.recipients }}",
                                        "item_alias": "user",
                                        "body_subgraph": {
                                            "nodes": [
                                                {
                                                    "id": "send_one",
                                                    "type": "action",
                                                    "action_id": "demo.send_email",
                                                    "params": {
                                                        "user": "{{ loop.user }}",
                                                        "content": "{{ result_of.prepare_email.body }}",
                                                    },
                                                }
                                            ]
                                        },
                                    },
                                },
                            ],
                        }
                    ]
                },
            },
        ],
        "edges": [
            {"from": "start", "to": "check"},
            {"from": "check", "to": "route", "condition": True},
            {"from": "route", "to": "parallel_group", "condition": "email"},
            {"from": "route", "to": "fallback", "condition": "default"},
        ],
    }

    builder = WorkflowBuilder()
    _hydrate_builder_from_workflow(builder=builder, workflow=existing_workflow)

    workflow = builder.to_workflow()
    nodes = workflow["nodes"]
    all_nodes = _collect_all_nodes(workflow)

    assert workflow["workflow_name"] == "branches"
    assert _find(nodes, "check").get("true_to_node") == "route"
    assert set(_find(nodes, "route").get("depends_on")) == {"check", "start"}
    assert _find(nodes, "parallel_group").get("depends_on") == ["route"]

    parallel_node = _find(nodes, "parallel_group")
    branches = (parallel_node.get("params") or {}).get("branches", [])
    assert branches and isinstance(branches[0].get("sub_graph_nodes"), list)
    assert any(node.get("id") == "prepare_email" for node in branches[0].get("sub_graph_nodes"))

    loop_node = next(node for node in branches[0].get("sub_graph_nodes") if node.get("id") == "loop_email")
    loop_body_nodes = (loop_node.get("params") or {}).get("body_subgraph", {}).get("nodes") or []

    assert any(node.get("id") == "send_one" for node in loop_body_nodes)
    assert "send_one" in all_nodes
    assert set(all_nodes["send_one"].get("depends_on", [])) == {"prepare_email"}


def test_hydrate_builder_handles_condition_nodes_in_loop_body():
    existing_workflow = {
        "workflow_name": "conditions",
        "description": "loop with condition branches",
        "nodes": [
            {
                "id": "start",
                "type": "action",
                "action_id": "demo.start",
                "params": {"ready": True},
            },
            {
                "id": "loop_cond",
                "type": "loop",
                "params": {
                    "loop_kind": "sequential",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "gate",
                                "type": "condition",
                                "params": {"expression": "{{ result_of.start.ready }}"},
                            },
                            {"id": "branch_true", "type": "action", "action_id": "demo.true"},
                            {"id": "branch_false", "type": "action", "action_id": "demo.false"},
                        ],
                        "entry": "gate",
                    },
                },
            },
        ],
        "edges": [
            {"from": "start", "to": "loop_cond"},
            {"from": "start", "to": "gate"},
            {"from": "gate", "to": "branch_true", "condition": True},
            {"from": "gate", "to": "branch_false", "condition": False},
        ],
    }

    builder = WorkflowBuilder()
    _hydrate_builder_from_workflow(builder=builder, workflow=existing_workflow)

    workflow = builder.to_workflow()
    loop_node = _find(workflow["nodes"], "loop_cond")
    body_nodes = (loop_node.get("params") or {}).get("body_subgraph", {}).get("nodes") or []

    assert not any(node.get("id") == "gate" for node in workflow["nodes"])

    gate = _find(body_nodes, "gate")
    true_branch = _find(body_nodes, "branch_true")
    false_branch = _find(body_nodes, "branch_false")

    assert gate.get("params") == {"expression": "{{ result_of.start.ready }}"}
    assert gate.get("true_to_node") == "branch_true"
    assert gate.get("false_to_node") == "branch_false"
    assert gate.get("depends_on") == ["start"]

    assert true_branch.get("parent_node_id") == "loop_cond"
    assert false_branch.get("parent_node_id") == "loop_cond"
    assert true_branch.get("depends_on") == ["gate"]
    assert false_branch.get("depends_on") == ["gate"]
