from velvetflow.planner.structure import _hydrate_builder_from_workflow
from velvetflow.planner.workflow_builder import WorkflowBuilder


def _find(nodes, node_id):
    return next(node for node in nodes if node.get("id") == node_id)


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
