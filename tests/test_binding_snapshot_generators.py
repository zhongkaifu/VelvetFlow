from velvetflow.bindings import BindingContext
from velvetflow.models import Workflow, Node


def _fake_workflow() -> Workflow:
    return Workflow(
        workflow_name="demo",
        nodes=[Node(id="n1", type="action", action_id="noop", params={})],
    )


def test_binding_snapshot_materializes_generators():
    workflow = _fake_workflow()
    # generator that would normally fail copy.deepcopy
    gen_value = (i for i in range(3))
    ctx = BindingContext(workflow, {"n1": {"value": gen_value}})

    snapshot = ctx.snapshot()

    assert snapshot["results"]["n1"]["value"] == [0, 1, 2]
    restored = BindingContext.from_snapshot(workflow, snapshot)
    assert restored.results == {"n1": {"value": [0, 1, 2]}}

