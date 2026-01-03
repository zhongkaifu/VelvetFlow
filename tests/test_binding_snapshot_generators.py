from velvetflow.bindings import BindingContext
from velvetflow.models import Workflow, Node
from jinja2.filters import sync_do_unique
from jinja2 import Environment


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


def test_binding_snapshot_handles_unhashable_unique_generator():
    workflow = _fake_workflow()
    # Jinja's unique filter generator can raise TypeError if it encounters
    # unhashable items (e.g., dict elements) while deduplicating. Ensure
    # snapshotting falls back gracefully.
    env = Environment()
    gen_value = sync_do_unique(env, [{"a": 1}, {"a": 2}], attribute=None)
    ctx = BindingContext(workflow, {"n1": {"value": gen_value}})

    snapshot = ctx.snapshot()

    # Falls back to the original sequence rather than raising during snapshot.
    assert snapshot["results"]["n1"]["value"] == [{"a": 1}, {"a": 2}]

