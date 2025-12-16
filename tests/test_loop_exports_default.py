from velvetflow.executor.loops import LoopExecutionMixin


class _DummyLoopExecutor(LoopExecutionMixin):
    def __init__(self) -> None:
        self.workflow = {"workflow_name": "dummy", "nodes": []}

    def _get_field_value(self, obj, field):
        return obj.get(field) if isinstance(obj, dict) else None


def test_items_list_exports_use_last_action_as_source():
    dummy = _DummyLoopExecutor()
    body_nodes = [
        {"id": "check", "type": "condition"},
        {"id": "collect_high_temp", "type": "action"},
    ]
    default_source = dummy._infer_default_export_source(body_nodes, {})

    items_out = []
    dummy._apply_loop_exports(
        ["employee_id"],
        None,
        {"collect_high_temp": {"employee_id": "e1", "temperature": 39}},
        items_out,
        {},
        {},
        default_source,
    )

    assert items_out == [{"employee_id": "e1"}]


def test_items_mapping_without_from_node_uses_default():
    dummy = _DummyLoopExecutor()
    body_nodes = [
        {"id": "check", "type": "condition"},
        {"id": "collect_high_temp", "type": "action"},
    ]
    default_source = dummy._infer_default_export_source(body_nodes, {})

    items_out = []
    dummy._apply_loop_exports(
        {"fields": ["employee_id"]},
        None,
        {"collect_high_temp": {"employee_id": "e2", "temperature": 37}},
        items_out,
        {},
        {},
        default_source,
    )

    assert items_out == [{"employee_id": "e2"}]


def test_items_default_from_node_with_jinja_literal_type():
    dummy = _DummyLoopExecutor()
    body_nodes = [
        {"id": "check", "type": "{{ 'condition' }}"},
        {"id": "collect_high_temp", "type": "{{ 'action' }}"},
    ]
    default_source = dummy._infer_default_export_source(body_nodes, {})

    items_out = []
    dummy._apply_loop_exports(
        ["employee_id"],
        None,
        {"collect_high_temp": {"employee_id": "e3", "temperature": 40}},
        items_out,
        {},
        {},
        default_source,
    )

    assert items_out == [{"employee_id": "e3"}]
