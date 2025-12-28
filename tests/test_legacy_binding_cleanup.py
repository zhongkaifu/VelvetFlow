from velvetflow.planner.legacy_cleanup import LegacyBindingError, strip_legacy_bindings_to_jinja


def test_strip_legacy_bindings_converts_from_dict():
    raw = {
        "workflow_name": "demo",
        "nodes": [
            {
                "id": "a",
                "type": "action",
                "action_id": "demo.echo",
                "params": {"text": {"__from__": "result_of.seed.output"}},
            }
        ],
    }

    cleaned = strip_legacy_bindings_to_jinja(raw)

    assert cleaned["nodes"][0]["params"]["text"] == "{{ result_of.seed.output }}"


def test_strip_legacy_bindings_rejects_agg_usage():
    raw = {"params": {"text": {"__from__": "result_of.a.out", "__agg__": "identity"}}}

    try:
        strip_legacy_bindings_to_jinja(raw)
    except LegacyBindingError as exc:
        assert "__agg__" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("__agg__ 应被拒绝")
