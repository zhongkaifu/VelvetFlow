import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from velvetflow.bindings import BindingContext
from velvetflow.models import Workflow


def _build_ctx(results):
    workflow = Workflow.model_validate(
        {
            "workflow_name": "binding-errors",
            "nodes": [
                {"id": "a", "type": "action", "params": {}},
            ],
            "edges": [],
        }
    )
    return BindingContext(workflow, results)


def test_missing_result_path_raises_descriptive_error():
    ctx = _build_ctx(results={})

    with pytest.raises(KeyError, match=r"result_of\.a: 上游节点未执行或没有结果"):
        ctx.resolve_binding({"__from__": "result_of.a.output"})


def test_missing_field_on_dict_gives_precise_path():
    ctx = _build_ctx(results={"a": {"output": {"value": 1}}})

    with pytest.raises(KeyError, match=r"result_of\.a\.output\.missing: 字段不存在"):
        ctx.resolve_binding({"__from__": "result_of.a.output.missing"})


def test_type_mismatch_reports_actual_type():
    ctx = _build_ctx(results={"a": {"output": 3}})

    with pytest.raises(TypeError, match=r"值类型为 int，不支持字段 'id' 访问"):
        ctx.resolve_binding({"__from__": "result_of.a.output.id"})
