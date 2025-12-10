import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from velvetflow.models import Workflow
from velvetflow.verification.llm_repair import (
    LLMClientProtocol,
    LLMRepairConfig,
    repair_workflow_types_with_local_and_llm,
)


class _NoCallLLM(LLMClientProtocol):
    def complete_json(self, system_prompt: str, user_prompt: str):  # pragma: no cover - protocol
        raise AssertionError("LLM should not be invoked when local repair succeeds")


class _EchoFixLLM(LLMClientProtocol):
    def __init__(self, fixed_workflow: dict):
        self.fixed_workflow = fixed_workflow
        self.called = False

    def complete_json(self, system_prompt: str, user_prompt: str):
        self.called = True
        return {"workflow": json.loads(json.dumps(self.fixed_workflow))}


def _build_action_registry():
    actions = [
        {
            "action_id": "producer",
            "arg_schema": {"type": "object", "properties": {}},
            "output_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "value": {"type": "number"}},
            },
        },
        {
            "action_id": "consumer",
            "arg_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "output_schema": {"type": "object"},
        },
    ]
    return {a["action_id"]: a for a in actions}


def test_local_repair_runs_before_llm():
    action_registry = _build_action_registry()
    workflow = Workflow.model_validate(
        {
            "workflow_name": "demo",
            "nodes": [
                {"id": "producer", "type": "action", "action_id": "producer", "params": {}},
                {
                    "id": "consumer",
                    "type": "action",
                    "action_id": "consumer",
                    "params": {"text": {"__from__": "producer.output"}},
                },
            ],
            "edges": [],
        }
    )

    result = repair_workflow_types_with_local_and_llm(
        workflow=workflow,
        action_registry=action_registry,
        llm_client=_NoCallLLM(),
    )

    assert result.success is True
    consumer = next(node for node in result.workflow.nodes if node.id == "consumer")
    assert consumer.params["text"]["__from__"] == "producer.output.text"
    assert result.local_repair is not None and not result.local_repair.remaining_errors


def test_llm_runs_when_local_repair_insufficient():
    action_registry = _build_action_registry()
    workflow_dict = {
        "workflow_name": "demo",
        "nodes": [
            {
                "id": "producer",
                "type": "action",
                "action_id": "producer",
                "params": {},
            },
            {
                "id": "consumer",
                "type": "action",
                "action_id": "consumer",
                # 绑定到 number，将触发类型错误，且本地修复无法处理
                "params": {"text": {"__from__": "producer.output.value"}},
            },
        ],
        "edges": [],
    }
    workflow = Workflow.model_validate(workflow_dict)

    fixed_workflow = workflow_dict | {"nodes": [*workflow_dict["nodes"]]}
    fixed_workflow["nodes"][1] = {
        **fixed_workflow["nodes"][1],
        "params": {"text": "fixed"},
    }

    llm_client = _EchoFixLLM(fixed_workflow)

    result = repair_workflow_types_with_local_and_llm(
        workflow=workflow,
        action_registry=action_registry,
        llm_client=llm_client,
        config=LLMRepairConfig(max_rounds=1, pretty_workflow_json=False),
    )

    assert result.success is True
    assert llm_client.called is True
    consumer = next(node for node in result.workflow.nodes if node.id == "consumer")
    assert consumer.params["text"] == "fixed"
    assert result.local_repair is not None and result.local_repair.remaining_errors
