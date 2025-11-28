"""从已保存的 workflow JSON 与模拟数据执行工作流。"""

import argparse
import json
import os

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.ir import build_ir_from_workflow, generate_target_specs, materialize_workflow_from_ir, optimize_ir
from velvetflow.models import ValidationError, Workflow
from velvetflow.verification import validate_completed_workflow

DEFAULT_WORKFLOW_JSON = "workflow_output.json"


def load_workflow_from_file(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Workflow.model_validate(data)


def validate_workflow_for_execution(workflow: Workflow) -> list[ValidationError]:
    """Run structural + semantic checks before executing a workflow."""

    workflow_dict = workflow.model_dump(by_alias=True)
    return validate_completed_workflow(workflow_dict, BUSINESS_ACTIONS)


def compile_and_optimize(workflow: Workflow) -> tuple[Workflow, dict[str, object]]:
    """Normalize the workflow into IR, apply optimizations, and return targets."""

    ir = build_ir_from_workflow(workflow, action_registry=BUSINESS_ACTIONS)
    optimized_ir = optimize_ir(ir)
    targets = generate_target_specs(optimized_ir)
    optimized_dict = materialize_workflow_from_ir(optimized_ir)
    optimized_workflow = Workflow.model_validate(optimized_dict)
    return optimized_workflow, targets


def main():
    parser = argparse.ArgumentParser(description="从已保存的 JSON 执行 workflow")
    parser.add_argument(
        "--workflow-json",
        default=DEFAULT_WORKFLOW_JSON,
        help="工作流 JSON 文件路径，默认使用 workflow_output.json",
    )
    args = parser.parse_args()

    workflow_path = os.path.abspath(args.workflow_json)
    if not os.path.exists(workflow_path):
        print(f"未找到工作流文件：{workflow_path}")
        return

    try:
        workflow = load_workflow_from_file(workflow_path)
    except Exception as e:
        print(f"加载工作流 JSON 失败：{e}")
        return

    errors = validate_workflow_for_execution(workflow)
    if errors:
        print("workflow 校验失败，无法执行：")
        for idx, err in enumerate(errors, start=1):
            location_bits = []
            if err.node_id:
                location_bits.append(f"node={err.node_id}")
            if err.field:
                location_bits.append(f"field={err.field}")
            location = f" ({', '.join(location_bits)})" if location_bits else ""
            print(f"{idx}. [{err.code}]{location} {err.message}")
        return

    optimized_workflow, targets = compile_and_optimize(workflow)
    summary_bits = []
    pruned = targets.get("native", {}).get("pruned_nodes") if isinstance(targets, dict) else None
    if pruned:
        summary_bits.append(f"裁剪未使用节点: {pruned}")
    layers = targets.get("native", {}).get("parallel_layers") if isinstance(targets, dict) else None
    if layers:
        summary_bits.append(f"拓扑层级={len(layers)}, 可并行批次={layers}")
    if summary_bits:
        print("IR 优化结果: " + "; ".join(summary_bits))

    simulation_path = os.path.join(os.path.dirname(__file__), "velvetflow", "simulation_data.json")
    simulation_data = load_simulation_data(simulation_path)

    executor = DynamicActionExecutor(optimized_workflow, simulations=simulation_data)
    executor.run()


if __name__ == "__main__":
    main()
