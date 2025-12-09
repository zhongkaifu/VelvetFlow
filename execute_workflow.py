# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""从已保存的 workflow JSON 与模拟数据执行工作流。

支持两种模式：
- 默认运行模式：执行 workflow，若遇到异步工具挂起，会在返回时把 suspension 持久化到文件。
- 恢复模式：通过此前保存的 suspension 文件恢复执行，可选提供异步工具的结果文件。
"""

import argparse
import json
import os

from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.executor.async_runtime import WorkflowSuspension
from velvetflow.metrics import RunManager
from velvetflow.models import Workflow

DEFAULT_WORKFLOW_JSON = "workflow_output.json"
DEFAULT_SUSPENSION_JSON = "workflow_suspension.json"


def load_workflow_from_file(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Workflow.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="从已保存的 JSON 执行 workflow")
    parser.add_argument(
        "--workflow-json",
        default=DEFAULT_WORKFLOW_JSON,
        help="工作流 JSON 文件路径，默认使用 workflow_output.json",
    )
    parser.add_argument(
        "--simulation-file",
        default=None,
        help="模拟数据 JSON 文件路径，默认使用仓库根目录下的 simulation_data.json",
    )
    parser.add_argument(
        "--suspension-file",
        default=DEFAULT_SUSPENSION_JSON,
        help="异步挂起时保存 suspension 的文件路径（也可用于恢复模式的默认输入）",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="从指定的 suspension JSON 文件恢复 workflow",
    )
    parser.add_argument(
        "--tool-result-file",
        default=None,
        help="异步工具完成后结果的 JSON 文件路径，恢复模式可选",
    )
    args = parser.parse_args()

    workflow_path = os.path.abspath(args.workflow_json)
    if not os.path.exists(workflow_path):
        print(f"未找到工作流文件：{workflow_path}")
        return None

    try:
        workflow = load_workflow_from_file(workflow_path)
    except Exception as e:
        print(f"加载工作流 JSON 失败：{e}")
        return None

    simulation_path = (
        os.path.abspath(args.simulation_file)
        if args.simulation_file
        else os.path.join(os.path.dirname(__file__), "simulation_data.json")
    )
    simulation_data = load_simulation_data(simulation_path)

    suspension_path = os.path.abspath(args.resume_from or args.suspension_file)
    tool_result = None
    if args.tool_result_file:
        tool_result_path = os.path.abspath(args.tool_result_file)
        with open(tool_result_path, "r", encoding="utf-8") as f:
            tool_result = json.load(f)

    with RunManager(workflow_name=workflow.workflow_name) as run_manager:
        executor = DynamicActionExecutor(
            workflow, simulations=simulation_data, run_manager=run_manager
        )

        if args.resume_from:
            suspension = WorkflowSuspension.load_from_file(suspension_path)
            result = executor.resume_from_suspension(
                suspension, tool_result=tool_result
            )
        else:
            result = executor.run()

        if isinstance(result, WorkflowSuspension):
            os.makedirs(os.path.dirname(suspension_path) or ".", exist_ok=True)
            result.save_to_file(suspension_path)
            print(f"workflow 挂起，已保存至：{suspension_path}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

        return result


if __name__ == "__main__":
    main()
