"""从已保存的 workflow JSON 与模拟数据执行工作流。"""

import argparse
import json
import os

from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.models import Workflow

DEFAULT_WORKFLOW_JSON = "workflow_output.json"


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

    simulation_path = os.path.join(os.path.dirname(__file__), "velvetflow", "simulation_data.json")
    simulation_data = load_simulation_data(simulation_path)

    executor = DynamicActionExecutor(workflow, simulations=simulation_data)
    executor.run()


if __name__ == "__main__":
    main()
