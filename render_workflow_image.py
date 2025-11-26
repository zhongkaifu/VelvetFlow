"""Render a workflow JSON file into a JPEG DAG image.

Usage:
    python render_workflow_image.py --workflow-json path/to/workflow.json --output workflow.jpg
"""

import argparse
import json
from pathlib import Path
from typing import Any

from velvetflow.models import Workflow
from velvetflow.visualization import render_workflow_dag


def load_workflow(path: Path) -> Workflow:
    with path.open("r", encoding="utf-8") as fp:
        data: Any = json.load(fp)
    return Workflow.model_validate(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render workflow JSON to JPEG")
    parser.add_argument(
        "--workflow-json",
        required=True,
        dest="workflow_json",
        type=Path,
        help="Path to the workflow JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output JPEG path (default: <workflow_json_basename>.jpg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = load_workflow(args.workflow_json)
    output_path = args.output or args.workflow_json.with_suffix(".jpg")
    saved = render_workflow_dag(workflow, output_path=str(output_path))
    print(f"Workflow image saved to: {saved}")


if __name__ == "__main__":
    main()
