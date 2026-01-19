## Quickstart (English)

> 中文版本: [quickstart.md](quickstart.md)

This guide helps you install dependencies, build the action index, and run an end-to-end demo in minutes.

### Environment Setup
- Python 3.10+ (recommended to use a virtual environment):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install agents  # Planner relies on the OpenAI Agent SDK
  ```
- Or use [uv](https://github.com/astral-sh/uv) to sync dependencies from `pyproject.toml`:
  ```bash
  uv venv --python 3.10
  source .venv/bin/activate
  uv sync
  uv add agents
  ```
- Set your OpenAI credentials:
  ```bash
  export OPENAI_API_KEY="<your_api_key>"
  ```

### Build the Action Index (optional but recommended)
Rebuild the keyword + vector index when `tools/business_actions/` changes:
```bash
python build_action_index.py --output tools/action_index.json --model text-embedding-3-large
```

### Run the End-to-End Generation Demo
1. Run the generation script and provide a natural-language requirement (press Enter to use the default sample):
   ```bash
   python build_workflow.py
   ```
2. Outputs:
   - `workflow_output.json`: Workflow DSL after LLM planning and validation.
   - `workflow_dag.jpg`: Visualized DAG.

### Execute and Visualize an Existing Workflow
- Execute from JSON using simulated data:
  ```bash
  python execute_workflow.py --workflow-json workflow_output.json
  ```
- Render any Workflow JSON to JPEG:
  ```bash
  python render_workflow_image.py --workflow-json workflow_output.json --output workflow_dag.jpg
  ```

### Incrementally Update
- Append new requirements to an existing workflow:
  ```bash
  python update_workflow.py path/to/workflow.json --requirement "add an approval step" --output workflow_updated.json
  ```
