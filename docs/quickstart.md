<!-- Language toggle tabs -->
<style>
.lang-tabs { border: 1px solid #d0d7de; border-radius: 6px; padding: 0.75rem; }
.lang-tabs input[type="radio"] { display: none; }
.lang-tabs label { padding: 0.35rem 0.75rem; border: 1px solid #d0d7de; border-bottom: none; border-radius: 6px 6px 0 0; margin-right: 0.25rem; cursor: pointer; background: #f6f8fa; font-weight: 600; }
.lang-tabs input[type="radio"]:checked + label { background: #fff; border-bottom: 1px solid #fff; }
.lang-tabs .tabs-body { border-top: 1px solid #d0d7de; padding-top: 0.75rem; }
.lang-tabs .tab-content { display: none; }
#quickstart-lang-zh:checked ~ .tabs-body #quickstart-tab-zh,
#quickstart-lang-en:checked ~ .tabs-body #quickstart-tab-en { display: block; }
</style>
<div class="lang-tabs">
<input type="radio" id="quickstart-lang-zh" name="quickstart-lang" checked>
<label for="quickstart-lang-zh">中文</label>
<input type="radio" id="quickstart-lang-en" name="quickstart-lang">
<label for="quickstart-lang-en">English</label>
<div class="tabs-body">
<div class="tab-content" id="quickstart-tab-zh">
# 快速开始

本指南帮助你在几分钟内完成依赖安装、索引构建与端到端演示运行。

## 环境准备
- Python 3.10+，建议使用虚拟环境：
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install agents  # Planner 使用 OpenAI Agent SDK 的 Agent/Runner/function_tool
  ```
- 或使用 [uv](https://github.com/astral-sh/uv) 同步 `pyproject.toml` 中的依赖：
  ```bash
  uv venv --python 3.10
  source .venv/bin/activate
  uv sync
  uv add agents
  ```
- 设置 OpenAI 凭证：
  ```bash
  export OPENAI_API_KEY="<your_api_key>"
  ```

## 构建动作索引（可选但推荐）
如果更新过 `tools/business_actions/`，可离线重建关键词 + 向量索引：
```bash
python build_action_index.py --output tools/action_index.json --model text-embedding-3-large
```

## 运行端到端生成示例
1. 执行生成脚本，输入自然语言需求（直接回车使用默认示例）：
   ```bash
   python build_workflow.py
   ```
2. 产物：
   - `workflow_output.json`：LLM 规划 + 校验后的 Workflow DSL。
   - `workflow_dag.jpg`：可视化的 DAG。

## 执行与可视化已有 Workflow
- 从 JSON 执行并使用模拟数据：
  ```bash
  python execute_workflow.py --workflow-json workflow_output.json
  ```
- 将任意 Workflow JSON 渲染为 JPEG：
  ```bash
  python render_workflow_image.py --workflow-json workflow_output.json --output workflow_dag.jpg
  ```

## 校验或增量更新
- 校验并打印归一化 DSL：
  ```bash
  python validate_workflow.py path/to/workflow.json --print-normalized
  ```
- 在已有 workflow 上追加需求：
  ```bash
  python update_workflow.py path/to/workflow.json --requirement "新增审批环节" --output workflow_updated.json
  ```

</div>
<div class="tab-content" id="quickstart-tab-en">
## Quickstart (English)
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

### Validate or Incrementally Update
- Validate and print normalized DSL:
  ```bash
  python validate_workflow.py path/to/workflow.json --print-normalized
  ```
- Append new requirements to an existing workflow:
  ```bash
  python update_workflow.py path/to/workflow.json --requirement "add an approval step" --output workflow_updated.json
  ```

</div>
</div>
</div>
