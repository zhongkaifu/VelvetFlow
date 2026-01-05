# 快速开始

> English version: [quickstart.en.md](quickstart.en.md)


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
