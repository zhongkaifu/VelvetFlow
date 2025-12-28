# Legacy Binding Surface Area (__from__/__agg__)

下列模块和文档仍然保留了 legacy 绑定（`__from__` / `__agg__` 等）相关的代码路径或提示，可在彻底移除旧 DSL 时一并删除或重写。

## 运行时解析与校验
- `velvetflow/bindings.py`: 解析、验证 `__from__`/聚合绑定并提供各种路径正则；删除 legacy 后可精简绑定解析入口与错误消息。
- `velvetflow/executor/conditions.py`: 针对条件节点输入的 `__from__` 绑定做转换逻辑。
- `velvetflow/models.py`: `NodeParams`/`Binding` 数据模型仍包含 `__from__` 字段并在构造时做路径规范化。
- `velvetflow/verification/*.py`: 校验、Jinja 校正与修复建议中大量检测/修复 `__from__` 的分支（如 `binding_checks.py`, `workflow_validation.py`, `jinja_validation.py`, `node_rules.py`, `repair_tools.py`, `repair_suggestions.py`）。

## 规划阶段提示与工具
- `velvetflow/planner/params_tools.py`: 明确提示“严禁使用 __from__/__agg__”，但仍保留 legacy 术语，可在完全删除后改写文案与分支。
- `velvetflow/planner/orchestrator.py` & `structure.py`: 仍有针对 `__from__` 输入的检查、推断和错误信息；彻底弃用时可删除这些校验并精简依赖的 helper。
- `velvetflow/planner/repair.py` / `repair_tools.py`: 仍有关于禁止 fallback 到 `__from__/__agg__` 的提示和兼容处理。

## 前端与接口
- `webapp/js/core.js`: `bindingToTemplate` 等 helper 会将对象形式的 `__from__` 转换为 Jinja；如果后端不再接受 legacy 对象，可以移除这类分支。

## 文档与示例
- `README.md`、`docs/core_concepts.md`、`docs/workflow_dsl_schema.md`、`docs/troubleshooting.md`: 部分段落仍描述 legacy 绑定或示例，可在完成代码清理后同步删除。
- 示例与测试数据（如 `examples/async_suspend_resume/workflow_async_health.json`）仍包含 `__from__` 对象，可替换为 Jinja 字符串或移除。

## 测试覆盖
- 测试集中有大量针对 `__from__`/`__agg__` 的解析、校验、修复用例（如 `tests/test_reference_templates.py`, `tests/test_loop_body_validation.py`, `tests/test_aggregation_mini_expr.py` 等）；完成移除后需要整体重写或删除对应测试。

> 本清单仅列出主要入口，`rg "__from__"` 可继续审查残留。统一迁移到 Jinja 后，可按上述路径逐层删除解析/验证逻辑并更新文档、前端与测试。
