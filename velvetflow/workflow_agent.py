# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""OpenAI Agent 集成：为构建/编排/校验/修复/更新 workflow 提供工具函数。

本模块通过 OpenAI Agent SDK（``Agent``）暴露一组可调用的工具：
- ``build_workflow``：基于自然语言需求生成 workflow DSL；
- ``validate_workflow``：对给定 DSL 做静态校验并返回错误列表；
- ``repair_workflow``：在校验失败时调用 LLM 进行自动修复；
- ``update_workflow``：在原有 workflow 上增量更新新需求。

使用方法（伪代码）::

    from openai import OpenAI
    from velvetflow.workflow_agent import create_workflow_agent

    client = OpenAI()
    agent = create_workflow_agent(client=client)
    response = agent.run("请为 HR 健康检查场景构建流程")

如果当前环境未安装支持 ``Agent`` 的 OpenAI Agent SDK，将在构建阶段抛出
``AgentSdkNotInstalled`` 并提示安装/升级。为了避免导入时立刻失败，Agent
相关的依赖仅在 ``create_workflow_agent`` 被调用时按需加载。
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from velvetflow.action_registry import (
    BUSINESS_ACTIONS,
    load_actions_from_path,
    validate_actions,
)
from velvetflow.config import OPENAI_MODEL
from velvetflow.models import ValidationError, Workflow
from velvetflow.planner import plan_workflow_with_two_pass, update_workflow_with_two_pass
from velvetflow.planner.repair import repair_workflow_with_llm
from velvetflow.search import HybridActionSearchService, build_search_service_from_actions
from velvetflow.workflow_parser import parse_workflow_source
from velvetflow.verification import validate_completed_workflow
from validate_workflow import validate_workflow_data

DEFAULT_AGENT_INSTRUCTIONS = """
你是 VelvetFlow Workflow Agent，负责用业务动作库来规划与修复工作流。
当用户描述需求时，请调用 build_workflow 生成 DSL；如果用户给出了现有 DSL，
先使用 validate_workflow 做静态检查，必要时调用 repair_workflow 尝试自动修复；
当用户提出增量需求时，使用 update_workflow 在原有 DSL 上更新。所有工具都会
返回结构化 JSON，便于后续推理与渲染。
"""


INTERNAL_AGENT_INSTRUCTIONS = """
你是 VelvetFlow Workflow Internal Agent。你必须基于提供的工具完成任务：
- build_workflow(requirement): 为自然语言需求规划完整的 workflow DSL。
- validate_workflow(workflow_raw): 对 workflow 进行静态校验。
- repair_workflow(workflow_raw): 针对校验失败的 workflow 自动修复。
- update_workflow(workflow_raw, requirement): 按新需求更新已有 workflow。

每次用户都会明确指出需要调用的工具与参数。请严格按照用户指定的工具调用，
并直接返回工具输出的 JSON，不要添加解释或额外文本。
"""


class AgentSdkNotInstalled(RuntimeError):
    """Raised when the OpenAI Agent SDK cannot be imported."""


@dataclass
class WorkflowAgentConfig:
    """Configuration for creating an OpenAI Agent bound to VelvetFlow tools."""

    model: str = OPENAI_MODEL
    instructions: str = DEFAULT_AGENT_INSTRUCTIONS
    action_registry_path: Path | None = None
    max_plan_rounds: int = 50
    max_repair_rounds: int = 3


class _FallbackAgent:
    """Lightweight fallback when Agent SDK is unavailable.

    This class simply routes ``run`` to the first provided tool to avoid
    hard failures in environments缺失 Agent SDK. It is primarily used by CLI
    调用，以便在未安装 Agent SDK 的情况下依然可运行核心流程。
    """

    _is_fallback = True

    def __init__(self, *, tools: list[Callable[..., Any]], **_: Any) -> None:
        if not tools:
            raise ValueError("FallbackAgent requires at least one tool")
        self.tools = tools

    def run(self, prompt: str):  # noqa: D401 - simple passthrough
        # 尽力从 prompt 中推断要用的工具；若无法判断则使用第一个
        lower = prompt.lower()
        preferred_order = [
            ("build_workflow", lambda name: "build_workflow" in name),
            ("validate_workflow", lambda name: "validate" in name),
            ("repair_workflow", lambda name: "repair" in name),
            ("update_workflow", lambda name: "update" in name),
        ]

        def _find_tool() -> Callable[..., Any]:
            for keyword, matcher in preferred_order:
                if keyword in lower:
                    for t in self.tools:
                        tname = getattr(t, "__name__", "").lower()
                        if matcher(tname):
                            return t
            return self.tools[0]

        tool = _find_tool()
        return tool(prompt)


def _import_openai_agent() -> tuple[type, Callable[[Callable[..., Any]], Any] | None, bool, Any | None, bool]:
    """Load Agent SDK lazily; fall back to a local stub when unavailable.

    Returns (agent_cls, tool_wrapper, used_fallback, runner_cls, uses_agents_pkg)
    """

    # 1) 优先尝试新的 ``agents`` 包（示例中的调用方式）
    last_error: Exception | None = None
    try:
        agents_mod = importlib.import_module("agents")
        agent_cls = getattr(agents_mod, "Agent", None)
        runner_cls = getattr(agents_mod, "Runner", None)
        if agent_cls is not None and runner_cls is not None:
            tool_wrapper = getattr(agents_mod, "tool", None)
            return agent_cls, tool_wrapper, False, runner_cls, True  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - depends on external SDK
        last_error = exc

    # 2) 回退到 openai agent SDK
    candidates = [
        "openai.agents",
        "openai.agent",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on external SDK
            last_error = exc
            continue

        agent_cls = getattr(module, "Agent", None)
        tool_wrapper = getattr(module, "tool", None)
        if agent_cls is not None:
            return agent_cls, tool_wrapper, False, None, False  # type: ignore[arg-type]

    # 3) 本地 fallback
    if last_error:
        message = (
            "未检测到 OpenAI Agent SDK（需要 openai>=1.60 或官方 agent 扩展）；"
            f"使用本地 FallbackAgent 继续运行。最后一次导入错误: {last_error}"
        )
    else:
        message = "未检测到 OpenAI Agent SDK（需要 openai>=1.60 或官方 agent 扩展）；使用本地 FallbackAgent 继续运行。"

    print(message)
    return _FallbackAgent, None, True, None, False


def coerce_agent_output(payload: Any) -> dict[str, Any]:
    """Best-effort convert Agent 输出为 dict。

    Agent SDK 的 ``run`` 结果可能是字符串、dict 或带 ``output`` 属性的对象；
    该工具尝试将常见格式解析为 JSON/dict，若无法解析则抛出 ``TypeError``。
    """

    if isinstance(payload, Mapping):
        return dict(payload)

    if hasattr(payload, "output"):
        output = getattr(payload, "output")
        try:
            return coerce_agent_output(output)
        except Exception:  # noqa: BLE001 - 避免屏蔽下方处理
            pass

    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TypeError(f"无法从 Agent 输出解析 JSON: {exc}") from exc

    raise TypeError(f"无法解析的 Agent 输出类型: {type(payload)}")


def _load_action_registry(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return BUSINESS_ACTIONS
    actions = load_actions_from_path(path)
    return validate_actions(actions)


def _normalize_workflow_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Workflow):
        return payload.model_dump(by_alias=True)

    if isinstance(payload, Mapping):
        return dict(payload)

    if isinstance(payload, Path):
        return json.loads(payload.read_text(encoding="utf-8"))

    if isinstance(payload, str):
        # Prefer file path if it exists; otherwise treat as JSON string.
        path = Path(payload)
        if path.exists():
            return _normalize_workflow_payload(path)
        return json.loads(payload)

    raise TypeError(f"Unsupported workflow payload type: {type(payload)}")


class _WorkflowAgentRuntime:
    """Holds shared services for the Agent tools."""

    def __init__(
        self,
        *,
        action_registry: list[dict[str, Any]],
        search_service: HybridActionSearchService,
        model: str,
        max_plan_rounds: int,
        max_repair_rounds: int,
        agent_cls: type,
        tool_wrapper: Callable[[Callable[..., Any]], Any] | None,
        runner_cls: Any | None,
        uses_agents_pkg: bool,
    ) -> None:
        self.action_registry = action_registry
        self.search_service = search_service
        self.model = model
        self.max_plan_rounds = max_plan_rounds
        self.max_repair_rounds = max_repair_rounds
        self._agent_cls = agent_cls
        self._tool_wrapper = tool_wrapper
        self._internal_agent = self._build_internal_agent()
        self.uses_fallback_agent = getattr(agent_cls, "_is_fallback", False)
        self._runner_cls = runner_cls
        self._uses_agents_pkg = uses_agents_pkg
        self._runner_cls = runner_cls
        self._uses_agents_pkg = uses_agents_pkg

    # --- 原子工具：直接调用规划/校验/修复/更新流水线（不经过 Agent 推理） ---
    def _plan_workflow_direct(self, requirement: str) -> dict[str, Any]:
        workflow = plan_workflow_with_two_pass(
            nl_requirement=requirement,
            search_service=self.search_service,
            action_registry=self.action_registry,
            max_rounds=self.max_plan_rounds,
            max_repair_rounds=self.max_repair_rounds,
            model=self.model,
        )
        return workflow.model_dump(by_alias=True)

    def _update_workflow_direct(self, workflow_raw: Any, requirement: str) -> dict[str, Any]:
        normalized = _normalize_workflow_payload(workflow_raw)
        updated = update_workflow_with_two_pass(
            existing_workflow=normalized,
            requirement=requirement,
            search_service=self.search_service,
            action_registry=self.action_registry,
            max_repair_rounds=self.max_repair_rounds,
            model=self.model,
        )
        return updated.model_dump(by_alias=True)

    def _validate_workflow_direct(self, workflow_raw: Any) -> dict[str, Any]:
        normalized = _normalize_workflow_payload(workflow_raw)
        parse_result = parse_workflow_source(normalized)
        errors = validate_workflow_data(
            parse_result.ast or {},
            self.action_registry,
            parser_result=parse_result,
        )

        response: dict[str, Any] = {
            "errors": [err.model_dump() for err in errors],
            "ok": not errors,
        }

        if errors:
            if parse_result.syntax_errors:
                response["syntax_errors"] = [asdict(err) for err in parse_result.syntax_errors]
            if parse_result.grammar_issues:
                response["grammar_issues"] = [asdict(issue) for issue in parse_result.grammar_issues]
            response["recovered"] = parse_result.recovered
        else:
            workflow_model = Workflow.model_validate(parse_result.ast or normalized)
            response["normalized_workflow"] = workflow_model.model_dump(by_alias=True)
        return response

    def _repair_workflow_direct(self, workflow_raw: Any) -> dict[str, Any]:
        validation = self._validate_workflow_direct(workflow_raw)
        normalized = validation.get("normalized_workflow") or _normalize_workflow_payload(workflow_raw)
        errors_before = validation.get("errors") or []

        if not errors_before:
            return {"workflow": normalized, "errors_before": []}

        repaired = repair_workflow_with_llm(
            broken_workflow=normalized,
            validation_errors=[ValidationError.model_validate(err) for err in errors_before],
            action_registry=self.action_registry,
            model=self.model,
        )
        remaining_errors = validate_completed_workflow(repaired, self.action_registry)
        return {
            "workflow": repaired,
            "errors_before": errors_before,
            "errors_after": [err.model_dump() for err in remaining_errors],
        }

    # --- Internal Agent：用 Agent SDK 调度上述原子工具 ---
    def _build_internal_agent(self) -> Any:
        build_tool = _wrap_tool(self._tool_wrapper, self._plan_workflow_direct)
        validate_tool = _wrap_tool(self._tool_wrapper, self._validate_workflow_direct)
        repair_tool = _wrap_tool(self._tool_wrapper, self._repair_workflow_direct)
        update_tool = _wrap_tool(self._tool_wrapper, self._update_workflow_direct)

        if self._uses_agents_pkg:
            return self._agent_cls(
                name="workflow_internal_agent",
                instructions=INTERNAL_AGENT_INSTRUCTIONS,
                tools=[build_tool, validate_tool, repair_tool, update_tool],
            )

        return self._agent_cls(
            model=self.model,
            instructions=INTERNAL_AGENT_INSTRUCTIONS,
            tools=[build_tool, validate_tool, repair_tool, update_tool],
        )

    def _invoke_internal_agent(self, prompt: str, fallback: Callable[[], dict[str, Any]]):
        try:
            if not self.uses_fallback_agent:
                print("[Agent] 调用 OpenAI Agent 执行工具，prompt 已截断展示: ", prompt[:200])
            if self._runner_cls is not None:
                raw = asyncio.run(self._runner_cls.run(self._internal_agent, input=prompt))
                raw = getattr(raw, "final_output", raw)
            else:
                raw = self._internal_agent.run(prompt)
            return coerce_agent_output(raw)
        except Exception as exc:  # noqa: BLE001 - Agent 推理可能失败，兜底使用直接路径
            if not self.uses_fallback_agent:
                print(f"[Agent] Agent 调用失败，回退到本地实现，原因: {exc}")
            return fallback()

    # --- 对外暴露的 Agent 工具：依赖 Internal Agent 完成任务 ---
    def plan_workflow(self, requirement: str) -> dict[str, Any]:
        """Plan a brand-new workflow from a natural language requirement via Agent."""

        prompt = (
            "需求如下，请仅调用 build_workflow 工具完成规划，并返回工具 JSON 输出。"
            f"\n需求: {requirement}"
        )
        if not self.uses_fallback_agent:
            print("[Agent] 开始规划 workflow（build_workflow）。")
        return self._invoke_internal_agent(prompt, lambda: self._plan_workflow_direct(requirement))

    def update_workflow(self, workflow_raw: Any, requirement: str) -> dict[str, Any]:
        """Update an existing workflow based on a new requirement via Agent."""

        payload_text = json.dumps(_normalize_workflow_payload(workflow_raw), ensure_ascii=False)
        prompt = (
            "请调用 update_workflow 工具，以下 JSON 为现有 workflow，另附新需求。"
            "务必返回工具原样输出的 JSON。"
            f"\n现有 workflow: {payload_text}\n新需求: {requirement}"
        )
        if not self.uses_fallback_agent:
            print("[Agent] 开始更新 workflow（update_workflow）。")
        return self._invoke_internal_agent(
            prompt,
            lambda: self._update_workflow_direct(workflow_raw, requirement),
        )

    def validate_workflow(self, workflow_raw: Any) -> dict[str, Any]:
        """Run static validation on a workflow payload via Agent."""

        payload_text = json.dumps(_normalize_workflow_payload(workflow_raw), ensure_ascii=False)
        prompt = (
            "请调用 validate_workflow 工具，对以下 JSON 进行静态校验，并返回工具输出。"
            f"\nworkflow: {payload_text}"
        )
        if not self.uses_fallback_agent:
            print("[Agent] 开始校验 workflow（validate_workflow）。")
        return self._invoke_internal_agent(prompt, lambda: self._validate_workflow_direct(workflow_raw))

    def repair_workflow(self, workflow_raw: Any) -> dict[str, Any]:
        """Attempt LLM-driven repair when validation fails via Agent."""

        payload_text = json.dumps(_normalize_workflow_payload(workflow_raw), ensure_ascii=False)
        prompt = (
            "workflow 校验失败时，请调用 repair_workflow 工具尝试自动修复；"
            "若已通过校验则直接返回当前结果。请返回工具输出的 JSON。"
            f"\nworkflow: {payload_text}"
        )
        if not self.uses_fallback_agent:
            print("[Agent] 开始修复 workflow（repair_workflow）。")
        return self._invoke_internal_agent(prompt, lambda: self._repair_workflow_direct(workflow_raw))


def _wrap_tool(tool_wrapper: Callable[[Callable[..., Any]], Any] | None, func: Callable[..., Any]) -> Any:
    if tool_wrapper is None or not callable(tool_wrapper):
        return func
    return tool_wrapper(func)


def create_workflow_agent(
    *,
    client: Any | None = None,
    config: WorkflowAgentConfig | None = None,
    search_service: HybridActionSearchService | None = None,
    action_registry: list[dict[str, Any]] | None = None,
) -> Any:
    """Create an OpenAI Agent wired with VelvetFlow workflow tools.

    Parameters
    ----------
    client:
        An optional ``OpenAI`` client instance to be passed to the Agent ctor
        (if supported by the installed Agent SDK).
    config:
        Runtime configuration such as model name and planner iterations.
    search_service:
        Prebuilt ``HybridActionSearchService`` to reuse; if omitted, one will
        be built from the resolved action registry.
    action_registry:
        Custom action registry list. Defaults to built-in ``BUSINESS_ACTIONS``
        or the registry loaded from ``config.action_registry_path``.
    """

    agent_cls, tool_wrapper, used_fallback, runner_cls, uses_agents_pkg = _import_openai_agent()
    config = config or WorkflowAgentConfig()
    registry = action_registry or _load_action_registry(config.action_registry_path)
    service = search_service or build_search_service_from_actions(registry)
    runtime = _WorkflowAgentRuntime(
        action_registry=registry,
        search_service=service,
        model=config.model,
        max_plan_rounds=config.max_plan_rounds,
        max_repair_rounds=config.max_repair_rounds,
        agent_cls=agent_cls,
        tool_wrapper=tool_wrapper,
        runner_cls=runner_cls,
        uses_agents_pkg=uses_agents_pkg,
    )

    build_workflow_tool = _wrap_tool(tool_wrapper, runtime.plan_workflow)
    validate_workflow_tool = _wrap_tool(tool_wrapper, runtime.validate_workflow)
    repair_workflow_tool = _wrap_tool(tool_wrapper, runtime.repair_workflow)
    update_workflow_tool = _wrap_tool(tool_wrapper, runtime.update_workflow)

    if uses_agents_pkg:
        agent_kwargs = {
            "name": "workflow_orchestrator",
            "instructions": config.instructions,
            "tools": [
                build_workflow_tool,
                validate_workflow_tool,
                repair_workflow_tool,
                update_workflow_tool,
            ],
        }
    else:
        agent_kwargs = {
            "model": config.model,
            "instructions": config.instructions,
            "tools": [
                build_workflow_tool,
                validate_workflow_tool,
                repair_workflow_tool,
                update_workflow_tool,
            ],
        }

        if client is not None:
            # Only attach client if the Agent constructor accepts it.
            try:
                signature = inspect.signature(agent_cls)  # type: ignore[arg-type]
                if "client" in signature.parameters:
                    agent_kwargs["client"] = client
            except (TypeError, ValueError):  # pragma: no cover - depends on SDK impl
                pass

    agent = agent_cls(**agent_kwargs)
    if used_fallback:
        setattr(agent, "_is_fallback", True)
        print("[Agent] 未检测到官方 Agent SDK，使用 FallbackAgent 运行。")
    else:
        print("[Agent] 已加载 OpenAI Agent SDK，将使用官方 Agent 调度工具。")
    return agent


__all__ = [
    "AgentSdkNotInstalled",
    "WorkflowAgentConfig",
    "coerce_agent_output",
    "create_workflow_agent",
]
