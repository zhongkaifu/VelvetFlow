"""Action execution helpers."""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict

from velvetflow.logging_utils import log_debug, log_info, log_warn
from tools import get_registered_tool

from .async_runtime import AsyncToolHandle, GLOBAL_ASYNC_RESULT_STORE

from .simulation import SimulationData


class ActionExecutionMixin:
    simulation_data: SimulationData

    def _simulate_action(self, action_id: str, resolved_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic simulation payload for a given action."""

        payload = self.simulation_data.get(action_id)
        if not isinstance(payload, dict):
            return {
                "status": "simulated",
                "action_id": action_id,
                "params_used": resolved_params,
            }

        defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
        params_for_render = {**defaults, **resolved_params}

        template = payload.get("result")
        if template is None:
            return {
                "status": "simulated",
                "action_id": action_id,
                "params_used": resolved_params,
            }

        return copy.deepcopy(self._render_template(template, params_for_render))

    def _should_simulate(self, action_id: str) -> bool:
        """Check if the current action_id has an explicit simulation entry."""

        return action_id in self.simulation_data

    def _invoke_tool(self, tool_name: str, resolved_params: Dict[str, Any], action_id: str) -> Dict[str, Any]:
        """Execute a registered tool synchronously or asynchronously.

        支持两种调用模式：
        - 同步（默认）：直接调用工具函数，返回结果。
        - 异步：优先使用 Tool.async_function；如未提供，则调用同步函数并自动包装
          为 AsyncToolHandle，写入 GLOBAL_ASYNC_RESULT_STORE，方便后续恢复。
        """

        tool = get_registered_tool(tool_name)
        if not tool:
            log_warn(f"[action:{action_id}] tool '{tool_name}' not registered")
            return {
                "status": "tool_not_registered",
                "tool_name": tool_name,
                "action_id": action_id,
            }

        invoke_mode = str(resolved_params.pop("__invoke_mode", "")).lower()
        force_async = bool(resolved_params.pop("__async__", False)) or invoke_mode == "async"
        call_params = {k: v for k, v in resolved_params.items() if not k.startswith("__")}

        log_info(f"-> 调用工具: {tool_name} with params={call_params} (async={force_async})")
        try:
            if force_async and getattr(tool, "async_function", None):
                output = tool.async_function(**call_params)  # type: ignore[attr-defined]
            else:
                output = tool(**call_params)
            log_debug(f"<- 工具返回: {output}")
        except Exception as exc:  # pragma: no cover - runtime safety
            log_warn(f"[action:{action_id}] tool '{tool_name}' 执行失败: {exc}")
            return {
                "status": "tool_error",
                "tool_name": tool_name,
                "action_id": action_id,
                "error": str(exc),
            }

        if isinstance(output, AsyncToolHandle):
            # 工具本身返回了异步句柄，无论是否显式要求 async，都按异步流程处理。
            GLOBAL_ASYNC_RESULT_STORE.register(output)
            payload: Dict[str, Any] = {
                "status": "async_pending",
                "request_id": output.request_id,
                "tool_name": output.tool_name,
                "params": dict(output.params),
            }
            if output.metadata:
                payload["metadata"] = dict(output.metadata)
            return payload

        if force_async:
            # 没有显式 AsyncToolHandle 时，自动包装同步输出
            request_id = f"{tool_name}-{uuid.uuid4().hex}"
            handle = AsyncToolHandle(
                request_id=request_id, tool_name=tool_name, params=call_params
            )
            GLOBAL_ASYNC_RESULT_STORE.register(handle)
            GLOBAL_ASYNC_RESULT_STORE.complete(request_id, output)
            return {
                "status": "async_pending",
                "request_id": request_id,
                "tool_name": tool_name,
                "params": call_params,
            }

        if isinstance(output, dict):
            return output
        return {"value": output, "tool_name": tool_name}

    def _record_node_metrics(self, result: Any) -> None:
        if not getattr(self, "run_manager", None):
            return

        success = True
        if isinstance(result, dict):
            status = result.get("status")
            normalized_status = str(status).lower()
            if status and not (
                normalized_status in {"ok", "success", "allowed"}
                or normalized_status.startswith("async_")
            ):
                success = False
        self.run_manager.metrics.record_node(success=success)
