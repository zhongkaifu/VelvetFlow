"""Action execution helpers."""
from __future__ import annotations

import copy
from typing import Any, Dict

from velvetflow.logging_utils import log_debug, log_info, log_warn
from tools import get_registered_tool

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
        """Execute a registered tool and normalize its output."""

        tool = get_registered_tool(tool_name)
        if not tool:
            log_warn(f"[action:{action_id}] tool '{tool_name}' not registered")
            return {
                "status": "tool_not_registered",
                "tool_name": tool_name,
                "action_id": action_id,
            }

        log_info(f"-> 调用工具: {tool_name} with params={resolved_params}")
        try:
            output = tool(**resolved_params)
            log_debug(f"<- 工具返回: {output}")
        except Exception as exc:  # pragma: no cover - runtime safety
            log_warn(f"[action:{action_id}] tool '{tool_name}' 执行失败: {exc}")
            return {
                "status": "tool_error",
                "tool_name": tool_name,
                "action_id": action_id,
                "error": str(exc),
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
            if status and str(status).lower() not in {"ok", "success", "allowed"}:
                success = False
        self.run_manager.metrics.record_node(success=success)
