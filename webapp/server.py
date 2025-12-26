"""FastAPI server for the VelvetFlow web playground.

This server exposes two endpoints:
- POST /api/plan: build or update a workflow from a natural-language requirement.
- POST /api/run: execute a validated workflow with the simulated executor.

Static assets in the same directory are also served, so running
`python webapp/server.py` will provide both the API and the front-end.
"""
from __future__ import annotations
"""FastAPI entrypoint that powers the VelvetFlow playground UI."""

import sys
import traceback
import asyncio
import json
import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import anyio

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Ensure the local package is importable when running via `python webapp/server.py`.
    sys.path.insert(0, str(REPO_ROOT))

import velvetflow.logging_utils as logging_utils
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.config import OPENAI_MODEL
from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.executor.async_runtime import WorkflowSuspension
from velvetflow.models import Workflow
from velvetflow.planner import plan_workflow_with_two_pass, update_workflow_with_two_pass
from velvetflow.search import (
    HybridActionSearchService,
    build_default_search_service,
    get_openai_client,
)

SIMULATION_PATH = REPO_ROOT / "simulation_data.json"

app = FastAPI(title="VelvetFlow Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlanRequest(BaseModel):
    requirement: str
    existing_workflow: Optional[Dict[str, Any]] = None


class PlanResponse(BaseModel):
    workflow: Dict[str, Any]
    logs: List[str]
    suggestions: Optional[List[str]] = None
    tool_gap_message: Optional[str] = None
    tool_gap_suggestions: Optional[List[str]] = None


class RunRequest(BaseModel):
    workflow: Dict[str, Any]
    simulation_data: Optional[Dict[str, Any]] = None


class RunResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    logs: List[str]


class ActionInfo(BaseModel):
    action_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    params_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


_search_service: HybridActionSearchService | None = None


def _get_search_service() -> HybridActionSearchService:
    """Lazily construct a shared search service for planner endpoints."""

    # Building the default service may hit disk/network, so reuse a singleton
    # across requests.
    global _search_service
    if _search_service is None:
        _search_service = build_default_search_service()
    return _search_service


def _is_empty_workflow(workflow: Workflow | Dict[str, Any]) -> bool:
    """Check whether a workflow contains any nodes regardless of object type."""

    # Accept either Pydantic object or raw dict payloads from the API body.
    nodes = getattr(workflow, "nodes", None)
    if nodes is None and isinstance(workflow, dict):
        nodes = workflow.get("nodes")
    return not nodes


def _summarize_workflow(requirement: str, workflow_dict: Dict[str, Any]):
    """Derive UI suggestions/tool-gap hints from a workflow result."""

    suggestions: List[str] = []
    tool_gap_message = None
    tool_gap_suggestions: List[str] = []
    needs_tool_gap = _workflow_missing_business_tools(workflow_dict, BUSINESS_ACTIONS)
    if needs_tool_gap:
        tool_gap_message, tool_gap_suggestions = _suggest_tool_gap_guidance(requirement)
    elif _is_empty_workflow(workflow_dict):
        suggestions = _suggest_requirement_additions(requirement)

    return suggestions, tool_gap_message, tool_gap_suggestions or None


def _run_planner_for_request(
    req: PlanRequest,
    search_service: HybridActionSearchService,
    require_existing: bool = False,
    *,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
):
    """Dispatch a planning or updating request based on presence of a DAG."""

    # Basic input validation before calling into planner logic.
    if not req.requirement.strip():
        raise HTTPException(status_code=400, detail="requirement 不能为空")

    if require_existing and not req.existing_workflow:
        raise HTTPException(status_code=400, detail="existing_workflow 不能为空")

    # When an existing workflow is provided, route through the updater to
    # preserve and extend the current DAG. Otherwise, build a fresh plan.
    if req.existing_workflow:
        return update_workflow_with_two_pass(
            existing_workflow=req.existing_workflow,
            requirement=req.requirement,
            search_service=search_service,
            action_registry=BUSINESS_ACTIONS,
            max_rounds=100,
            max_repair_rounds=3,
            progress_callback=progress_callback,
        )

    return plan_workflow_with_two_pass(
        nl_requirement=req.requirement,
        search_service=search_service,
        action_registry=BUSINESS_ACTIONS,
        max_rounds=100,
        max_repair_rounds=3,
        progress_callback=progress_callback,
    )


def _suggest_requirement_additions(user_requirement: str, *, model: str = OPENAI_MODEL) -> List[str]:
    """Use an LLM to propose clarifications that make the workflow plannable."""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是业务流程规划助手，会帮用户把含糊的需求拆解得更具体。"
                        "请基于用户输入的需求、想法和目的给出 3 条补充思路，帮助用户明确：数据来源/触发时机、"
                        "关键条件或过滤规则、具体输出/通知形式、成功判定标准。"
                        "请用简短中文短句返回 JSON 数组，不要添加其他说明。"
                    ),
                },
                {"role": "user", "content": user_requirement},
            ],
        )

        if not resp.choices:
            raise RuntimeError("未获得补充建议")

        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)
    except Exception as exc:  # pragma: no cover - 网络/模型依赖环境
        logging_utils.log_error(f"[suggestion] 无法从 LLM 获取补充建议：{exc}")
        return []


def _workflow_missing_business_tools(
    workflow: Workflow | Dict[str, Any], action_registry: List[Dict[str, Any]]
) -> bool:
    nodes = getattr(workflow, "nodes", None)
    if nodes is None and isinstance(workflow, dict):
        nodes = workflow.get("nodes")
    nodes = nodes or []
    if not nodes:
        return True

    actions_by_id = {action.get("action_id") for action in action_registry if action.get("action_id")}
    for node in nodes:
        if isinstance(node, dict):
            if node.get("type") != "action":
                continue
            action_id = node.get("action_id")
        else:
            if getattr(node, "type", None) != "action":
                continue
            action_id = getattr(node, "action_id", None)
        if action_id and action_id in actions_by_id:
            return False
    return True


def _suggest_tool_gap_guidance(
    user_requirement: str, *, model: str = OPENAI_MODEL
) -> tuple[str, List[str]]:
    fallback_message = "当前动作库中暂未找到适合该需求的业务工具，可能需要调整需求描述或提供可用的系统信息。"
    fallback_suggestions = [
        "补充你期望使用的业务系统/数据源名称或接口类型。",
        "说明可接受的替代流程（例如改为通知/记录/导出）。",
        "明确触发时机、输入字段与输出目标，便于匹配到可用动作。",
    ]

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是业务流程规划助手。当前动作库里没有发现能够支撑该需求的业务工具。"
                        "请输出 JSON 对象，包含 message 和 suggestions 两个字段："
                        "message 用 1-2 句中文明确告知没有匹配的业务工具，并从用户输入的需求、想法和目的出发提示可以如何调整；"
                        "suggestions 提供 3 条修改建议（简短中文短句），必须基于用户的需求意图。"
                        "仅返回 JSON，不要添加其他说明。"
                    ),
                },
                {"role": "user", "content": user_requirement},
            ],
        )

        if not resp.choices:
            raise RuntimeError("未获得业务工具缺失建议")

        content = (resp.choices[0].message.content or "").strip()
        payload = json.loads(content)
        message = payload.get("message") if isinstance(payload, dict) else None
        suggestions = payload.get("suggestions") if isinstance(payload, dict) else None

        normalized_message = message.strip() if isinstance(message, str) and message.strip() else fallback_message
        normalized_suggestions = (
            [item for item in suggestions if isinstance(item, str) and item.strip()]
            if isinstance(suggestions, list)
            else fallback_suggestions
        )
        if not normalized_suggestions:
            normalized_suggestions = fallback_suggestions
        return normalized_message, normalized_suggestions
    except Exception as exc:  # pragma: no cover - 网络/模型依赖环境
        logging_utils.log_error(f"[tool-gap] 无法从 LLM 获取业务工具缺失建议：{exc}")
        return fallback_message, fallback_suggestions


@contextmanager
def _capture_logs():
    """Capture console logs emitted through logging_utils.console_log."""

    logs: List[str] = []
    original = logging_utils.console_log

    def patched(level: str, message: str) -> None:
        logs.append(f"[{level.upper()}] {message}")
        try:
            original(level, message)
        except Exception:
            # Avoid breaking the request if console output fails.
            pass

    logging_utils.console_log = patched
    try:
        yield logs
    finally:
        logging_utils.console_log = original


@contextmanager
def _capture_logs_stream(
    queue: "asyncio.Queue[Dict[str, Any]]", loop: asyncio.AbstractEventLoop
):
    """Capture console logs and forward them to an async queue for streaming."""

    original = logging_utils.console_log

    def patched(level: str, message: str) -> None:
        line = f"[{level.upper()}] {message}"
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "message": line})
        except Exception:
            # Avoid breaking the request if queue operations fail.
            pass
        try:
            original(level, message)
        except Exception:
            # Avoid breaking the request if console output fails.
            pass

    logging_utils.console_log = patched
    try:
        yield
    finally:
        logging_utils.console_log = original


def _serialize_workflow(workflow: Workflow) -> Dict[str, Any]:
    normalized = workflow.model_dump(by_alias=True)
    normalized["edges"] = [edge.model_dump(by_alias=True) for edge in workflow.edges]
    return normalized


def _iter_workflow_snapshots(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate incremental workflow snapshots that reveal nodes step by step."""

    full_workflow = copy.deepcopy(workflow) or {}
    nodes = full_workflow.get("nodes") or []
    edges = full_workflow.get("edges") or []
    base_meta = {
        "workflow_name": full_workflow.get("workflow_name", ""),
        "description": full_workflow.get("description", ""),
    }

    snapshots: List[Dict[str, Any]] = []
    partial_nodes: List[Dict[str, Any]] = []

    if not nodes:
        snapshots.append(
            {"workflow": full_workflow, "visible_subgraph": {**base_meta, "nodes": [], "edges": []}}
        )
        return snapshots

    for node in nodes:
        partial_nodes.append(node)
        visible_ids = {item.get("id") for item in partial_nodes if item}
        partial_edges = [
            edge
            for edge in edges
            if edge.get("from_node") in visible_ids and edge.get("to_node") in visible_ids
        ]
        snapshots.append(
            {
                "workflow": full_workflow,
                "visible_subgraph": {
                    **base_meta,
                    "nodes": copy.deepcopy(partial_nodes),
                    "edges": copy.deepcopy(partial_edges),
                },
            }
        )

    return snapshots


@app.post("/api/plan", response_model=PlanResponse)
def plan_workflow(req: PlanRequest) -> PlanResponse:
    try:
        search_service = _get_search_service()
    except Exception as exc:  # noqa: BLE001 - surface startup errors
        raise HTTPException(status_code=500, detail=f"构建检索服务失败: {exc}") from exc

    try:
        with _capture_logs() as logs:
            workflow = _run_planner_for_request(req, search_service)
    except Exception as exc:  # noqa: BLE001 - keep API message concise
        raise HTTPException(
            status_code=500,
            detail=f"规划或自修复失败: {exc} | trace: {traceback.format_exc()}",
        ) from exc

    workflow_dict = _serialize_workflow(workflow)
    suggestions, tool_gap_message, tool_gap_suggestions = _summarize_workflow(req.requirement, workflow_dict)

    return PlanResponse(
        workflow=workflow_dict,
        logs=logs,
        suggestions=suggestions,
        tool_gap_message=tool_gap_message,
        tool_gap_suggestions=tool_gap_suggestions or None,
    )


@app.post("/api/update", response_model=PlanResponse)
def update_workflow(req: PlanRequest) -> PlanResponse:
    try:
        search_service = _get_search_service()
    except Exception as exc:  # noqa: BLE001 - surface startup errors
        raise HTTPException(status_code=500, detail=f"构建检索服务失败: {exc}") from exc

    try:
        with _capture_logs() as logs:
            workflow = _run_planner_for_request(req, search_service, require_existing=True)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - keep API message concise
        raise HTTPException(
            status_code=500,
            detail=f"规划或自修复失败: {exc} | trace: {traceback.format_exc()}",
        ) from exc

    workflow_dict = _serialize_workflow(workflow)
    suggestions, tool_gap_message, tool_gap_suggestions = _summarize_workflow(req.requirement, workflow_dict)

    return PlanResponse(
        workflow=workflow_dict,
        logs=logs,
        suggestions=suggestions,
        tool_gap_message=tool_gap_message,
        tool_gap_suggestions=tool_gap_suggestions or None,
    )


def _build_streaming_response(req: PlanRequest, require_existing: bool = False) -> StreamingResponse:
    if not req.requirement.strip():
        raise HTTPException(status_code=400, detail="requirement 不能为空")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def run_planner() -> None:
        try:
            search_service = _get_search_service()
        except Exception as exc:  # noqa: BLE001 - surface startup errors
            await queue.put({"type": "error", "message": f"构建检索服务失败: {exc}"})
            await queue.put({"type": "end"})
            return

        def emit_progress(label: str, workflow_state: Dict[str, Any]) -> None:
            if not workflow_state:
                return
            try:
                normalized = (
                    workflow_state
                    if isinstance(workflow_state, dict)
                    else _serialize_workflow(Workflow.model_validate(workflow_state))
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "snapshot", "stage": label, "workflow": normalized},
                )
            except Exception:
                # Avoid interrupting the planner when progress reporting fails.
                pass

        try:
            with _capture_logs_stream(queue, loop):
                workflow = await asyncio.to_thread(
                    _run_planner_for_request,
                    req,
                    search_service,
                    require_existing,
                    progress_callback=emit_progress,
                )

            workflow_dict = _serialize_workflow(workflow)
            snapshots = _iter_workflow_snapshots(workflow_dict)
            total_nodes = max(len(workflow_dict.get("nodes", [])) or 1, 1)
            for idx, snapshot in enumerate(snapshots, start=1):
                payload = {
                    "type": "snapshot",
                    "workflow": snapshot.get("workflow") or workflow_dict,
                    "progress": idx / total_nodes,
                }
                if snapshot.get("visible_subgraph") is not None:
                    payload["visible_subgraph"] = snapshot["visible_subgraph"]
                await queue.put(payload)

            suggestions, tool_gap_message, tool_gap_suggestions = _summarize_workflow(
                req.requirement, workflow_dict
            )
            needs_more_detail = bool(tool_gap_message) or _is_empty_workflow(workflow_dict)
            await queue.put(
                {
                    "type": "result",
                    "workflow": workflow_dict,
                    "needs_more_detail": needs_more_detail,
                    "suggestions": suggestions,
                    "tool_gap_message": tool_gap_message,
                    "tool_gap_suggestions": tool_gap_suggestions,
                }
            )
        except HTTPException as exc:
            await queue.put({"type": "error", "message": exc.detail})
        except Exception as exc:  # noqa: BLE001 - keep API message concise
            await queue.put(
                {
                    "type": "error",
                    "message": f"规划或自修复失败: {exc} | trace: {traceback.format_exc()}",
                }
            )
        finally:
            await queue.put({"type": "end"})

    async def event_stream():
        planner_task = asyncio.create_task(run_planner())
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "end":
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            planner_task.cancel()
            with anyio.CancelScope(shield=True):
                try:
                    await planner_task
                except Exception:
                    pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/plan/stream")
async def plan_workflow_stream(req: PlanRequest) -> StreamingResponse:
    """Stream planning logs and the final workflow over SSE."""

    return await _build_streaming_response(req)


@app.post("/api/update/stream")
async def update_workflow_stream(req: PlanRequest) -> StreamingResponse:
    """Stream updates when users refine an existing workflow over SSE."""

    return await _build_streaming_response(req, require_existing=True)


@app.post("/api/run", response_model=RunResponse)
def run_workflow(req: RunRequest) -> RunResponse:
    try:
        workflow = Workflow.model_validate(req.workflow)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail={"message": "workflow 校验失败", "errors": exc.errors()})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    simulation_data = req.simulation_data or load_simulation_data(str(SIMULATION_PATH))

    with _capture_logs() as logs:
        executor = DynamicActionExecutor(workflow, simulations=simulation_data)
        result = executor.run()

    payload: Dict[str, Any] = jsonable_encoder(result)
    status = "suspended" if isinstance(result, WorkflowSuspension) else "completed"
    return RunResponse(status=status, result=payload, logs=logs)


def _execute_workflow(workflow: Workflow, simulation_data: Dict[str, Any]) -> Any:
    executor = DynamicActionExecutor(workflow, simulations=simulation_data)
    return executor.run()


@app.post("/api/run/stream")
async def run_workflow_stream(req: RunRequest) -> StreamingResponse:
    try:
        workflow = Workflow.model_validate(req.workflow)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail={"message": "workflow 校验失败", "errors": exc.errors()})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    simulation_data = req.simulation_data or load_simulation_data(str(SIMULATION_PATH))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def run_executor() -> None:
        try:
            with _capture_logs_stream(queue, loop):
                result = await asyncio.to_thread(_execute_workflow, workflow, simulation_data)
            payload: Dict[str, Any] = jsonable_encoder(result)
            status = "suspended" if isinstance(result, WorkflowSuspension) else "completed"
            await queue.put({"type": "result", "status": status, "result": payload})
        except Exception as exc:  # noqa: BLE001 - runtime failures should surface to the UI
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put({"type": "end"})

    async def event_stream():
        task = asyncio.create_task(run_executor())
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "end":
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            task.cancel()
            with anyio.CancelScope(shield=True):
                try:
                    await task
                except Exception:
                    pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/actions", response_model=List[ActionInfo])
def list_actions() -> List[ActionInfo]:
    return [
        ActionInfo(
            action_id=action.get("action_id", ""),
            name=action.get("name"),
            description=action.get("description"),
            params_schema=action.get("params_schema") or action.get("arg_schema"),
            output_schema=action.get("output_schema") or action.get("out_params_schema"),
        )
        for action in BUSINESS_ACTIONS
    ]


@app.get("/logo.jpg")
def serve_logo() -> FileResponse:
    """Serve the VelvetFlow logo from the repository root."""

    return FileResponse(REPO_ROOT / "logo.jpg", media_type="image/jpeg")


app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="static")


if __name__ == "__main__":
    import asyncio

    try:
        from hypercorn.asyncio import serve
        from hypercorn.config import Config
    except ModuleNotFoundError as exc:  # noqa: PERF203 - clearer guidance for users
        missing = "hypercorn"
        message = (
            f"缺少依赖 {missing}，请先运行 `pip install -r requirements.txt` 再启动: {exc}"
        )
        print(message)
        raise SystemExit(1)

    config = Config()
    config.bind = ["0.0.0.0:8000"]
    asyncio.run(serve(app, config))
