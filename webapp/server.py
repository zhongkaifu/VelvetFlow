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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.executor.async_runtime import WorkflowSuspension
from velvetflow.models import Workflow
from velvetflow.planner import plan_workflow_with_two_pass, update_workflow_with_two_pass
from velvetflow.search import HybridActionSearchService, build_default_search_service

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
    global _search_service
    if _search_service is None:
        _search_service = build_default_search_service()
    return _search_service


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


@app.post("/api/plan", response_model=PlanResponse)
def plan_workflow(req: PlanRequest) -> PlanResponse:
    if not req.requirement.strip():
        raise HTTPException(status_code=400, detail="requirement 不能为空")

    try:
        search_service = _get_search_service()
    except Exception as exc:  # noqa: BLE001 - surface startup errors
        raise HTTPException(status_code=500, detail=f"构建检索服务失败: {exc}") from exc

    try:
        with _capture_logs() as logs:
            if req.existing_workflow:
                workflow = update_workflow_with_two_pass(
                    existing_workflow=req.existing_workflow,
                    requirement=req.requirement,
                    search_service=search_service,
                    action_registry=BUSINESS_ACTIONS,
                    # Align with the CLI demo defaults to reduce discrepancies.
                    max_repair_rounds=3,
                )
            else:
                workflow = plan_workflow_with_two_pass(
                    nl_requirement=req.requirement,
                    search_service=search_service,
                    action_registry=BUSINESS_ACTIONS,
                    # Mirror build_workflow.py defaults for consistent planning results.
                    max_rounds=100,
                    max_repair_rounds=3,
                )
    except Exception as exc:  # noqa: BLE001 - keep API message concise
        raise HTTPException(
            status_code=500,
            detail=f"规划或自修复失败: {exc} | trace: {traceback.format_exc()}",
        ) from exc

    return PlanResponse(workflow=_serialize_workflow(workflow), logs=logs)


@app.post("/api/plan/stream")
async def plan_workflow_stream(req: PlanRequest) -> StreamingResponse:
    """Stream planning logs and the final workflow over SSE."""

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

        try:
            with _capture_logs_stream(queue, loop):
                if req.existing_workflow:
                    workflow = await asyncio.to_thread(
                        update_workflow_with_two_pass,
                        existing_workflow=req.existing_workflow,
                        requirement=req.requirement,
                        search_service=search_service,
                        action_registry=BUSINESS_ACTIONS,
                        max_repair_rounds=3,
                    )
                else:
                    workflow = await asyncio.to_thread(
                        plan_workflow_with_two_pass,
                        nl_requirement=req.requirement,
                        search_service=search_service,
                        action_registry=BUSINESS_ACTIONS,
                        max_rounds=100,
                        max_repair_rounds=3,
                    )

            await queue.put({"type": "result", "workflow": _serialize_workflow(workflow)})
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
