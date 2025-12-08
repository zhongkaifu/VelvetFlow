"""Helpers for suspending and resuming workflow execution around async tools."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


@dataclass
class AsyncToolHandle:
    """Lightweight envelope returned by tools that execute asynchronously."""

    request_id: str
    tool_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "params": dict(self.params),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class AsyncResultStore:
    """In-memory bookkeeping for async tool requests and their results."""

    def __init__(self) -> None:
        self._pending: Dict[str, AsyncToolHandle] = {}
        self._results: Dict[str, Any] = {}

    def register(self, handle: AsyncToolHandle) -> None:
        self._pending[handle.request_id] = handle

    def complete(self, request_id: str, result: Any) -> None:
        """Mark an async request as completed and store its result."""

        if request_id in self._pending:
            self._pending.pop(request_id, None)
        self._results[request_id] = result

    def pop_result(self, request_id: str) -> Any:
        """Retrieve and clear a finished async result if present."""

        return self._results.pop(request_id, None)

    def pending_requests(self) -> Iterable[AsyncToolHandle]:
        return list(self._pending.values())


GLOBAL_ASYNC_RESULT_STORE = AsyncResultStore()


@dataclass
class ExecutionCheckpoint:
    """Serializable snapshot of executor state for resumption."""

    workflow_dict: Dict[str, Any]
    binding_snapshot: Dict[str, Any]
    visited: List[str]
    reachable: List[str]
    pending_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": copy.deepcopy(self.workflow_dict),
            "binding": copy.deepcopy(self.binding_snapshot),
            "visited": list(self.visited),
            "reachable": list(self.reachable),
            "pending_ids": list(self.pending_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionCheckpoint":
        return cls(
            workflow_dict=copy.deepcopy(payload.get("workflow", {})),
            binding_snapshot=copy.deepcopy(payload.get("binding", {})),
            visited=list(payload.get("visited", [])),
            reachable=list(payload.get("reachable", [])),
            pending_ids=list(payload.get("pending_ids", [])),
        )


@dataclass
class WorkflowSuspension:
    """Returned when execution is paused waiting for an async tool result."""

    checkpoint: ExecutionCheckpoint
    node_id: str
    request_id: str
    tool_name: str
    reason: str = "async_tool_pending"

