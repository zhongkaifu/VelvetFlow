# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Utility helpers for console + structured logging with trace context."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

# Basic ANSI color codes for a slightly nicer CLI experience without extra deps.
RESET = "\033[0m"
COLORS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m",
    "dim": "\033[2m",
}

ICONS = {
    "INFO": "ℹ️",
    "SUCCESS": "✅",
    "WARN": "⚠️",
    "ERROR": "❌",
    "DEBUG": "🔍",
}

def _default_log_file() -> Path:
    if env_path := os.environ.get("VELVETFLOW_LOG_FILE"):
        return Path(env_path)

    log_dir = Path(os.environ.get("VELVETFLOW_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"velvetflow_{ts}.log"


LOG_FILE_PATH = _default_log_file()
_TRACE_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)
_SPAN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_id", default=None
)
_SPAN_NAME: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_name", default=None
)
_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "run_id", default=None
)
_LLM_USAGE_RECORDER: Callable[[Any], None] | None = None


class TraceContext:
    """Small container for correlating logs across modules."""

    def __init__(self, trace_id: str, span_id: str, span_name: str | None = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.span_name = span_name

    @classmethod
    def create(
        cls, trace_id: str | None = None, span_id: str | None = None, span_name: str | None = None
    ) -> "TraceContext":
        return cls(trace_id or _generate_trace_id(), span_id or _generate_span_id(), span_name)

    def child(self, span_name: str | None = None) -> "TraceContext":
        return TraceContext(self.trace_id, _generate_span_id(), span_name or self.span_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "span_name": self.span_name,
        }


def _generate_trace_id() -> str:
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    return uuid.uuid4().hex


def current_trace_context() -> TraceContext | None:
    trace_id = _TRACE_ID.get()
    if not trace_id:
        return None
    return TraceContext(trace_id, _SPAN_ID.get() or _generate_span_id(), _SPAN_NAME.get())


def current_run_id() -> str | None:
    return _RUN_ID.get()


def configure_run_logging(
    run_id: str | None = None, log_file: str | Path | None = None
) -> contextvars.Token[str | None] | None:
    """Configure per-run context for structured logs.

    Parameters
    ----------
    run_id:
        Unique identifier for the workflow run. When provided, it will be injected into
        every structured log record emitted by :func:`log_event`.
    log_file:
        Optional override for the JSONL log destination. When omitted, the default
        timestamped path is kept.
    """

    token: contextvars.Token[str | None] | None = None
    if run_id is not None:
        token = _RUN_ID.set(run_id)

    global LOG_FILE_PATH
    if log_file is not None:
        LOG_FILE_PATH = Path(log_file)

    return token


@contextlib.contextmanager
def use_trace_context(context: TraceContext | None):
    if context is None:
        yield None
        return

    token_trace = _TRACE_ID.set(context.trace_id)
    token_span = _SPAN_ID.set(context.span_id)
    token_name = _SPAN_NAME.set(context.span_name)
    try:
        yield context
    finally:
        _TRACE_ID.reset(token_trace)
        _SPAN_ID.reset(token_span)
        _SPAN_NAME.reset(token_name)


@contextlib.contextmanager
def child_span(span_name: str | None = None):
    parent = current_trace_context() or TraceContext.create()
    child_ctx = parent.child(span_name=span_name)
    with use_trace_context(child_ctx):
        yield child_ctx

LEVEL_COLOR = {
    "INFO": COLORS["cyan"],
    "SUCCESS": COLORS["green"],
    "WARN": COLORS["yellow"],
    "ERROR": COLORS["red"],
    "DEBUG": COLORS["magenta"],
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_lines(message: str) -> str:
    lines = message.split("\n")
    if len(lines) <= 1:
        return message
    # Align multi-line messages under the prefix for readability
    return ("\n" + " " * 18).join(lines)


def console_log(level: str, message: str) -> None:
    level = level.upper()
    color = LEVEL_COLOR.get(level, "")
    icon = ICONS.get(level, "➡️")
    trace = current_trace_context()
    trace_prefix = ""
    if trace:
        trace_prefix = f" [trace={trace.trace_id} span={trace.span_id}]"
    prefix = f"{color}{icon} [{level:>5}] {_timestamp()}{trace_prefix} |{RESET} "
    _safe_print(prefix + _format_lines(message))


def _safe_print(text: str) -> None:
    """Print text while tolerating encoding issues on non-UTF-8 consoles."""

    output = text if text.endswith("\n") else text + "\n"
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.buffer.write(output.encode(encoding, errors="replace"))
    except Exception:
        # Fallback to UTF-8 to ensure logs are emitted even on misconfigured streams.
        sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    try:
        sys.stdout.flush()
    except Exception:
        pass


def log_info(*messages: str) -> None:
    """Log informational messages.

    Accepts multiple message parts for convenience and joins them with spaces
    before delegating to :func:`console_log`.
    """

    message = " ".join(str(part) for part in messages) if messages else ""
    console_log("INFO", message)


def log_success(message: str) -> None:
    console_log("SUCCESS", message)


def log_warn(message: str) -> None:
    console_log("WARN", message)


def log_error(message: str) -> None:
    console_log("ERROR", message)


def log_debug(message: str) -> None:
    console_log("DEBUG", message)


def set_llm_usage_recorder(recorder: Callable[[Any], None] | None) -> None:
    global _LLM_USAGE_RECORDER
    _LLM_USAGE_RECORDER = recorder


def log_section(title: str, subtitle: str | None = None, char: str = "=") -> None:
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    line = char * width
    print(f"\n{COLORS['dim']}{line}{RESET}")
    console_log("INFO", title)
    if subtitle:
        console_log("DEBUG", subtitle)
    print(f"{COLORS['dim']}{line}{RESET}\n")


def log_kv(label: str, value: Any, level: str = "INFO") -> None:
    console_log(level, f"{label}：{value}")


def log_json(label: str, data: Mapping[str, Any] | list[Any] | Any) -> None:
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        payload = str(data)
    console_log("DEBUG", f"{label}:\n{payload}")


def _stringify_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, (dict, list)):
        try:
            return json.dumps(args, ensure_ascii=False)
        except TypeError:
            return str(args)
    return str(args)


def log_tool_call(
    source: str,
    tool_name: str,
    *,
    args: Any | None = None,
    tool_call_id: str | None = None,
) -> None:
    """Highlight LLM tool invocations in logs for easier debugging."""

    metadata_parts = []
    if tool_call_id:
        metadata_parts.append(f"id={tool_call_id}")
    metadata = f" ({', '.join(metadata_parts)})" if metadata_parts else ""

    args_text = _stringify_args(args)
    args_suffix = f" args={args_text}" if args_text else ""

    tool_name_colored = f"{COLORS['blue']}{tool_name}{RESET}"

    console_log(
        "INFO",
        f"{COLORS['yellow']}🛠️{RESET} [LLM ToolCall] {source}: {tool_name_colored}{metadata}{args_suffix}",
    )


def _persist_record(record: dict[str, Any]) -> None:
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_event(
    message: str,
    payload: Mapping[str, Any] | None = None,
    *,
    level: str = "INFO",
    workflow_run_id: str | None = None,
    node_id: str | None = None,
    action_id: str | None = None,
    context: TraceContext | None = None,
) -> None:
    """Emit machine-readable JSON logs with consistent fields.

    The output is a JSON line containing the timestamp, level, workflow_run_id,
    node_id, action_id, and message so that downstream tools can easily parse and
    aggregate logs.
    """

    ctx = context or current_trace_context()
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level.upper(),
        "workflow_run_id": workflow_run_id or current_run_id(),
        "node_id": node_id,
        "action_id": action_id,
        "message": message,
    }
    if payload:
        record["payload"] = dict(payload)
        if isinstance(payload, Mapping):
            record.setdefault("node_id", payload.get("node_id"))
            record.setdefault("action_id", payload.get("action_id"))
    if ctx:
        record.update(ctx.to_dict())

    json_line = json.dumps(record, ensure_ascii=False)
    _safe_print(json_line)
    _persist_record(record)


def log_llm_usage(model: str, usage: Any, operation: str | None = None) -> None:
    if usage is None:
        return

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if isinstance(usage, Mapping):
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        total_tokens = usage.get("total_tokens", total_tokens)

    payload = {
        "model": model,
        "operation": operation or "unknown",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if _LLM_USAGE_RECORDER:
        _LLM_USAGE_RECORDER(payload)
    log_event("llm_usage", payload)


def _serialize_tool_call(tc: Any) -> dict[str, Any]:
    if isinstance(tc, Mapping):
        return dict(tc)

    func = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", None),
        "type": getattr(tc, "type", None),
        "function": {
            "name": getattr(func, "name", None) if func else None,
            "arguments": getattr(func, "arguments", None) if func else None,
        },
    }


def _serialize_llm_message(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)

    payload: dict[str, Any] = {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
    }
    name = getattr(message, "name", None)
    if name:
        payload["name"] = name

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [_serialize_tool_call(tc) for tc in tool_calls]

    return payload


def log_llm_message(
    model: str,
    message: Any,
    *,
    operation: str | None = None,
    workflow_run_id: str | None = None,
    node_id: str | None = None,
    action_id: str | None = None,
) -> None:
    """Persist and print LLM reasoning messages for debugging and tracing."""

    serialized = _serialize_llm_message(message)
    role = serialized.get("role", "assistant")
    content = serialized.get("content")
    console_summary = f"[LLM {operation or model}] ({role})"
    if content:
        console_summary += f" {content}"
    console_log("INFO", console_summary)

    payload = {"model": model, "operation": operation or "unknown", "message": serialized}
    log_event(
        "llm_message",
        payload,
        level="INFO",
        workflow_run_id=workflow_run_id,
        node_id=node_id,
        action_id=action_id,
    )
