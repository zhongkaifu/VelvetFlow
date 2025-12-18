# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Lightweight compatibility layer for the OpenAI Agent SDK.

The planner prefers to import the official ``agents`` package that ships
``Agent``, ``Runner`` and ``function_tool``. When that package is not
available (for example in minimal CI environments), we fall back to a
shim that mirrors the same interface on top of Chat Completions. The shim
intentionally strips ``additionalProperties`` from generated JSON Schemas
to avoid the OpenAI Agents service rejecting tools with strict object
schemas.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from openai import OpenAI
from pydantic import create_model


try:  # pragma: no cover - prefer the official SDK when available
    from agents import Agent, Runner, function_tool as _official_function_tool  # type: ignore

    USING_OFFICIAL_AGENTS = True
except ImportError:  # pragma: no cover - exercised in environments without the SDK
    USING_OFFICIAL_AGENTS = False


def _strip_additional_properties(schema: Any) -> Any:
    if isinstance(schema, MutableMapping):
        return {k: _strip_additional_properties(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_properties(item) for item in schema]
    return schema


def _build_parameter_schema(func: Callable[..., Any]) -> Mapping[str, Any]:
    """Derive a JSON schema for the function signature without additionalProperties."""

    fields: Dict[str, tuple[Any, Any]] = {}
    signature = inspect.signature(func)
    for name, param in signature.parameters.items():
        annotation = param.annotation if param.annotation is not inspect._empty else Any
        default = param.default if param.default is not inspect._empty else ...
        fields[name] = (annotation, default)

    model = create_model(f"{func.__name__.capitalize()}Params", **fields)
    schema = model.model_json_schema()
    return _strip_additional_properties(schema)


def _attach_sanitized_schema(func: Callable[..., Any]) -> None:
    """Attach sanitized schemas for both fallback and official runners."""

    schema = _build_parameter_schema(func)
    sanitized = _strip_additional_properties(schema)
    # The fallback runner reads __tool_schema__, while the official SDK may look
    # for various internal attributes. We proactively attach a few common names
    # to ensure the sanitized schema is discoverable regardless of the runner
    # implementation.
    func.__tool_schema__ = sanitized  # type: ignore[attr-defined]
    func.__agents_schema__ = sanitized  # type: ignore[attr-defined]
    func.__openai_schema__ = sanitized  # type: ignore[attr-defined]


if USING_OFFICIAL_AGENTS:
    def function_tool(func: Callable[..., Any]) -> Callable[..., Any]:  # pragma: no cover - passthrough wrapper
        decorated = _official_function_tool(func)
        sanitized_schema = _build_parameter_schema(func)
        cleaned_schema = _strip_additional_properties(sanitized_schema)

        _attach_sanitized_schema(decorated)
        # Patch the official FunctionTool instance to avoid strict schema errors
        # caused by additionalProperties.
        if hasattr(decorated, "params_json_schema"):
            try:
                decorated.params_json_schema = cleaned_schema  # type: ignore[attr-defined]
            except Exception:
                pass
        if getattr(decorated, "strict_json_schema", None):
            try:
                decorated.strict_json_schema = False  # type: ignore[attr-defined]
            except Exception:
                pass
        return decorated

else:

    @dataclass
    class Agent:  # pragma: no cover - thin data holder
        name: str
        instructions: str
        tools: Sequence[Callable[..., Any]]
        model: str | None = None


    def function_tool(func: Callable[..., Any]) -> Callable[..., Any]:  # pragma: no cover - simple decorator
        _attach_sanitized_schema(func)
        return func


    class Runner:  # pragma: no cover - explicit error to surface missing SDK
        @staticmethod
        def run(*_: Any, **__: Any) -> Any:
            raise RuntimeError("The official `agents` SDK is required to run planners; please install `agents`.")


__all__ = [
    "Agent",
    "Runner",
    "USING_OFFICIAL_AGENTS",
    "function_tool",
]
