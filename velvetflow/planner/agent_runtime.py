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
    from agents import Agent, Runner, function_tool  # type: ignore

    USING_OFFICIAL_AGENTS = True
    RunnerResult = None  # type: ignore[assignment]
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


if not USING_OFFICIAL_AGENTS:

    @dataclass
    class Agent:  # pragma: no cover - thin data holder
        name: str
        instructions: str
        tools: Sequence[Callable[..., Any]]
        model: str | None = None


    def function_tool(func: Callable[..., Any]) -> Callable[..., Any]:  # pragma: no cover - simple decorator
        func.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": inspect.getdoc(func) or "",
                "parameters": _build_parameter_schema(func),
            },
        }
        return func


    @dataclass
    class RunnerResult:  # pragma: no cover - basic return wrapper
        final_output: str | None
        messages: List[Mapping[str, Any]]


    class Runner:  # pragma: no cover - synchronous shim for Chat Completions
        @staticmethod
        def run(
            agent: Agent,
            prompt_or_messages: str | Sequence[Mapping[str, Any]],
            *,
            client: OpenAI | None = None,
            max_rounds: int = 12,
            temperature: float = 0.2,
        ) -> RunnerResult:
            client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            if isinstance(prompt_or_messages, str):
                messages: List[Mapping[str, Any]] = [{"role": "user", "content": prompt_or_messages}]
            else:
                messages = list(prompt_or_messages)

            if agent.instructions:
                messages = [{"role": "system", "content": agent.instructions}] + messages

            tools = []
            tool_lookup: Dict[str, Callable[..., Any]] = {}
            for tool in agent.tools:
                schema = getattr(tool, "__tool_schema__", None)
                if schema:
                    tools.append(schema)
                tool_lookup[getattr(tool, "__name__", "")] = tool

            for _ in range(max_rounds):
                resp = client.chat.completions.create(
                    model=agent.model or os.environ.get("OPENAI_MODEL"),
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                )
                msg = resp.choices[0].message
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })

                if not msg.tool_calls:
                    return RunnerResult(final_output=msg.content, messages=messages)

                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    args_raw = tc.function.arguments
                    try:
                        args = json.loads(args_raw) if args_raw else {}
                    except json.JSONDecodeError:
                        args = {}

                    tool_fn = tool_lookup.get(func_name)
                    if not tool_fn:
                        tool_result = {"status": "error", "message": f"未知工具 {func_name}"}
                    else:
                        try:
                            tool_result = tool_fn(**args)
                        except Exception as exc:  # pragma: no cover - defensive path
                            tool_result = {
                                "status": "error",
                                "message": f"执行工具 {func_name} 时出错: {exc}",
                            }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )

            return RunnerResult(final_output=None, messages=messages)


__all__ = [
    "Agent",
    "Runner",
    "RunnerResult",
    "USING_OFFICIAL_AGENTS",
    "function_tool",
]
