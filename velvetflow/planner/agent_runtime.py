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

try:  # Prefer the official Agents SDK when available
    from agents import Agent, Runner, function_tool
except Exception as import_error:  # pragma: no cover - exercised in CI without agents
    import functools
    import warnings

    warnings.warn(
        "Falling back to the built-in Agents shim because the optional "
        "'agents' package is missing or incompatible. Install the official "
        "package alongside openai>=1.30.0 to enable full agent execution.",
        RuntimeWarning,
    )

    class Agent:
        def __init__(self, *, name: str, instructions: str, tools: Sequence[Callable[..., Any]], model: str, **_: Any):
            self.name = name
            self.instructions = instructions
            self.tools = list(tools)
            self.model = model

    def function_tool(strict_mode: bool = True, **decorator_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            # Preserve metadata that downstream code may introspect
            _wrapper.__original_function__ = func  # type: ignore[attr-defined]
            _wrapper.__function_tool_strict_mode__ = strict_mode  # type: ignore[attr-defined]
            _wrapper.__function_tool_kwargs__ = decorator_kwargs  # type: ignore[attr-defined]
            return _wrapper

        return _decorator

    class Runner:
        @staticmethod
        def run_sync(agent: Agent, run_input: Any, *, max_turns: int | None = None) -> None:
            raise RuntimeError(
                "agents shim does not implement execution. Install the 'agents' package "
                "compatible with openai>=1.30.0 to run agents."
            ) from import_error

        @staticmethod
        async def run(agent: Agent, run_input: Any, *, max_turns: int | None = None) -> None:
            raise RuntimeError(
                "agents shim does not implement execution. Install the 'agents' package "
                "compatible with openai>=1.30.0 to run agents."
            ) from import_error
