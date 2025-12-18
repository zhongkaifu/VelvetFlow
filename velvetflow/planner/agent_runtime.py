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

from agents import Agent, FunctionTool, Runner, function_tool
