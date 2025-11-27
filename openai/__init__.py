"""Minimal OpenAI client stub for offline environments."""

from __future__ import annotations

from typing import Any, Optional


class _DummyResponse:
    def __init__(self, data: Any = None):
        self.data = data or {}


class _DummyCompletions:
    def create(self, *args: Any, **kwargs: Any) -> _DummyResponse:  # noqa: ANN401
        raise RuntimeError("OpenAI stub: network calls are disabled in this environment")


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


class OpenAI:
    def __init__(self, api_key: Optional[str] = None, **_: Any):  # noqa: ANN401
        self.api_key = api_key
        self.chat = _DummyChat()


__all__ = ["OpenAI"]
