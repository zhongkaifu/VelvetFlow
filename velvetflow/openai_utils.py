# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Helpers for interacting with the OpenAI API safely."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, BadRequestError, OpenAI


def _is_temperature_unsupported_error(exc: BadRequestError) -> bool:
    """Return True when an OpenAI error indicates temperature is unsupported.

    Some newer models only accept the default temperature (1). This helper
    checks the structured error payload first and falls back to inspecting the
    error message to detect this scenario.
    """

    error_body = getattr(exc, "body", None)
    if isinstance(error_body, dict):
        error_info = error_body.get("error") or {}
        message = error_info.get("message", "")
        if (
            "temperature" in message
            and "default (1) value is supported" in message
        ):
            return True

    return "temperature" in str(exc) and "unsupported" in str(exc)


def safe_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    **kwargs: Any,
):
    """Create a chat completion with graceful fallback for unsupported params.

    The OpenAI ``chat.completions.create`` endpoint now rejects non-default
    temperature values for certain models. This helper retries the request
    without a temperature override when the API signals that only the default
    is supported, preserving existing behavior for compatible models.
    """

    params: Dict[str, Any] = {"model": model, "messages": messages, **kwargs}
    if temperature is not None:
        params["temperature"] = temperature

    try:
        return client.chat.completions.create(**params)
    except BadRequestError as exc:
        if temperature is not None and _is_temperature_unsupported_error(exc):
            params.pop("temperature", None)
            return client.chat.completions.create(**params)
        raise


async def async_safe_chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    **kwargs: Any,
):
    """Async counterpart of :func:`safe_chat_completion`."""

    params: Dict[str, Any] = {"model": model, "messages": messages, **kwargs}
    if temperature is not None:
        params["temperature"] = temperature

    try:
        return await client.chat.completions.create(**params)
    except BadRequestError as exc:
        if temperature is not None and _is_temperature_unsupported_error(exc):
            params.pop("temperature", None)
            return await client.chat.completions.create(**params)
        raise


__all__ = ["safe_chat_completion", "async_safe_chat_completion"]
