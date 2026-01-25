# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from __future__ import annotations

"""A small set of real, ready-to-use business tools."""

import html
import json
import os
import re
import textwrap
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
import uuid
from typing import Any, Dict, List, Mapping, Optional, Tuple
from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_info, log_json

from tools.base import Tool
from tools.business import (
    register_finance_tools,
    register_hr_tools,
    register_it_tools,
    register_marketing_tools,
    register_sales_tools,
    register_web_scraper_tools,
)
from tools.registry import get_registered_tool, register_tool


def _normalize_web_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        return ""

    if url.startswith("//"):
        url = f"https:{url}"
    elif not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = f"https://{url}"

    return url


def _resolve_search_url(raw_url: str) -> str:
    if not raw_url:
        return ""

    if raw_url.startswith("/l/?"):
        parsed = urllib.parse.urlparse(raw_url)
        uddg = urllib.parse.parse_qs(parsed.query).get("uddg", [])
        if uddg:
            return _normalize_web_url(urllib.parse.unquote(uddg[0]))
    return _normalize_web_url(raw_url)


class _DuckDuckGoParser(HTMLParser):
    """Minimal parser to extract search results from DuckDuckGo HTML."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: List[Dict[str, str]] = []
        self._capturing_title = False
        self._capturing_snippet = False
        self._current_href: str | None = None
        self._title_parts: List[str] = []
        self._snippet_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:  # pragma: no cover - HTML parsing
        attrs_dict = {k: v for k, v in attrs}
        if tag == "a" and "result__a" in (attrs_dict.get("class") or ""):
            self._capturing_title = True
            self._current_href = attrs_dict.get("href")
            self._title_parts = []
        if tag in {"p", "div", "span"} and "result__snippet" in (attrs_dict.get("class") or ""):
            self._capturing_snippet = True
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - HTML parsing
        if self._capturing_title:
            self._title_parts.append(data.strip())
        if self._capturing_snippet:
            self._snippet_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:  # pragma: no cover - HTML parsing
        if self._capturing_title and tag == "a":
            title = " ".join(self._title_parts).strip()
            if title and self._current_href:
                self.results.append({"title": title, "url": self._current_href})
            self._capturing_title = False
            self._title_parts = []
            self._current_href = None
            if len(self.results) >= self.limit:
                raise StopIteration
        if self._capturing_snippet and tag in {"p", "div", "span"}:
            snippet = " ".join(self._snippet_parts).strip()
            if snippet and self.results:
                if "snippet" not in self.results[-1]:
                    self.results[-1]["snippet"] = snippet
            self._capturing_snippet = False
            self._snippet_parts = []
            if len(self.results) >= self.limit:
                raise StopIteration


def search_web(query: str, limit: int = 5, timeout: int = 8) -> Dict[str, List[Dict[str, str]]]:
    """Perform a DuckDuckGo HTML search and return top results with snippets."""

    def _clean_text(value: str) -> str:
        text_no_tags = re.sub(r"<[^>]+>", " ", value)
        text_unescaped = html.unescape(text_no_tags)
        return re.sub(r"\s+", " ", text_unescaped).strip()

    if not query:
        raise ValueError("query is required for web search")

    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded_q}&kl=us-en"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; VelvetFlow/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - network-dependent
        raise RuntimeError(f"web search failed: {exc}") from exc

    parser = _DuckDuckGoParser(limit)
    try:
        parser.feed(raw_html)
    except StopIteration:  # pragma: no cover - parser stops early when limit reached
        pass

    results: List[Dict[str, str]] = []
    for entry in parser.results:
        title_value = _clean_text(entry.get("title", ""))
        url_value = _resolve_search_url(entry.get("url", ""))
        snippet_value = _clean_text(entry.get("snippet", "")) or title_value

        if title_value and url_value:
            results.append(
                {
                    "title": title_value,
                    "url": url_value,
                    "snippet": snippet_value,
                }
            )

        if len(results) >= limit:
            break

    return {"results": results}


def search_news(query: str, limit: int = 5, timeout: int = 8) -> Dict[str, List[Dict[str, str]]]:
    """Search recent news headlines via Bing and return titles, URLs, and snippets."""

    def _clean_text(value: str) -> str:
        text_no_tags = re.sub(r"<[^>]+>", " ", value)
        text_unescaped = html.unescape(text_no_tags)
        return re.sub(r"\s+", " ", text_unescaped).strip()

    if not query:
        raise ValueError("query is required for news search")

    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/news/search?q={encoded_q}&format=rss"
    req = urllib.request.Request(url, headers={"User-Agent": "VelvetFlow/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_feed = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - network-dependent
        raise RuntimeError(f"news search failed: {exc}") from exc

    try:
        root = ET.fromstring(raw_feed)
    except ET.ParseError as exc:  # pragma: no cover - network-dependent
        raise RuntimeError(f"failed to parse news feed: {exc}") from exc

    results: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        title = _clean_text(item.findtext("title", default=""))
        url_value = item.findtext("link", default="")
        description = item.findtext("description", default="")
        snippet = _clean_text(description) or title

        if title and url_value:
            results.append({"title": title, "url": url_value, "snippet": snippet})
        if len(results) >= limit:
            break

    return {"results": results}


def _prepare_ask_ai_prompt(
    *,
    prompt: str | None,
    question: str | None,
    context: Dict[str, Any] | None,
    expected_format: str | Mapping[str, Any] | None,
) -> str:
    if prompt:
        return prompt

    parts = ["You are a helpful assistant that produces concise JSON answers."]
    parts.append(f"Question: {question}")
    if context:
        context_json = json.dumps(context, ensure_ascii=False, indent=2)
        parts.append("Context (for reference):")
        parts.append(context_json)
    if expected_format:
        expected_format_text = (
            expected_format
            if isinstance(expected_format, str)
            else json.dumps(expected_format, ensure_ascii=False, indent=2)
        )
        parts.append("Return JSON following this guidance:")
        parts.append(expected_format_text)
    parts.append("Respond with valid JSON only.")
    return "\n\n".join(parts)


def _build_call_tool_schema(action: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    action_id = action.get("action_id") or action.get("tool_name") or "tool"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", action.get("tool_name") or action_id)
    description = action.get("description") or action.get("name") or action_id
    parameters = action.get("arg_schema") or action.get("params_schema")
    if not isinstance(parameters, Mapping):
        return None

    return {
        "type": "function",
        "function": {
            "name": safe_name,
            "description": description,
            "parameters": parameters,
        },
    }


def _execute_business_tool(
    *, action: Mapping[str, Any], args: Mapping[str, Any], call_name: str
) -> Dict[str, Any]:
    tool_fn_name = action.get("tool_name") or action.get("action_id") or call_name
    registered_tool = get_registered_tool(tool_fn_name)

    if not registered_tool:
        return {
            "status": "unavailable",
            "action_id": action.get("action_id"),
            "message": f"Business tool '{tool_fn_name}' is not registered; returning the original arguments for reference.",
            "echo": args,
        }

    try:
        output = registered_tool(**args)
        return {
            "status": "ok",
            "action_id": action.get("action_id"),
            "output": output,
        }
    except Exception as exc:  # pragma: no cover - runtime errors are surfaced to LLM
        return {
            "status": "error",
            "action_id": action.get("action_id"),
            "message": str(exc),
        }


def ask_ai(
    *,
    system_prompt: str | None = None,
    prompt: str | None = None,
    question: str | None = None,
    context: Dict[str, Any] | None = None,
    expected_format: str | Mapping[str, Any] | None = None,
    tool: List[Mapping[str, Any]] | None = None,
    model: str | None = None,
    max_tokens: int = 256,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """LLM helper that can optionally call business tools and analyze results."""

    if not (prompt or question):
        raise ValueError("either prompt or question must be provided for ask_ai")

    user_prompt = _prepare_ask_ai_prompt(
        prompt=prompt, question=question, context=context, expected_format=expected_format
    )
    system_text = system_prompt or "You are an intelligent assistant skilled at invoking business tools to solve problems."

    client = OpenAI()
    chat_model = model or OPENAI_MODEL
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]

    tools_spec: List[Dict[str, Any]] = []
    tool_lookup: Dict[str, Mapping[str, Any]] = {}
    for item in tool or []:
        schema = _build_call_tool_schema(item)
        if schema:
            tool_name = schema["function"]["name"]
            tools_spec.append(schema)
            tool_lookup[tool_name] = item

    for round_idx in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            tools=tools_spec or None,
            tool_choice="auto" if tools_spec else None,
            max_completion_tokens=max_tokens,
        )
        if not response.choices:
            raise RuntimeError("ask_ai did not return any choices")

        message = response.choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": message.tool_calls,
        }
        messages.append(assistant_msg)
        log_info(f"[ask_ai] round={round_idx} assistant response received")
        if assistant_msg.get("content"):
            log_json("[ask_ai] assistant content", assistant_msg.get("content"))
        else:
            reminder = (
                "Your previous response had no content. "
                "Please provide a concise response that satisfies the user's requirements."
            )
            messages.append({"role": "user", "content": reminder})
            log_info("[ask_ai] assistant content empty; prompting for a complete response")

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                func_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {"__raw__": raw_args or ""}

                action_def = tool_lookup.get(func_name)
                if not action_def:
                    tool_result = {
                        "status": "error",
                        "message": f"Unknown business tool invocation: {func_name}",
                        "echo": parsed_args,
                    }
                else:
                    tool_result = _execute_business_tool(
                        action=action_def, args=parsed_args, call_name=func_name
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
                log_info(f"[ask_ai] round={round_idx} tool_call={func_name}")
                log_json("[ask_ai] tool_args", parsed_args)
                log_json("[ask_ai] tool_result", tool_result)
            continue

        final_content = (message.content or "").strip()
        if not final_content:
            continue

        try:
            parsed_content = json.loads(final_content)
            results = parsed_content if isinstance(parsed_content, dict) else {"answer": parsed_content}
        except json.JSONDecodeError:
            results = {"answer": final_content}

        return {"status": "ok", "results": results, "messages": messages}

    return {
        "status": "error",
        "results": {"message": "ask_ai could not produce an answer within the allowed rounds"},
        "messages": messages,
    }


def classify_text(
    text: str,
    labels: List[str],
    *,
    instruction: str | None = None,
    allow_multiple: bool = False,
    model: str | None = None,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """Classify text into one or more labels using the configured LLM."""

    if not text:
        raise ValueError("text is required for classification")
    if not labels:
        raise ValueError("labels must be a non-empty list")

    client = OpenAI()
    chat_model = model or OPENAI_MODEL
    label_list = ", ".join(labels)
    system_prompt = [
        "You are a classification assistant.",
        f"Valid labels: {label_list}.",
        "Return JSON with keys 'labels' (array of selected labels) and 'reason'.",
    ]
    if instruction:
        system_prompt.append(f"Additional guidance: {instruction}")
    if not allow_multiple:
        system_prompt.append("Select exactly one label unless the text is ambiguous.")

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "\n".join(system_prompt)},
            {"role": "user", "content": text},
        ],
        max_completion_tokens=max_tokens,
    )
    if not response.choices:
        raise RuntimeError("classify_text did not return any choices")

    message = response.choices[0].message
    content = message.content or ""
    try:
        parsed = json.loads(content)
        labels_out = parsed.get("labels") if isinstance(parsed, dict) else None
        reason = parsed.get("reason") if isinstance(parsed, dict) else None
    except Exception:
        parsed = None
        labels_out = None
        reason = None

    labels_value: List[str]
    if isinstance(labels_out, list) and labels_out:
        labels_value = [str(l) for l in labels_out]
    else:
        first_label = str(content).strip() if content else labels[0]
        labels_value = [first_label]
    return {
        "labels": labels_value,
        "reason": reason or "",
        "model": chat_model,
        "finish_reason": response.choices[0].finish_reason,
    }


def list_files(
    directory: str = ".", *, include_hidden: bool = False, max_entries: int = 200
) -> Dict[str, List[Dict[str, Any]]]:
    """Return metadata for files under a directory."""

    path = Path(directory).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")

    entries: List[Dict[str, Any]] = []
    for entry in path.iterdir():
        if len(entries) >= max_entries:
            break
        if not include_hidden and entry.name.startswith('.'):
            continue
        stat = entry.stat()
        entries.append(
            {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return {"entries": entries}


def read_file(path: str, *, max_bytes: int = 65536, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read a text file with a safety limit."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    size = file_path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"file is too large to read safely (size={size}, limit={max_bytes})")

    with file_path.open("r", encoding=encoding, errors="replace") as f:
        content = f.read()
    return {
        "path": str(file_path),
        "size": size,
        "content": content,
    }


def join_list(items: List[str], separator: str = ",") -> Dict[str, str]:
    """Join a list of strings using the provided separator."""

    if not isinstance(items, list):
        raise TypeError("items must be a list of strings")

    if not all(isinstance(item, str) for item in items):
        raise TypeError("items must contain only strings")

    if not isinstance(separator, str):
        raise TypeError("separator must be a string")

    return {"result": separator.join(items)}


def split_list(text: str, separator: str = ",") -> Dict[str, List[str]]:
    """Split a string into a list using the provided separator."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(separator, str) or separator == "":
        raise ValueError("separator must be a non-empty string")

    return {"items": text.split(separator)}


def summarize(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    """Summarize a block of text using the configured LLM."""

    if not text:
        raise ValueError("text is required for summarization")

    normalized_text = textwrap.dedent(text).strip()
    sentences = re.split(r"(?<=[。！？.!?])\s+", normalized_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    client = OpenAI()
    chat_model = OPENAI_MODEL
    prompt = textwrap.dedent(
        f"""
        Provide a concise summary of the following content in no more than {max_sentences} sentences.
        Keep the summary factual and omit speculation.

        Content:
        {normalized_text}
        """
    ).strip()

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max(64, max_sentences * 64),
    )
    if not response.choices:
        raise RuntimeError("summarize did not return any choices")

    summary = (response.choices[0].message.content or "").strip()
    return {"summary": summary, "sentence_count": len(sentences)}


def compose_outlook_email(email_content: str, emailTo: Optional[str] = None) -> Dict[str, str]:
    """Open Outlook, start a new draft, and paste the provided HTML body."""

    if not isinstance(email_content, str) or not email_content.strip():
        raise ValueError("email_content must be a non-empty string")

    if emailTo is not None and (not isinstance(emailTo, str) or not emailTo.strip()):
        raise ValueError("emailTo must be a non-empty string when provided")

    if os.name != "nt":  # Outlook automation only available on Windows
        raise RuntimeError(
            "Outlook automation requires a Windows host with Microsoft Outlook installed."
        )

    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "pywin32 is required for Outlook automation. Install it with 'pip install pywin32'."
        ) from exc

    co_initialized = False
    try:
        pythoncom.CoInitialize()
        co_initialized = True

        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
        except Exception as exc:  # pragma: no cover - relies on Outlook installation
            raise RuntimeError(f"Failed to start Outlook: {exc}") from exc

        try:
            mail_item = outlook.CreateItem(0)  # 0 == olMailItem
            mail_item.Display()  # Bring up the draft window for user visibility
            if emailTo:
                mail_item.To = emailTo.strip()
            mail_item.HTMLBody = email_content.strip() + (mail_item.HTMLBody or "")
        except Exception as exc:  # pragma: no cover - relies on Outlook COM APIs
            raise RuntimeError(f"Failed to compose Outlook email: {exc}") from exc

    finally:
        if co_initialized:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    return {
        "status": "success",
        "message": "Outlook opened, new draft created, and content inserted.",
    }


def create_incident(title: str, description: str | None = None, severity: str | None = None) -> Dict[str, str]:
    """Simulate creating an incident ticket and return a deterministic payload."""

    incident_id = f"inc-{uuid.uuid4().hex[:8]}"
    return {
        "incident_id": incident_id,
        "status": "created",
        "title": title,
        "description": description or "",
        "severity": severity or "",
    }


def create_lead(name: str, email: str | None = None, source: str | None = None) -> Dict[str, str]:
    """Record a lightweight CRM lead entry and echo back normalized fields."""

    lead_id = f"lead-{uuid.uuid4().hex[:8]}"
    return {
        "lead_id": lead_id,
        "status": "created",
        "name": name,
        "email": email or "",
        "source": source or "",
    }


def log_interaction(lead_id: str, note: str) -> Dict[str, str]:
    """Append an interaction note to a CRM lead timeline."""

    interaction_id = f"note-{uuid.uuid4().hex[:8]}"
    return {
        "status": "logged",
        "interaction_id": interaction_id,
        "lead_id": lead_id,
        "note": note,
    }


def get_today_temperatures(date: str) -> Dict[str, Any]:
    """Return a mock temperature list for the provided date."""

    sample = [
        {"employee_id": "emp-1001", "temperature": 36.6},
        {"employee_id": "emp-1002", "temperature": 37.2},
    ]
    return {"date": date, "data": sample}


def record_health_event(event_type: str, date: str | None = None, abnormal_count: int | None = None) -> Dict[str, Any]:
    """Store a simplified health event record."""

    event_id = f"health-{uuid.uuid4().hex[:8]}"
    return {
        "status": "recorded",
        "event_id": event_id,
        "event_type": event_type,
        "date": date or "",
        "abnormal_count": abnormal_count if abnormal_count is not None else 0,
    }


def update_employee_health_profile(
    employee_id: str,
    last_check_date: str | None = None,
    last_temperature: float | None = None,
    status: str | None = None,
) -> Dict[str, Any]:
    """Upsert a minimal employee health profile record."""

    return {
        "employee_id": employee_id,
        "last_temperature": last_temperature if last_temperature is not None else 0.0,
        "status": status or "updated",
        "last_check_date": last_check_date or "",
    }


def async_stub_tool(payload: str) -> "AsyncToolHandle":
    """Return an async handle to simulate delayed execution for demo actions."""

    from velvetflow.executor.async_runtime import AsyncToolHandle

    request_id = f"demo-async-{uuid.uuid4().hex}"
    return AsyncToolHandle(
        request_id=request_id,
        tool_name="async_stub_tool",
        params={"payload": payload},
        metadata={"demo": True},
    )


def persist_async_stub(payload: str) -> "AsyncToolHandle":
    """Return an async handle that marks payloads for persistence testing."""

    from velvetflow.executor.async_runtime import AsyncToolHandle

    request_id = f"demo-persist-{uuid.uuid4().hex}"
    return AsyncToolHandle(
        request_id=request_id,
        tool_name="persist_async_stub",
        params={"payload": payload},
        metadata={"demo": True, "persist": True},
    )


def register_builtin_tools() -> None:
    """Register all built-in tools in the global registry."""

    register_tool(
        Tool(
            name="search_web",
            description="Perform a Bing search and return titles, URLs, and snippets.",
            function=search_web,
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "timeout": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        )
    )

    register_tool(
        Tool(
            name="search_news",
            description="Search recent news headlines and return titles, URLs, and snippets.",
            function=search_news,
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "timeout": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        )
    )

    register_tool(
        Tool(
            name="ask_ai",
            description=(
                "LLM helper that can run multi-round chat with optional business tool calls "
                "and return normalized status/results output."
            ),
            function=ask_ai,
            args_schema={
                "type": "object",
                "properties": {
                    "system_prompt": {"type": "string", "nullable": True},
                    "prompt": {"type": "string", "nullable": True},
                    "question": {"type": "string", "nullable": True},
                    "context": {"type": "object", "nullable": True},
                    "expected_format": {"type": "string", "nullable": True},
                    "tool": {"type": "array", "items": {"type": "object"}, "nullable": True},
                    "model": {"type": "string", "nullable": True},
                    "max_tokens": {"type": "integer", "default": 256},
                    "max_rounds": {"type": "integer", "default": 3},
                },
            },
        )
    )

    register_tool(
        Tool(
            name="classify_text",
            description="Classify text into one or more labels via OpenAI chat completion.",
            function=classify_text,
            args_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "instruction": {"type": "string", "nullable": True},
                    "allow_multiple": {"type": "boolean", "default": False},
                    "model": {"type": "string", "nullable": True},
                    "max_tokens": {"type": "integer", "default": 128},
                },
                "required": ["text", "labels"],
            },
        )
    )

    register_tool(
        Tool(
            name="list_files",
            description="List files and folders under a directory with size metadata.",
            function=list_files,
            args_schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                    "include_hidden": {"type": "boolean", "default": False},
                    "max_entries": {"type": "integer", "default": 200},
                },
            },
        )
    )

    register_tool(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file with a size guard.",
            function=read_file,
            args_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer", "default": 65536},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
            },
        )
    )

    register_tool(
        Tool(
            name="join_list",
            description="Join a list of strings using a separator.",
            function=join_list,
            args_schema={
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                    "separator": {"type": "string", "default": ","},
                },
                "required": ["items"],
            },
        )
    )

    register_tool(
        Tool(
            name="split_list",
            description="Split a string into a list using a separator.",
            function=split_list,
            args_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "separator": {"type": "string", "default": ","},
                },
                "required": ["text"],
            },
        )
    )

    register_tool(
        Tool(
            name="summarize",
            description="Summarize text with the LLM for concise overviews.",
            function=summarize,
            args_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_sentences": {"type": "integer", "default": 3},
                },
                "required": ["text"],
            },
        )
    )

    register_tool(
        Tool(
            name="compose_outlook_email",
            description=(
                "Launch Outlook, open a new draft, and fill in the provided email content."
            ),
            function=compose_outlook_email,
            args_schema={
                "type": "object",
                "properties": {
                    "email_content": {"type": "string"},
                    "emailTo": {"type": "string"},
                },
                "required": ["email_content"],
            },
        )
    )

    register_tool(
        Tool(
            name="record_health_event",
            description="Record an employee health event entry for auditing.",
            function=record_health_event,
            args_schema={
                "type": "object",
                "properties": {
                    "event_type": {"type": "string"},
                    "date": {"type": "string", "nullable": True},
                    "abnormal_count": {"type": "integer", "nullable": True},
                },
                "required": ["event_type"],
            },
        )
    )

    register_tool(
        Tool(
            name="get_today_temperatures",
            description="Retrieve temperature readings for a given date.",
            function=get_today_temperatures,
            args_schema={
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                },
                "required": ["date"],
            },
        )
    )

    register_tool(
        Tool(
            name="update_employee_health_profile",
            description="Update an employee health profile with latest vitals.",
            function=update_employee_health_profile,
            args_schema={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "last_check_date": {"type": "string", "nullable": True},
                    "last_temperature": {"type": "number", "nullable": True},
                    "status": {"type": "string", "nullable": True},
                },
                "required": ["employee_id"],
            },
        )
    )

    register_tool(
        Tool(
            name="create_incident",
            description="Open an incident ticket in the ops system.",
            function=create_incident,
            args_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string", "nullable": True},
                    "severity": {"type": "string", "nullable": True},
                },
                "required": ["title"],
            },
        )
    )

    register_tool(
        Tool(
            name="create_lead",
            description="Add a new lead into the CRM system.",
            function=create_lead,
            args_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "nullable": True},
                    "source": {"type": "string", "nullable": True},
                },
                "required": ["name"],
            },
        )
    )

    register_tool(
        Tool(
            name="log_interaction",
            description="Log an interaction note against a CRM lead.",
            function=log_interaction,
            args_schema={
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["lead_id", "note"],
            },
        )
    )

    register_tool(
        Tool(
            name="async_stub_tool",
            description="Dispatch an async stub request for demo workflows.",
            function=async_stub_tool,
            args_schema={
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
        )
    )

    register_tool(
        Tool(
            name="persist_async_stub",
            description="Create an async stub used to test persistence/recovery flows.",
            function=persist_async_stub,
            args_schema={
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
        )
    )

    # Namespace-specific business tool packs
    register_sales_tools()
    register_marketing_tools()
    register_finance_tools()
    register_hr_tools()
    register_it_tools()
    register_web_scraper_tools()


__all__ = [
    "register_builtin_tools",
    "search_web",
    "search_news",
    "ask_ai",
    "classify_text",
    "list_files",
    "read_file",
    "join_list",
    "split_list",
    "summarize",
    "compose_outlook_email",
    "record_health_event",
    "get_today_temperatures",
    "update_employee_health_profile",
    "async_stub_tool",
    "persist_async_stub",
    "create_incident",
    "create_lead",
    "log_interaction",
]
