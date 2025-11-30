from __future__ import annotations

"""A small set of real, ready-to-use business tools."""

import asyncio
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
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from openai import OpenAI

from velvetflow.config import OPENAI_MODEL

from tools.base import Tool
from tools.registry import get_registered_tool, register_tool


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
    """Perform a lightweight DuckDuckGo HTML search and return top results."""

    if not query:
        raise ValueError("query is required for web search")

    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded_q}&kl=wt-wt"
    req = urllib.request.Request(url, headers={"User-Agent": "VelvetFlow/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - network-dependent
        raise RuntimeError(f"web search failed: {exc}") from exc

    parser = _DuckDuckGoParser(limit=limit)
    try:
        parser.feed(raw_html)
    except StopIteration:
        pass

    results: List[Dict[str, str]] = []
    for item in parser.results[:limit]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            }
        )
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
    expected_format: str | None,
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
        parts.append("Return JSON following this guidance:")
        parts.append(expected_format)
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
            "message": f"业务工具 '{tool_fn_name}' 未注册，返回原始参数供参考。",
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
    expected_format: str | None = None,
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
    system_text = system_prompt or "你是一个善于调用业务工具解决问题的智能助手。"

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

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            tools=tools_spec or None,
            tool_choice="auto" if tools_spec else None,
            max_tokens=max_tokens,
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
                        "message": f"未知的业务工具调用: {func_name}",
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
            continue

        final_content = (message.content or "").strip()
        if not final_content:
            continue

        try:
            parsed_content = json.loads(final_content)
            results = parsed_content if isinstance(parsed_content, dict) else {"answer": parsed_content}
        except json.JSONDecodeError:
            results = {"answer": final_content}

        return {"status": "ok", "results": results}

    return {
        "status": "error",
        "results": {"message": "ask_ai 未能在限定轮次内给出答案"},
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
        max_tokens=max_tokens,
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
        max_tokens=max(64, max_sentences * 64),
    )
    if not response.choices:
        raise RuntimeError("summarize did not return any choices")

    summary = (response.choices[0].message.content or "").strip()
    return {"summary": summary, "sentence_count": len(sentences)}


def _run_coroutine(factory: Callable[[], Awaitable[Any]]) -> Any:
    """Safely execute an async coroutine factory from sync code."""

    try:
        return asyncio.run(factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(factory())
        finally:
            loop.close()


def scrape_web_page(
    url: str,
    user_request: str,
    *,
    llm_instruction: Optional[str] = None,
    llm_provider: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Download and analyze a web page according to a natural-language request."""

    if not url:
        raise ValueError("url is required for scraping")
    if not user_request:
        raise ValueError("user_request is required for scraping")

    instruction = llm_instruction or textwrap.dedent(
        f"""
        You are a web analysis agent. Read the page content and analyze it for the user's request.
        User request: {user_request}

        Return JSON with fields:
        - summary: a concise overview of the page relevant to the user request
        - key_points: bullet points highlighting important facts or insights
        - relevance: short explanation of how the content addresses the request
        """
    ).strip()

    api_token = os.getenv("OPENAI_API_KEY")
    use_llm = bool(api_token)

    async def _scrape() -> tuple[Any, List[Dict[str, Any]]]:
        browser_conf = BrowserConfig(headless=True, verbose=False)
        attempts: List[Dict[str, Any]] = []

        async with AsyncWebCrawler(config=browser_conf) as crawler:
            if use_llm:
                llm_conf = LLMConfig(provider=llm_provider, api_token=api_token)
                llm_strategy = LLMExtractionStrategy(
                    llm_config=llm_conf,
                    schema=None,
                    extraction_type="text",
                    instruction=instruction,
                    input_format="markdown",
                    verbose=False,
                )
                llm_run = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, extraction_strategy=llm_strategy
                )
                llm_result = await crawler.arun(url=url, config=llm_run)
                attempts.append(
                    {
                        "method": "llm_extraction",
                        "success": llm_result.success,
                        "status_code": llm_result.status_code,
                        "error": llm_result.error_message,
                    }
                )
                if llm_result.success:
                    return llm_result, attempts

            fallback_run = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            fallback_result = await crawler.arun(url=url, config=fallback_run)
            attempts.append(
                {
                    "method": "raw_scrape",
                    "success": fallback_result.success,
                    "status_code": fallback_result.status_code,
                    "error": fallback_result.error_message,
                }
            )
            return fallback_result, attempts

    result, attempts = _run_coroutine(_scrape)

    raw_content = result.extracted_content or ""
    analysis: Any
    try:
        analysis = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
    except json.JSONDecodeError:
        analysis = raw_content.strip() if isinstance(raw_content, str) else raw_content

    if not analysis and getattr(result, "markdown", None):
        analysis = result.markdown

    status = "ok" if result.success else "error"

    return {
        "status": status,
        "url": url,
        "analysis": analysis,
        "markdown": getattr(result, "markdown", "") or "",
        "attempts": attempts,
        "llm_used": use_llm,
        "llm_provider": llm_provider if use_llm else None,
        "user_request": user_request,
    }


def register_builtin_tools() -> None:
    """Register all built-in tools in the global registry."""

    register_tool(
        Tool(
            name="search_web",
            description="Perform a DuckDuckGo search and return titles, URLs, and snippets.",
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
            name="scrape_web_page",
            description="Download a web page and analyze it per a natural-language request.",
            function=scrape_web_page,
            args_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "user_request": {"type": "string"},
                    "llm_instruction": {"type": "string", "nullable": True},
                    "llm_provider": {"type": "string", "default": "openai/gpt-4o-mini"},
                },
                "required": ["url", "user_request"],
            },
        )
    )


__all__ = [
    "register_builtin_tools",
    "search_web",
    "search_news",
    "ask_ai",
    "classify_text",
    "list_files",
    "read_file",
    "summarize",
    "scrape_web_page",
]
