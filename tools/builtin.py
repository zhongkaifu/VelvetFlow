from __future__ import annotations

"""A small set of real, ready-to-use business tools."""

import html
import asyncio
import importlib
import importlib.util
import json
import re
import textwrap
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import inspect

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL

from tools.base import Tool
from tools.registry import register_tool


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


def ask_ai(
    *,
    prompt: str | None = None,
    question: str | None = None,
    context: Dict[str, Any] | None = None,
    expected_format: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """Send a prompt to the configured OpenAI model and return structured output.

    The tool accepts either a raw ``prompt`` or high-level ``question`` +
    ``context`` fields that mirror the ``common.ask_ai.v1`` action schema. When
    structured inputs are provided, the tool generates a prompt that requests
    JSON output matching ``expected_format`` (if supplied) and attempts to parse
    the model response back into an object. On parsing failure, the raw content
    is wrapped under ``{"answer": ...}`` to satisfy the action's output
    contract.
    """

    if not (prompt or question):
        raise ValueError("either prompt or question must be provided for ask_ai")

    if prompt is None:
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
        prompt = "\n\n".join(parts)

    client = OpenAI()
    chat_model = model or OPENAI_MODEL
    response = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    if not response.choices:
        raise RuntimeError("ask_ai did not return any choices")

    message = response.choices[0].message
    content = message.content or ""

    parsed: Dict[str, Any]
    try:
        parsed_content = json.loads(content)
        parsed = parsed_content if isinstance(parsed_content, dict) else {"answer": parsed_content}
    except json.JSONDecodeError:
        parsed = {"answer": content}

    return {
        "result": parsed,
        "reasoning": getattr(message, "refusal", None) or "",
        "model": chat_model,
        "finish_reason": response.choices[0].finish_reason,
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


def _load_crawl4ai_module():
    """Ensure ``crawl4ai`` is available before attempting to use it."""

    if importlib.util.find_spec("crawl4ai") is None:
        raise RuntimeError(
            "crawl4ai is required for this tool. Install it with `pip install crawl4ai` and retry."
        )
    return importlib.import_module("crawl4ai")


def _resolve_crawl4ai_class(name: str):
    """Best-effort lookup for Crawl4AI classes across known modules."""

    for module_name in ("crawl4ai", "crawl4ai.crawler", "crawl4ai.web_crawler"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    return None


def _extract_crawl4ai_content(result: Any) -> str:
    """Extract the richest text payload returned by Crawl4AI."""

    if result is None:
        return ""

    candidates = []
    for attr in ("markdown_v2", "markdown", "cleaned_html", "html"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            candidates.append(value)
        elif value is not None:
            raw_markdown = getattr(value, "raw_markdown", None)
            if isinstance(raw_markdown, str):
                candidates.append(raw_markdown)

    for candidate in candidates:
        normalized = textwrap.dedent(candidate).strip()
        if normalized:
            return normalized

    return ""


def _normalize_url(url: str) -> str:
    """Ensure the URL has a supported scheme, defaulting to HTTPS."""

    trimmed = url.strip()
    allowed_prefixes = ("http://", "https://", "file://")
    if not trimmed.lower().startswith(allowed_prefixes):
        return f"https://{trimmed}"
    return trimmed


def _extract_query_answer(result: Any) -> str:
    """Return a query-focused answer if Crawl4AI produced one."""

    if result is None:
        return ""

    candidates = []
    for attr in (
        "answer",
        "query_answer",
        "qa_text",
        "qa_markdown",
        "qa_markdown_v2",
    ):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            candidates.append(value)

    for candidate in candidates:
        normalized = textwrap.dedent(candidate).strip()
        if normalized:
            return normalized

    return ""


def _summarize_head(content: str, max_sentences: int) -> str:
    """Provide a brief head summary when the crawler does not answer directly."""

    sentences = re.split(r"(?<=[。！？.!?])\s+", content)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:max_sentences]).strip()


def crawl_and_summarize(
    url: str, query: str, *, max_chars: int = 2000, max_sentences: int = 5
) -> Dict[str, Any]:
    """Crawl a web page with Crawl4AI and return a query-focused summary."""

    if not url:
        raise ValueError("url is required for crawling")
    if not query:
        raise ValueError("query is required to focus the extraction")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    url = _normalize_url(url)

    _load_crawl4ai_module()
    WebCrawler = _resolve_crawl4ai_class("WebCrawler")
    AsyncWebCrawler = _resolve_crawl4ai_class("AsyncWebCrawler")

    if WebCrawler is not None:
        crawler = WebCrawler(verbose=False)
        warmup = getattr(crawler, "warmup", None)
        if callable(warmup):
            warmup()
        result = crawler.run(url=url, query=query)
    elif AsyncWebCrawler is not None:
        async def _async_crawl() -> Any:
            async with AsyncWebCrawler(verbose=False) as crawler:
                warmup = getattr(crawler, "warmup", None)
                if callable(warmup):
                    maybe_coroutine = warmup()
                    if inspect.iscoroutine(maybe_coroutine):
                        await maybe_coroutine
                return await crawler.arun(url=url, query=query)

        result = asyncio.run(_async_crawl())
    else:
        raise RuntimeError(
            "crawl4ai does not provide WebCrawler/AsyncWebCrawler in the installed version; "
            "upgrade crawl4ai or use a compatible release."
        )
    content = _extract_crawl4ai_content(result)
    if not content:
        raise RuntimeError("crawl4ai did not return any textual content for the requested page")

    answer = _extract_query_answer(result)
    summary = answer or _summarize_head(content, max_sentences=max_sentences)
    preview = textwrap.shorten(content, width=max_chars, placeholder="...")

    return {
        "url": url,
        "query": query,
        "summary": summary,
        "answer": answer,
        "content_preview": preview,
        "content_length": len(content),
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
            description="Send a prompt to OpenAI chat completion and return the answer.",
            function=ask_ai,
            args_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "nullable": True},
                    "question": {"type": "string", "nullable": True},
                    "context": {"type": "object", "nullable": True},
                    "expected_format": {"type": "string", "nullable": True},
                    "model": {"type": "string", "nullable": True},
                    "max_tokens": {"type": "integer", "default": 256},
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
            name="crawl_and_summarize",
            description="Use Crawl4AI to fetch a web page, extract text, and summarize it for a query.",
            function=crawl_and_summarize,
            args_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 2000},
                    "max_sentences": {"type": "integer", "default": 5},
                },
                "required": ["url", "query"],
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
    "crawl_and_summarize",
]
