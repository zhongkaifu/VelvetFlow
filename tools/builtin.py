from __future__ import annotations

"""A small set of real, ready-to-use business tools."""

import re
import textwrap
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL

from tools.base import Tool
from tools.registry import register_tool


class _DuckDuckGoParser(HTMLParser):
    """Minimal parser to extract search results from DuckDuckGo HTML."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: List[Tuple[str, str]] = []
        self._capturing = False
        self._current_href: str | None = None
        self._text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:  # pragma: no cover - HTML parsing
        if tag != "a":
            return
        attrs_dict = {k: v for k, v in attrs}
        if "result__a" in (attrs_dict.get("class") or ""):
            self._capturing = True
            self._current_href = attrs_dict.get("href")
            self._text_parts = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - HTML parsing
        if self._capturing:
            self._text_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:  # pragma: no cover - HTML parsing
        if self._capturing and tag == "a":
            title = " ".join(self._text_parts).strip()
            if title and self._current_href:
                self.results.append((title, self._current_href))
            self._capturing = False
            self._text_parts = []
            self._current_href = None
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
    for title, href in parser.results[:limit]:
        results.append({"title": title, "url": href})
    return {"results": results}


def ask_ai(prompt: str, *, model: str | None = None, max_tokens: int = 256) -> Dict[str, Any]:
    """Send a prompt to the configured OpenAI model and return the response."""

    if not prompt:
        raise ValueError("prompt is required for ask_ai")

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
    return {
        "model": chat_model,
        "answer": message.content or "",
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
    """Summarize a block of text using a simple sentence selection heuristic."""

    if not text:
        raise ValueError("text is required for summarization")

    sentences = re.split(r"(?<=[。！？.!?])\s+", textwrap.dedent(text).strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    summary = " ".join(sentences[:max_sentences])
    return {"summary": summary, "sentence_count": len(sentences)}


def register_builtin_tools() -> None:
    """Register all built-in tools in the global registry."""

    register_tool(
        Tool(
            name="search_web",
            description="Perform a DuckDuckGo search and return titles + URLs.",
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
            name="ask_ai",
            description="Send a prompt to OpenAI chat completion and return the answer.",
            function=ask_ai,
            args_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model": {"type": "string", "nullable": True},
                    "max_tokens": {"type": "integer", "default": 256},
                },
                "required": ["prompt"],
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
            description="Summarize text by selecting the first few sentences.",
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


__all__ = [
    "register_builtin_tools",
    "search_web",
    "ask_ai",
    "list_files",
    "read_file",
    "summarize",
]
