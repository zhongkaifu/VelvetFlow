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

try:
    from crawl4ai import BrowserConfig, CacheMode, CrawlerRunConfig, LLMConfig
    try:  # Prefer native sync crawler when available
        from crawl4ai import WebCrawler
    except ImportError:  # pragma: no cover - backward compatibility
        WebCrawler = None

    try:
        from crawl4ai import AsyncWebCrawler as _AsyncWebCrawler
    except ImportError:  # pragma: no cover - async crawler not available
        _AsyncWebCrawler = None

    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    _CRAWL4AI_AVAILABLE = True

    if WebCrawler is None and _AsyncWebCrawler is not None:  # pragma: no cover - exercised when sync crawler missing
        import asyncio

        class WebCrawler:  # type: ignore[misc]
            """Synchronous shim built on crawl4ai.AsyncWebCrawler.

            A dedicated event loop is kept alive for the lifetime of the crawler
            so Playwright connections do not get tied to short-lived loops.
            """

            def __init__(self, config: Any) -> None:
                self._config = config
                self._crawler: Any | None = None
                self._loop: asyncio.AbstractEventLoop | None = None

            def __enter__(self) -> "WebCrawler":
                async def _start() -> Any:
                    crawler = _AsyncWebCrawler(config=self._config)
                    await crawler.__aenter__()
                    return crawler

                # Keep a dedicated loop alive for the crawler lifecycle to avoid
                # Playwright connections being associated with a closed loop.
                self._loop = asyncio.new_event_loop()
                self._crawler = self._loop.run_until_complete(_start())
                return self

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if self._crawler is None or self._loop is None:  # pragma: no cover - defensive guard
                    raise RuntimeError("Crawler not initialized")

                async def _run() -> Any:
                    return await self._crawler.arun(*args, **kwargs)

                return self._loop.run_until_complete(_run())

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                if self._crawler is None or self._loop is None:  # pragma: no cover - defensive guard
                    return

                async def _stop() -> None:
                    await self._crawler.__aexit__(exc_type, exc, tb)

                try:
                    self._loop.run_until_complete(_stop())
                finally:
                    self._loop.close()

    AsyncWebCrawler = _AsyncWebCrawler
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without crawl4ai
    _CRAWL4AI_AVAILABLE = False

    class _MissingCrawlerDependency:
        def __getattr__(self, name: str) -> None:  # pragma: no cover - defensive fallback
            raise ImportError(
                "crawl4ai is required for web scraping tools; install it with 'pip install crawl4ai'."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive fallback
            raise ImportError(
                "crawl4ai is required for web scraping tools; install it with 'pip install crawl4ai'."
            )

    class _MissingCacheMode:
        BYPASS = "bypass"

    AsyncWebCrawler = BrowserConfig = CrawlerRunConfig = LLMConfig = WebCrawler = _MissingCrawlerDependency()
    CacheMode = _MissingCacheMode()
    LLMExtractionStrategy = _MissingCrawlerDependency()


def _require_crawl4ai() -> None:
    if not _CRAWL4AI_AVAILABLE:
        raise ImportError(
            "crawl4ai is required for web scraping tools; install it with 'pip install crawl4ai'."
        )
from openai import OpenAI

from velvetflow.config import OPENAI_MODEL

from tools.base import Tool
from tools.business import (
    register_finance_tools,
    register_hr_tools,
    register_it_tools,
    register_marketing_tools,
    register_sales_tools,
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
    if raw_url.startswith("/url") or "google.com/url" in raw_url:
        parsed = urllib.parse.urlparse(raw_url)
        query_params = urllib.parse.parse_qs(parsed.query)
        target = query_params.get("q") or query_params.get("url")
        if target:
            return _normalize_web_url(target[0])
    return _normalize_web_url(raw_url)


class _GoogleParser(HTMLParser):
    """Minimal parser to extract organic search results from Google HTML."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: List[Dict[str, str]] = []
        self._in_result = False
        self._result_depth = 0
        self._capturing_title = False
        self._capturing_snippet = False
        self._current_href: str | None = None
        self._title_parts: List[str] = []
        self._snippet_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:  # pragma: no cover - HTML parsing
        attrs_dict = {k: v or "" for k, v in attrs}
        classes = attrs_dict.get("class", "")

        if tag == "div":
            if not self._in_result and "tF2Cxc" in classes:
                self._in_result = True
                self._result_depth = 1
                self._current_href = None
                self._title_parts = []
                self._snippet_parts = []
            elif self._in_result:
                self._result_depth += 1

        if not self._in_result:
            return

        if tag == "a" and not self._current_href:
            href = attrs_dict.get("href")
            if href:
                self._current_href = href

        if tag == "h3":
            self._capturing_title = True
            self._title_parts = []

        snippet_classes = {"VwiC3b", "aCOpRe", "yXK7lf"}
        if tag in {"div", "span"} and snippet_classes.intersection(classes.split()):
            if not self._capturing_snippet:
                self._snippet_parts = []
            self._capturing_snippet = True

    def handle_data(self, data: str) -> None:  # pragma: no cover - HTML parsing
        if self._capturing_title:
            self._title_parts.append(data.strip())
        if self._capturing_snippet:
            self._snippet_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:  # pragma: no cover - HTML parsing
        if self._capturing_title and tag == "h3":
            self._capturing_title = False

        if self._capturing_snippet and tag in {"div", "span"}:
            self._capturing_snippet = False

        if self._in_result and tag == "div":
            self._result_depth -= 1
            if self._result_depth == 0:
                title = " ".join(self._title_parts).strip()
                snippet = " ".join(self._snippet_parts).strip()
                if title and self._current_href:
                    self.results.append({"title": title, "url": self._current_href, "snippet": snippet})
                self._in_result = False
                self._current_href = None
                self._title_parts = []
                self._snippet_parts = []
                if len(self.results) >= self.limit:
                    raise StopIteration


def search_web(query: str, limit: int = 5, timeout: int = 8) -> Dict[str, List[Dict[str, str]]]:
    """Perform a Google HTML search and return top results with snippets.

    The function prefers a Selenium-backed fetch (to better mirror user browser
    behavior and bypass bot detection) when selenium and a chromedriver are
    available. It falls back to a lightweight HTTP request otherwise, keeping
    the output format consistent either way.
    """

    def _clean_text(value: str) -> str:
        text_no_tags = re.sub(r"<[^>]+>", " ", value)
        text_unescaped = html.unescape(text_no_tags)
        return re.sub(r"\s+", " ", text_unescaped).strip()

    def _fetch_html() -> str:
        encoded_q = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_q}&num={max(limit, 1)}&hl=en"

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
        except Exception:  # pragma: no cover - selenium not installed
            webdriver = None  # type: ignore
            Service = None  # type: ignore

        if webdriver and Service:
            chromedriver_path = os.getenv("GOOGLE_CHROMEDRIVER_PATH") or os.getenv(
                "CHROMEDRIVER_PATH"
            )
            service = Service(chromedriver_path) if chromedriver_path else Service()
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            driver = None
            try:
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_page_load_timeout(timeout)
                driver.get(url)
                return driver.page_source
            except Exception:
                # If Selenium fails (e.g., chromedriver missing), fall back to HTTP.
                pass
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; VelvetFlow/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    if not query:
        raise ValueError("query is required for web search")

    try:
        raw_html = _fetch_html()
    except Exception as exc:  # pragma: no cover - network-dependent
        raise RuntimeError(f"web search failed: {exc}") from exc

    parser = _GoogleParser(limit)
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
        "results": {"message": "ask_ai could not produce an answer within the allowed rounds"},
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
        max_tokens=max(64, max_sentences * 64),
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


def _normalize_internal_url(candidate: str, base_url: str) -> str:
    """Resolve and validate that the candidate URL stays within the base domain."""

    if not candidate:
        return ""

    joined = urllib.parse.urljoin(base_url, candidate)
    parsed_base = urllib.parse.urlparse(base_url)
    parsed_candidate = urllib.parse.urlparse(joined)

    if parsed_candidate.scheme not in {"http", "https"}:
        return ""

    base_host = parsed_base.netloc.lower().lstrip("www.")
    candidate_host = parsed_candidate.netloc.lower().lstrip("www.")

    if base_host != candidate_host:
        return ""

    return joined


def _extract_internal_links(result: Any, base_url: str) -> List[str]:
    """Best-effort extraction of in-domain links from a crawl result."""

    raw_collections: List[Any] = []

    for attr in ("links", "hyperlinks", "anchors"):
        value = getattr(result, attr, None)
        if value:
            raw_collections.append(value)

    candidates: List[str] = []

    for collection in raw_collections:
        if isinstance(collection, Mapping):
            candidates.extend(str(item) for item in collection.values())
        elif isinstance(collection, list):
            for entry in collection:
                if isinstance(entry, str):
                    candidates.append(entry)
                elif isinstance(entry, Mapping):
                    href = entry.get("url") or entry.get("href") or ""
                    if href:
                        candidates.append(str(href))
                elif hasattr(entry, "url"):
                    href = getattr(entry, "url")
                    if href:
                        candidates.append(str(href))
        elif isinstance(collection, str):
            candidates.append(collection)

    normalized: List[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        normalized_candidate = _normalize_internal_url(candidate, base_url)
        if normalized_candidate and normalized_candidate not in seen:
            normalized.append(normalized_candidate)
            seen.add(normalized_candidate)

    return normalized


def _truncate_for_llm(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _should_continue_crawling(
    user_request: str,
    pages: List[Dict[str, Any]],
    remaining_links: List[str],
    api_token: Optional[str],
    llm_provider: str,
    max_pages: int,
) -> Tuple[bool, List[str]]:
    """Use an LLM to decide whether deeper in-site crawling is needed."""

    if not api_token:
        return len(pages) >= 1, []

    model = llm_provider
    if model.startswith("openai/"):
        model = model.split("/", 1)[1]

    client = OpenAI(api_key=api_token)

    page_summaries = []
    for page in pages[-3:]:  # limit context size
        content = str(page.get("extracted_content", ""))
        page_summaries.append(
            {
                "url": page.get("url"),
                "content": _truncate_for_llm(content),
                "status": page.get("status"),
            }
        )

    prompt = textwrap.dedent(
        f"""
        You are deciding if the current crawl satisfies the user's request or needs to go deeper.
        User request: {user_request}

        Pages already crawled (most recent first, trimmed content included):
        {json.dumps(page_summaries, ensure_ascii=False)}

        Remaining in-site links that can be crawled next (respect domain boundaries only):
        {json.dumps(remaining_links, ensure_ascii=False)}

        Respond with strict JSON using keys:
        - satisfied (boolean): whether the existing pages already satisfy the request.
        - reason (string): brief justification.
        - next_links (array of strings): optional prioritized subset of remaining links to crawl next.

        If the request is already satisfied or no useful links remain, set satisfied to true.
        """
    ).strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a crawl coordinator."}, {"role": "user", "content": prompt}],
        max_tokens=256,
    )

    content = (response.choices[0].message.content or "{}") if response.choices else "{}"

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    satisfied = bool(data.get("satisfied")) or len(pages) >= max_pages or not remaining_links
    next_links_raw = data.get("next_links") or []

    prioritized: List[str] = []
    for link in next_links_raw:
        if isinstance(link, str) and link in remaining_links and link not in prioritized:
            prioritized.append(link)

    return satisfied, prioritized


def _crawl_page_once(
    url: str,
    instruction: str,
    api_token: Optional[str],
    llm_provider: str,
) -> Dict[str, Any]:
    """Crawl a single URL and return extracted content plus in-domain links."""

    browser_conf = BrowserConfig(headless=True, verbose=False)
    attempts: List[Dict[str, Any]] = []

    with WebCrawler(config=browser_conf) as crawler:
        result = None

        if api_token:
            llm_conf = LLMConfig(provider=llm_provider, api_token=api_token)
            llm_strategy = LLMExtractionStrategy(
                llm_config=llm_conf,
                schema=None,
                extraction_type="text",
                instruction=instruction,
                input_format="markdown",
                verbose=False,
            )
            llm_run = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=llm_strategy)
            llm_result = crawler.run(url=url, config=llm_run)
            attempts.append(
                {
                    "method": "llm_extraction",
                    "success": getattr(llm_result, "success", False),
                    "status_code": getattr(llm_result, "status_code", None),
                    "error": getattr(llm_result, "error_message", None),
                }
            )
            if getattr(llm_result, "success", False):
                result = llm_result

        if result is None or not getattr(result, "success", False):
            fallback_run = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            fallback_result = crawler.run(url=url, config=fallback_run)
            attempts.append(
                {
                    "method": "raw_scrape",
                    "success": getattr(fallback_result, "success", False),
                    "status_code": getattr(fallback_result, "status_code", None),
                    "error": getattr(fallback_result, "error_message", None),
                }
            )
            result = fallback_result

    # result is guaranteed to be set after the crawler context
    assert result is not None

    raw_content = getattr(result, "extracted_content", "") or ""
    if isinstance(raw_content, (dict, list)):
        extracted_content = json.dumps(raw_content, ensure_ascii=False)
    else:
        extracted_content = str(raw_content).strip()

    if not extracted_content and getattr(result, "markdown", None):
        extracted_content = str(getattr(result, "markdown")).strip()

    links = _extract_internal_links(result, url)
    status = "ok" if getattr(result, "success", False) else "error"

    return {
        "status": status,
        "extracted_content": extracted_content,
        "links": links,
    }


# Preserve the default implementation so tests can detect when it has been patched.
_DEFAULT_CRAWL_PAGE_ONCE = _crawl_page_once


def _scrape_single_url(
    url: str,
    user_request: str,
    *,
    llm_instruction: Optional[str] = None,
    llm_provider: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Download and analyze a single web page according to a request."""

    if _CRAWL4AI_AVAILABLE or _crawl_page_once is not _DEFAULT_CRAWL_PAGE_ONCE:
        pass
    else:
        _require_crawl4ai()

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
    max_pages = 6

    visited: set[str] = set()
    to_visit: List[str] = [url]
    collected: List[Dict[str, Any]] = []
    aggregated_contents: List[str] = []

    while to_visit and len(collected) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue

        visited.add(current)
        page_result = _crawl_page_once(current, instruction, api_token, llm_provider)
        page_result["url"] = current
        collected.append(page_result)

        content = page_result.get("extracted_content")
        if content:
            aggregated_contents.append(str(content))

        for link in page_result.get("links", []):
            normalized_link = _normalize_internal_url(link, current)
            if normalized_link and normalized_link not in visited and normalized_link not in to_visit:
                to_visit.append(normalized_link)

        satisfied, prioritized_links = _should_continue_crawling(
            user_request,
            collected,
            to_visit,
            api_token,
            llm_provider,
            max_pages,
        )

        if prioritized_links:
            remaining = [link for link in to_visit if link not in prioritized_links]
            to_visit = prioritized_links + remaining

        if satisfied:
            break

    successes = [item.get("status") == "ok" for item in collected]
    overall_status = "ok" if any(successes) else "error"

    aggregated_content = "\n".join(aggregated_contents)

    return {
        "status": overall_status,
        "extracted_content": aggregated_content,
    }


def _aggregate_scrape_results(results: List[Dict[str, Any]], user_request: str) -> str:
    """Aggregate per-URL scrape outputs into a human-readable summary."""

    if not results:
        return "No pages were scraped."

    bullet_points: List[str] = []
    for item in results:
        url = item.get("url", "")
        status = item.get("status", "unknown")
        analysis = item.get("analysis") or item.get("markdown") or ""

        if isinstance(analysis, (dict, list)):
            analysis_text = json.dumps(analysis, ensure_ascii=False)
        else:
            analysis_text = str(analysis).strip()

        if analysis_text:
            bullet_points.append(f"- [{status}] {url}: {analysis_text}")
        else:
            bullet_points.append(f"- [{status}] {url}: No analysis available")

    header = f"Aggregate summary for request: {user_request}".strip()
    return "\n".join([header, *bullet_points])


def scrape_web_page(
    urls: List[str],
    user_request: str,
    *,
    llm_instruction: Optional[str] = None,
    llm_provider: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Download and analyze one or more web pages according to a natural-language request."""

    if isinstance(urls, str):
        urls = [urls]

    if not isinstance(urls, list) or not urls:
        raise ValueError("urls must be a non-empty list of strings")
    if not user_request:
        raise ValueError("user_request is required for scraping")

    normalized_urls: List[str] = []
    for url in urls:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("each url must be a non-empty string")
        normalized_urls.append(url.strip())

    per_page_results: List[Dict[str, Any]] = []
    extracted_parts: List[str] = []
    for url in normalized_urls:
        page_result = _scrape_single_url(
            url,
            user_request,
            llm_instruction=llm_instruction,
            llm_provider=llm_provider,
        )
        per_page_results.append(page_result)
        content = page_result.get("extracted_content")
        if content:
            extracted_parts.append(str(content))

    successes = [item.get("status") == "ok" for item in per_page_results]
    overall_status = "ok" if all(successes) else "partial" if any(successes) else "error"

    aggregated_content = "\n".join(extracted_parts)

    return {
        "status": overall_status,
        "extracted_content": aggregated_content,
    }


def business_action(
    query: str,
    *,
    search_limit: int = 5,
    llm_provider: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Search the web, crawl relevant pages, and summarize findings."""

    if not query:
        raise ValueError("query is required for business_action")

    search_output = search_web(query, limit=search_limit)
    search_results = search_output.get("results", []) if isinstance(search_output, Mapping) else []

    if not search_results:
        return {
            "status": "not_found",
            "summary": f"No search results found for query: {query}",
            "results": [],
        }

    crawled: List[Dict[str, Any]] = []

    for entry in search_results:
        url = entry.get("url") if isinstance(entry, Mapping) else None
        if not url:
            continue

        scrape_result = _scrape_single_url(url, query, llm_provider=llm_provider)

        crawled.append(
            {
                "title": entry.get("title", "") if isinstance(entry, Mapping) else "",
                "url": url,
                "snippet": entry.get("snippet", "") if isinstance(entry, Mapping) else "",
                "scrape": scrape_result,
            }
        )

    if not crawled:
        return {
            "status": "not_found",
            "summary": f"Search results lacked crawlable URLs for query: {query}",
            "results": [],
        }

    analysis_inputs = []
    successes: List[bool] = []

    for item in crawled:
        scrape = item.get("scrape") or {}
        status = scrape.get("status")
        successes.append(status == "ok")
        analysis_inputs.append(
            {
                "url": item.get("url"),
                "status": status,
                "analysis": scrape.get("extracted_content"),
            }
        )

    ok_count = sum(successes)
    if ok_count == len(successes):
        overall_status = "ok"
    elif ok_count > 0:
        overall_status = "partial"
    else:
        overall_status = "error"

    summary = _aggregate_scrape_results(analysis_inputs, query)

    return {
        "status": overall_status,
        "summary": summary,
        "results": crawled,
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


def register_builtin_tools() -> None:
    """Register all built-in tools in the global registry."""

    register_tool(
        Tool(
            name="search_web",
            description="Perform a Google search and return titles, URLs, and snippets.",
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
            name="scrape_web_page",
            description="Download a web page and analyze it per a natural-language request.",
            function=scrape_web_page,
            args_schema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "user_request": {"type": "string"},
                    "llm_instruction": {"type": "string", "nullable": True},
                    "llm_provider": {"type": "string", "default": "openai/gpt-4o-mini"},
                },
                "required": ["urls", "user_request"],
            },
        )
    )

    register_tool(
        Tool(
            name="search_and_crawl_web_content",
            description=(
                "Search Google for a natural-language query, crawl in-domain pages with crawl4ai, let an LLM assess coverage, and return a multi-page summary."
            ),
            function=business_action,
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_limit": {"type": "integer", "default": 5},
                    "llm_provider": {"type": "string", "default": "openai/gpt-4o-mini"},
                },
                "required": ["query"],
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

    # Namespace-specific business tool packs
    register_sales_tools()
    register_marketing_tools()
    register_finance_tools()
    register_hr_tools()
    register_it_tools()


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
    "scrape_web_page",
    "business_action",
    "record_health_event",
    "get_today_temperatures",
    "update_employee_health_profile",
    "create_incident",
    "create_lead",
    "log_interaction",
]
