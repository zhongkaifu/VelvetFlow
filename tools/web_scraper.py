# crawler_llm_v4_fixed_stateful.py
# -*- coding: utf-8 -*-

"""
Stateful LLM-driven crawler (Fix: current_summary always empty)
Key design fixes:
1) Summary state is owned by Python (NOT by LLM).
2) LLM returns DELTA updates only; Python merges them.
3) current_summary is always provided (compressed) to LLM.
4) Summary is persisted to disk (summary_state.json) for recovery.
5) Anti-WAF Playwright fetcher for CSDN-like sites; blocked pages are NOT summarized.
6) Full observability: logs LLM input/output + summary before/after.

Run:
  set OPENAI_API_KEY=...
  python crawler_llm_v4_fixed_stateful.py --seed "https://docs.python.org/" --goal "..." --same_domain --max_pages 30

Notes:
- Some sites (CSDN) may still block by IP/rate-limit. This code does best-effort browser-like fetch.
"""

from __future__ import annotations

# ============================================================
# Compatibility & sanity checks
# ============================================================

import sys, types, os, re, json, time, uuid, asyncio, argparse, heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Literal
from urllib.parse import urljoin, urldefrag, urlparse

# ---- lxml.html.clean shim (lxml>=5) ----
try:
    from lxml_html_clean import Cleaner as _Cleaner  # noqa
    shim = types.ModuleType("lxml.html.clean")
    shim.Cleaner = _Cleaner
    sys.modules["lxml.html.clean"] = shim
except Exception:
    pass

# ---- readability sanity check (must be readability-lxml) ----
try:
    from readability import Document  # noqa
except Exception as e:
    raise ImportError(
        "Failed to import Document from 'readability'.\n"
        "You likely installed the WRONG package 'readability'.\n\n"
        "Fix:\n"
        "  python -m pip uninstall -y readability\n"
        "  python -m pip install -U readability-lxml\n"
    ) from e

# ============================================================
# Third-party imports
# ============================================================

from bs4 import BeautifulSoup
from markdownify import markdownify as md
import trafilatura
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# Logging helpers
# ============================================================

def ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)

def log_llm_io(tag: str, prompt: dict, response: str, max_resp_chars: int = 12000):
    log(f"[LLM:{tag}] ===== INPUT =====")
    try:
        log(json.dumps(prompt, ensure_ascii=False, indent=2))
    except Exception:
        log(str(prompt))
    log(f"[LLM:{tag}] ===== OUTPUT =====")
    if len(response) <= max_resp_chars:
        log(response)
    else:
        log(response[: max_resp_chars // 2] + "\n...[TRUNCATED]...\n" + response[-max_resp_chars // 2 :])
    log(f"[LLM:{tag}] ==================")


# ============================================================
# Utils
# ============================================================

def now_ts() -> float:
    return time.time()

def same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)

def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith(("javascript:", "mailto:", "tel:")):
        return None
    full = urljoin(base, href)
    full, _ = urldefrag(full)
    return full

def looks_like_asset(url: str) -> bool:
    return bool(re.search(r"\.(png|jpg|jpeg|gif|webp|svg|pdf|zip|rar|7z|tar|gz|mp4|mp3|avi|mov|woff2?|ttf|eot)(\?|$)", url, re.I))

def shrink_text(text: str, max_chars: int) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n...[TRUNCATED]...\n\n" + tail

def tokenize_keywords(s: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", s)
    return list(dict.fromkeys([t.lower() for t in toks]))


# ============================================================
# HTML -> Clean Markdown
# ============================================================

MIN_MD_CHARS = 250

def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title:
        return soup.title.get_text(strip=True)
    return ""

def extract_markdown_trafilatura(html: str, url: str) -> str:
    out = trafilatura.extract(
        html,
        url=url,
        include_links=True,
        include_images=False,
        output_format="markdown",
        favor_precision=True,
        deduplicate=True,
    )
    return (out or "").strip()

def extract_markdown_readability(html: str, url: str) -> str:
    doc = Document(html)
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "aside"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        a["href"] = urljoin(url, a["href"])
    markdown = md(str(soup), heading_style="ATX")
    return (markdown or "").strip()

def extract_clean_markdown(html: str, url: str, max_chars: int) -> Tuple[str, str]:
    title = extract_title(html)
    md_text = extract_markdown_trafilatura(html, url)
    if not md_text or len(md_text) < MIN_MD_CHARS:
        md_text = extract_markdown_readability(html, url)
    md_text = shrink_text(md_text, max_chars=max_chars)
    return title, md_text

def extract_links_with_anchor(html: str, base_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(base_url, a["href"])
        if not u or looks_like_asset(u):
            continue
        anchor = a.get_text(" ", strip=True)[:200]
        out.append({"url": u, "anchor": anchor})
    # de-dup by url keep best anchor
    seen: Dict[str, Dict[str, str]] = {}
    for x in out:
        if x["url"] not in seen or len(x["anchor"]) > len(seen[x["url"]]["anchor"]):
            seen[x["url"]] = x
    return list(seen.values())


# ============================================================
# Playwright Fetcher (Anti-WAF)
# ============================================================

class PlaywrightFetcher:
    def __init__(self, timeout_ms: int = 20000, wait_until: str = "domcontentloaded"):
        self.timeout_ms = timeout_ms
        self.wait_until = wait_until

    async def __aenter__(self):
        log("[PLAYWRIGHT] starting")
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        log("[PLAYWRIGHT] browser launched")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        log("[PLAYWRIGHT] stopping")
        await self._browser.close()
        await self._pw.stop()

    async def fetch(self, url: str) -> Tuple[str, str]:
        USER_AGENT = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )

        ctx = await self._browser.new_context(
            user_agent=USER_AGENT,
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1280, "height": 800},
            extra_http_headers={
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        page = await ctx.new_page()

        try:
            # hide webdriver
            await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            await page.goto(url, wait_until=self.wait_until, timeout=self.timeout_ms)

            # simulate human behavior
            await page.wait_for_timeout(1200)
            await page.mouse.wheel(0, 800)
            await page.wait_for_timeout(800)

            html = await page.content()

            # detect WAF pages
            waf_markers = ["403 Forbidden", "WAF", "captcha", "安全验证", "访问异常", "Request blocked"]
            if any(m in html for m in waf_markers):
                return "blocked", ""

            return "ok", html

        except PlaywrightTimeoutError:
            return "timeout", ""
        except Exception:
            return "error", ""
        finally:
            await page.close()
            await ctx.close()


# ============================================================
# Summary State (Python-owned, persistent)
# ============================================================

class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    type: Literal["fact", "relation", "code", "open_question"]
    text: str
    sources: List[str] = Field(default_factory=list)

class SummarizerState(BaseModel):
    # IMPORTANT: use default_factory to avoid mutable default pitfalls
    facts: List[EvidenceItem] = Field(default_factory=list)
    relations: List[EvidenceItem] = Field(default_factory=list)
    code: List[EvidenceItem] = Field(default_factory=list)
    open_questions: List[EvidenceItem] = Field(default_factory=list)

    def to_markdown(self, max_items_each: int = 60) -> str:
        def sec(title: str, items: List[EvidenceItem]):
            lines = [f"## {title}"]
            for it in items[:max_items_each]:
                src = ", ".join(it.sources[:3]) + (" ..." if len(it.sources) > 3 else "")
                lines.append(f"- ({it.id}) {it.text}  \n  Sources: {src}")
            return "\n".join(lines)

        parts = ["# Evidence Summary\n"]
        if self.facts: parts.append(sec("Facts", self.facts))
        if self.relations: parts.append(sec("Relations", self.relations))
        if self.code: parts.append(sec("Code", self.code))
        if self.open_questions: parts.append(sec("Open Questions", self.open_questions))
        return "\n\n".join(parts).strip()

    def compact_for_prompt(self, max_each: int = 25) -> dict:
        # compress to reduce tokens (rolling summary)
        return {
            "facts": [x.text for x in self.facts[:max_each]],
            "relations": [x.text for x in self.relations[:max_each]],
            "code": [x.text for x in self.code[:max_each]],
            "open_questions": [x.text for x in self.open_questions[:max_each]],
        }

    def _dedup_append(self, arr: List[EvidenceItem], new_item: EvidenceItem):
        # de-dup by normalized text
        nt = re.sub(r"\s+", " ", new_item.text.strip().lower())
        for it in arr:
            if re.sub(r"\s+", " ", it.text.strip().lower()) == nt:
                # merge sources
                merged = list(dict.fromkeys(it.sources + new_item.sources))
                it.sources = merged
                return
        arr.append(new_item)

    def apply_delta(self, delta: dict, page_url: str):
        # delta schema:
        # add_facts/add_relations/add_code/add_open_questions as list of {"text":..., "sources":[...optional]}
        for key, target_type, target_list in [
            ("add_facts", "fact", self.facts),
            ("add_relations", "relation", self.relations),
            ("add_code", "code", self.code),
            ("add_open_questions", "open_question", self.open_questions),
        ]:
            for obj in (delta.get(key) or [])[:200]:
                if not isinstance(obj, dict):
                    continue
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                sources = obj.get("sources") or [page_url]
                if page_url not in sources:
                    sources = [page_url] + list(sources)
                item = EvidenceItem(type=target_type, text=text, sources=list(dict.fromkeys(sources)))
                self._dedup_append(target_list, item)

        # optionally remove_open_questions
        remove = delta.get("remove_open_questions") or []
        if remove:
            remove_norm = set(re.sub(r"\s+", " ", r.strip().lower()) for r in remove if isinstance(r, str))
            self.open_questions = [
                it for it in self.open_questions
                if re.sub(r"\s+", " ", it.text.strip().lower()) not in remove_norm
            ]


# ============================================================
# LLM Modules
# ============================================================

class DeltaSummarizer:
    """
    Fix: LLM returns DELTA only; Python merges into SummarizerState.
    This guarantees current_summary is never lost.
    """
    def __init__(self, model: str, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.state = SummarizerState()

    def load_state_if_exists(self, path: str):
        if os.path.exists(path):
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
                self.state = SummarizerState(**data)
                log(f"[STATE] loaded summary state from {path}")
            except Exception as e:
                log(f"[STATE] failed to load {path}: {e}")

    def save_state(self, path: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"[STATE] failed to save {path}: {e}")

    async def update_with_page(self, goal: str, page_url: str, page_markdown: str):
        before_md = self.state.to_markdown()

        system = (
            "You are an incremental evidence summarizer.\n"
            "You MUST update the summary ONLY by returning a JSON DELTA (changes).\n"
            "DO NOT return the full summary.\n\n"
            "Rules:\n"
            "- Base all additions strictly on the given page_markdown.\n"
            "- Add atomic items.\n"
            "- If you add items, include sources=[page_url] (or omit sources and we will inject).\n"
            "- If an open question is answered by this page, list it in remove_open_questions.\n"
            "- Output ONLY valid JSON with keys:\n"
            "  add_facts, add_relations, add_code, add_open_questions, remove_open_questions\n"
        )

        prompt = {
            "goal": goal,
            "page_url": page_url,
            "current_summary": self.state.compact_for_prompt(max_each=30),
            "page_markdown": page_markdown[:8000],
            "delta_schema": {
                "add_facts": [{"text": "string", "sources": ["url(optional)"]}],
                "add_relations": [{"text": "string", "sources": ["url(optional)"]}],
                "add_code": [{"text": "python code or pseudo", "sources": ["url(optional)"]}],
                "add_open_questions": [{"text": "string", "sources": ["url(optional)"]}],
                "remove_open_questions": ["string"],
            },
        }

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.1,
        )

        raw = resp.choices[0].message.content or ""
        log_llm_io("SUMMARY_DELTA", {"goal": goal, "page_url": page_url, "current_summary": prompt["current_summary"]}, raw)

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            log("[SUMMARY] delta invalid JSON; skip")
            return

        try:
            delta = json.loads(m.group(0))
        except Exception as e:
            log(f"[SUMMARY] delta parse error: {e}")
            return

        # Apply merge in Python (the actual fix)
        self.state.apply_delta(delta, page_url)

        after_md = self.state.to_markdown()
        log("[SUMMARY] ===== BEFORE =====")
        log(before_md)
        log("[SUMMARY] ===== AFTER =====")
        log(after_md)


class SummaryCritique(BaseModel):
    goal_status: Literal["satisfied", "almost_satisfied", "not_satisfied", "unclear"]
    missing_information: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

class SummarySelfCritic:
    def __init__(self, model: str, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def critique(self, goal: str, summary_md: str) -> SummaryCritique:
        system = (
            "You are a strict evaluator.\n"
            "Decide if the summary satisfies the goal.\n"
            "Output ONLY valid JSON."
        )
        prompt = {
            "goal": goal,
            "summary": summary_md[:9000],
            "schema": {
                "goal_status": "satisfied|almost_satisfied|not_satisfied|unclear",
                "missing_information": ["string"],
                "contradictions": ["string"],
                "confidence": "0.0-1.0",
            },
        }
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or ""
        log_llm_io("SELF_CRITIC", {"goal": goal, "summary_head": summary_md[:1200]}, raw)

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return SummaryCritique(goal_status="unclear", missing_information=["invalid JSON"], contradictions=[], confidence=0.2)
        try:
            return SummaryCritique(**json.loads(m.group(0)))
        except Exception:
            return SummaryCritique(goal_status="unclear", missing_information=["parse error"], contradictions=[], confidence=0.2)


class URLScore(BaseModel):
    url: str
    score: int = Field(..., ge=0, le=100)
    reason: str

class URLScoringResponse(BaseModel):
    scores: List[URLScore]

class SummaryDrivenURLScorer:
    def __init__(self, model: str, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def score_urls(self, goal: str, summary_compact: dict, candidates: List[Dict[str, str]], top_k: int = 15) -> List[URLScore]:
        system = (
            "You are selecting which URLs to crawl next.\n"
            "Score based on likely usefulness for the goal, given what's missing in the summary.\n"
            "Use only URL patterns and anchor text.\n"
            "Output ONLY valid JSON."
        )
        prompt = {
            "goal": goal,
            "current_summary_compact": summary_compact,
            "candidates": candidates[:60],
            "schema": {"scores": [{"url": "string", "score": "0-100", "reason": "string"}]},
        }
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or ""
        log_llm_io("URL_SCORER", prompt, raw)

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        try:
            parsed = URLScoringResponse(**json.loads(m.group(0)))
        except Exception:
            return []
        return sorted(parsed.scores, key=lambda x: x.score, reverse=True)[:top_k]


class AnswerWithCitations(BaseModel):
    answer_markdown: str
    warnings: List[str] = Field(default_factory=list)

class EvidenceAwareAnswerGenerator:
    def __init__(self, model: str, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, goal: str, state: SummarizerState) -> AnswerWithCitations:
        system = (
            "You are an evidence-grounded answer writer.\n"
            "Only use the evidence items provided.\n"
            "Cite evidence IDs in brackets like [abc123].\n"
            "If missing info, list it in warnings.\n"
            "Output ONLY valid JSON."
        )

        # Provide full items (still bounded)
        items = []
        for it in state.facts[:120] + state.relations[:120] + state.code[:80]:
            items.append(it.model_dump())

        prompt = {
            "goal": goal,
            "evidence_items": items,
            "schema": {"answer_markdown": "string", "warnings": ["string"]},
        }
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        log_llm_io("ANSWER", {"goal": goal, "evidence_count": len(items)}, raw)

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return AnswerWithCitations(answer_markdown="无法生成答案：LLM 未返回有效 JSON。", warnings=["invalid JSON"])
        try:
            return AnswerWithCitations(**json.loads(m.group(0)))
        except Exception:
            return AnswerWithCitations(answer_markdown="无法生成答案：JSON 解析失败。", warnings=["parse error"])


# ============================================================
# Records & Crawler
# ============================================================

@dataclass
class PageRecord:
    url: str
    status: str
    fetched_at: float
    title: str = ""
    markdown: str = ""
    out_links: List[Dict[str, str]] = field(default_factory=list)

class GoalDrivenCrawler:
    def __init__(
        self,
        seed_url: str,
        goal: str,
        max_pages: int,
        same_domain_only: bool,
        politeness_delay_s: float,
        page_md_chars: int,
        llm_model: str,
        api_key: str,
        concurrency: int = 2,
        goal_check_interval: int = 10,
        min_enqueue_score: int = 55,
        state_path: str = "summary_state.json",
        pages_path: str = "pages.jsonl",
    ):
        self.seed_url = seed_url
        self.goal = goal
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        self.politeness_delay_s = politeness_delay_s
        self.page_md_chars = page_md_chars
        self.concurrency = max(1, concurrency)
        self.goal_check_interval = max(1, goal_check_interval)
        self.min_enqueue_score = min_enqueue_score

        self.state_path = state_path
        self.pages_path = pages_path

        self.visited: Set[str] = set()
        self.pages: List[PageRecord] = []
        self.host_last_fetch: Dict[str, float] = {}

        self.heap: List[Tuple[int, str]] = []  # (-score, url)
        self.heap_lock = asyncio.Lock()
        self.sem = asyncio.Semaphore(self.concurrency)

        self.summarizer = DeltaSummarizer(model=llm_model, api_key=api_key)
        self.summarizer.load_state_if_exists(self.state_path)

        self.self_critic = SummarySelfCritic(model=llm_model, api_key=api_key)
        self.url_scorer = SummaryDrivenURLScorer(model=llm_model, api_key=api_key)
        self.answer_gen = EvidenceAwareAnswerGenerator(model=llm_model, api_key=api_key)

        self._stop = asyncio.Event()
        self._final_answer: Optional[str] = None

        self._goal_keywords = tokenize_keywords(goal)[:30]
        self._last_progress = now_ts()

    def allowed(self, url: str) -> bool:
        if looks_like_asset(url):
            return False
        if self.same_domain_only and not same_domain(self.seed_url, url):
            return False
        return True

    async def _politeness(self, url: str):
        host = urlparse(url).netloc
        last = self.host_last_fetch.get(host, 0.0)
        wait = self.politeness_delay_s - (now_ts() - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self.host_last_fetch[host] = now_ts()

    async def _heap_push(self, url: str, score: int):
        async with self.heap_lock:
            heapq.heappush(self.heap, (-score, url))

    async def _heap_pop(self) -> Optional[str]:
        async with self.heap_lock:
            while self.heap:
                _neg, url = heapq.heappop(self.heap)
                if url not in self.visited:
                    return url
        return None

    async def _seed(self):
        await self._heap_push(self.seed_url, 100)
        log(f"[SEED] {self.seed_url}")

    def _heuristic_shortlist(self, links: List[Dict[str, str]], limit: int = 60) -> List[Dict[str, str]]:
        # quick prefilter without LLM
        scored = []
        for x in links:
            u = x["url"].lower()
            a = (x.get("anchor") or "").lower()
            s = 5
            if any(k in u for k in ["asyncio", "task", "future", "event-loop", "eventloop", "loop"]):
                s += 25
            for kw in self._goal_keywords[:25]:
                if kw in u or kw in a:
                    s += 2
            scored.append((s, x))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [x for _s, x in scored[:limit]]

    async def _enqueue_links_scored(self, links: List[Dict[str, str]]):
        links = [x for x in links if self.allowed(x["url"]) and x["url"] not in self.visited]
        if not links:
            return

        shortlist = self._heuristic_shortlist(links, limit=60)
        compact = self.summarizer.state.compact_for_prompt(max_each=30)

        llm_scores = await self.url_scorer.score_urls(
            goal=self.goal,
            summary_compact=compact,
            candidates=shortlist,
            top_k=20,
        )

        pushed = 0
        for s in llm_scores:
            if s.score >= self.min_enqueue_score and s.url not in self.visited:
                await self._heap_push(s.url, s.score)
                pushed += 1
        log(f"[URL_SCORE] enqueued={pushed} from candidates={len(shortlist)}")

        # fallback if scorer fails badly
        if pushed == 0:
            for x in shortlist[:15]:
                await self._heap_push(x["url"], 25)
            log("[URL_SCORE] fallback enqueue 15 urls")

    async def _maybe_stop_by_self_critic(self):
        if len(self.pages) == 0 or (len(self.pages) % self.goal_check_interval != 0):
            return

        summary_md = self.summarizer.state.to_markdown()
        log(f"[SELF_CRITIC] running at pages={len(self.pages)}")
        critique = await self.self_critic.critique(self.goal, summary_md)

        log(f"[SELF_CRITIC] status={critique.goal_status} conf={critique.confidence:.2f}")
        if critique.missing_information:
            # steer next steps by adding open questions (Python-owned)
            for q in critique.missing_information[:10]:
                self.summarizer.state.apply_delta({"add_open_questions": [{"text": q, "sources": []}]}, page_url="")

        if critique.goal_status == "satisfied" and critique.confidence >= 0.8:
            log("[STOP] satisfied by self-critic; generating answer")
            ans = await self.answer_gen.generate(self.goal, self.summarizer.state)
            self._final_answer = ans.answer_markdown
            self._stop.set()

    async def _save_page_record(self, rec: PageRecord):
        try:
            with open(self.pages_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"[SAVE] pages.jsonl failed: {e}")

    async def _fetch_one(self, fetcher: PlaywrightFetcher, url: str):
        async with self.sem:
            if self._stop.is_set():
                return
            if url in self.visited:
                return
            self.visited.add(url)

            log(f"[FETCH] start {url}")
            await self._politeness(url)
            status, html = await fetcher.fetch(url)
            log(f"[FETCH] done  {url} status={status}")

            rec = PageRecord(url=url, status=status, fetched_at=now_ts())

            if status == "ok" and html:
                title, markdown = extract_clean_markdown(html, url, max_chars=self.page_md_chars)
                links = extract_links_with_anchor(html, url)

                rec.title = title
                rec.markdown = markdown
                rec.out_links = links

                log(f"[EXTRACT] title='{title[:70]}' md_chars={len(markdown)} links={len(links)}")

                # IMPORTANT: only summarize real content
                if markdown.strip():
                    await self.summarizer.update_with_page(self.goal, url, markdown)
                    self.summarizer.save_state(self.state_path)

                await self._enqueue_links_scored(links)

            else:
                # blocked/timeout/error => do not summarize
                log("[SUMMARY] skip (non-ok fetch)")

            self.pages.append(rec)
            await self._save_page_record(rec)
            await self._maybe_stop_by_self_critic()

    async def run(self, fetcher: PlaywrightFetcher) -> Tuple[bool, str]:
        await self._seed()

        tasks: Set[asyncio.Task] = set()
        log("[CRAWLER] run loop entered")

        while not self._stop.is_set() and len(self.pages) < self.max_pages:
            url = await self._heap_pop()
            if not url:
                # wait inflight tasks
                if not tasks:
                    log("[CRAWLER] heap empty and no inflight tasks -> stop")
                    break
                await asyncio.sleep(0.2)
                done = {t for t in tasks if t.done()}
                tasks -= done
                continue

            if not self.allowed(url):
                continue

            t = asyncio.create_task(self._fetch_one(fetcher, url))
            tasks.add(t)

            done = {t for t in tasks if t.done()}
            tasks -= done

            await asyncio.sleep(0)

        # Cleanup
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # If no final answer, best-effort
        if not self._final_answer:
            log("[FINAL] generating best-effort answer")
            ans = await self.answer_gen.generate(self.goal, self.summarizer.state)
            self._final_answer = ans.answer_markdown

        satisfied = self._stop.is_set()
        return satisfied, self._final_answer


# ============================================================
# CLI
# ============================================================

def build_argparser():
    ap = argparse.ArgumentParser(description="Stateful LLM crawler (fix summary state ownership + anti-WAF)")
    ap.add_argument("--seed", required=True, help="Seed URL")
    ap.add_argument("--goal", required=True, help="Goal in natural language")
    ap.add_argument("--max_pages", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--same_domain", action="store_true")
    ap.add_argument("--politeness_delay", type=float, default=0.8)
    ap.add_argument("--page_md_chars", type=int, default=9000)
    ap.add_argument("--goal_check_interval", type=int, default=10)
    ap.add_argument("--min_enqueue_score", type=int, default=55)
    ap.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--openai_api_key", type=str, default=None)
    ap.add_argument("--state_path", type=str, default="summary_state.json")
    ap.add_argument("--pages_path", type=str, default="pages.jsonl")
    ap.add_argument("--answer_path", type=str, default="answer.md")
    return ap

async def main_async(args):
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OpenAI API key. Provide --openai_api_key or set env OPENAI_API_KEY")

    # Reset outputs if you want clean run
    # (Keep summary_state.json if you want resume)
    if not os.path.exists(args.pages_path):
        pass

    crawler = GoalDrivenCrawler(
        seed_url=args.seed,
        goal=args.goal,
        max_pages=args.max_pages,
        same_domain_only=args.same_domain,
        politeness_delay_s=args.politeness_delay,
        page_md_chars=args.page_md_chars,
        llm_model=args.llm_model,
        api_key=api_key,
        concurrency=args.concurrency,
        goal_check_interval=args.goal_check_interval,
        min_enqueue_score=args.min_enqueue_score,
        state_path=args.state_path,
        pages_path=args.pages_path,
    )

    async with PlaywrightFetcher(timeout_ms=20000, wait_until="domcontentloaded") as fetcher:
        satisfied, answer_md = await crawler.run(fetcher)

    with open(args.answer_path, "w", encoding="utf-8") as f:
        f.write(answer_md)

    log("========== RESULT ==========")
    log(f"SATISFIED(by self-critic stop): {satisfied}")
    log(f"PAGES CRAWLED: {len(crawler.pages)} / {args.max_pages}")
    log(f"Saved: state={args.state_path} pages={args.pages_path} answer={args.answer_path}")
    print("\n----- ANSWER (Markdown) -----\n")
    print(answer_md)

def main():
    args = build_argparser().parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
