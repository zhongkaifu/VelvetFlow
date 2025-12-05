# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Search and embedding utilities for VelvetFlow."""
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:  # pragma: no cover - import guard for environments without real numpy/faiss
    import faiss  # type: ignore
    _faiss_import_error: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - we want module import to survive in tests
    faiss = None  # type: ignore
    _faiss_import_error = exc
from openai import OpenAI

from velvetflow.action_registry import get_action_by_id
from velvetflow.logging_utils import child_span, log_event
from velvetflow.search_index import (
    DEFAULT_INDEX_PATH,
    ActionIndex,
    build_action_index,
    build_vocab_from_actions,
    load_default_actions,
)


class FakeElasticsearch:
    """
    极简 ES 模拟：只支持按关键词对 name/description/domain/tags 做计分。
    """

    def __init__(self, actions: List[Dict[str, Any]]):
        self.actions = actions

    def search(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        query_obj = body.get("query", {})
        size = body.get("size", 50)

        q_text = ""
        bool_q = query_obj.get("bool", {})
        musts = bool_q.get("must", [])
        for clause in musts:
            if "multi_match" in clause:
                q_text = clause["multi_match"].get("query", "")
                break

        q = q_text.lower()
        tokens = q.split()

        filters = bool_q.get("filter", [])
        filter_domain = None
        for f in filters:
            if "term" in f and "domain" in f["term"]:
                filter_domain = f["term"]["domain"]

        hits = []
        for action in self.actions:
            if not action.get("enabled", True):
                continue
            if filter_domain and action.get("domain") != filter_domain:
                continue

            text = (
                (action.get("name", "") or "")
                + " "
                + (action.get("description", "") or "")
                + " "
                + (action.get("domain", "") or "")
                + " "
                + " ".join(action.get("tags", []) or [])
            ).lower()

            score = 0.0
            for tok in tokens:
                if tok in text:
                    score += 1.0
            if score <= 0:
                continue

            hits.append({
                "_source": action,
                "_score": float(score),
            })

        hits.sort(key=lambda h: h["_score"], reverse=True)
        hits = hits[:size]

        return {"hits": {"hits": hits}}


class FaissVectorIndex:
    """
    使用 Faiss 的向量检索实现，用于在线余弦相似度搜索。

    - 通过 IndexIDMap 保存 action_id 到内部整数 ID 的映射，便于去重和更新；
    - 使用内积并在入库前做 L2 归一化，相当于计算余弦相似度；
    - 支持增量 upsert（同 action_id 会覆盖旧向量）。
    """

    def __init__(self, dim: Optional[int]):
        self.dim = dim
        self._index: Optional[Any] = None
        self._id_map: Dict[str, int] = {}
        self._reverse_id_map: Dict[int, str] = {}
        self._next_internal_id = 0

    def _ensure_index(self) -> None:
        if faiss is None:
            raise ImportError(
                "Faiss is unavailable; ensure faiss is installed with a real numpy or install optional dependency."
            ) from _faiss_import_error
        if self._index is None:
            if self.dim is None:
                raise ValueError("向量维度未知，无法初始化 Faiss 索引")
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def upsert(self, action_id: str, embedding: List[float]) -> None:
        vec = np.array(embedding, dtype=np.float32)
        if self.dim is None:
            self.dim = vec.shape[0]
        if vec.shape[0] != self.dim:
            raise ValueError(f"向量维度不匹配: expected {self.dim}, got {vec.shape[0]}")

        self._ensure_index()

        if action_id in self._id_map:
            old_internal_id = self._id_map[action_id]
            self._index.remove_ids(np.array([old_internal_id], dtype=np.int64))

        internal_id = self._next_internal_id
        self._next_internal_id += 1
        self._id_map[action_id] = internal_id
        self._reverse_id_map[internal_id] = action_id

        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        self._index.add_with_ids(vec, np.array([internal_id], dtype=np.int64))

    def search(self, query_emb: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        if self.dim and q.shape[1] != self.dim:
            raise ValueError(f"查询向量维度不匹配: expected {self.dim}, got {q.shape[1]}")

        faiss.normalize_L2(q)
        scores, ids = self._index.search(q, top_k)
        results: List[Tuple[str, float]] = []
        for internal_id, score in zip(ids[0], scores[0]):
            if internal_id == -1:
                continue
            aid = self._reverse_id_map.get(int(internal_id))
            if aid is None:
                continue
            results.append((aid, float(score)))
        return results


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def embed_text_openai(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    client = get_openai_client()
    response = client.embeddings.create(model=model, input=text or "")
    if not response.data:
        raise ValueError("未从 OpenAI 获取到有效的 embedding 结果")
    return list(response.data[0].embedding)


def _select_query_embed_fn(index: ActionIndex):
    model = index.embedding_model or DEFAULT_EMBEDDING_MODEL
    return lambda text: embed_text_openai(text, model=model)


class HybridActionSearchService:
    def __init__(
        self, es: FakeElasticsearch, vector_index: FaissVectorIndex, embed_fn, alpha: float = 0.6
    ):
        self.es = es
        self.vector_index = vector_index
        self.embed_fn = embed_fn
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5,
               tenant_id: str = None,
               domain: str = None) -> List[Dict[str, Any]]:
        start = time.perf_counter()
        with child_span("action_search") as span_ctx:
            q_emb = self.embed_fn(query)

            must_clauses = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["name^3", "description^2", "tags^2", "domain"],
                        "type": "best_fields",
                    }
                }
            ]
            filter_clauses = [{"term": {"enabled": True}}]
            if domain:
                filter_clauses.append({"term": {"domain": domain}})

            es_body = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses,
                    }
                },
                "_source": ["action_id", "name", "description", "domain", "tags", "arg_schema", "output_schema"],
                "size": 50,
            }

            es_resp = self.es.search(index="action_registry", body=es_body)
            bm25_scores: Dict[str, float] = {}
            es_actions: Dict[str, Dict[str, Any]] = {}
            for hit in es_resp["hits"]["hits"]:
                src = hit["_source"]
                aid = src["action_id"]
                bm25_scores[aid] = hit["_score"]
                es_actions[aid] = src

            vec_results = self.vector_index.search(q_emb, top_k=50)
            vec_scores: Dict[str, float] = {aid: score for aid, score in vec_results}

            all_ids = set(bm25_scores.keys()) | set(vec_scores.keys())
            if not all_ids:
                return []

            def normalize(scores: Dict[str, float]) -> Dict[str, float]:
                if not scores:
                    return {}
                vals = list(scores.values())
                mn, mx = min(vals), max(vals)
                if mx - mn < 1e-8:
                    return {k: 1.0 for k in scores}
                return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

            bm25_norm = normalize(bm25_scores)
            vec_norm = normalize(vec_scores)

            hybrid: List[Tuple[str, float]] = []
            for aid in all_ids:
                b = bm25_norm.get(aid, 0.0)
                v = vec_norm.get(aid, 0.0)
                score = self.alpha * b + (1.0 - self.alpha) * v
                hybrid.append((aid, score))

            hybrid.sort(key=lambda x: x[1], reverse=True)
            top_ids = [aid for aid, _ in hybrid[:top_k]]
            hybrid_scores = {aid: score for aid, score in hybrid}

            results = []
            for aid in top_ids:
                src = es_actions.get(aid)
                if not src:
                    src = get_action_by_id(aid) or {"action_id": aid}
                results.append({
                    "action_id": src["action_id"],
                    "name": src.get("name", ""),
                    "description": src.get("description", ""),
                    "domain": src.get("domain", ""),
                    "tags": src.get("tags", []),
                    "arg_schema": src.get("arg_schema"),
                    "output_schema": src.get("output_schema"),
                    "score": hybrid_scores.get(aid),
                })

            duration_ms = (time.perf_counter() - start) * 1000
            log_event(
                "search_metrics",
                {
                    "query": query,
                    "domain": domain,
                    "tenant_id": tenant_id,
                    "top_k": top_k,
                    "latency_ms": round(duration_ms, 2),
                    "result_size": len(results),
                },
                context=span_ctx,
            )
            return results


def build_search_service_from_index(
    index: ActionIndex,
    *,
    embed_fn,
    alpha: float = 0.6,
) -> HybridActionSearchService:
    fake_es = FakeElasticsearch(index.actions)
    vec_client = FaissVectorIndex(dim=index.embedding_dim or None)
    for action_id, emb in index.embeddings.items():
        vec_client.upsert(action_id, emb)

    return HybridActionSearchService(
        es=fake_es,
        vector_index=vec_client,
        embed_fn=embed_fn,
        alpha=alpha,
    )


def build_search_service_from_actions(
    actions: List[Dict[str, Any]],
    *,
    alpha: float = 0.6,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> HybridActionSearchService:
    vocab = build_vocab_from_actions(actions)
    embed_fn = lambda text: embed_text_openai(text, model=embedding_model)
    index = build_action_index(
        actions,
        embed_fn=embed_fn,
        vocab=vocab,
        embedding_model=embedding_model,
    )
    query_embed_fn = _select_query_embed_fn(index)
    return build_search_service_from_index(index, embed_fn=query_embed_fn, alpha=alpha)


def build_default_search_service(
    *, index_path: str = str(DEFAULT_INDEX_PATH), alpha: float = 0.6
) -> HybridActionSearchService:
    try:
        index = ActionIndex.load(index_path)
    except FileNotFoundError:
        actions = load_default_actions()
        vocab = build_vocab_from_actions(actions)
        embed_fn = lambda text: embed_text_openai(text, model=DEFAULT_EMBEDDING_MODEL)
        index = build_action_index(
            actions,
            embed_fn=embed_fn,
            vocab=vocab,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
        )
    query_embed_fn = _select_query_embed_fn(index)
    return build_search_service_from_index(index, embed_fn=query_embed_fn, alpha=alpha)
