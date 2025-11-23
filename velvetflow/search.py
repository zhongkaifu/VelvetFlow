"""Search and embedding utilities for VelvetFlow."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from velvetflow.action_registry import BUSINESS_ACTIONS, get_action_by_id


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


class VectorClient:
    """
    内存版向量库：用于演示向量相似度。
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._store: Dict[str, np.ndarray] = {}

    def upsert(self, action_id: str, embedding: List[float]):
        v = np.array(embedding, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError(f"向量维度不匹配: expected {self.dim}, got {v.shape[0]}")
        self._store[action_id] = v

    def search(self, query_emb: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._store:
            return []
        q = np.array(query_emb, dtype=np.float32)
        results: List[Tuple[str, float]] = []
        for aid, emb in self._store.items():
            num = float(np.dot(q, emb))
            denom = float(np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8)
            sim = num / denom
            results.append((aid, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def build_vocab_from_actions(actions: List[Dict[str, Any]]) -> List[str]:
    vocab = set()
    for a in actions:
        text = (
            (a.get("name", "") or "")
            + " "
            + (a.get("description", "") or "")
            + " "
            + (a.get("domain", "") or "")
            + " "
            + " ".join(a.get("tags", []) or [])
        ).lower()
        tokens = text.split()
        vocab.update(tokens)
    return sorted(vocab)


GLOBAL_VOCAB = build_vocab_from_actions(BUSINESS_ACTIONS)
VOCAB_INDEX = {tok: idx for idx, tok in enumerate(GLOBAL_VOCAB)}


def embed_text_local(text: str) -> List[float]:
    tokens = (text or "").lower().split()
    if not tokens or not GLOBAL_VOCAB:
        return [0.0] * len(GLOBAL_VOCAB)
    vec = np.zeros(len(GLOBAL_VOCAB), dtype=np.float32)
    for t in tokens:
        idx = VOCAB_INDEX.get(t)
        if idx is not None:
            vec[idx] += 1.0
    vec = vec / (len(tokens) + 1e-8)
    return vec.tolist()


class HybridActionSearchService:
    def __init__(self, es: FakeElasticsearch, vector_client: VectorClient,
                 embed_fn, alpha: float = 0.6):
        self.es = es
        self.vector_client = vector_client
        self.embed_fn = embed_fn
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5,
               tenant_id: str = None,
               domain: str = None) -> List[Dict[str, Any]]:
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

        vec_results = self.vector_client.search(q_emb, top_k=50)
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
            })
        return results
