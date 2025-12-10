"""Business action indexing utilities.

This module splits the offline indexing flow from the online search flow.
The index stores lightweight keyword vocabulary and pre-computed embeddings
for business actions so that online search can focus on recall and ranking.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from velvetflow.action_registry import BUSINESS_ACTIONS


DEFAULT_OFFLINE_EMBEDDING_MODEL = "text-embedding-3-large"


@dataclass
class ActionIndex:
    """Container for offline-built search assets."""

    actions: List[Dict[str, Any]]
    embeddings: Dict[str, List[float]]
    feature_embeddings: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    feature_keywords: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    vocab: List[str] = field(default_factory=list)
    embedding_model: str = DEFAULT_OFFLINE_EMBEDDING_MODEL
    created_ts: float = field(default_factory=lambda: time.time())

    @property
    def embedding_dim(self) -> int:
        if not self.embeddings:
            return 0
        first = next(iter(self.embeddings.values()))
        return len(first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": self.actions,
            "embeddings": self.embeddings,
            "feature_embeddings": self.feature_embeddings,
            "feature_keywords": self.feature_keywords,
            "vocab": self.vocab,
            "embedding_model": self.embedding_model,
            "created_ts": self.created_ts,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ActionIndex":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            actions=data.get("actions", []),
            embeddings=data.get("embeddings", {}),
            feature_embeddings=data.get("feature_embeddings", {}),
            feature_keywords=data.get("feature_keywords", {}),
            vocab=data.get("vocab", []),
            embedding_model=data.get(
                "embedding_model", DEFAULT_OFFLINE_EMBEDDING_MODEL
            ),
            created_ts=data.get("created_ts", time.time()),
        )


def build_vocab_from_actions(actions: Iterable[Dict[str, Any]]) -> List[str]:
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


def _schema_keywords(schema: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(schema, dict):
        return []

    props = schema.get("properties")
    if isinstance(props, dict):
        return list(props.keys())
    return []


def _stringify_schema(schema: Optional[Dict[str, Any]]) -> str:
    keywords = _schema_keywords(schema)
    return " ".join(keywords)


def _split_namespace(action_id: str) -> str:
    if not action_id:
        return ""
    parts = action_id.split(".")
    if len(parts) <= 1:
        return action_id
    return ".".join(parts[:-1])


def _extract_feature_texts(action: Dict[str, Any]) -> Dict[str, str]:
    tags = action.get("tags", []) or []
    return {
        "namespace": _split_namespace(action.get("action_id", "")),
        "category": action.get("category") or action.get("domain", ""),
        "tags": " ".join(tags),
        "description": action.get("description", ""),
        "name": action.get("name", ""),
        "inputs": _stringify_schema(action.get("arg_schema")),
        "outputs": _stringify_schema(action.get("output_schema")),
    }


def build_action_index(
    actions: List[Dict[str, Any]],
    *,
    embed_fn: Callable[[str], List[float]],
    vocab: Optional[Iterable[str]] = None,
    embedding_model: str = DEFAULT_OFFLINE_EMBEDDING_MODEL,
) -> ActionIndex:
    """Offline build of keyword vocabulary and embedding index."""

    vocab_list = list(vocab) if vocab is not None else build_vocab_from_actions(actions)
    embeddings: Dict[str, List[float]] = {}
    feature_embeddings: Dict[str, Dict[str, List[float]]] = {}
    feature_keywords: Dict[str, Dict[str, List[str]]] = {}
    for action in actions:
        text = (
            (action.get("name", "") or "")
            + " "
            + (action.get("description", "") or "")
            + " "
            + (action.get("domain", "") or "")
            + " "
            + " ".join(action.get("tags", []) or [])
        )
        embeddings[action["action_id"]] = embed_fn(text)

        features = _extract_feature_texts(action)
        feature_embeddings[action["action_id"]] = {
            fname: embed_fn(ftext)
            for fname, ftext in features.items()
            if ftext
        }
        feature_keywords[action["action_id"]] = {
            fname: ftext.lower().split()
            for fname, ftext in features.items()
            if ftext
        }

    return ActionIndex(
        actions=actions,
        embeddings=embeddings,
        feature_embeddings=feature_embeddings,
        feature_keywords=feature_keywords,
        vocab=vocab_list,
        embedding_model=embedding_model,
    )


def load_default_actions() -> List[Dict[str, Any]]:
    return BUSINESS_ACTIONS


DEFAULT_INDEX_PATH = Path("tools/action_index.json")


def build_and_save_default_index(
    *,
    embed_fn: Callable[[str], List[float]],
    vocab: Optional[Iterable[str]] = None,
    embedding_model: str = DEFAULT_OFFLINE_EMBEDDING_MODEL,
    path: Path | str = DEFAULT_INDEX_PATH,
) -> ActionIndex:
    index = build_action_index(
        load_default_actions(),
        embed_fn=embed_fn,
        vocab=vocab,
        embedding_model=embedding_model,
    )
    index.save(path)
    return index
