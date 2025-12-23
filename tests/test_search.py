# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import numpy as np
import pytest

from velvetflow.search import FakeElasticsearch, FaissVectorIndex, FeatureRanker


def test_fake_elasticsearch_filters_and_scores() -> None:
    actions = [
        {
            "action_id": "a1",
            "name": "alpha beta",
            "description": "",
            "domain": "sales",
            "tags": [],
            "enabled": True,
        },
        {
            "action_id": "a2",
            "name": "alpha",
            "description": "",
            "domain": "sales",
            "tags": [],
            "enabled": True,
        },
        {
            "action_id": "a3",
            "name": "alpha beta",
            "description": "",
            "domain": "sales",
            "tags": [],
            "enabled": False,
        },
        {
            "action_id": "a4",
            "name": "alpha beta",
            "description": "",
            "domain": "finance",
            "tags": [],
            "enabled": True,
        },
    ]

    es = FakeElasticsearch(actions)
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": "alpha beta",
                            "fields": ["name", "description"],
                        }
                    }
                ],
                "filter": [
                    {"term": {"enabled": True}},
                    {"term": {"domain": "sales"}},
                ],
            }
        },
        "size": 10,
    }

    resp = es.search(index="action_registry", body=body)
    hits = resp["hits"]["hits"]
    assert [hit["_source"]["action_id"] for hit in hits] == ["a1", "a2"]


def test_feature_ranker_empty_inputs_score_zero() -> None:
    ranker = FeatureRanker(
        feature_keywords={"a1": {"name": ["alpha"]}},
        feature_embeddings={"a1": {"name": [1.0, 0.0]}},
    )

    assert ranker.score("a1", [], [], {}) == 0.0


def test_feature_ranker_cosine_handles_zero_and_tiny_norms() -> None:
    zero_vec = np.array([0.0, 0.0], dtype=np.float32)
    unit_vec = np.array([1.0, 0.0], dtype=np.float32)
    tiny_vec = np.array([1e-13, 0.0], dtype=np.float32)

    assert FeatureRanker._cosine(zero_vec, unit_vec) == 0.0
    assert FeatureRanker._cosine(tiny_vec, tiny_vec) == 0.0


def test_faiss_vector_index_upsert_requires_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("velvetflow.search.faiss", None)
    monkeypatch.setattr("velvetflow.search._faiss_import_error", Exception("x"))

    index = FaissVectorIndex(dim=2)
    with pytest.raises(ImportError, match="Faiss is unavailable"):
        index.upsert("a1", [0.1, 0.2])
