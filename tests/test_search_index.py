import json

from velvetflow.search_index import (
    ActionIndex,
    build_vocab_from_actions,
    expand_tokens_with_phrases,
    filter_tokens,
)


def test_filter_tokens_filters_stopwords_digits_and_single_chars():
    tokens = ["a", "the", "3", "b", "AI", "data", "的", "数据"]

    assert filter_tokens(tokens) == ["ai", "data", "数据"]


def test_filter_tokens_respects_chinese_stopwords():
    tokens = ["的", "了", "和", "产品", "客户"]

    assert filter_tokens(tokens) == ["产品", "客户"]


def test_expand_tokens_with_phrases_adds_adjacent_short_phrases_and_dedupes():
    tokens = ["api", "doc", "api", "ops"]

    assert expand_tokens_with_phrases(tokens) == [
        "api",
        "doc",
        "ops",
        "api doc",
        "doc api",
        "api ops",
    ]


def test_build_vocab_from_actions_builds_from_fields_with_case_and_empty_tolerance():
    actions = [
        {
            "name": "AI Tool",
            "description": "Notify User",
            "domain": "CRM",
            "tags": ["VIP"],
        },
        {
            "name": None,
            "description": "",
            "domain": "",
            "tags": None,
        },
        {
            "name": "短语 词组",
            "description": "数据 处理",
            "domain": "BI",
            "tags": ["AI"],
        },
    ]

    vocab = build_vocab_from_actions(actions)

    assert vocab == sorted(
        {
            "ai",
            "tool",
            "notify",
            "user",
            "crm",
            "vip",
            "crm vip",
            "短语",
            "词组",
            "数据",
            "处理",
            "bi",
            "ai",
            "短语 词组",
            "词组 数据",
            "数据 处理",
            "处理 bi",
            "bi ai",
        }
    )


def test_action_index_save_load_round_trip(tmp_path):
    index = ActionIndex(
        actions=[{"action_id": "a1", "name": "Test"}],
        embeddings={"a1": [0.1, 0.2]},
        feature_embeddings={"a1": {"name": [0.1, 0.2]}},
        feature_keywords={"a1": {"name": ["test"]}},
        feature_keyword_embeddings={"a1": {"name": {"test": [0.1, 0.2]}}},
        vocab=["test"],
        embedding_model="test-model",
        created_ts=123.0,
    )
    path = tmp_path / "index.json"

    index.save(path)
    loaded = ActionIndex.load(path)

    assert loaded.to_dict() == json.loads(path.read_text(encoding="utf-8"))
    assert loaded.to_dict() == index.to_dict()


def test_action_index_embedding_dim_empty_embeddings():
    index = ActionIndex(actions=[], embeddings={})

    assert index.embedding_dim == 0
