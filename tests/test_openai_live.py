import os
import urllib.parse

import pytest
from openai import OpenAI


LIVE_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LIVE_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _normalize_base_url(raw_base_url: str | None) -> str | None:
    if not raw_base_url:
        return None

    cleaned = raw_base_url.strip()
    if not cleaned:
        return None

    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme:
        return cleaned

    # Allow users to provide a host without scheme; default to https.
    candidate = f"https://{cleaned.lstrip('/')}"
    parsed_candidate = urllib.parse.urlparse(candidate)
    return candidate if parsed_candidate.netloc else None


def _require_live_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Set OPENAI_API_KEY to run live OpenAI endpoint tests.")

    base_url = _normalize_base_url(os.getenv("OPENAI_BASE_URL"))
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def test_openai_embedding_live_dimensions():
    client = _require_live_openai_client()

    response = client.embeddings.create(
        model=LIVE_EMBEDDING_MODEL,
        input="VelvetFlow live embedding smoke test.",
    )

    assert response.data, "Expected an embedding response"
    embedding = response.data[0].embedding
    assert isinstance(embedding, list) and len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.parametrize(
    "user_prompt",
    [
        "Reply only with the word VelvetFlow.",
        "Say VelvetFlow and nothing else.",
    ],
)
def test_openai_chat_completion_live(user_prompt: str):
    client = _require_live_openai_client()

    response = client.chat.completions.create(
        model=LIVE_CHAT_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=10,
    )

    message_content = response.choices[0].message.content
    assert message_content is not None
    assert "velvetflow" in message_content.lower()
