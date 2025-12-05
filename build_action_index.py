"""CLI tool to build offline business action index."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from velvetflow.search import DEFAULT_EMBEDDING_MODEL, embed_text_openai
from velvetflow.search_index import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    DEFAULT_OFFLINE_EMBEDDING_MODEL,
    build_and_save_default_index,
    build_vocab_from_actions,
    load_default_actions,
)


def main():
    parser = argparse.ArgumentParser(description="Build business action search index")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to save the generated index (JSON)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OFFLINE_EMBEDDING_MODEL,
        help=(
            "OpenAI embedding model to use for offline indexing (default: "
            f"{DEFAULT_EMBEDDING_MODEL})"
        ),
    )
    args = parser.parse_args()

    actions = load_default_actions()
    vocab = build_vocab_from_actions(actions)
    embed_fn = lambda text: embed_text_openai(text, model=args.model)
    build_and_save_default_index(
        embed_fn=embed_fn,
        vocab=vocab,
        embedding_model=args.model,
        path=args.output,
    )
    print(f"Index saved to {args.output}")


if __name__ == "__main__":
    main()
