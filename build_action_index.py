"""CLI tool to build offline business action index."""
from __future__ import annotations

import argparse
import logging
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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

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

    logger.info("Starting business action index build")
    logger.info("Embedding model: %s", args.model)
    logger.info("Output path: %s", args.output)

    logger.info("Loading default actions")
    actions = load_default_actions()
    logger.info("Loaded %d actions", len(actions))

    logger.info("Building vocabulary from actions")
    vocab = build_vocab_from_actions(actions)
    logger.info("Vocabulary size: %d", len(vocab))

    logger.info("Creating embed function with specified model")
    embed_fn = lambda text: embed_text_openai(text, model=args.model)
    logger.info("Building and saving index")
    build_and_save_default_index(
        embed_fn=embed_fn,
        vocab=vocab,
        embedding_model=args.model,
        path=args.output,
    )
    logger.info("Index saved to %s", args.output)


if __name__ == "__main__":
    main()
