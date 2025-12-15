"""CLI tool to fetch Composio tool definitions and schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.composio import collect_composio_tool_specs


def _dump_json(payload: dict, *, output: Path | None, pretty: bool) -> None:
    indent = 2 if pretty or output else None
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + ("\n" if indent else ""), encoding="utf-8")
        print(f"Saved {len(payload.get('tools', []))} tool definitions to {output}")
    else:
        print(text)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch Composio tools and their schemas")
    parser.add_argument(
        "--actions",
        "-a",
        nargs="*",
        help="Optional subset of actions to include (default: all actions)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save the JSON payload (defaults to stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation",
    )
    args = parser.parse_args(argv)

    specs = collect_composio_tool_specs(selected_actions=args.actions)
    payload = {
        "selected_actions": args.actions or [],
        "tool_count": len(specs),
        "tools": specs,
    }
    _dump_json(payload, output=args.output, pretty=args.pretty)


if __name__ == "__main__":
    main()

