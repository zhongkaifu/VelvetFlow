"""Utility helpers for structured and console-friendly logging."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from typing import Any, Mapping

# Basic ANSI color codes for a slightly nicer CLI experience without extra deps.
RESET = "\033[0m"
COLORS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m",
    "dim": "\033[2m",
}

ICONS = {
    "INFO": "ℹ️",
    "SUCCESS": "✅",
    "WARN": "⚠️",
    "ERROR": "❌",
    "DEBUG": "🔍",
}

LEVEL_COLOR = {
    "INFO": COLORS["cyan"],
    "SUCCESS": COLORS["green"],
    "WARN": COLORS["yellow"],
    "ERROR": COLORS["red"],
    "DEBUG": COLORS["magenta"],
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_lines(message: str) -> str:
    lines = message.split("\n")
    if len(lines) <= 1:
        return message
    # Align multi-line messages under the prefix for readability
    return ("\n" + " " * 18).join(lines)


def console_log(level: str, message: str) -> None:
    level = level.upper()
    color = LEVEL_COLOR.get(level, "")
    icon = ICONS.get(level, "➡️")
    prefix = f"{color}{icon} [{level:>5}] {_timestamp()} |{RESET} "
    print(prefix + _format_lines(message))


def log_info(message: str) -> None:
    console_log("INFO", message)


def log_success(message: str) -> None:
    console_log("SUCCESS", message)


def log_warn(message: str) -> None:
    console_log("WARN", message)


def log_error(message: str) -> None:
    console_log("ERROR", message)


def log_debug(message: str) -> None:
    console_log("DEBUG", message)


def log_section(title: str, subtitle: str | None = None, char: str = "=") -> None:
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    line = char * width
    print(f"\n{COLORS['dim']}{line}{RESET}")
    console_log("INFO", title)
    if subtitle:
        console_log("DEBUG", subtitle)
    print(f"{COLORS['dim']}{line}{RESET}\n")


def log_kv(label: str, value: Any, level: str = "INFO") -> None:
    console_log(level, f"{label}：{value}")


def log_json(label: str, data: Mapping[str, Any] | list[Any] | Any) -> None:
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        payload = str(data)
    console_log("DEBUG", f"{label}:\n{payload}")


def log_event(event_type: str, payload: dict) -> None:
    """Emit machine-readable JSON logs while still being human-friendly."""

    record = {
        "time": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    print(json.dumps(record, ensure_ascii=False))
