# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Action registry definitions for VelvetFlow."""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _with_security_defaults(action: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all actions carry the security metadata fields."""

    action.setdefault("requires_approval", False)
    action.setdefault("allowed_roles", [])
    return action


_ACTIONS_DATA_DIR = Path(__file__).resolve().parents[1] / "tools" / "business_actions"
_LEGACY_ACTIONS_FILE = _ACTIONS_DATA_DIR.with_suffix(".json")


def _iter_action_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.json"))
        return

    if path.is_file():
        yield path
        return

    if path == _ACTIONS_DATA_DIR and _LEGACY_ACTIONS_FILE.exists():
        yield _LEGACY_ACTIONS_FILE
        return

    raise FileNotFoundError(f"Action registry path not found: {path}")


def load_actions_from_path(path: Path = _ACTIONS_DATA_DIR) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    for file_path in _iter_action_files(path):
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Action registry file {file_path} must contain a list of actions")

        actions.extend(data)

    return actions


def _normalized_params_schema(action: Dict[str, Any]) -> Any:
    if "arg_schema" in action:
        return action["arg_schema"]

    if "params_schema" in action:
        action["arg_schema"] = action["params_schema"]
        return action["arg_schema"]

    return None


def _validate_action(action: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Validate and normalize a raw action definition."""

    if not isinstance(action, dict):
        raise ValueError(f"Action at index {index} must be a mapping, got {type(action).__name__}")

    normalized = dict(action)

    for field in ("action_id", "description"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Action at index {index} is missing required field '{field}' or it is empty")

    params_schema = _normalized_params_schema(normalized)
    action_id = normalized.get("action_id", "<unknown>")
    if not isinstance(params_schema, dict):
        raise ValueError(
            f"Action at index {index} (action_id='{action_id}') must include a params_schema/arg_schema object"
        )

    normalized.setdefault("params_schema", params_schema)

    requires_approval = normalized.get("requires_approval", False)
    if not isinstance(requires_approval, bool):
        raise ValueError(
            f"Action at index {index} (action_id='{action_id}') must set requires_approval as a boolean"
        )
    normalized["requires_approval"] = requires_approval

    allowed_roles = normalized.get("allowed_roles", [])
    if not isinstance(allowed_roles, list):
        raise ValueError(
            f"Action at index {index} (action_id='{action_id}') must set allowed_roles as a list"
        )
    normalized["allowed_roles"] = allowed_roles

    return _with_security_defaults(normalized)


def validate_actions(raw_actions: List[Any]) -> List[Dict[str, Any]]:
    """Validate and normalize a list of action definitions."""

    validated: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, raw_action in enumerate(raw_actions):
        action = _validate_action(raw_action, idx)
        action_id = action["action_id"]
        if action_id in seen_ids:
            raise ValueError(f"Duplicate action_id '{action_id}' found at index {idx}")
        seen_ids.add(action_id)
        validated.append(action)

    return validated


BUSINESS_ACTIONS: List[Dict[str, Any]] = validate_actions(load_actions_from_path())


def register_dynamic_actions(raw_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and append additional actions at runtime.

    Existing ``action_id`` entries are preserved to keep registration idempotent.
    """

    validated = validate_actions(raw_actions)

    existing_ids = {action.get("action_id") for action in BUSINESS_ACTIONS}
    for action in validated:
        action_id = action.get("action_id")
        if action_id in existing_ids:
            continue
        BUSINESS_ACTIONS.append(action)
        existing_ids.add(action_id)

    return validated


def get_action_by_id(action_id: str) -> Optional[Dict[str, Any]]:
    for a in BUSINESS_ACTIONS:
        if a["action_id"] == action_id:
            return a
    return None


__all__ = [
    "load_actions_from_path",
    "validate_actions",
    "BUSINESS_ACTIONS",
    "get_action_by_id",
    "register_dynamic_actions",
]
