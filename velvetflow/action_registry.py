"""Action registry definitions for VelvetFlow."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _with_security_defaults(action: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all actions carry the security metadata fields."""

    action.setdefault("requires_approval", False)
    action.setdefault("allowed_roles", [])
    return action


_ACTIONS_DATA_PATH = Path(__file__).with_name("business_actions.json")


def _load_business_actions() -> List[Dict[str, Any]]:
    with _ACTIONS_DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("business_actions.json must contain a list of actions")

    return data


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


BUSINESS_ACTIONS: List[Dict[str, Any]] = validate_actions(_load_business_actions())


def get_action_by_id(action_id: str) -> Optional[Dict[str, Any]]:
    for a in BUSINESS_ACTIONS:
        if a["action_id"] == action_id:
            return a
    return None
