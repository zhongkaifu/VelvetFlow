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


BUSINESS_ACTIONS: List[Dict[str, Any]] = [
    _with_security_defaults(dict(action)) for action in _load_business_actions()
]


def get_action_by_id(action_id: str) -> Optional[Dict[str, Any]]:
    for a in BUSINESS_ACTIONS:
        if a["action_id"] == action_id:
            return a
    return None
