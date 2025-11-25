from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from velvetflow.action_registry import validate_actions


@pytest.fixture
def valid_actions():
    return [
        {
            "action_id": "action.one",
            "description": "Test action",
            "arg_schema": {},
            "output_schema": {},
        },
        {
            "action_id": "action.two",
            "description": "Test action 2",
            "params_schema": {},
        },
    ]


def test_validate_actions_accepts_valid(valid_actions):
    validated = validate_actions(valid_actions)

    assert len(validated) == 2
    assert validated[0]["requires_approval"] is False
    assert validated[0]["allowed_roles"] == []
    assert validated[1]["arg_schema"] == {}


def test_validate_actions_rejects_missing_action_id(valid_actions):
    invalid = [dict(valid_actions[0])]
    invalid[0].pop("action_id")

    with pytest.raises(ValueError, match=r"index 0"):
        validate_actions(invalid)


def test_validate_actions_rejects_missing_params_schema(valid_actions):
    invalid = [dict(valid_actions[0])]
    invalid[0].pop("arg_schema")

    with pytest.raises(ValueError, match=r"params_schema/arg_schema"):
        validate_actions(invalid)


def test_validate_actions_rejects_duplicate_ids(valid_actions):
    invalid = valid_actions + [dict(valid_actions[0])]

    with pytest.raises(ValueError, match=r"Duplicate action_id 'action.one' found at index 2"):
        validate_actions(invalid)


def test_validate_actions_requires_boolean_security_flags(valid_actions):
    invalid = [dict(valid_actions[0])]
    invalid[0]["requires_approval"] = "yes"

    with pytest.raises(ValueError, match=r"requires_approval"):
        validate_actions(invalid)
