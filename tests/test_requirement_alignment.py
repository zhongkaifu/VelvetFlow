import json

import pytest

from velvetflow.planner.requirement_alignment import check_missing_requirements


class _DummyCompletions:
    def __init__(self, captured_kwargs: dict):
        self._captured_kwargs = captured_kwargs

    def create(self, **kwargs):
        # Capture the raw kwargs to ensure no unsupported parameters are sent.
        self._captured_kwargs.update(kwargs)

        class _Message:
            content = json.dumps({"missing_requirements": []})

        class _Choice:
            def __init__(self):
                self.message = _Message()

        class _Response:
            def __init__(self):
                self.choices = [_Choice()]
                self.usage = None

        return _Response()


class _DummyChat:
    def __init__(self, captured_kwargs: dict):
        self.completions = _DummyCompletions(captured_kwargs)


class _DummyClient:
    def __init__(self, captured_kwargs: dict):
        self.chat = _DummyChat(captured_kwargs)


def test_check_missing_requirements_avoids_fixed_temperature(monkeypatch):
    captured_kwargs: dict = {}

    def _fake_openai_client(*args, **kwargs):
        return _DummyClient(captured_kwargs)

    monkeypatch.setattr(
        "velvetflow.planner.requirement_alignment.OpenAI", _fake_openai_client
    )

    workflow = {"workflow_name": "demo", "nodes": []}
    requirement_plan = {"requirements": [{"description": "do something"}]}

    missing = check_missing_requirements(workflow, requirement_plan)

    assert missing == []
    # Ensure the OpenAI call does not pass a temperature that may be unsupported.
    assert "temperature" not in captured_kwargs

