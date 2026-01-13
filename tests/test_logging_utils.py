import json

from velvetflow import logging_utils


def test_configure_run_logging_sets_and_resets_run_id():
    assert logging_utils.current_run_id() is None

    token = logging_utils.configure_run_logging(run_id="run-123")
    assert token is not None
    assert logging_utils.current_run_id() == "run-123"

    logging_utils._RUN_ID.reset(token)
    assert logging_utils.current_run_id() is None


def test_use_trace_context_sets_current_and_resets():
    context = logging_utils.TraceContext("trace-1", "span-1", "span-name")

    with logging_utils.use_trace_context(context):
        current = logging_utils.current_trace_context()
        assert current is not None
        assert current.trace_id == "trace-1"
        assert current.span_id == "span-1"
        assert current.span_name == "span-name"

    assert logging_utils.current_trace_context() is None


def test_log_event_includes_context_and_payload_override(tmp_path, capsys):
    log_path = tmp_path / "events.jsonl"
    token = logging_utils.configure_run_logging(run_id="run-123", log_file=log_path)
    context = logging_utils.TraceContext("trace-1", "span-1", "span-name")

    with logging_utils.use_trace_context(context):
        logging_utils.log_event(
            "hello",
            payload={"node_id": "node-1", "action_id": "action-1", "extra": "data"},
        )

    output = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(output)
    assert record["workflow_run_id"] == "run-123"
    assert record["trace_id"] == "trace-1"
    assert record["span_id"] == "span-1"
    assert record["node_id"] == "node-1"
    assert record["action_id"] == "action-1"
    assert record["payload"]["extra"] == "data"

    logging_utils._RUN_ID.reset(token)


def test_log_llm_message_serializes_tool_calls(tmp_path, capsys):
    class DummyFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class DummyToolCall:
        def __init__(self, tool_id, tool_type, function):
            self.id = tool_id
            self.type = tool_type
            self.function = function

    class DummyMessage:
        def __init__(self, role, content, tool_calls):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls

    logging_utils.configure_run_logging(log_file=tmp_path / "llm.jsonl")

    function = DummyFunction("search", {"query": "hello"})
    tool_call = DummyToolCall("tool-1", "function", function)
    message = DummyMessage("assistant", "hi", [tool_call])
    logging_utils.log_llm_message("gpt-test", message, operation="demo")

    output = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(output)
    tool_calls = record["payload"]["message"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "search"
    assert tool_calls[0]["function"]["arguments"] == {"query": "hello"}
