from types import SimpleNamespace

import pytest

from cc_agent.config import parse_extra_body
from cc_agent.llm import OpenAICompatibleChat
from helpers import make_config


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_call = SimpleNamespace(id="call_9", function=SimpleNamespace(name="read_file", arguments='{"path": "a.py"}'))
        message = SimpleNamespace(content="", tool_calls=[tool_call])
        usage = SimpleNamespace(
            prompt_tokens=120, completion_tokens=30, completion_tokens_details=SimpleNamespace(reasoning_tokens=0)
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], usage=usage)


def fake_client():
    completions = FakeCompletions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_chat_parses_tool_calls_and_usage(tmp_path):
    client, completions = fake_client()
    llm = OpenAICompatibleChat(make_config(tmp_path, extra_body={"enable_thinking": False}), client=client)
    result = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "read_file"}}])
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == '{"path": "a.py"}'
    assert (result.prompt_tokens, result.completion_tokens, result.reasoning_tokens) == (120, 30, 0)
    request = completions.calls[0]
    assert request["extra_body"] == {"enable_thinking": False}
    assert request["tools"][0]["function"]["name"] == "read_file"


def test_assistant_message_keeps_the_tool_call_ids(tmp_path):
    client, _ = fake_client()
    message = OpenAICompatibleChat(make_config(tmp_path), client=client).chat([{"role": "user", "content": "hi"}]).assistant_message()
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "call_9"


def test_optional_fields_are_omitted(tmp_path):
    client, completions = fake_client()
    OpenAICompatibleChat(make_config(tmp_path), client=client).chat([{"role": "user", "content": "hi"}])
    assert "extra_body" not in completions.calls[0]
    assert "tools" not in completions.calls[0]


def test_parse_extra_body():
    assert parse_extra_body("") is None
    assert parse_extra_body('{"enable_thinking": false}') == {"enable_thinking": False}
    with pytest.raises(RuntimeError):
        parse_extra_body("[1, 2]")
