from __future__ import annotations

from pathlib import Path
from typing import Any

from cc_agent.config import AgentConfig
from cc_agent.llm import LLMResult, ToolCall


def make_config(trace_dir: Path, **overrides: Any) -> AgentConfig:
    """A config built in code, so tests never read the real .env."""
    values: dict[str, Any] = dict(
        provider="openai",
        api_key="test-key",
        base_url=None,
        model="fake-model",
        temperature=0.0,
        extra_body=None,
        request_timeout=30.0,
        max_retries=0,
        max_steps=6,
        command_timeout=60,
        trace_dir=trace_dir,
        tool_mode="native",
        verify_mode="tests",
        max_verify_retries=2,
        tool_output_chars=6000,
        context_char_budget=60000,
    )
    values.update(overrides)
    return AgentConfig(**values)


class ScriptedLLM:
    """A fake chat model: replays scripted replies and records every request it receives."""

    def __init__(self, replies: list[LLMResult]):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResult:
        self.requests.append({"messages": [dict(message) for message in messages], "tools": tools})
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


def text(content: str) -> LLMResult:
    return LLMResult(content=content, prompt_tokens=10, completion_tokens=5)


def call(name: str, arguments: str, call_id: str = "call_1") -> LLMResult:
    return LLMResult(content="", tool_calls=(ToolCall(call_id, name, arguments),), prompt_tokens=20, completion_tokens=8)


FIX = '{"path": "calculator.py", "old": "return a + b", "new": "return a - b"}'
