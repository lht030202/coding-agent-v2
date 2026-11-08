from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from cc_agent.config import AgentConfig


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text produced by the model


@dataclass(frozen=True)
class LLMResult:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None

    def assistant_message(self) -> dict[str, Any]:
        """The assistant turn to append to the conversation in native tool mode."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResult: ...


class OpenAICompatibleChat:
    """The single entry point for model calls: provider, extra fields, retries and usage all live here."""

    def __init__(self, config: AgentConfig, client: Any | None = None):
        config.require_valid_provider()
        self.config = config
        if client is None:
            config.require_api_key()
            kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "timeout": config.request_timeout,
                # The SDK retries connection errors, 408/409/429 and 5xx with exponential backoff.
                "max_retries": config.max_retries,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            client = OpenAI(**kwargs)
        self.client = client

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResult:
        request: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools
        if self.config.extra_body:
            # Provider-specific fields (e.g. SiliconFlow's enable_thinking) are not part of the
            # OpenAI schema, so they travel in extra_body.
            request["extra_body"] = self.config.extra_body

        start = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        latency = time.perf_counter() - start

        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        calls = tuple(
            ToolCall(
                id=call.id or f"call_{index}",
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for index, call in enumerate(getattr(message, "tool_calls", None) or [])
        )
        return LLMResult(
            content=message.content or "",
            tool_calls=calls,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            latency_s=latency,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def complete(self, system: str, user: str) -> LLMResult:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])


def build_chat_model(config: AgentConfig) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(config)
