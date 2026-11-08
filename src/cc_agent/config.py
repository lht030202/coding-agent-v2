from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

TOOL_MODES = ("native", "json")
VERIFY_MODES = ("tests", "rule")


def parse_extra_body(raw: str) -> dict[str, Any] | None:
    """Parse OPENAI_EXTRA_BODY: provider-specific request fields such as thinking switches."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OPENAI_EXTRA_BODY must be a JSON object, got {raw!r}") from exc
    if not isinstance(value, dict):
        raise RuntimeError('OPENAI_EXTRA_BODY must be a JSON object, e.g. {"enable_thinking": false}')
    return value


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise RuntimeError(f"{name} must be one of {allowed}, got {value!r}")
    return value


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str
    temperature: float
    extra_body: dict[str, Any] | None
    request_timeout: float
    max_retries: int
    max_steps: int
    command_timeout: int
    trace_dir: Path
    tool_mode: str
    verify_mode: str
    max_verify_retries: int
    tool_output_chars: int
    context_char_budget: int

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AgentConfig":
        if project_root is not None:
            load_dotenv(project_root / ".env")
        load_dotenv()

        return cls(
            provider=os.getenv("CC_AGENT_LLM_PROVIDER", "openai").strip().lower(),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            extra_body=parse_extra_body(os.getenv("OPENAI_EXTRA_BODY", "")),
            request_timeout=float(os.getenv("CC_AGENT_LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("CC_AGENT_LLM_MAX_RETRIES", "3")),
            max_steps=int(os.getenv("CC_AGENT_MAX_STEPS", "10")),
            command_timeout=int(os.getenv("CC_AGENT_COMMAND_TIMEOUT", "60")),
            trace_dir=Path(os.getenv("CC_AGENT_TRACE_DIR", "traces")),
            tool_mode=_choice("CC_AGENT_TOOL_MODE", "native", TOOL_MODES),
            verify_mode=_choice("CC_AGENT_VERIFY_MODE", "tests", VERIFY_MODES),
            max_verify_retries=int(os.getenv("CC_AGENT_MAX_VERIFY_RETRIES", "2")),
            tool_output_chars=int(os.getenv("CC_AGENT_TOOL_OUTPUT_CHARS", "6000")),
            context_char_budget=int(os.getenv("CC_AGENT_CONTEXT_CHARS", "60000")),
        )

    def require_valid_provider(self) -> None:
        if self.provider != "openai":
            raise RuntimeError(
                f"Unsupported CC_AGENT_LLM_PROVIDER={self.provider!r}. "
                "Use any OpenAI-compatible chat API via CC_AGENT_LLM_PROVIDER=openai."
            )

    def require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Copy .env.example to .env and set "
                "OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL."
            )
