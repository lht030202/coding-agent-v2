from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str
    temperature: float
    max_steps: int
    command_timeout: int
    trace_dir: Path

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AgentConfig":
        if project_root is not None:
            load_dotenv(project_root / ".env")
        load_dotenv()

        provider = os.getenv("CC_AGENT_LLM_PROVIDER", "openai").strip().lower()
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        max_steps = int(os.getenv("CC_AGENT_MAX_STEPS", "8"))
        command_timeout = int(os.getenv("CC_AGENT_COMMAND_TIMEOUT", "60"))
        trace_dir = Path(os.getenv("CC_AGENT_TRACE_DIR", "traces"))

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_steps=max_steps,
            command_timeout=command_timeout,
            trace_dir=trace_dir,
        )

    def require_valid_provider(self) -> None:
        if self.provider != "openai":
            raise RuntimeError(
                f"Unsupported CC_AGENT_LLM_PROVIDER={self.provider!r}. "
                "This public version supports OpenAI-compatible chat APIs via CC_AGENT_LLM_PROVIDER=openai."
            )

    def require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Copy .env.example to .env and set "
                "OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL."
            )
