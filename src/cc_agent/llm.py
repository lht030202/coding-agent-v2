from __future__ import annotations

from openai import OpenAI

from cc_agent.config import AgentConfig


class OpenAICompatibleChat:
    def __init__(self, config: AgentConfig):
        config.require_valid_provider()
        config.require_api_key()
        self.config = config
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        return content or ""


def build_chat_model(config: AgentConfig) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(config)
