from __future__ import annotations

import json
from typing import Any


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and the tail of a long output; errors usually sit at one of the two ends."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def message_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)


def compact_messages(
    messages: list[dict[str, Any]],
    budget_chars: int,
    keep_recent: int = 6,
    preview_chars: int = 400,
) -> tuple[list[dict[str, Any]], int]:
    """Shrink old tool outputs once the conversation exceeds the budget.

    Messages are never dropped: every assistant tool call must stay paired with its tool
    result, so only the content of older tool messages is replaced by a short preview.
    Returns the (possibly new) message list and how many messages were compacted.
    """
    if budget_chars <= 0 or message_chars(messages) <= budget_chars:
        return messages, 0

    compacted = [dict(message) for message in messages]
    cutoff = max(0, len(compacted) - keep_recent)
    count = 0
    for index in range(cutoff):
        message = compacted[index]
        content = message.get("content") or ""
        if message.get("role") == "tool" and len(content) > preview_chars:
            message["content"] = truncate_middle(content, preview_chars) + "\n[compacted to save context]"
            count += 1
            if message_chars(compacted) <= budget_chars:
                break
    return compacted, count
