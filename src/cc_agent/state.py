from __future__ import annotations

from typing import Any, TypedDict


class ToolRecord(TypedDict, total=False):
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: str
    blocked: bool
    reason: str


class AgentState(TypedDict, total=False):
    repo_path: str
    task: str
    test_command: str | None
    repo_context: str
    project_rules: str
    plan: str
    steps: int
    max_steps: int
    last_action: dict[str, Any]
    tool_history: list[ToolRecord]
    # Native tool mode keeps the whole conversation: assistant tool calls paired with tool results.
    messages: list[dict[str, Any]]
    no_tool_streak: int
    verify_retries: int
    tests_passed: bool | None
    stop_reason: str | None
    usage: dict[str, Any]
    counters: dict[str, int]
    review: str
    final_answer: str
    done: bool
    trace_path: str
