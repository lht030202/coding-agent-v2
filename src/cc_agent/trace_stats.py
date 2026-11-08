from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceStats:
    trace_count: int
    runs: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    llm_seconds: float
    tool_calls: int
    successful_tool_calls: int
    blocked_tool_calls: int
    format_errors: int
    tests_run: int
    tests_passed: int
    files_edited: int
    tool_counts: dict[str, int]

    def render(self) -> str:
        tool_lines = "\n".join(f"- {name}: {count}" for name, count in sorted(self.tool_counts.items()))
        return f"""Trace files: {self.trace_count}
Completed runs: {self.runs}
LLM calls: {self.llm_calls}
Tokens: prompt {self.prompt_tokens}, completion {self.completion_tokens} (reasoning {self.reasoning_tokens})
LLM time: {self.llm_seconds:.1f}s
Tool calls: {self.tool_calls}, success rate {_pct(self.successful_tool_calls, self.tool_calls)}
Blocked tool calls: {self.blocked_tool_calls}
Format errors: {self.format_errors}
Tests: {self.tests_passed}/{self.tests_run} passed ({_pct(self.tests_passed, self.tests_run)})
Files edited: {self.files_edited}
Tool usage:
{tool_lines or '- none'}
""".strip()


def compute_trace_stats(path: str | Path) -> TraceStats:
    root = Path(path)
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    totals: Counter[str] = Counter()
    llm_seconds = 0.0
    tool_counts: Counter[str] = Counter()

    for trace_file in files:
        if not trace_file.exists():
            continue
        for line in trace_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            payload = record.get("payload", {})
            if event == "llm_call":
                totals["llm_calls"] += 1
                for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                    totals[key] += int(payload.get(key, 0) or 0)
                llm_seconds += float(payload.get("latency_s", 0.0) or 0.0)
            elif event == "run_summary":
                totals["runs"] += 1
                totals["format_errors"] += int((payload.get("counters") or {}).get("format_errors", 0))
            elif event == "tool_call":
                result = payload.get("result", {})
                name = result.get("name", "unknown")
                ok = bool(result.get("ok"))
                tool_counts[name] += 1
                totals["tool_calls"] += 1
                totals["successful_tool_calls"] += int(ok)
                totals["blocked_tool_calls"] += int(bool(result.get("blocked")))
                if name == "run_tests":
                    totals["tests_run"] += 1
                    totals["tests_passed"] += int(ok)
                if name in {"write_file", "replace_in_file"} and ok:
                    totals["files_edited"] += 1

    return TraceStats(
        trace_count=len([file for file in files if file.exists()]),
        runs=totals["runs"],
        llm_calls=totals["llm_calls"],
        prompt_tokens=totals["prompt_tokens"],
        completion_tokens=totals["completion_tokens"],
        reasoning_tokens=totals["reasoning_tokens"],
        llm_seconds=llm_seconds,
        tool_calls=totals["tool_calls"],
        successful_tool_calls=totals["successful_tool_calls"],
        blocked_tool_calls=totals["blocked_tool_calls"],
        format_errors=totals["format_errors"],
        tests_run=totals["tests_run"],
        tests_passed=totals["tests_passed"],
        files_edited=totals["files_edited"],
        tool_counts=dict(tool_counts),
    )


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"
