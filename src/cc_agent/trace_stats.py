from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceStats:
    trace_count: int
    tool_calls: int
    successful_tool_calls: int
    blocked_tool_calls: int
    tests_run: int
    tests_passed: int
    files_edited: int
    tool_counts: dict[str, int]

    def render(self) -> str:
        success_rate = _pct(self.successful_tool_calls, self.tool_calls)
        test_rate = _pct(self.tests_passed, self.tests_run)
        tool_lines = "\n".join(f"- {name}: {count}" for name, count in sorted(self.tool_counts.items()))
        return f"""Trace files: {self.trace_count}
Tool calls: {self.tool_calls}
Tool success rate: {success_rate}
Blocked tool calls: {self.blocked_tool_calls}
Tests: {self.tests_passed}/{self.tests_run} passed ({test_rate})
Files edited: {self.files_edited}
Tool usage:
{tool_lines or '- none'}
""".strip()


def compute_trace_stats(path: str | Path) -> TraceStats:
    root = Path(path)
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    tool_calls = 0
    successful = 0
    blocked = 0
    tests_run = 0
    tests_passed = 0
    files_edited = 0
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
            if record.get("event") != "tool_call":
                continue
            result = record.get("payload", {}).get("result", {})
            name = result.get("name", "unknown")
            ok = bool(result.get("ok"))
            is_blocked = bool(result.get("blocked"))
            tool_calls += 1
            successful += int(ok)
            blocked += int(is_blocked)
            tool_counts[name] += 1
            if name == "run_tests":
                tests_run += 1
                tests_passed += int(ok)
            if name in {"write_file", "replace_in_file"} and ok:
                files_edited += 1

    return TraceStats(
        trace_count=len([file for file in files if file.exists()]),
        tool_calls=tool_calls,
        successful_tool_calls=successful,
        blocked_tool_calls=blocked,
        tests_run=tests_run,
        tests_passed=tests_passed,
        files_edited=files_edited,
        tool_counts=dict(tool_counts),
    )


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"
