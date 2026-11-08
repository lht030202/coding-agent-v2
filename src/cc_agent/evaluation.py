"""Evaluation harness: run every task under several agent configurations and judge with tests.

A run is judged successful only when, after the agent stops, the ORIGINAL visible tests plus the
hidden tests all pass. Restoring the original tests prevents an agent from "fixing" a task by
editing the tests; hidden tests catch fixes that only satisfy the visible cases.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from cc_agent.config import AgentConfig
from cc_agent.graph import run_agent
from cc_agent.hooks import sanitize_env

CONFIGS: dict[str, dict[str, str]] = {
    "v0": {"tool_mode": "json", "verify_mode": "rule", "label": "原版：JSON 文本解析 + 规则验证"},
    "v1": {"tool_mode": "native", "verify_mode": "rule", "label": "+ 原生 function calling"},
    "v2": {"tool_mode": "native", "verify_mode": "tests", "label": "+ 测试驱动验证（完整版）"},
}
COUNTER_KEYS = ("tool_calls", "tool_errors", "blocked_calls", "test_failures", "format_errors", "no_tool_replies", "compactions")


@dataclass(frozen=True)
class EvalTask:
    id: str
    category: str
    task: str
    repo: Path
    hidden_tests: Path | None
    test_command: str


def load_tasks(manifest: Path, project_root: Path, task_ids: list[str] | None = None) -> list[EvalTask]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    tasks = [
        EvalTask(
            id=row["id"],
            category=row["category"],
            task=row["task"],
            repo=project_root / row["repo"],
            hidden_tests=(project_root / row["hidden_tests"]) if row.get("hidden_tests") else None,
            test_command=row.get("test_command", "pytest -q"),
        )
        for row in rows
    ]
    if task_ids:
        wanted = set(task_ids)
        missing = wanted - {task.id for task in tasks}
        if missing:
            raise ValueError(f"Unknown task ids: {sorted(missing)}")
        tasks = [task for task in tasks if task.id in wanted]
    return tasks


def prepare_workspace(task: EvalTask) -> Path:
    """Copy the task repo into a fresh temp dir and commit it, so every run starts from the same state."""
    workdir = Path(tempfile.mkdtemp(prefix=f"ccagent_{task.id}_"))
    repo = workdir / "repo"
    shutil.copytree(task.repo, repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    git = ["git", "-c", "user.email=eval@local", "-c", "user.name=eval"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "task"], cwd=repo, check=True)
    return repo


def judge(repo: Path, task: EvalTask, timeout: int) -> tuple[bool, str]:
    """Restore the original tests, add the hidden tests, and run the whole suite."""
    tests_dir = repo / "tests"
    shutil.rmtree(tests_dir, ignore_errors=True)
    shutil.copytree(task.repo / "tests", tests_dir, ignore=shutil.ignore_patterns("__pycache__"))
    if task.hidden_tests:
        for path in sorted(task.hidden_tests.glob("*.py")):
            shutil.copy2(path, tests_dir / path.name)
    env = sanitize_env({**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "judge timed out"
    return completed.returncode == 0, (completed.stdout + "\n" + completed.stderr)[-1500:]


def run_one(
    task: EvalTask,
    config_name: str,
    run_index: int,
    base_config: AgentConfig,
    project_root: Path,
    traces_dir: Path,
) -> dict[str, Any]:
    spec = CONFIGS[config_name]
    config = replace(
        base_config,
        tool_mode=spec["tool_mode"],
        verify_mode=spec["verify_mode"],
        trace_dir=traces_dir / config_name,
    )
    repo = prepare_workspace(task)
    started = time.perf_counter()
    final: dict[str, Any] = {}
    error = None
    try:
        final = dict(run_agent(repo, task.task, task.test_command, config=config, project_root=project_root))
    except Exception as exc:  # noqa: BLE001 - one failed run must not stop the evaluation
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    passed, judge_output = judge(repo, task, timeout=max(60, config.command_timeout * 2))
    shutil.rmtree(repo.parent, ignore_errors=True)

    usage = final.get("usage") or {}
    counters = final.get("counters") or {}
    trace_path = final.get("trace_path")
    if trace_path:
        try:  # keep reports free of absolute local paths
            trace_path = Path(trace_path).resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "task": task.id,
        "category": task.category,
        "config": config_name,
        "run": run_index,
        "success": passed and error is None,
        "steps": int(final.get("steps", 0)),
        "llm_calls": int(usage.get("llm_calls", 0)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": int(usage.get("reasoning_tokens", 0)),
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_seconds": float(usage.get("llm_seconds", 0.0)),
        "wall_seconds": round(wall, 2),
        **{key: int(counters.get(key, 0)) for key in COUNTER_KEYS},
        "verify_retries": int(final.get("verify_retries", 0)),
        "stop_reason": final.get("stop_reason") or ("error" if error else None),
        "agent_tests_passed": final.get("tests_passed"),
        "error": error,
        "trace_path": trace_path,
        "judge_tail": "" if passed else judge_output[-600:],
    }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def summarize(records: list[dict[str, Any]], config_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in config_names:
        rows = [row for row in records if row["config"] == name]
        per_task: dict[str, list[bool]] = {}
        per_category: dict[str, list[bool]] = {}
        for row in rows:
            per_task.setdefault(row["task"], []).append(row["success"])
            per_category.setdefault(row["category"], []).append(row["success"])
        successes = sum(row["success"] for row in rows)
        summary[name] = {
            "label": CONFIGS[name]["label"],
            "runs": len(rows),
            "successes": successes,
            "success_rate": round(successes / len(rows), 4) if rows else 0.0,
            "tasks": len(per_task),
            "tasks_solved_every_run": sum(all(values) for values in per_task.values()),
            "tasks_solved_any_run": sum(any(values) for values in per_task.values()),
            "avg_steps": _mean([row["steps"] for row in rows]),
            "avg_llm_calls": _mean([row["llm_calls"] for row in rows]),
            "avg_prompt_tokens": _mean([row["prompt_tokens"] for row in rows]),
            "avg_completion_tokens": _mean([row["completion_tokens"] for row in rows]),
            "avg_total_tokens": _mean([row["total_tokens"] for row in rows]),
            "avg_wall_seconds": _mean([row["wall_seconds"] for row in rows]),
            "format_errors": sum(row["format_errors"] for row in rows),
            "no_tool_replies": sum(row["no_tool_replies"] for row in rows),
            "aborted_by_format": sum(1 for row in rows if row["stop_reason"] in {"parse_error", "no_tool_call"}),
            "verify_send_backs": sum(row["verify_retries"] for row in rows),
            "runs_rescued_by_verify": sum(1 for row in rows if row["verify_retries"] > 0 and row["success"]),
            "errors": sum(1 for row in rows if row["error"]),
            "stop_reasons": dict(Counter(str(row["stop_reason"]) for row in rows)),
            "by_category": {
                category: round(sum(values) / len(values), 4) for category, values in sorted(per_category.items())
            },
            "by_task": {task: f"{sum(values)}/{len(values)}" for task, values in sorted(per_task.items())},
        }
    return summary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(summary: dict[str, Any], meta: dict[str, Any], config_names: list[str]) -> str:
    lines = [
        "# 评测报告",
        "",
        f"- 日期：{meta['finished_at'][:10]}",
        f"- 模型：`{meta['model']}`，temperature {meta['temperature']}，extra_body `{json.dumps(meta['extra_body'], ensure_ascii=False)}`",
        f"- 任务：{meta['tasks']} 个（{meta['task_breakdown']}），每个任务在每种配置下运行 {meta['runs_per_task']} 次，共 {meta['total_runs']} 次运行",
        f"- 共同设置：最多 {meta['max_steps']} 步；测试验证最多打回 {meta['max_verify_retries']} 次；三种配置共用同一套工具、hooks 和上下文截断规则",
        "- 判定：Agent 结束后恢复原始测试文件（防止改测试），再加入隐藏测试，全部通过才算成功",
    ]
    if meta.get("note"):
        lines.append(f"- 说明：{meta['note']}")
    lines += [
        "",
        "## 总体结果",
        "",
        "| 配置 | 说明 | 成功率 | 每次都成功的任务 | 平均步数 | 平均模型调用 | 平均 token | 平均耗时(秒) | 格式错误(次) | 因格式问题中止(次) | 测试验证打回(次) | 被打回后修好(次) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in config_names:
        row = summary[name]
        lines.append(
            f"| {name} | {row['label']} | {_pct(row['success_rate'])} ({row['successes']}/{row['runs']}) "
            f"| {row['tasks_solved_every_run']}/{row['tasks']} | {row['avg_steps']} | {row['avg_llm_calls']} "
            f"| {row['avg_total_tokens']:.0f} | {row['avg_wall_seconds']} | {row['format_errors'] + row['no_tool_replies']} "
            f"| {row['aborted_by_format']} | {row['verify_send_backs']} | {row['runs_rescued_by_verify']} |"
        )

    categories = sorted({category for name in config_names for category in summary[name]["by_category"]})
    lines += ["", "## 按任务类别的成功率", "", "| 类别 | " + " | ".join(config_names) + " |", "| --- |" + " ---: |" * len(config_names)]
    for category in categories:
        cells = [_pct(summary[name]["by_category"].get(category, 0.0)) for name in config_names]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    tasks = sorted({task for name in config_names for task in summary[name]["by_task"]})
    lines += ["", "## 每个任务的通过次数", "", "| 任务 | " + " | ".join(config_names) + " |", "| --- |" + " ---: |" * len(config_names)]
    for task in tasks:
        cells = [summary[name]["by_task"].get(task, "-") for name in config_names]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    lines += ["", "## 结束原因", ""]
    for name in config_names:
        reasons = ", ".join(f"{reason} {count}" for reason, count in sorted(summary[name]["stop_reasons"].items()))
        errors = summary[name]["errors"]
        lines.append(f"- {name}：{reasons}" + (f"；运行异常 {errors} 次" if errors else ""))
    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    manifest: Path,
    config_names: list[str],
    runs: int,
    workers: int,
    task_ids: list[str] | None,
    output_root: Path,
    base_config: AgentConfig,
    project_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, str]:
    unknown = [name for name in config_names if name not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configurations: {unknown}; choose from {list(CONFIGS)}")
    tasks = load_tasks(manifest, project_root, task_ids)
    started = datetime.now()
    results_dir = output_root / started.strftime("%Y%m%d_%H%M%S")
    traces_dir = results_dir / "traces"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_path = results_dir / "runs.jsonl"

    # Interleave configurations so they share the same API conditions over time.
    jobs = [(task, name, index) for index in range(runs) for task in tasks for name in config_names]
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run_one, task, name, index, base_config, project_root, traces_dir) for task, name, index in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with lock:
                records.append(record)
                with runs_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress(
                f"[{count}/{len(jobs)}] {record['config']} {record['task']} run{record['run']}: "
                f"{'PASS' if record['success'] else 'FAIL'} steps={record['steps']} "
                f"tokens={record['total_tokens']} stop={record['stop_reason']}"
            )

    finished = datetime.now()
    categories = Counter(task.category for task in tasks)
    meta = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "model": base_config.model,
        "temperature": base_config.temperature,
        "extra_body": base_config.extra_body,
        "max_steps": base_config.max_steps,
        "max_verify_retries": base_config.max_verify_retries,
        "tasks": len(tasks),
        "task_breakdown": "，".join(f"{category} {count}" for category, count in sorted(categories.items())),
        "runs_per_task": runs,
        "total_runs": len(jobs),
        "configs": {name: CONFIGS[name] for name in config_names},
    }
    summary = summarize(records, config_names)
    (results_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "configs": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(summary, meta, config_names)
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    return results_dir, report
