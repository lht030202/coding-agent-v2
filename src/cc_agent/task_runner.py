from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cc_agent.config import AgentConfig
from cc_agent.graph import run_agent


@dataclass(frozen=True)
class TaskRunResult:
    total: int
    runnable: int
    ran: int
    skipped: int
    output_path: Path

    def render(self) -> str:
        return (
            f"Total tasks: {self.total}\n"
            f"Runnable local tasks: {self.runnable}\n"
            f"Ran: {self.ran}\n"
            f"Skipped: {self.skipped}\n"
            f"Output: {self.output_path}"
        )


def run_task_manifest(
    manifest_path: str | Path,
    config: AgentConfig,
    project_root: Path,
    limit: int = 10,
    dry_run: bool = False,
    output_path: str | Path | None = None,
) -> TaskRunResult:
    manifest = Path(manifest_path)
    output = Path(output_path) if output_path else manifest.parent / f"{manifest.stem}_run_results.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_read_jsonl(manifest))
    runnable_rows = [row for row in rows if row.get("repo") and row.get("task")]
    ran = 0
    skipped = 0

    with output.open("w", encoding="utf-8") as out:
        for row in runnable_rows[:limit]:
            repo = Path(str(row["repo"]))
            if not repo.is_absolute():
                repo = (project_root / repo).resolve()
            record: dict[str, Any] = {
                "id": row.get("id"),
                "source": row.get("source"),
                "repo": str(repo),
                "task": row.get("task"),
                "test_command": row.get("test_command", "pytest -q"),
                "dry_run": dry_run,
            }
            if not repo.exists():
                record.update({"ok": False, "skipped": True, "reason": "repo path does not exist"})
                skipped += 1
            elif dry_run:
                record.update({"ok": True, "skipped": True, "reason": "dry run"})
                skipped += 1
            else:
                try:
                    final_state = run_agent(
                        repo_path=repo,
                        task=str(row["task"]),
                        test_command=str(row.get("test_command") or "pytest -q"),
                        config=config,
                        project_root=project_root,
                    )
                    record.update(
                        {
                            "ok": True,
                            "skipped": False,
                            "trace_path": final_state.get("trace_path"),
                            "final_answer": final_state.get("final_answer"),
                        }
                    )
                    ran += 1
                except Exception as exc:  # noqa: BLE001 - batch runner should continue on single-task failure
                    record.update(
                        {
                            "ok": False,
                            "skipped": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    ran += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return TaskRunResult(
        total=len(rows),
        runnable=len(runnable_rows),
        ran=ran,
        skipped=skipped + max(0, len(runnable_rows[:limit]) - ran - skipped),
        output_path=output,
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
