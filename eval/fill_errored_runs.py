"""Fill in the runs an interrupted evaluation lost to API errors, then rebuild the report.

Used when an evaluation is cut short by API errors (rate limits, an empty account balance): the
completed runs are kept, the missing (task, configuration) runs are re-run separately with the same
code, model and settings, and the pieces are merged so that every task again has the planned number
of runs. The script refuses to write a report if the runs do not add up.

    .venv/bin/python eval/fill_errored_runs.py eval/results/<partial_run> eval/results/<re_run> [<re_run> ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cc_agent.evaluation import render_report, summarize  # noqa: E402


def load_runs(results_dir: Path) -> list[dict]:
    lines = (results_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_meta(results_dir: Path) -> dict:
    return json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))["meta"]


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    partial_dir, *rerun_dirs = (Path(arg).resolve() for arg in sys.argv[1:])
    meta = load_meta(partial_dir)
    config_names = list(meta["configs"])
    planned = int(meta["runs_per_task"])

    partial = load_runs(partial_dir)
    reasons = Counter(row["error"][:80] for row in partial if row.get("error"))
    kept = [row for row in partial if not row.get("error")]
    added = [row for directory in rerun_dirs for row in load_runs(directory) if not row.get("error")]

    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    records = []
    for row in kept + added:
        key = (row["task"], row["config"])
        records.append({**row, "run": counts[key]})
        counts[key] += 1
    records.sort(key=lambda row: (config_names.index(row["config"]), row["task"], row["run"]))

    tasks = sorted({row["task"] for row in partial})
    wrong = {f"{task}/{cfg}": counts[(task, cfg)] for task in tasks for cfg in config_names if counts[(task, cfg)] != planned}
    if wrong:
        raise SystemExit(f"Runs do not add up to {planned} per task and configuration: {wrong}")

    output = partial_dir.parent / f"merged_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True)
    with (output / "runs.jsonl").open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    rerun_metas = [load_meta(directory) for directory in rerun_dirs]
    reason = "；".join(f"{text}（{count} 次）" for text, count in reasons.most_common(2))
    merged_meta = {
        **meta,
        "finished_at": max(item["finished_at"] for item in rerun_metas),
        "total_runs": len(records),
        "note": (
            f"这轮评测开始于 {meta['started_at']}，中途因接口错误中断：{reason}。"
            f"已完成的 {len(kept)} 次运行保留；缺失的 {len(added)} 次用相同的代码、模型和配置分 {len(rerun_dirs)} 批补跑后合并。"
        ),
        "sources": [str(path.relative_to(PROJECT_ROOT)) for path in (partial_dir, *rerun_dirs)],
    }
    summary = summarize(records, config_names)
    (output / "summary.json").write_text(
        json.dumps({"meta": merged_meta, "configs": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(summary, merged_meta, config_names)
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"merged {len(kept)} kept + {len(added)} re-run = {len(records)} runs into {output.relative_to(PROJECT_ROOT)}")
    print(report)


if __name__ == "__main__":
    main()
