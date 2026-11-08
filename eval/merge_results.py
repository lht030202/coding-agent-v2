"""Merge a re-run of some tasks into a main evaluation result and rebuild the report.

Used when a task had to be fixed after the main run: the re-run records of those tasks replace
the originals, everything else comes from the main run, and the report says so.

    .venv/bin/python eval/merge_results.py eval/results/<main_run> eval/results/<re_run>
"""
from __future__ import annotations

import json
import sys
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
    main_dir, rerun_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    main_meta, rerun_meta = load_meta(main_dir), load_meta(rerun_dir)
    rerun = load_runs(rerun_dir)
    rerun_tasks = sorted({row["task"] for row in rerun})
    records = [row for row in load_runs(main_dir) if row["task"] not in rerun_tasks] + rerun

    output = PROJECT_ROOT / "eval" / "results" / f"final_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True)
    with (output / "runs.jsonl").open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    config_names = list(main_meta["configs"])
    meta = {
        **main_meta,
        "finished_at": rerun_meta["finished_at"],
        "total_runs": len(records),
        "note": (
            f"{'、'.join(rerun_tasks)} 在主评测后发现测试文件缺少导入（任务不可解），修复后单独重跑；"
            f"其余任务来自主评测。两次使用相同的代码、模型和配置"
            f"（主评测开始于 {main_meta['started_at']}，重跑开始于 {rerun_meta['started_at']}）。"
        ),
        "sources": [str(main_dir.relative_to(PROJECT_ROOT)), str(rerun_dir.relative_to(PROJECT_ROOT))],
    }
    summary = summarize(records, config_names)
    (output / "summary.json").write_text(
        json.dumps({"meta": meta, "configs": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(summary, meta, config_names)
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"merged {len(records)} runs into {output.relative_to(PROJECT_ROOT)}")
    print(report)


if __name__ == "__main__":
    main()
