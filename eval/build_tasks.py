"""Build the evaluation task repos plus eval/tasks.json, and check that every task is valid.

Run from the project root:
    .venv/bin/python eval/build_tasks.py

Checks:
1. Every task is solvable: the correct minishop codebase passes its visible and hidden tests, and
   every function task passes its tests once the reference solution is filled in.
2. Every generated task starts with at least one failing visible test (so no task is trivial).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BASE_DIR = EVAL_DIR / "base_repos"
TASKS_DIR = EVAL_DIR / "tasks"
MINISHOP = BASE_DIR / "minishop"
HIDDEN_TESTS = MINISHOP / "hidden_tests"

sys.path.insert(0, str(EVAL_DIR))
from task_defs import FUNCTION_TASKS, MINISHOP_TASKS  # noqa: E402

IGNORE = shutil.ignore_patterns("__pycache__", ".pytest_cache", "hidden_tests")


def pytest_passes(repo: Path, *paths: str) -> bool:
    env = {**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def build_minishop_task(spec: dict) -> dict:
    repo = TASKS_DIR / spec["id"] / "repo"
    shutil.rmtree(repo.parent, ignore_errors=True)
    shutil.copytree(MINISHOP, repo, ignore=IGNORE)
    for rel_path, old, new in spec["injections"]:
        path = repo / rel_path
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{spec['id']}: injection target appears {count} times in {rel_path}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return {
        "id": spec["id"],
        "category": spec["category"],
        "task": spec["task"],
        "repo": relative(repo),
        "hidden_tests": relative(HIDDEN_TESTS),
        "test_command": "pytest -q",
    }


def build_function_task(spec: dict) -> dict:
    repo = TASKS_DIR / spec["id"] / "repo"
    shutil.rmtree(repo.parent, ignore_errors=True)
    shutil.copytree(BASE_DIR / spec["source"], repo, ignore=IGNORE)
    return {
        "id": spec["id"],
        "category": spec["category"],
        "task": spec["task"],
        "repo": relative(repo),
        "hidden_tests": None,
        "test_command": "pytest -q",
    }


def reference_passes(source: Path, reference: str) -> bool:
    """Fill the reference solution into a temporary copy and run the task's tests."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(source, repo, ignore=IGNORE)
        solution = repo / "solution.py"
        head, placeholder, tail = solution.read_text(encoding="utf-8").rpartition("    pass\n")
        if not placeholder:
            raise SystemExit(f"{source}: solution.py has no `pass` placeholder")
        solution.write_text(head + reference + tail, encoding="utf-8")
        return pytest_passes(repo, "tests")


def main() -> None:
    if not pytest_passes(MINISHOP, "tests", "hidden_tests"):
        raise SystemExit("The correct minishop codebase does not pass its own tests.")
    print("ok  base minishop passes visible + hidden tests")
    for spec in FUNCTION_TASKS:
        if not reference_passes(BASE_DIR / spec["source"], spec["reference"]):
            raise SystemExit(f"{spec['id']}: the reference solution does not pass the tests, the task is unsolvable")
        print(f"ok  {spec['id']:<28} reference solution passes")

    entries = [build_minishop_task(spec) for spec in MINISHOP_TASKS]
    entries += [build_function_task(spec) for spec in FUNCTION_TASKS]
    for entry in entries:
        if pytest_passes(PROJECT_ROOT / entry["repo"], "tests"):
            raise SystemExit(f"{entry['id']}: visible tests already pass, the task would be meaningless")
        print(f"ok  {entry['id']:<28} {entry['category']}  (visible tests fail before the fix)")

    manifest = EVAL_DIR / "tasks.json"
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} tasks to {relative(manifest)}")


if __name__ == "__main__":
    main()
