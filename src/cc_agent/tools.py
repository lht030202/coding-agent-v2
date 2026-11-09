from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cc_agent.hooks import (
    HookViolation,
    ensure_inside_repo,
    is_protected_file,
    post_tool_check,
    sanitize_env,
    validate_test_command,
    validate_tool_call,
)
from cc_agent.repo_indexer import RepoIndexer

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".venv", "venv"}
# read_file stays this far below the agent's output budget, so its paging note fits and the
# context layer never has to cut the middle out of a file.
READ_NOTE_RESERVE = 400
DEFAULT_READ_LIMIT = 12000  # only used when no output budget is configured


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    blocked: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """One tool, described once and used by both protocols (JSON text and native function calling)."""

    name: str
    description: str
    parameters: dict[str, Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def _string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def _integer(description: str) -> dict[str, str]:
    return {"type": "integer", "description": description}


# Descriptions mirror the original prompt text, so the ablation compares protocols, not wording.
# Exception: read_file's line range, added after the 2026-11-09 evaluation (see docs/改进说明.md).
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("list_files", "List repository files under path.", _object({"path": _string("Directory relative to the repository root, default '.'")}, [])),
    ToolSpec(
        "read_file",
        "Read a text file, or a line range of it.",
        _object(
            {
                "path": _string("File path relative to the repository root"),
                "start_line": _integer("First line to read (1-based), e.g. a line number from grep; default 1"),
                "end_line": _integer("Last line to read (inclusive); default end of file"),
                "limit": _integer("Maximum characters to return; long files come back page by page"),
            },
            ["path"],
        ),
    ),
    ToolSpec(
        "grep",
        "Search text files with a regex.",
        _object({"pattern": _string("Python regular expression"), "path": _string("Directory or file to search, default '.'")}, ["pattern"]),
    ),
    ToolSpec(
        "retrieve_context",
        "Retrieve relevant code snippets with a lightweight lexical index.",
        _object({"query": _string("Bug description or symbol names"), "top_k": _integer("Number of files to return, default 5")}, ["query"]),
    ),
    ToolSpec(
        "replace_in_file",
        "Make a focused exact replacement.",
        _object(
            {
                "path": _string("File path relative to the repository root"),
                "old": _string("Exact existing text to replace"),
                "new": _string("Replacement text"),
            },
            ["path", "old", "new"],
        ),
    ),
    ToolSpec(
        "write_file",
        "Overwrite a file safely.",
        _object({"path": _string("File path relative to the repository root"), "content": _string("Full new file content")}, ["path", "content"]),
    ),
    ToolSpec("run_tests", "Run an allowlisted test command.", _object({"command": _string("Test command, e.g. 'pytest -q'")}, [])),
    ToolSpec("git_diff", "Show git diff if available.", _object({}, [])),
    ToolSpec("finish", "Finish when the task is complete.", _object({"summary": _string("Final answer for the user")}, ["summary"])),
)

# The original JSON-mode prompt, kept verbatim for the baseline configuration.
# Same single exception as TOOL_SPECS, read_file's line range: without it the baseline never used
# start_line, as its history keeps 1800 characters per output and cuts off the paging note.
LEGACY_TOOL_DESCRIPTIONS = """
Available tools:
- list_files {"path": "."}: list repository files under path.
- read_file {"path": "relative/path.py"}: read a text file. Optional "start_line" and "end_line" (1-based, inclusive) read a line range, e.g. around a line number from grep; long files come back page by page.
- grep {"pattern": "regex", "path": "."}: search text files with a regex.
- retrieve_context {"query": "bug or symbol name", "top_k": 5}: retrieve relevant code snippets with a lightweight lexical index.
- replace_in_file {"path": "relative/path.py", "old": "exact old text", "new": "replacement text"}: make a focused exact replacement.
- write_file {"path": "relative/path.py", "content": "full new file content"}: overwrite a file safely.
- run_tests {"command": "pytest -q"}: run an allowlisted test command.
- git_diff {}: show git diff if available, otherwise a best-effort status.
- finish {"summary": "final answer"}: finish when the task is complete.
""".strip()


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Check required fields and basic types; unknown fields are dropped, integers are coerced."""
    properties: dict[str, Any] = spec.parameters.get("properties", {})
    for name in spec.parameters.get("required", []):
        if name not in arguments:
            return None, f"Missing required argument '{name}' for {spec.name}."
    cleaned: dict[str, Any] = {}
    for name, value in arguments.items():
        if name not in properties:
            continue
        expected = properties[name].get("type")
        if expected == "string" and not isinstance(value, str):
            return None, f"Argument '{name}' for {spec.name} must be a string."
        if expected == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return None, f"Argument '{name}' for {spec.name} must be an integer."
        cleaned[name] = value
    return cleaned, None


class RepoTools:
    def __init__(self, repo_path: str | Path, timeout: int = 60, max_output_chars: int = 6000):
        self.repo_root = Path(repo_path).resolve()
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self._specs = {spec.name: spec for spec in TOOL_SPECS}
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "grep": self._grep,
            "retrieve_context": self._retrieve_context,
            "replace_in_file": self._replace_in_file,
            "write_file": self._write_file,
            "run_tests": self._run_tests,
            "git_diff": self._git_diff,
            "finish": self._finish,
        }

    @property
    def tool_names(self) -> list[str]:
        return list(self._specs)

    def legacy_descriptions(self) -> str:
        return LEGACY_TOOL_DESCRIPTIONS

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in TOOL_SPECS]

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(False, f"Unknown tool: {name}. Available tools: {', '.join(self._specs)}.")
        if not isinstance(arguments, dict):
            return ToolResult(False, f"Arguments for {name} must be a JSON object.")
        cleaned, error = validate_arguments(spec, arguments)
        if error or cleaned is None:
            return ToolResult(False, error or "Invalid arguments.")
        try:
            validate_tool_call(self.repo_root, name, cleaned)
            result = self._handlers[name](cleaned)
            note = post_tool_check(name, result.ok, result.output)
            if note:
                return ToolResult(result.ok, f"{result.output}\n\nHook note: {note}", result.blocked, result.reason)
            return result
        except HookViolation as exc:
            return ToolResult(ok=False, output=str(exc), blocked=True, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - tool errors become observations for the model
            return ToolResult(ok=False, output=f"Tool error: {type(exc).__name__}: {exc}")

    def _iter_files(self, base: Path):
        targets = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in targets:
            if any(part in SKIP_DIRS for part in path.relative_to(self.repo_root).parts):
                continue
            if path.is_file():
                yield path

    def _list_files(self, arguments: dict[str, Any]) -> ToolResult:
        base = ensure_inside_repo(self.repo_root, arguments.get("path", "."))
        if not base.exists():
            return ToolResult(False, f"Path does not exist: {arguments.get('path', '.')}")
        files: list[str] = []
        for path in self._iter_files(base):
            files.append(path.relative_to(self.repo_root).as_posix())
            if len(files) >= 120:
                files.append("... output truncated")
                break
        return ToolResult(True, "\n".join(files) or "No files found.")

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = ensure_inside_repo(self.repo_root, arguments["path"])
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {arguments['path']}")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total = len(lines)
        start = max(1, int(arguments.get("start_line", 1)))
        if total and start > total:
            return ToolResult(False, f"start_line {start} is past the end of the file ({total} lines).")
        end = min(total, max(start, int(arguments.get("end_line", total))))
        selected = "".join(lines[start - 1 : end])

        # Stay inside the agent's output budget; otherwise the context layer would cut the middle out.
        cap = max(500, self.max_output_chars - READ_NOTE_RESERVE) if self.max_output_chars > 0 else None
        requested = int(arguments.get("limit", cap or DEFAULT_READ_LIMIT))
        budget = max(1, min(requested, cap) if cap else requested)
        ranged = "start_line" in arguments or "end_line" in arguments
        if len(selected) <= budget:
            return ToolResult(True, f"[lines {start}-{end} of {total}]\n{selected}" if ranged else selected)

        cut = selected.rfind("\n", 0, budget) + 1  # stop at a line boundary
        if cut == 0:
            return ToolResult(True, f"{selected[:budget]}\n... [line {start} is longer than {budget} characters; use grep to find what you need]")
        shown_end = start + selected[:cut].count("\n") - 1
        return ToolResult(
            True,
            f"{selected[:cut]}... [showing lines {start}-{shown_end} of {total}; {len(text)} characters in the file] "
            f"Read on with start_line={shown_end + 1}, or use grep to find the lines you need.",
        )

    def _grep(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return ToolResult(False, "grep requires a pattern")
        base = ensure_inside_repo(self.repo_root, arguments.get("path", "."))
        regex = re.compile(pattern)
        matches: list[str] = []
        for path in self._iter_files(base):
            if is_protected_file(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(f"{path.relative_to(self.repo_root).as_posix()}:{line_no}: {line}")
                    if len(matches) >= 80:
                        return ToolResult(True, "\n".join(matches) + "\n... matches truncated")
        return ToolResult(True, "\n".join(matches) or "No matches found.")

    def _retrieve_context(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", ""))
        if not query.strip():
            return ToolResult(False, "retrieve_context requires a non-empty query")
        top_k = int(arguments.get("top_k", 5))
        return ToolResult(True, RepoIndexer(self.repo_root).retrieve(query=query, top_k=top_k))

    def _replace_in_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = ensure_inside_repo(self.repo_root, arguments["path"])
        old, new = arguments["old"], arguments["new"]
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {arguments['path']}")
        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old)
        if count == 0:
            return ToolResult(False, "old text not found; inspect the file before retrying")
        if count > 1:
            return ToolResult(False, f"old text occurs {count} times; provide a more specific snippet")
        updated = content.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return ToolResult(True, self._diff(path, content, updated))

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = ensure_inside_repo(self.repo_root, arguments["path"])
        content = arguments["content"]
        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, self._diff(path, old, content))

    def _diff(self, path: Path, before: str, after: str) -> str:
        rel = path.relative_to(self.repo_root).as_posix()
        diff = "\n".join(
            difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="")
        )
        return diff or f"Updated {rel} with no textual diff."

    def _run_tests(self, arguments: dict[str, Any]) -> ToolResult:
        tokens = validate_test_command(arguments.get("command") or "pytest -q")
        lowered = [token.lower() for token in tokens]
        if lowered[:1] == ["pytest"]:
            tokens = [sys.executable, "-m", "pytest", *tokens[1:]]
        elif lowered[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            tokens = [sys.executable, "-m", "pytest", *tokens[3:]]
        env = sanitize_env(
            {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(self.repo_root)}
        )
        try:
            completed = subprocess.run(
                tokens,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Test command timed out after {self.timeout}s")
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return ToolResult(completed.returncode == 0, output or f"Command exited with {completed.returncode}")

    def _git_diff(self, arguments: dict[str, Any]) -> ToolResult:
        completed = subprocess.run(
            ["git", "diff", "--"], cwd=self.repo_root, text=True, capture_output=True, timeout=self.timeout, check=False
        )
        if completed.returncode != 0:
            return ToolResult(False, completed.stderr.strip() or "git diff failed; repository may not be a git repo")
        return ToolResult(True, completed.stdout.strip() or "No git diff.")

    def _finish(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(True, str(arguments.get("summary", "Finished.")))
