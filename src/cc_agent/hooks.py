from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

DANGEROUS_COMMAND_PARTS = [
    "rm -rf",
    "sudo",
    "mkfs",
    "dd ",
    ":(){",
    "chmod -R 777",
    "chown -R",
    "curl ",
    "wget ",
    "| sh",
    "| bash",
    "killall",
    "pkill",
]

PROTECTED_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519"}
ALLOWED_TEST_PREFIXES = (
    "pytest",
    "python -m pytest",
    "conda run",
    "npm test",
    "npm run test",
    "pnpm test",
    "yarn test",
)


class HookViolation(RuntimeError):
    pass


def ensure_inside_repo(repo_root: Path, candidate: str | Path) -> Path:
    root = repo_root.resolve()
    path = (root / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
    if path != root and root not in path.parents:
        raise HookViolation(f"Path escapes repository root: {path}")
    return path


def validate_tool_call(repo_root: Path, tool_name: str, arguments: dict[str, Any]) -> None:
    if tool_name in {"read_file", "write_file", "replace_in_file"}:
        path = str(arguments.get("path", ""))
        if not path:
            raise HookViolation(f"{tool_name} requires a path")
        resolved = ensure_inside_repo(repo_root, path)
        if tool_name in {"write_file", "replace_in_file"}:
            validate_write_path(resolved)

    if tool_name == "run_tests":
        command = str(arguments.get("command", ""))
        validate_test_command(command)


def validate_write_path(path: Path) -> None:
    if path.name in PROTECTED_NAMES:
        raise HookViolation(f"Refusing to edit protected file: {path.name}")
    if any(part in {".git", "node_modules", "__pycache__"} for part in path.parts):
        raise HookViolation(f"Refusing to edit generated or internal path: {path}")


def validate_test_command(command: str) -> None:
    normalized = " ".join(shlex.split(command)) if command.strip() else "pytest -q"
    lowered = normalized.lower()
    if any(part in lowered for part in DANGEROUS_COMMAND_PARTS):
        raise HookViolation(f"Dangerous command blocked: {command}")
    if not normalized.startswith(ALLOWED_TEST_PREFIXES):
        raise HookViolation(
            "Test command is not in allowlist. Use pytest, python -m pytest, npm test, pnpm test, yarn test, or conda run."
        )


def post_tool_check(tool_name: str, ok: bool, output: str) -> str | None:
    if tool_name == "run_tests" and not ok:
        return "Tests failed. Inspect the output before editing again."
    if tool_name in {"write_file", "replace_in_file"} and ok:
        return "File edited. Run tests or inspect git diff before finishing."
    return None
