from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Mapping

# Defence in depth only: commands never go through a shell, and they must match the allowlist.
DANGEROUS_COMMAND_PARTS = (
    "rm -rf",
    "rm -r ",
    "sudo",
    "mkfs",
    "dd ",
    ":(){",
    "chmod -r 777",
    "chown -r",
    "curl ",
    "wget ",
    "killall",
    "pkill",
    "shutdown",
    "reboot",
)
SHELL_METACHARACTERS = (";", "&&", "||", "|", "`", "$(", ">", "<")
# Exact token prefixes. The original prefix check accepted "conda run <anything>" and "pytest_evil".
ALLOWED_TEST_COMMANDS = (
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("npm", "test"),
    ("npm", "run", "test"),
    ("pnpm", "test"),
    ("yarn", "test"),
)
PROTECTED_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", ".netrc", ".pypirc"}
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}
SECRET_ENV_PATTERN = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
INTERNAL_DIRS = {".git", "node_modules", "__pycache__"}


class HookViolation(RuntimeError):
    pass


def ensure_inside_repo(repo_root: Path, candidate: str | Path) -> Path:
    root = repo_root.resolve()
    path = Path(candidate)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path != root and root not in path.parents:
        raise HookViolation(f"Path escapes repository root: {path}")
    return path


def is_protected_file(path: Path) -> bool:
    """Secrets must never be read into the model context or edited."""
    name = path.name
    if name in SAFE_ENV_TEMPLATES:
        return False
    return name in PROTECTED_NAMES or name.startswith(".env")


def validate_tool_call(repo_root: Path, tool_name: str, arguments: dict[str, Any]) -> None:
    if tool_name in {"read_file", "write_file", "replace_in_file"}:
        path = str(arguments.get("path", ""))
        if not path:
            raise HookViolation(f"{tool_name} requires a path")
        resolved = ensure_inside_repo(repo_root, path)
        if is_protected_file(resolved):
            action = "read" if tool_name == "read_file" else "edit"
            raise HookViolation(f"Refusing to {action} protected file: {resolved.name}")
        if tool_name in {"write_file", "replace_in_file"}:
            validate_write_path(resolved)

    if tool_name in {"list_files", "grep"}:
        ensure_inside_repo(repo_root, str(arguments.get("path", ".")))

    if tool_name == "run_tests":
        validate_test_command(str(arguments.get("command", "")))


def validate_write_path(path: Path) -> None:
    if any(part in INTERNAL_DIRS for part in path.parts):
        raise HookViolation(f"Refusing to edit generated or internal path: {path}")


def validate_test_command(command: str) -> list[str]:
    raw = command.strip() or "pytest -q"
    for meta in SHELL_METACHARACTERS:
        if meta in raw:
            raise HookViolation(f"Shell operators are not allowed in test commands: {meta!r}")
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise HookViolation(f"Cannot parse test command: {exc}") from exc
    lowered = [token.lower() for token in tokens]
    normalized = " ".join(lowered) + " "
    for part in DANGEROUS_COMMAND_PARTS:
        if part in normalized:
            raise HookViolation(f"Dangerous command blocked: {command}")
    if not any(lowered[: len(prefix)] == list(prefix) for prefix in ALLOWED_TEST_COMMANDS):
        raise HookViolation(
            "Test command is not in the allowlist. Use pytest, python -m pytest, npm test, pnpm test or yarn test."
        )
    return tokens


def sanitize_env(env: Mapping[str, str]) -> dict[str, str]:
    """Drop secrets before running repository code, so a test cannot print an API key into the context."""
    return {key: value for key, value in env.items() if not SECRET_ENV_PATTERN.search(key)}


def post_tool_check(tool_name: str, ok: bool, output: str) -> str | None:
    if tool_name == "run_tests" and not ok:
        return "Tests failed. Inspect the output before editing again."
    if tool_name in {"write_file", "replace_in_file"} and ok:
        return "File edited. Run tests or inspect git diff before finishing."
    return None
