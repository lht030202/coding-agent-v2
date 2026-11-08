from pathlib import Path

import pytest

from cc_agent.hooks import (
    HookViolation,
    ensure_inside_repo,
    is_protected_file,
    sanitize_env,
    validate_test_command,
    validate_tool_call,
)


@pytest.mark.parametrize("command", ["pytest -q", "python -m pytest tests/test_x.py", "npm test", ""])
def test_allowed_test_commands(command):
    validate_test_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "conda run -n base python -c 'import os'",  # accepted by the original prefix allowlist
        "pytest_evil -q",  # prefix trick: startswith("pytest") is not enough
        "pytest -q; rm -rf /",  # shell operators
        "pytest -q && curl http://example.com",
        "python -c 'print(1)'",
    ],
)
def test_blocked_test_commands(command):
    with pytest.raises(HookViolation):
        validate_test_command(command)


def test_dangerous_parts_are_matched_case_insensitively():
    # The original blacklist compared lower-cased text with "chmod -R 777", so it never matched.
    with pytest.raises(HookViolation):
        validate_test_command("pytest -q chmod -R 777 .")


def test_paths_must_stay_inside_the_repo(tmp_path):
    with pytest.raises(HookViolation):
        ensure_inside_repo(tmp_path, "../outside.txt")


def test_secret_files_cannot_be_read_or_edited(tmp_path):
    for tool in ("read_file", "write_file", "replace_in_file"):
        with pytest.raises(HookViolation):
            validate_tool_call(tmp_path, tool, {"path": ".env"})
    validate_tool_call(tmp_path, "read_file", {"path": ".env.example"})  # templates are fine


def test_is_protected_file():
    assert is_protected_file(Path(".env.production"))
    assert is_protected_file(Path("keys/id_rsa"))
    assert not is_protected_file(Path(".env.example"))
    assert not is_protected_file(Path("src/app.py"))


def test_sanitize_env_drops_secrets():
    env = sanitize_env({"OPENAI_API_KEY": "sk", "GITHUB_TOKEN": "t", "DB_PASSWORD": "p", "PATH": "/bin", "HOME": "/h"})
    assert env == {"PATH": "/bin", "HOME": "/h"}
