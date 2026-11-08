from cc_agent.tools import TOOL_SPECS, RepoTools, validate_arguments


def test_openai_schemas_cover_every_tool(tmp_path):
    tools = RepoTools(tmp_path)
    schemas = tools.openai_tools()
    assert [schema["function"]["name"] for schema in schemas] == tools.tool_names
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


def test_replace_requires_a_unique_match(buggy_repo):
    tools = RepoTools(buggy_repo)
    (buggy_repo / "dup.py").write_text("x = 1\nx = 1\n")
    assert "occurs 2 times" in tools.run("replace_in_file", {"path": "dup.py", "old": "x = 1", "new": "x = 2"}).output
    assert "not found" in tools.run("replace_in_file", {"path": "dup.py", "old": "y = 1", "new": "y = 2"}).output
    result = tools.run("replace_in_file", {"path": "calculator.py", "old": "return a + b", "new": "return a - b"})
    assert result.ok
    assert "+    return a - b" in result.output


def test_missing_arguments_become_observations(buggy_repo):
    result = RepoTools(buggy_repo).run("replace_in_file", {"path": "calculator.py"})
    assert not result.ok
    assert "Missing required argument 'old'" in result.output


def test_unknown_tool_lists_available_tools(buggy_repo):
    result = RepoTools(buggy_repo).run("delete_repo", {})
    assert not result.ok
    assert "Available tools" in result.output


def test_validate_arguments_drops_unknown_fields_and_coerces_integers():
    spec = next(spec for spec in TOOL_SPECS if spec.name == "read_file")
    cleaned, error = validate_arguments(spec, {"path": "a.py", "limit": "50", "reason": "look"})
    assert error is None
    assert cleaned == {"path": "a.py", "limit": 50}


def test_grep_skips_secret_files(buggy_repo):
    (buggy_repo / ".env").write_text("OPENAI_API_KEY=sk-secret\n")
    assert "No matches found" in RepoTools(buggy_repo).run("grep", {"pattern": "sk-secret"}).output


def test_run_tests_does_not_leak_api_keys(buggy_repo, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    (buggy_repo / "calculator.py").write_text("def subtract(a, b):\n    return a - b\n")
    (buggy_repo / "tests" / "test_env.py").write_text(
        "import os\n\n\ndef test_no_key():\n    assert 'OPENAI_API_KEY' not in os.environ\n"
    )
    result = RepoTools(buggy_repo).run("run_tests", {"command": "pytest -q"})
    assert result.ok, result.output


def test_read_file_truncation_tells_how_to_read_more(buggy_repo):
    (buggy_repo / "big.py").write_text("x" * 50)
    result = RepoTools(buggy_repo).run("read_file", {"path": "big.py", "limit": 10})
    assert result.ok
    assert "pass a larger limit" in result.output
