from cc_agent.tools import LEGACY_TOOL_DESCRIPTIONS, TOOL_SPECS, RepoTools, validate_arguments


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


def test_overlong_single_line_is_cut_with_a_hint(buggy_repo):
    (buggy_repo / "big.py").write_text("x" * 50)
    result = RepoTools(buggy_repo).run("read_file", {"path": "big.py", "limit": 10})
    assert result.ok
    assert result.output.startswith("x" * 10 + "\n")
    assert "use grep" in result.output


def test_read_file_reads_a_line_range(buggy_repo):
    (buggy_repo / "big.py").write_text("".join(f"line {n}\n" for n in range(1, 101)))
    result = RepoTools(buggy_repo).run("read_file", {"path": "big.py", "start_line": "40", "end_line": 42})
    assert result.output == "[lines 40-42 of 100]\nline 40\nline 41\nline 42\n"
    properties = next(spec for spec in TOOL_SPECS if spec.name == "read_file").parameters["properties"]
    assert {"start_line", "end_line"} <= set(properties)


def test_long_files_are_paged_inside_the_output_budget(buggy_repo):
    (buggy_repo / "big.py").write_text("".join(f"value_{n} = {n}\n" for n in range(1, 2001)))
    tools = RepoTools(buggy_repo, max_output_chars=1000)
    first = tools.run("read_file", {"path": "big.py", "limit": 50000})
    assert len(first.output) <= 1000  # short enough that the context layer never cuts its middle out
    assert first.output.startswith("value_1 = 1\n")
    next_line = int(first.output.split("start_line=")[1].split(",")[0])
    second = tools.run("read_file", {"path": "big.py", "start_line": next_line})
    assert second.output.startswith(f"value_{next_line} = {next_line}\n")
    assert "Read on with start_line=" in second.output


def test_start_line_past_the_end_is_an_observation(buggy_repo):
    result = RepoTools(buggy_repo).run("read_file", {"path": "calculator.py", "start_line": 999})
    assert not result.ok
    assert "past the end" in result.output


def test_both_protocols_document_the_line_range():
    # When only the schema had start_line, the text-protocol baseline never used it (2026-11-09 evaluation).
    schema = next(spec for spec in TOOL_SPECS if spec.name == "read_file").parameters["properties"]
    text = next(line for line in LEGACY_TOOL_DESCRIPTIONS.splitlines() if line.startswith("- read_file "))
    for name in ("start_line", "end_line"):
        assert name in schema
        assert f'"{name}"' in text
