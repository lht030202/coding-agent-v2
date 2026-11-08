"""Agent-loop tests with a scripted fake model: deterministic, offline, no API cost."""
import json
from pathlib import Path

from cc_agent.graph import run_agent
from helpers import FIX, ScriptedLLM, call, make_config, text


def run(repo, llm, tmp_path, **overrides):
    config = make_config(tmp_path / "traces", **overrides)
    return run_agent(repo, "修复 subtract", "pytest -q", config=config, project_root=tmp_path, llm=llm)


def test_native_mode_fixes_the_bug_and_verifies(buggy_repo, tmp_path):
    llm = ScriptedLLM(
        [
            text("1. read calculator.py 2. fix it 3. run tests"),  # plan
            call("read_file", '{"path": "calculator.py"}'),  # act 1
            call("replace_in_file", FIX, "call_2"),  # act 2
            call("finish", '{"summary": "fixed"}', "call_3"),  # act 3
            text("looks good"),  # review
            text("summary"),  # summarize
        ]
    )
    final = run(buggy_repo, llm, tmp_path)
    assert "return a - b" in (buggy_repo / "calculator.py").read_text()
    assert final["stop_reason"] == "finish"
    assert final["tests_passed"] is True
    assert final["steps"] == 3
    assert final["usage"]["llm_calls"] == 6
    # Act requests carry the tool schemas, and tool results come back as tool messages.
    second_act = llm.requests[2]
    assert second_act["tools"]
    assert second_act["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": second_act["messages"][-1]["content"]}
    # Every model call is in the trace with its token counts.
    events = [json.loads(line) for line in Path(final["trace_path"]).read_text().splitlines()]
    assert sum(event["event"] == "llm_call" for event in events) == 6


def test_failed_finish_is_sent_back_with_the_test_output(buggy_repo, tmp_path):
    llm = ScriptedLLM(
        [
            text("plan"),
            call("finish", '{"summary": "done"}'),  # premature: the tests still fail
            call("replace_in_file", FIX, "call_2"),
            call("finish", '{"summary": "fixed"}', "call_3"),
            text("review"),
            text("summary"),
        ]
    )
    final = run(buggy_repo, llm, tmp_path)
    assert final["verify_retries"] == 1
    assert final["tests_passed"] is True
    feedback = llm.requests[2]["messages"][-1]
    assert feedback["role"] == "user"
    assert "Automatic verification failed" in feedback["content"]


def test_rule_verify_accepts_a_premature_finish(buggy_repo, tmp_path):
    llm = ScriptedLLM([text("plan"), call("finish", '{"summary": "done"}'), text("review"), text("summary")])
    final = run(buggy_repo, llm, tmp_path, verify_mode="rule")
    assert final["stop_reason"] == "finish"
    assert final["tests_passed"] is None
    assert "return a + b" in (buggy_repo / "calculator.py").read_text()  # the bug is still there


def test_plain_text_reply_is_nudged_back_to_the_tools(buggy_repo, tmp_path):
    llm = ScriptedLLM(
        [
            text("plan"),
            text("I think the bug is in subtract."),  # no tool call
            call("replace_in_file", FIX),
            call("finish", '{"summary": "fixed"}', "call_2"),
            text("review"),
            text("summary"),
        ]
    )
    final = run(buggy_repo, llm, tmp_path)
    assert final["counters"]["no_tool_replies"] == 1
    assert final["tests_passed"] is True
    assert llm.requests[2]["messages"][-1]["content"].startswith("Continue by calling")


def test_invalid_tool_arguments_become_an_observation(buggy_repo, tmp_path):
    llm = ScriptedLLM(
        [
            text("plan"),
            call("read_file", '{"path": calculator.py}'),  # broken JSON
            call("replace_in_file", FIX, "call_2"),
            call("finish", '{"summary": "fixed"}', "call_3"),
            text("review"),
            text("summary"),
        ]
    )
    final = run(buggy_repo, llm, tmp_path)
    assert final["counters"]["format_errors"] == 1
    assert final["tests_passed"] is True
    assert "Invalid arguments" in llm.requests[2]["messages"][-1]["content"]


def test_json_mode_stops_when_the_reply_cannot_be_parsed(buggy_repo, tmp_path):
    llm = ScriptedLLM([text("plan"), text('<｜DSML｜ invoke name="read_file">'), text("review"), text("summary")])
    final = run(buggy_repo, llm, tmp_path, tool_mode="json", verify_mode="rule")
    assert final["stop_reason"] == "parse_error"
    assert final["counters"]["format_errors"] == 1


def test_json_mode_executes_parsed_tool_calls(buggy_repo, tmp_path):
    llm = ScriptedLLM(
        [
            text("plan"),
            text('{"tool": "replace_in_file", "arguments": ' + FIX + ', "reason": "fix"}'),
            text('```json\n{"tool": "finish", "arguments": {"summary": "done"}}\n```'),
            text("review"),
            text("summary"),
        ]
    )
    final = run(buggy_repo, llm, tmp_path, tool_mode="json", verify_mode="rule")
    assert final["stop_reason"] == "finish"
    assert "return a - b" in (buggy_repo / "calculator.py").read_text()
    assert llm.requests[1]["tools"] is None  # the original protocol sends no tool schema


def test_max_steps_stops_the_loop(buggy_repo, tmp_path):
    replies = [text("plan")]
    replies += [call("read_file", '{"path": "calculator.py"}', f"call_{index}") for index in range(3)]
    replies += [text("review"), text("summary")]
    final = run(buggy_repo, ScriptedLLM(replies), tmp_path, max_steps=3)
    assert final["stop_reason"] == "max_steps"
    assert final["steps"] == 3
    assert final["tests_passed"] is False
