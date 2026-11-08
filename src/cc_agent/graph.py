from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from cc_agent.config import AgentConfig
from cc_agent.context import compact_messages, truncate_middle
from cc_agent.json_utils import extract_json_object
from cc_agent.llm import ChatModel, LLMResult, build_chat_model
from cc_agent.repo_indexer import RepoIndexer
from cc_agent.state import AgentState, ToolRecord
from cc_agent.tools import RepoTools, ToolResult
from cc_agent.tracing import append_trace, create_trace_file


PLANNER_SYSTEM = """You are a careful AI coding planner.
Create a concise implementation plan before tools are used.
Prefer small, safe, testable edits. Do not invent file contents without inspecting files first.
"""

# Original protocol (tool_mode=json): the model writes the tool call as JSON text and the program parses it.
ACTOR_SYSTEM_JSON = """You are a Claude Code style coding agent.
You must respond with exactly one JSON object and no markdown.
Choose one tool call at a time. Inspect files before editing.
Use retrieve_context or grep to find relevant code before editing.
Prefer replace_in_file for focused edits. Use write_file only when full-file replacement is safer.
Run tests after code edits when a test command is available.

JSON schema:
{
  "tool": "tool_name",
  "arguments": {"key": "value"},
  "reason": "brief reason"
}
"""

# Native protocol (tool_mode=native): tools travel as JSON Schema and the API returns structured tool calls.
ACTOR_SYSTEM_NATIVE = """You are a Claude Code style coding agent.
Use the provided tools and choose one tool call at a time. Inspect files before editing.
Use retrieve_context or grep to find relevant code before editing.
Prefer replace_in_file for focused edits. Use write_file only when full-file replacement is safer.
Run tests after code edits when a test command is available.
Call finish with a short summary when the task is complete.
"""

REVIEWER_SYSTEM = """You are a conservative code-review subagent.
Review the tool history, diffs, and test results. Identify whether the task is actually complete.
Return a concise review with: correctness, tests, risks, and recommended follow-up.
"""

SUMMARIZER_SYSTEM = """You summarize coding-agent runs for a developer.
Be concise. Include files changed, tests run, result, review findings, and remaining risks.
"""

NO_TOOL_NUDGE = "Continue by calling one of the provided tools. If the task is complete, call finish with a short summary."
MAX_NO_TOOL_STREAK = 2
VERIFY_FEEDBACK = """Automatic verification failed: `{command}` did not pass after you called finish.
Fix the remaining problems, run the tests, and call finish again only when they pass.

Test output:
{output}"""


def new_usage() -> dict[str, Any]:
    return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "llm_seconds": 0.0}


def new_counters() -> dict[str, int]:
    return {
        "tool_calls": 0,
        "tool_errors": 0,
        "blocked_calls": 0,
        "test_failures": 0,
        "format_errors": 0,
        "no_tool_replies": 0,
        "compactions": 0,
    }


def _history_text(history: list[ToolRecord], limit: int = 9000) -> str:
    """Original single-turn memory: the last 12 tool records, each output capped at 1800 characters."""
    chunks: list[str] = []
    for item in history[-12:]:
        output = item.get("output", "")
        if len(output) > 1800:
            output = output[:1800] + "\n... output truncated"
        chunks.append(
            json.dumps(
                {
                    "tool": item.get("name"),
                    "arguments": item.get("arguments"),
                    "ok": item.get("ok"),
                    "blocked": item.get("blocked", False),
                    "output": output,
                },
                ensure_ascii=False,
            )
        )
    text = "\n".join(chunks)
    return text[-limit:]


def _count_tool(counters: dict[str, int], name: str, result: ToolResult) -> None:
    if name == "finish":
        return
    counters["tool_calls"] += 1
    if result.blocked:
        counters["blocked_calls"] += 1
    elif not result.ok:
        counters["test_failures" if name == "run_tests" else "tool_errors"] += 1


def _record(name: str, arguments: dict[str, Any], result: ToolResult) -> ToolRecord:
    return {
        "name": name,
        "arguments": arguments,
        "ok": result.ok,
        "output": result.output,
        "blocked": result.blocked,
        "reason": result.reason,
    }


class CodingAgentGraph:
    def __init__(self, config: AgentConfig, project_root: Path, llm: ChatModel | None = None):
        self.config = config
        self.project_root = project_root
        self.llm = llm if llm is not None else build_chat_model(config)

    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("index_repo", self.index_repo)
        graph.add_node("plan", self.plan)
        graph.add_node("act", self.act)
        graph.add_node("verify", self.verify)
        graph.add_node("review", self.review)
        graph.add_node("summarize", self.summarize)

        graph.set_entry_point("index_repo")
        graph.add_edge("index_repo", "plan")
        graph.add_edge("plan", "act")
        graph.add_edge("act", "verify")
        graph.add_conditional_edges("verify", self.route_after_verify, {"act": "act", "review": "review"})
        graph.add_edge("review", "summarize")
        graph.add_edge("summarize", END)
        return graph.compile()

    # ------------------------------------------------------------------ helpers

    def _tools(self, state: AgentState) -> RepoTools:
        return RepoTools(state["repo_path"], timeout=self.config.command_timeout)

    def _max_steps(self, state: AgentState) -> int:
        return int(state.get("max_steps", self.config.max_steps))

    def _call_llm(
        self,
        state: AgentState,
        node: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[LLMResult, dict[str, Any]]:
        """Every model call goes through here, so tokens and latency are always recorded."""
        result = self.llm.chat(messages, tools=tools)
        usage = dict(state.get("usage") or new_usage())
        usage["llm_calls"] += 1
        usage["prompt_tokens"] += result.prompt_tokens
        usage["completion_tokens"] += result.completion_tokens
        usage["reasoning_tokens"] += result.reasoning_tokens
        usage["llm_seconds"] = round(usage["llm_seconds"] + result.latency_s, 3)
        append_trace(
            state["trace_path"],
            "llm_call",
            {
                "node": node,
                "step": int(state.get("steps", 0)),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "latency_s": round(result.latency_s, 3),
                "finish_reason": result.finish_reason,
                "tool_calls": [call.name for call in result.tool_calls],
            },
        )
        return result, usage

    # ------------------------------------------------------------------ nodes

    def index_repo(self, state: AgentState) -> AgentState:
        repo_path = Path(state["repo_path"]).resolve()
        trace_dir = self.config.trace_dir
        if not trace_dir.is_absolute():
            trace_dir = self.project_root / trace_dir
        trace_path = create_trace_file(trace_dir)

        snapshot = RepoIndexer(repo_path).snapshot(query=state.get("task", ""))
        append_trace(
            trace_path,
            "run_config",
            {
                "model": self.config.model,
                "tool_mode": self.config.tool_mode,
                "verify_mode": self.config.verify_mode,
                "max_steps": self._max_steps(state),
                "extra_body": self.config.extra_body,
            },
        )
        append_trace(trace_path, "repo_indexed", {"repo_path": str(repo_path), "tree": snapshot.tree, "task": state.get("task", "")})
        return {
            **state,
            "repo_context": snapshot.as_context(),
            "project_rules": snapshot.project_rules,
            "tool_history": state.get("tool_history", []),
            "messages": [],
            "steps": state.get("steps", 0),
            "max_steps": self._max_steps(state),
            "trace_path": str(trace_path),
            "usage": new_usage(),
            "counters": new_counters(),
            "no_tool_streak": 0,
            "verify_retries": 0,
            "tests_passed": None,
            "stop_reason": None,
        }

    def plan(self, state: AgentState) -> AgentState:
        prompt = f"""
Task:
{state['task']}

Project rules:
{state.get('project_rules', '')}

Repository context:
{state.get('repo_context', '')}

Create a short plan. Mention which files should be inspected before editing.
""".strip()
        result, usage = self._call_llm(
            state, "plan", [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": prompt}]
        )
        append_trace(state["trace_path"], "plan", {"plan": result.content})
        return {**state, "plan": result.content, "usage": usage}

    def act(self, state: AgentState) -> AgentState:
        if self.config.tool_mode == "json":
            return self._act_json(state)
        return self._act_native(state)

    def _act_json(self, state: AgentState) -> AgentState:
        """Original protocol: rebuild one prompt per step and parse the tool call out of the reply text."""
        tools = self._tools(state)
        history = list(state.get("tool_history", []))
        counters = dict(state.get("counters") or new_counters())
        prompt = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Project rules:
{state.get('project_rules', '')}

Repository context:
{state.get('repo_context', '')[:10000]}

Test command:
{state.get('test_command') or 'pytest -q'}

{tools.legacy_descriptions()}

Recent tool history:
{_history_text(history)}

Choose the next single tool call. If complete, call finish.
""".strip()
        result, usage = self._call_llm(
            state, "act", [{"role": "system", "content": ACTOR_SYSTEM_JSON}, {"role": "user", "content": prompt}]
        )
        raw = result.content
        parse_failed = False
        try:
            action = extract_json_object(raw)
            tool_name = str(action.get("tool", ""))
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
        except Exception as exc:  # noqa: BLE001 - model output is untrusted
            parse_failed = True
            counters["format_errors"] += 1
            tool_name = "finish"
            arguments = {"summary": f"Stopped because model did not return valid JSON: {exc}. Raw output: {raw[:500]}"}
            action = {"tool": tool_name, "arguments": arguments, "reason": "invalid_json"}

        if tool_name == "run_tests" and not arguments.get("command"):
            arguments["command"] = state.get("test_command") or "pytest -q"

        tool_result = tools.run(tool_name, arguments)
        _count_tool(counters, tool_name, tool_result)
        record = _record(tool_name, arguments, tool_result)
        steps = int(state.get("steps", 0)) + 1
        finished = tool_name == "finish" and tool_result.ok
        if parse_failed:
            stop_reason: str | None = "parse_error"
        elif finished:
            stop_reason = "finish"
        elif steps >= self._max_steps(state):
            stop_reason = "max_steps"
        else:
            stop_reason = None
        append_trace(state["trace_path"], "tool_call", {"action": action, "result": record, "step": steps})
        return {
            **state,
            "last_action": action,
            "tool_history": [*history, record],
            "steps": steps,
            "done": stop_reason is not None,
            "stop_reason": stop_reason,
            "usage": usage,
            "counters": counters,
        }

    def _initial_messages(self, state: AgentState) -> list[dict[str, Any]]:
        user = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Project rules:
{state.get('project_rules', '')}

Repository context:
{state.get('repo_context', '')[:10000]}

Test command:
{state.get('test_command') or 'pytest -q'}

Work step by step with the provided tools. Call finish when the task is complete.
""".strip()
        return [{"role": "system", "content": ACTOR_SYSTEM_NATIVE}, {"role": "user", "content": user}]

    def _act_native(self, state: AgentState) -> AgentState:
        """Native function calling: structured tool calls, results returned as tool messages."""
        tools = self._tools(state)
        history = list(state.get("tool_history", []))
        counters = dict(state.get("counters") or new_counters())
        messages = list(state.get("messages") or []) or self._initial_messages(state)
        messages, compacted = compact_messages(messages, self.config.context_char_budget)
        counters["compactions"] += compacted

        result, usage = self._call_llm(state, "act", messages, tools=tools.openai_tools())
        steps = int(state.get("steps", 0)) + 1
        messages.append(result.assistant_message())

        finished = False
        no_tool_streak = int(state.get("no_tool_streak", 0))
        last_action: dict[str, Any] = {"tool": None, "arguments": {}, "reason": result.content[:500]}
        if result.tool_calls:
            no_tool_streak = 0
            for call in result.tool_calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as exc:  # json.JSONDecodeError is a ValueError
                    counters["format_errors"] += 1
                    tool_result = ToolResult(
                        False, f"Invalid arguments for {call.name}: {exc}. Send a JSON object that matches the tool schema."
                    )
                    arguments = {"_raw": call.arguments[:500]}
                else:
                    if call.name == "run_tests" and not arguments.get("command"):
                        arguments["command"] = state.get("test_command") or "pytest -q"
                    tool_result = tools.run(call.name, arguments)
                    if call.name == "finish" and tool_result.ok:
                        finished = True
                _count_tool(counters, call.name, tool_result)
                record = _record(call.name, arguments, tool_result)
                history.append(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": truncate_middle(tool_result.output, self.config.tool_output_chars),
                    }
                )
                last_action = {"tool": call.name, "arguments": arguments, "reason": result.content[:500]}
                append_trace(state["trace_path"], "tool_call", {"action": last_action, "result": record, "step": steps})
        else:
            # The model answered in plain text. Nudge it back to the tools instead of ending the run.
            no_tool_streak += 1
            counters["no_tool_replies"] += 1
            append_trace(state["trace_path"], "no_tool_call", {"step": steps, "content": result.content[:1000]})
            if no_tool_streak < MAX_NO_TOOL_STREAK:
                messages.append({"role": "user", "content": NO_TOOL_NUDGE})

        if finished:
            stop_reason: str | None = "finish"
        elif no_tool_streak >= MAX_NO_TOOL_STREAK:
            stop_reason = "no_tool_call"
        elif steps >= self._max_steps(state):
            stop_reason = "max_steps"
        else:
            stop_reason = None
        return {
            **state,
            "messages": messages,
            "tool_history": history,
            "steps": steps,
            "done": stop_reason is not None,
            "stop_reason": stop_reason,
            "last_action": last_action,
            "no_tool_streak": no_tool_streak,
            "usage": usage,
            "counters": counters,
        }

    def verify(self, state: AgentState) -> AgentState:
        if self.config.verify_mode == "tests":
            return self._verify_with_tests(state)

        # Original rule-based verify: decides whether to stop; the note is only written to the trace.
        history = state.get("tool_history", [])
        last = history[-1] if history else {}
        done = bool(state.get("done"))
        stop_reason = state.get("stop_reason")
        note = "continue"
        if stop_reason == "finish":
            note = "model_finished"
        elif stop_reason:
            note = stop_reason
        elif last.get("blocked"):
            note = "blocked_tool_call_continue_with_safer_action"
        elif last.get("name") == "run_tests" and last.get("ok"):
            note = "tests_passed_agent_may_finish_after_diff_summary"
        elif last.get("name") == "write_file" and last.get("ok"):
            note = "file_written_tests_recommended"
        append_trace(state["trace_path"], "verify", {"mode": "rule", "done": done, "note": note, "last_tool": last.get("name")})
        return {**state, "done": done}

    def _verify_with_tests(self, state: AgentState) -> AgentState:
        """When the run wants to stop, run the tests ourselves; a failed finish is sent back with the output."""
        done = bool(state.get("done"))
        command = state.get("test_command")
        if not done or not command:
            note = "continue" if not done else "no_test_command"
            append_trace(state["trace_path"], "verify", {"mode": "tests", "done": done, "note": note})
            return {**state, "done": done}

        result = self._tools(state).run("run_tests", {"command": command})
        retries = int(state.get("verify_retries", 0))
        finished = state.get("stop_reason") == "finish"
        can_retry = (
            finished
            and not result.ok
            and retries < self.config.max_verify_retries
            and int(state.get("steps", 0)) < self._max_steps(state)
        )
        append_trace(
            state["trace_path"],
            "verify",
            {
                "mode": "tests",
                "trigger": state.get("stop_reason"),
                "tests_passed": result.ok,
                "send_back": can_retry,
                "retries": retries,
                "output": truncate_middle(result.output, 1500),
            },
        )
        if not can_retry:
            return {**state, "done": True, "tests_passed": result.ok}

        feedback = VERIFY_FEEDBACK.format(command=command, output=truncate_middle(result.output, self.config.tool_output_chars))
        messages = list(state.get("messages") or [])
        if messages:
            messages.append({"role": "user", "content": feedback})
        record: ToolRecord = {
            "name": "auto_verify",
            "arguments": {"command": command},
            "ok": False,
            "output": truncate_middle(result.output, 1800),
            "blocked": False,
            "reason": "tests failed after finish",
        }
        return {
            **state,
            "done": False,
            "stop_reason": None,
            "tests_passed": False,
            "verify_retries": retries + 1,
            "messages": messages,
            "tool_history": [*state.get("tool_history", []), record],
        }

    def route_after_verify(self, state: AgentState) -> Literal["act", "review"]:
        return "review" if state.get("done") else "act"

    def review(self, state: AgentState) -> AgentState:
        prompt = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Tool history:
{_history_text(state.get('tool_history', []), limit=14000)}

Review whether the task is complete and safe to hand off.
""".strip()
        result, usage = self._call_llm(
            state, "review", [{"role": "system", "content": REVIEWER_SYSTEM}, {"role": "user", "content": prompt}]
        )
        append_trace(state["trace_path"], "review", {"review": result.content})
        return {**state, "review": result.content, "usage": usage}

    def summarize(self, state: AgentState) -> AgentState:
        prompt = f"""
Task:
{state['task']}

Plan:
{state.get('plan', '')}

Tool history:
{_history_text(state.get('tool_history', []), limit=14000)}

Review subagent findings:
{state.get('review', 'No review generated.')}

Trace file:
{state.get('trace_path')}

Summarize the run for the user.
""".strip()
        result, usage = self._call_llm(
            state, "summarize", [{"role": "system", "content": SUMMARIZER_SYSTEM}, {"role": "user", "content": prompt}]
        )
        append_trace(state["trace_path"], "final_summary", {"summary": result.content})
        final: AgentState = {**state, "final_answer": result.content, "done": True, "usage": usage}
        append_trace(
            state["trace_path"],
            "run_summary",
            {
                "steps": final.get("steps"),
                "stop_reason": final.get("stop_reason"),
                "tests_passed": final.get("tests_passed"),
                "verify_retries": final.get("verify_retries"),
                "usage": usage,
                "counters": final.get("counters"),
            },
        )
        return final


def run_agent(
    repo_path: str | Path,
    task: str,
    test_command: str | None,
    config: AgentConfig,
    project_root: Path,
    llm: ChatModel | None = None,
) -> AgentState:
    app = CodingAgentGraph(config=config, project_root=project_root, llm=llm).build()
    initial_state: AgentState = {
        "repo_path": str(Path(repo_path).resolve()),
        "task": task,
        "test_command": test_command,
        "tool_history": [],
        "steps": 0,
        "max_steps": config.max_steps,
    }
    # Each act/verify round is two graph steps; leave room for verification retries.
    return app.invoke(initial_state, config={"recursion_limit": 4 * config.max_steps + 20})
