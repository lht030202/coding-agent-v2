from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from cc_agent.config import TOOL_MODES, VERIFY_MODES, AgentConfig
from cc_agent.graph import run_agent
from cc_agent.repo_indexer import RepoIndexer
from cc_agent.trace_stats import compute_trace_stats

app = typer.Typer(help="Coding agent with native function calling, test-driven verification and an evaluation harness.")
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(tool_mode: str | None = None, verify_mode: str | None = None, max_steps: int | None = None) -> AgentConfig:
    config = AgentConfig.from_env(PROJECT_ROOT)
    if tool_mode is not None:
        if tool_mode not in TOOL_MODES:
            raise typer.BadParameter(f"--tool-mode must be one of {TOOL_MODES}")
        config = replace(config, tool_mode=tool_mode)
    if verify_mode is not None:
        if verify_mode not in VERIFY_MODES:
            raise typer.BadParameter(f"--verify-mode must be one of {VERIFY_MODES}")
        config = replace(config, verify_mode=verify_mode)
    if max_steps is not None:
        config = replace(config, max_steps=max_steps)
    return config


@app.command()
def index(
    repo: Path = typer.Option(..., "--repo", "-r", help="Target repository path."),
    query: str | None = typer.Option(None, "--query", "-q", help="Optional task query for lightweight retrieval."),
) -> None:
    """Preview the repository context the agent will see (no LLM call)."""
    snapshot = RepoIndexer(repo).snapshot(query=query)
    console.print(Panel(snapshot.tree, title="Repository Tree"))
    console.print(Panel(snapshot.symbols, title="Symbol Index"))
    console.print(Panel(snapshot.retrieval_notes, title="Retrieved Context"))
    console.print(Panel(snapshot.project_rules, title="Project Rules"))


@app.command()
def retrieve(
    repo: Path = typer.Option(..., "--repo", "-r", help="Target repository path."),
    query: str = typer.Option(..., "--query", "-q", help="Search query."),
    top_k: int = typer.Option(5, "--top-k", help="Number of retrieved files."),
) -> None:
    """Run the dependency-free lexical retrieval over the repository."""
    console.print(Panel(RepoIndexer(repo).retrieve(query=query, top_k=top_k), title="Retrieved Context"))


@app.command()
def run(
    repo: Path = typer.Option(..., "--repo", "-r", help="Target repository path."),
    task: str = typer.Option(..., "--task", "-t", help="Natural language coding task."),
    test_command: str | None = typer.Option(None, "--test-command", help="Allowlisted test command, e.g. pytest -q."),
    tool_mode: str | None = typer.Option(None, "--tool-mode", help="native (function calling) or json (original)."),
    verify_mode: str | None = typer.Option(None, "--verify-mode", help="tests (run tests on finish) or rule (original)."),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Maximum act steps."),
) -> None:
    """Run the coding agent on a repository."""
    config = _config(tool_mode, verify_mode, max_steps)
    final_state = run_agent(repo, task, test_command, config=config, project_root=PROJECT_ROOT)
    console.print(Panel(final_state.get("plan", "No plan generated."), title="Plan"))
    console.print(Panel(final_state.get("review", "No review generated."), title="Review Subagent"))
    console.print(Panel(final_state.get("final_answer", "No final summary."), title="Final Summary"))
    usage = final_state.get("usage", {})
    counters = final_state.get("counters", {})
    console.print(
        f"[bold]Run:[/bold] mode={config.tool_mode}/{config.verify_mode} steps={final_state.get('steps')} "
        f"stop={final_state.get('stop_reason')} tests_passed={final_state.get('tests_passed')} "
        f"verify_retries={final_state.get('verify_retries', 0)}"
    )
    console.print(
        f"[bold]Cost:[/bold] llm_calls={usage.get('llm_calls', 0)} prompt_tokens={usage.get('prompt_tokens', 0)} "
        f"completion_tokens={usage.get('completion_tokens', 0)} llm_time={usage.get('llm_seconds', 0)}s "
        f"format_errors={counters.get('format_errors', 0)}"
    )
    console.print(f"[bold]Trace:[/bold] {final_state.get('trace_path')}")


@app.command()
def stats(
    path: Path = typer.Option(PROJECT_ROOT / "traces", "--path", "-p", help="Trace file or trace directory."),
) -> None:
    """Summarize JSONL traces: tokens, latency, tool usage and test results."""
    console.print(Panel(compute_trace_stats(path).render(), title="Trace Stats"))


@app.command("eval")
def eval_command(
    configs: str = typer.Option("v0,v1,v2", "--configs", help="Comma-separated configurations to compare."),
    runs: int = typer.Option(2, "--runs", help="Runs per task and configuration."),
    workers: int = typer.Option(3, "--workers", help="Parallel runs."),
    tasks: str = typer.Option("", "--tasks", help="Optional comma-separated task ids."),
    manifest: Path = typer.Option(PROJECT_ROOT / "eval" / "tasks.json", "--manifest", help="Task manifest."),
    output: Path = typer.Option(PROJECT_ROOT / "eval" / "results", "--output", help="Results directory."),
) -> None:
    """Run the evaluation harness and write a report."""
    from cc_agent.evaluation import run_evaluation

    config_names = [name.strip() for name in configs.split(",") if name.strip()]
    task_ids = [task_id.strip() for task_id in tasks.split(",") if task_id.strip()] or None
    results_dir, report = run_evaluation(
        manifest=manifest,
        config_names=config_names,
        runs=runs,
        workers=workers,
        task_ids=task_ids,
        output_root=output,
        base_config=AgentConfig.from_env(PROJECT_ROOT),
        project_root=PROJECT_ROOT,
        progress=lambda message: console.print(message, markup=False, highlight=False),
    )
    console.print(f"Report: {results_dir / 'report.md'}", markup=False)
    console.print(report, markup=False, highlight=False)


if __name__ == "__main__":
    app()
