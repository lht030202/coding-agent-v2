# Project Agent Rules

This file plays a role similar to `CLAUDE.md`: it gives the coding agent persistent project-level instructions.

## Coding Rules

- Prefer minimal, focused changes.
- Do not modify files outside the target repository.
- Do not edit `.env`, credentials, private keys, lock files, or generated caches unless explicitly requested.
- After editing code, run the configured test command when available.
- Summarize the final change with touched files, test result, and remaining risks.

## Safety Rules

- Never run destructive shell commands such as `rm -rf`, `sudo`, disk formatting, process killing, or network pipe execution.
- Treat tool calls as privileged operations; use read-only tools before write tools.
- If unsure which file to edit, inspect more context before writing.

## Agent Behavior

- Plan first, then act.
- Prefer small patches over large rewrites.
- If tests fail, inspect the error before attempting another edit.
- Keep a JSONL trace of decisions and tool calls for later SFT data construction.
