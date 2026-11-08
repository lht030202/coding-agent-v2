from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_trace_file(trace_dir: Path) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return trace_dir / f"agent_trace_{timestamp}.jsonl"


def append_trace(trace_path: str | Path, event: str, payload: dict[str, Any]) -> None:
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
