"""导出 NeuralSoup 可读取的标准 run 目录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_directory(
    output_dir: str | Path,
    *,
    run_id: str,
    summary: dict[str, Any],
    topology: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> Path:
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "topology.json", topology)
    _write_json(run_dir / "manifest.json", manifest)
    _write_jsonl(run_dir / "events.jsonl", events)

    for relative_path, payload in artifacts.items():
        artifact_path = run_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.endswith(".jsonl"):
            _write_jsonl(artifact_path, payload)
        else:
            _write_json(artifact_path, payload)

    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")

