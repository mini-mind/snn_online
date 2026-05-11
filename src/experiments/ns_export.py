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


def topology_to_ir(run_id: str, topology: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(topology.get("metadata", {}))
    return {
        "schema_version": 1,
        "id": f"{run_id}-topology",
        "label": metadata.get("label", f"{run_id} topology"),
        "label_zh": metadata.get("label_zh", metadata.get("label", f"{run_id} 拓扑")),
        "node_sets": [_copy_node_set(node_set) for node_set in topology.get("node_sets", [])],
        "edge_sets": [_copy_edge_set(edge_set) for edge_set in topology.get("edge_sets", [])],
        "ports": [_copy_port(port) for port in topology.get("ports", [])],
        "annotations": list(topology.get("annotations", [])),
        "metadata": metadata,
    }


def append_topology_artifacts(
    *,
    run_id: str,
    topology: dict[str, Any],
    manifest_entries: list[dict[str, Any]],
    artifact_payloads: dict[str, Any],
) -> None:
    for spec in topology.get("artifact_specs", []):
        artifact_id = str(spec["artifact_id"])
        relative_path = str(spec["path"])
        manifest_entries.append(
            {
                "id": artifact_id,
                "kind": str(spec["kind"]),
                "uri": f"runs/{run_id}/{relative_path}",
                "media_type": str(spec.get("media_type", "application/json")),
                "summary": dict(spec.get("summary", {})),
            }
        )
        artifact_payloads[relative_path] = spec["payload"]


def _copy_node_set(node_set: dict[str, Any]) -> dict[str, Any]:
    payload = dict(node_set)
    raw_parameters = payload.get("parameters")
    if isinstance(raw_parameters, dict):
        payload["parameters"] = dict(raw_parameters)
    payload["count"] = int(payload.pop("size", payload.get("count", 0)))
    payload["kind"] = str(payload.pop("node_type", payload.get("kind", "generic")))
    return payload


def _copy_edge_set(edge_set: dict[str, Any]) -> dict[str, Any]:
    payload = dict(edge_set)
    source = dict(payload.get("source", {}))
    target = dict(payload.get("target", {}))
    representation = payload.pop("representation", {})
    payload["source"] = {"node_set": source.get("node_set")}
    payload["target"] = {"node_set": target.get("node_set")}
    if isinstance(representation, dict):
        edges = list(representation.get("edges", []))
        payload["connectivity"] = {
            "type": str(representation.get("kind", "explicit")),
            "edge_count": len(edges),
            "edges_ref": {
                "id": f"{payload['id']}-edges",
                "artifact_id": f"{payload['id']}-edges",
                "kind": "weight_summary",
            },
        }
    elif "connectivity" not in payload:
        payload["connectivity"] = {"type": "unknown"}
    return payload


def _copy_port(port: dict[str, Any]) -> dict[str, Any]:
    payload = dict(port)
    raw_params = payload.get("params")
    if isinstance(raw_params, dict):
        payload["params"] = dict(raw_params)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")
