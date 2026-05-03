"""CLI entry for the cognitive-map ETLP-like toy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.ns_export import write_run_directory
from models.toy_learning import CognitiveMapConfig, train_cognitive_map, train_cognitive_map_with_summary


def parse_args() -> tuple[CognitiveMapConfig, str]:
    parser = argparse.ArgumentParser(description="Run the Cognitive Map + ETLP-like toy.")
    parser.add_argument("--grid-size", type=int, default=CognitiveMapConfig.grid_size)
    parser.add_argument("--feature-dim", type=int, default=CognitiveMapConfig.feature_dim)
    parser.add_argument("--train-steps", type=int, default=CognitiveMapConfig.train_steps)
    parser.add_argument("--eval-every", type=int, default=CognitiveMapConfig.eval_every)
    parser.add_argument("--eval-pairs", type=int, default=CognitiveMapConfig.eval_pairs)
    parser.add_argument("--planning-horizon", type=int, default=CognitiveMapConfig.planning_horizon)
    parser.add_argument("--lr", type=float, default=CognitiveMapConfig.lr)
    parser.add_argument("--trace-decay", type=float, default=CognitiveMapConfig.trace_decay)
    parser.add_argument("--noise", type=float, default=CognitiveMapConfig.noise)
    parser.add_argument("--seed", type=int, default=CognitiveMapConfig.seed)
    parser.add_argument("--export-run-dir", type=str, default="")
    args = parser.parse_args()
    return CognitiveMapConfig(
        grid_size=args.grid_size,
        feature_dim=args.feature_dim,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        eval_pairs=args.eval_pairs,
        planning_horizon=args.planning_horizon,
        lr=args.lr,
        trace_decay=args.trace_decay,
        noise=args.noise,
        seed=args.seed,
    ), args.export_run_dir


def main() -> None:
    config, export_run_dir = parse_args()
    if export_run_dir:
        summary = train_cognitive_map_with_summary(config)
        export_neuralsoup_run(export_run_dir, config, summary)
        return
    train_cognitive_map(config)


def export_neuralsoup_run(
    export_run_dir: str,
    config: CognitiveMapConfig,
    summary: dict[str, Any],
) -> Path:
    run_id = str(summary["run_id"])
    episode_artifacts = list(summary["episode_artifacts"])
    trajectory_artifacts = dict(summary["trajectory_artifacts"])
    run_events = list(summary["events"])
    topology = dict(summary["topology"])

    manifest_entries = []
    artifact_payloads: dict[str, object] = {}
    for episode_row in episode_artifacts:
        episode = int(episode_row["episode"])
        summary_artifact_id = str(episode_row["summary_artifact_id"])
        trajectory_artifact_id = str(episode_row["trajectory_artifact_id"])
        summary_uri = f"episodes/{summary_artifact_id}.json"
        trajectory_uri = f"episodes/{trajectory_artifact_id}.json"
        manifest_entries.append(
            {
                "id": summary_artifact_id,
                "kind": "episode_summary",
                "uri": f"runs/{run_id}/{summary_uri}",
                "media_type": "application/json",
                "episode": episode,
                "summary": {
                    "reward": float(episode_row["reward"]),
                    "success": bool(episode_row["success"]),
                    "steps": int(episode_row["steps"]),
                    "prediction_mse": float(episode_row["prediction_mse"]),
                },
            }
        )
        manifest_entries.append(
            {
                "id": trajectory_artifact_id,
                "kind": "trajectory",
                "uri": f"runs/{run_id}/{trajectory_uri}",
                "media_type": "application/json",
                "episode": episode,
                "summary": {
                    "sample_count": len(trajectory_artifacts[trajectory_artifact_id]),
                },
            }
        )
        artifact_payloads[summary_uri] = {
            "episode": episode,
            "reward": float(episode_row["reward"]),
            "success": bool(episode_row["success"]),
            "steps": int(episode_row["steps"]),
            "prediction_mse": float(episode_row["prediction_mse"]),
            "transition_acc": float(episode_row["transition_acc"]),
            "path_efficiency": float(episode_row["path_efficiency"]),
        }
        artifact_payloads[trajectory_uri] = trajectory_artifacts[trajectory_artifact_id]

    run_summary = {
        "schema_version": 1,
        "run_id": run_id,
        "name": f"Cognitive Map Grid World / seed {config.seed}",
        "status": "completed",
        "started_at": "2026-05-03T00:00:00Z",
        "finished_at": "2026-05-03T00:00:00Z",
        "task": "cognitive_map_grid_world",
        "seed": config.seed,
        "headline_metrics": {
            "prediction_mse": float(summary["prediction_mse"]),
            "success_rate": float(summary["planning_success"]),
            "transition_acc": float(summary["transition_acc"]),
            "path_efficiency": float(summary["path_efficiency"]),
        },
    }
    topology_ir = to_topology_ir(run_id, topology)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "artifacts": manifest_entries,
    }
    return write_run_directory(
        export_run_dir,
        run_id=run_id,
        summary=run_summary,
        topology=topology_ir,
        manifest=manifest,
        events=run_events,
        artifacts=artifact_payloads,
    )


def to_topology_ir(run_id: str, topology: dict[str, object]) -> dict[str, object]:
    node_sets = [
        {
            "id": node_set["id"],
            "count": int(node_set["size"]),
            "kind": str(node_set.get("node_type", "generic")),
        }
        for node_set in topology.get("node_sets", [])
    ]
    edge_sets = [
        {
            "id": edge_set["id"],
            "source": {"node_set": edge_set["source"]["node_set"]},
            "target": {"node_set": edge_set["target"]["node_set"]},
            "connectivity": {
                "type": "explicit",
                "edges_ref": {
                    "id": f"{edge_set['id']}-edges",
                    "artifact_id": f"{edge_set['id']}-edges",
                    "kind": "weight_summary",
                },
            },
        }
        for edge_set in topology.get("edge_sets", [])
    ]
    return {
        "schema_version": 1,
        "id": f"{run_id}-topology",
        "node_sets": node_sets,
        "edge_sets": edge_sets,
        "ports": list(topology.get("ports", [])),
    }


if __name__ == "__main__":
    main()
