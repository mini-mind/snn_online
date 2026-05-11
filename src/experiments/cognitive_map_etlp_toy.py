"""CLI entry for the cognitive-map ETLP-like toy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from envs.grid_world import ACTIONS as GRID_ACTIONS
from experiments.ns_export import append_topology_artifacts, topology_to_ir, write_run_directory
from models.toy_learning import CognitiveMapConfig, train_cognitive_map, train_cognitive_map_with_summary


def _variable_spec(name: str, label: str, label_zh: str, *, role: str) -> dict[str, str]:
    return {
        "name": name,
        "label": label,
        "label_zh": label_zh,
        "role": role,
    }


def _grid_world_episode_summary_variables() -> list[dict[str, str]]:
    return [
        _variable_spec("episode", "episode", "回合编号", role="episode_summary"),
        _variable_spec("reward", "reward", "窗口回报", role="episode_summary"),
        _variable_spec("success", "success", "是否成功", role="episode_summary"),
        _variable_spec("steps", "steps", "步数", role="episode_summary"),
        _variable_spec("prediction_mse", "prediction mse", "预测均方误差", role="episode_summary"),
        _variable_spec("transition_acc", "transition accuracy", "转移一致率", role="episode_summary"),
        _variable_spec("path_efficiency", "path efficiency", "路径效率", role="episode_summary"),
    ]


def _grid_world_trajectory_variables() -> list[dict[str, str]]:
    return [
        _variable_spec("t", "time step", "时间步", role="trajectory"),
        _variable_spec("x", "next x", "下一位置横坐标", role="trajectory"),
        _variable_spec("y", "next y", "下一位置纵坐标", role="trajectory"),
        _variable_spec("state_x", "state x", "当前位置横坐标", role="trajectory"),
        _variable_spec("state_y", "state y", "当前位置纵坐标", role="trajectory"),
        _variable_spec("action", "action", "动作", role="trajectory"),
        _variable_spec("reward", "reward", "即时回报", role="trajectory"),
    ]


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
    grid_layout = dict(summary["grid_layout"])
    trajectory_variables = _grid_world_trajectory_variables()
    episode_summary_variables = _grid_world_episode_summary_variables()

    manifest_entries = []
    artifact_payloads: dict[str, object] = {}
    map_artifact_id = "grid-world-map"
    map_uri = f"maps/{map_artifact_id}.json"
    manifest_entries.append(
        {
            "id": map_artifact_id,
            "kind": "grid_world_map",
            "uri": f"runs/{run_id}/{map_uri}",
            "media_type": "application/json",
            "summary": {
                "rows": int(grid_layout["rows"]),
                "columns": int(grid_layout["columns"]),
                "obstacle_count": len(grid_layout.get("obstacles", [])),
            },
            "label": "Grid World Map",
            "label_zh": "网格世界地图",
        }
    )
    artifact_payloads[map_uri] = {
        "schema_version": 1,
        "label": "Grid World Map",
        "label_zh": "网格世界地图",
        "variables": [
            _variable_spec("rows", "rows", "行数", role="map"),
            _variable_spec("columns", "columns", "列数", role="map"),
            _variable_spec("obstacles", "obstacles", "障碍列表", role="map"),
            _variable_spec("walkable_states", "walkable states", "可行走状态", role="map"),
        ],
        "metadata": {"action_names": list(GRID_ACTIONS)},
        "layout": grid_layout,
    }

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
                "label": f"Episode {episode} Summary",
                "label_zh": f"第 {episode} 回合摘要",
                "variable_names": [spec["name"] for spec in episode_summary_variables],
                "variables": episode_summary_variables,
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
                "label": f"Episode {episode} Trajectory",
                "label_zh": f"第 {episode} 回合轨迹",
                "variable_names": [spec["name"] for spec in trajectory_variables],
                "variables": trajectory_variables,
            }
        )
        artifact_payloads[summary_uri] = {
            "schema_version": 1,
            "label": f"Episode {episode} Summary",
            "label_zh": f"第 {episode} 回合摘要",
            "episode": episode,
            "reward": float(episode_row["reward"]),
            "success": bool(episode_row["success"]),
            "steps": int(episode_row["steps"]),
            "prediction_mse": float(episode_row["prediction_mse"]),
            "transition_acc": float(episode_row["transition_acc"]),
            "path_efficiency": float(episode_row["path_efficiency"]),
            "variable_names": [spec["name"] for spec in episode_summary_variables],
            "variables": episode_summary_variables,
        }
        artifact_payloads[trajectory_uri] = {
            "schema_version": 1,
            "label": f"Episode {episode} Trajectory",
            "label_zh": f"第 {episode} 回合轨迹",
            "variable_names": [spec["name"] for spec in trajectory_variables],
            "variables": trajectory_variables,
            "samples": trajectory_artifacts[trajectory_artifact_id],
        }

    append_topology_artifacts(
        run_id=run_id,
        topology=topology,
        manifest_entries=manifest_entries,
        artifact_payloads=artifact_payloads,
    )

    run_summary = {
        "schema_version": 1,
        "run_id": run_id,
        "name": f"Cognitive Map Grid World / seed {config.seed}",
        "label_zh": f"认知地图网格世界 / seed {config.seed}",
        "status": "completed",
        "started_at": "2026-05-03T00:00:00Z",
        "finished_at": "2026-05-03T00:00:00Z",
        "task": "cognitive_map_grid_world",
        "seed": config.seed,
        "metadata": {
            "action_names": list(GRID_ACTIONS),
            "grid_size": config.grid_size,
            "feature_dim": config.feature_dim,
            "subgraph_tree_id": "grid-world-subgraph-tree",
        },
        "headline_metrics": {
            "prediction_mse": float(summary["prediction_mse"]),
            "success_rate": float(summary["planning_success"]),
            "transition_acc": float(summary["transition_acc"]),
            "path_efficiency": float(summary["path_efficiency"]),
        },
    }
    topology_ir = topology_to_ir(run_id, topology)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "metadata": {
            "label": "Cognitive Map Grid World Manifest",
            "label_zh": "认知地图网格世界导出清单",
            "topology_id": topology_ir["id"],
        },
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


if __name__ == "__main__":
    main()
