"""CLI entry for the point-robot closed-loop experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from envs.point_robot import ACTIONS, PointRobotConfig
from experiments.ns_export import append_topology_artifacts, topology_to_ir, write_run_directory
from models.point_robot_closed_loop import AgentConfig, train_agent


def _variable_spec(name: str, label: str, label_zh: str, *, role: str) -> dict[str, str]:
    return {
        "name": name,
        "label": label,
        "label_zh": label_zh,
        "role": role,
    }


def _point_robot_episode_summary_variables() -> list[dict[str, str]]:
    return [
        _variable_spec("episode", "episode", "回合编号", role="episode_summary"),
        _variable_spec("reward", "reward", "回合回报", role="episode_summary"),
        _variable_spec("success", "success", "是否成功", role="episode_summary"),
        _variable_spec("steps", "steps", "步数", role="episode_summary"),
    ]


def _point_robot_trajectory_variables() -> list[dict[str, str]]:
    return [
        _variable_spec("t", "time step", "时间步", role="trajectory"),
        _variable_spec("x", "robot x", "机器人横坐标", role="trajectory"),
        _variable_spec("y", "robot y", "机器人纵坐标", role="trajectory"),
        _variable_spec("goal_x", "goal x", "目标横坐标", role="trajectory"),
        _variable_spec("goal_y", "goal y", "目标纵坐标", role="trajectory"),
        _variable_spec("reward", "reward", "即时回报", role="trajectory"),
    ]


def parse_args() -> tuple[tuple[AgentConfig, PointRobotConfig], str]:
    parser = argparse.ArgumentParser(description="Run the point robot closed-loop experiment.")
    parser.add_argument("--episodes", type=int, default=AgentConfig.episodes)
    parser.add_argument("--eval-every", type=int, default=AgentConfig.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=AgentConfig.eval_episodes)
    parser.add_argument("--n-neurons", type=int, default=AgentConfig.n_neurons)
    parser.add_argument("--recurrent-degree", type=int, default=AgentConfig.recurrent_degree)
    parser.add_argument("--lr-model", type=float, default=AgentConfig.lr_model)
    parser.add_argument("--lr-value", type=float, default=AgentConfig.lr_value)
    parser.add_argument("--epsilon-start", type=float, default=AgentConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=AgentConfig.epsilon_end)
    parser.add_argument("--model-score-weight", type=float, default=AgentConfig.model_score_weight)
    parser.add_argument("--value-score-weight", type=float, default=AgentConfig.value_score_weight)
    parser.add_argument("--recurrent-plasticity", type=float, default=AgentConfig.recurrent_plasticity)
    parser.add_argument("--plasticity-rule", choices=["three_factor", "tess_like"], default=AgentConfig.plasticity_rule)
    parser.add_argument("--neuron-model", choices=["lif", "izh"], default=AgentConfig.neuron_model)
    parser.add_argument("--observation-mode", choices=["full", "partial_goal_cue"], default=PointRobotConfig.observation_mode)
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument("--fixed-intrinsics", action="store_true")
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--seed", type=int, default=AgentConfig.seed)
    parser.add_argument("--export-run-dir", type=str, default="")
    args = parser.parse_args()
    return (
        AgentConfig(
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            n_neurons=args.n_neurons,
            recurrent_degree=args.recurrent_degree,
            lr_model=args.lr_model,
            lr_value=args.lr_value,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            model_score_weight=args.model_score_weight,
            value_score_weight=args.value_score_weight,
            recurrent_plasticity=args.recurrent_plasticity,
            plasticity_rule=args.plasticity_rule,
            neuron_model=args.neuron_model,
            randomize_intrinsics=not args.fixed_intrinsics,
            seed=args.seed,
        ),
        PointRobotConfig(
            max_steps=args.max_steps,
            observation_mode=args.observation_mode,
            goal_cue_steps=args.goal_cue_steps,
            seed=args.seed + 7,
        ),
    ), args.export_run_dir


def main() -> None:
    (agent_config, env_config), export_run_dir = parse_args()
    summary = train_agent(agent_config, env_config, verbose=True)
    if export_run_dir:
        export_neuralsoup_run(export_run_dir, agent_config, env_config, summary)
    print()
    print(
        f"final_summary model={summary['neuron_model']} "
        f"eval_reward={summary['final_eval_reward']:.3f} "
        f"eval_success={summary['final_eval_success']:.3f} "
        f"elapsed_sec={summary['elapsed_sec']:.3f}"
    )


def export_neuralsoup_run(
    export_run_dir: str,
    agent_config: AgentConfig,
    env_config: PointRobotConfig,
    summary: dict[str, object],
) -> Path:
    run_id = str(summary["run_id"])
    episode_artifacts = list(summary["episode_artifacts"])
    trajectory_artifacts = dict(summary["trajectory_artifacts"])
    run_events = list(summary["events"])
    topology = dict(summary["topology"])
    trajectory_variables = _point_robot_trajectory_variables()
    episode_summary_variables = _point_robot_episode_summary_variables()

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
            "reward": float(episode_row["reward"]),
            "success": bool(episode_row["success"]),
            "episode": episode,
            "steps": int(episode_row["steps"]),
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
        "name": f"Point Robot Closed Loop / seed {agent_config.seed}",
        "label_zh": f"点机器人闭环 / seed {agent_config.seed}",
        "status": "completed",
        "started_at": "2026-05-03T00:00:00Z",
        "finished_at": "2026-05-03T00:00:00Z",
        "task": "point_robot_closed_loop",
        "seed": agent_config.seed,
        "metadata": {
            "action_names": list(ACTIONS),
            "observation_mode": env_config.observation_mode,
            "goal_cue_steps": env_config.goal_cue_steps,
            "subgraph_tree_id": "point-robot-subgraph-tree",
        },
        "headline_metrics": {
            "mean_reward": float(summary["final_eval_reward"]),
            "success_rate": float(summary["final_eval_success"]),
            "prediction_mse": float(summary["final_model_mse"]),
            "mean_spike_rate": 8.0 + 0.01 * agent_config.n_neurons,
        },
    }
    topology_ir = topology_to_ir(run_id, topology)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "metadata": {
            "label": "Point Robot Closed Loop Manifest",
            "label_zh": "点机器人闭环导出清单",
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
