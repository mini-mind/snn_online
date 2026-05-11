"""Compare mainline historical stages on point-robot control."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from envs.point_robot import PointRobotConfig
from models.common import mean
from models.point_robot_closed_loop import AgentConfig, train_agent


STAGES = {
    "h1_three_factor_recurrent": {
        "plasticity_rule": "three_factor",
        "modulation_mode": "scalar",
        "recurrent_delay_line": False,
    },
    "h2_tess_recurrent": {
        "plasticity_rule": "tess_like",
        "modulation_mode": "scalar",
        "recurrent_delay_line": False,
    },
    "h3_tess_recurrent_delay": {
        "plasticity_rule": "tess_like",
        "modulation_mode": "scalar",
        "recurrent_delay_line": True,
    },
    "h4_eprop_like_v0": {
        "plasticity_rule": "tess_like",
        "modulation_mode": "per_neuron",
        "recurrent_delay_line": True,
    },
}


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def run_history(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
    output_jsonl: Path | None = None,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, list[dict[str, object]]] = {}
    for stage, settings in STAGES.items():
        print(f"stage={stage}")
        summaries[stage] = []
        for seed in seeds:
            agent_config = replace(base_config, seed=seed, **settings)
            run_env_config = replace(env_config, seed=seed + 7)
            summary = train_agent(agent_config, run_env_config, verbose=False)
            summaries[stage].append(summary)
            print(
                f"  seed={seed} "
                f"eval_reward={summary['final_eval_reward']:.3f} "
                f"eval_success={summary['final_eval_success']:.3f} "
                f"elapsed_sec={summary['elapsed_sec']:.3f}"
            )
            if output_jsonl is not None:
                append_jsonl(output_jsonl, run_row(stage, agent_config, run_env_config, summary))
        print()

    aggregated = {
        stage: {
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }
        for stage, rows in summaries.items()
    }
    if output_jsonl is not None:
        append_jsonl(
            output_jsonl,
            {
                "type": "summary",
                "config": config_to_dict(base_config, env_config, seeds),
                "stages": aggregated,
                "best_by_reward": best_stage(aggregated, "mean_eval_reward"),
                "best_by_success": best_stage(aggregated, "mean_eval_success"),
            },
        )
    return aggregated


def run_row(
    stage: str,
    agent_config: AgentConfig,
    env_config: PointRobotConfig,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "run",
        "stage": stage,
        "seed": agent_config.seed,
        "eval_reward": summary["final_eval_reward"],
        "eval_success": summary["final_eval_success"],
        "elapsed_sec": summary["elapsed_sec"],
        "episodes": agent_config.episodes,
        "eval_every": agent_config.eval_every,
        "eval_episodes": agent_config.eval_episodes,
        "n_neurons": agent_config.n_neurons,
        "recurrent_degree": agent_config.recurrent_degree,
        "plasticity_rule": agent_config.plasticity_rule,
        "neuron_model": agent_config.neuron_model,
        "modulation_mode": agent_config.modulation_mode,
        "recurrent_delay_line": agent_config.recurrent_delay_line,
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "task_family": summary["task_family"],
        "benchmark_id": summary["benchmark_id"],
        "observability_level": summary["observability_level"],
        "horizon_level": summary["horizon_level"],
        "reward_level": summary["reward_level"],
        "dynamics_level": summary["dynamics_level"],
        "goal_structure_level": summary["goal_structure_level"],
        "action_level": summary["action_level"],
        "distribution_level": summary["distribution_level"],
        "biological_params": summary["biological_params"],
    }


def best_stage(aggregated: dict[str, dict[str, float]], metric: str) -> dict[str, object]:
    if not aggregated:
        return {}
    stage = max(aggregated, key=lambda key: aggregated[key][metric])
    return {"stage": stage, metric: aggregated[stage][metric]}


def config_to_dict(base_config: AgentConfig, env_config: PointRobotConfig, seeds: list[int]) -> dict[str, object]:
    config = {
        "episodes": base_config.episodes,
        "eval_every": base_config.eval_every,
        "eval_episodes": base_config.eval_episodes,
        "n_neurons": base_config.n_neurons,
        "recurrent_degree": base_config.recurrent_degree,
        "neuron_model": base_config.neuron_model,
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "stages": list(STAGES),
        "seeds": seeds,
    }
    config.update(env_config.task_metadata())
    return config


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int], Path | None]:
    parser = argparse.ArgumentParser(
        description="Compare historical mainline stages on point robot control."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=64)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--neuron-model", choices=["lif", "izh"], default=AgentConfig.neuron_model)
    parser.add_argument(
        "--observation-mode",
        choices=["full", "partial_goal_cue"],
        default="partial_goal_cue",
    )
    parser.add_argument("--goal-cue-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=31)
    parser.add_argument("--output-jsonl", default="")
    args = parser.parse_args()
    return (
        AgentConfig(
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            n_neurons=args.n_neurons,
            recurrent_degree=args.recurrent_degree,
            neuron_model=args.neuron_model,
            randomize_intrinsics=True,
            seed=args.seed_start,
        ),
        PointRobotConfig(
            max_steps=args.max_steps,
            observation_mode=args.observation_mode,
            goal_cue_steps=args.goal_cue_steps,
            seed=args.seed_start + 7,
        ),
        [args.seed_start + offset for offset in range(args.seeds)],
        Path(args.output_jsonl) if args.output_jsonl else None,
    )


def main() -> None:
    agent_config, env_config, seeds, output_jsonl = parse_args()
    print(
        f"task observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"benchmark_id={env_config.task_metadata()['benchmark_id']} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"neuron_model={agent_config.neuron_model}"
    )
    if output_jsonl is not None:
        prepare_jsonl(output_jsonl)
        print(f"output_jsonl={output_jsonl}")
    results = run_history(agent_config, env_config, seeds, output_jsonl=output_jsonl)
    print("summary")
    for stage in sorted(results):
        metrics = results[stage]
        print(
            f"  {stage} mean_eval_reward={metrics['mean_eval_reward']:.3f} "
            f"mean_eval_success={metrics['mean_eval_success']:.3f} "
            f"mean_elapsed_sec={metrics['mean_elapsed_sec']:.3f}"
        )
    print(
        f"  best_by_reward={best_stage(results, 'mean_eval_reward')['stage']} "
        f"best_by_success={best_stage(results, 'mean_eval_success')['stage']}"
    )


if __name__ == "__main__":
    main()
