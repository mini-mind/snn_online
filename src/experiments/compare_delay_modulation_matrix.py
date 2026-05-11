"""Compare delay mechanisms under scalar and per-neuron modulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from envs.point_robot import PointRobotConfig
from models.common import mean
from models.point_robot_closed_loop import AgentConfig, biological_parameter_metadata, train_agent


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def condition_name(
    modulation_mode: str,
    delay_features: bool,
    recurrent_delay_line: bool,
) -> str:
    delay = "delay" if delay_features else "plain"
    line = "rline" if recurrent_delay_line else "no_rline"
    return f"{modulation_mode}_{delay}_{line}"


def condition_grid(include_recurrent_delay_line: bool) -> list[tuple[str, bool, bool]]:
    conditions = []
    for modulation_mode in ("scalar", "per_neuron"):
        for recurrent_delay_line in (False, True) if include_recurrent_delay_line else (False,):
            for delay_features in (False, True):
                conditions.append((modulation_mode, delay_features, recurrent_delay_line))
    return conditions


def run_matrix(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
    include_recurrent_delay_line: bool,
    output_jsonl: Path | None = None,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, list[dict[str, object]]] = {}
    task_metadata = env_config.task_metadata()
    for modulation_mode, delay_features, recurrent_delay_line in condition_grid(include_recurrent_delay_line):
        name = condition_name(modulation_mode, delay_features, recurrent_delay_line)
        print(f"condition={name}")
        summaries[name] = []
        for seed in seeds:
            agent_config = replace(
                base_config,
                modulation_mode=modulation_mode,
                delay_features=delay_features,
                recurrent_delay_line=recurrent_delay_line,
                seed=seed,
            )
            run_env_config = replace(env_config, seed=seed + 7)
            summary = train_agent(agent_config, run_env_config, verbose=False)
            summaries[name].append(summary)
            print(
                f"  seed={seed} "
                f"eval_reward={summary['final_eval_reward']:.3f} "
                f"eval_success={summary['final_eval_success']:.3f} "
                f"elapsed_sec={summary['elapsed_sec']:.3f}"
            )
            if output_jsonl is not None:
                append_jsonl(output_jsonl, run_row(name, agent_config, run_env_config, summary))
        print()

    aggregated = {
        name: {
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }
        for name, rows in summaries.items()
    }
    if output_jsonl is not None:
        append_jsonl(
            output_jsonl,
            {
                "type": "summary",
                "config": config_to_dict(base_config, env_config, seeds, include_recurrent_delay_line),
                "task_metadata": task_metadata,
                "conditions": aggregated,
                "best_by_reward": best_condition(aggregated, "mean_eval_reward"),
                "best_by_success": best_condition(aggregated, "mean_eval_success"),
            },
        )
    return aggregated


def run_row(
    condition: str,
    agent_config: AgentConfig,
    env_config: PointRobotConfig,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "run",
        "condition": condition,
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
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "delay_features": agent_config.delay_features,
        "recurrent_delay_line": agent_config.recurrent_delay_line,
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


def best_condition(aggregated: dict[str, dict[str, float]], metric: str) -> dict[str, object]:
    if not aggregated:
        return {}
    name = max(aggregated, key=lambda key: aggregated[key][metric])
    return {"condition": name, metric: aggregated[name][metric]}


def config_to_dict(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
    include_recurrent_delay_line: bool,
) -> dict[str, object]:
    config = {
        "episodes": base_config.episodes,
        "eval_every": base_config.eval_every,
        "eval_episodes": base_config.eval_episodes,
        "n_neurons": base_config.n_neurons,
        "recurrent_degree": base_config.recurrent_degree,
        "plasticity_rule": base_config.plasticity_rule,
        "neuron_model": base_config.neuron_model,
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "include_recurrent_delay_line": include_recurrent_delay_line,
        "seeds": seeds,
        "biological_params": biological_parameter_metadata(base_config),
    }
    config.update(env_config.task_metadata())
    return config


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int], bool, Path | None]:
    parser = argparse.ArgumentParser(
        description="Compare delay and recurrent-delay mechanisms under scalar/per-neuron modulation."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=64)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--plasticity-rule", choices=["three_factor", "tess_like"], default="tess_like")
    parser.add_argument("--neuron-model", choices=["lif", "izh"], default=AgentConfig.neuron_model)
    parser.add_argument(
        "--observation-mode",
        choices=["full", "partial_goal_cue"],
        default="partial_goal_cue",
    )
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--include-recurrent-delay-line", action="store_true")
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
            plasticity_rule=args.plasticity_rule,
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
        args.include_recurrent_delay_line,
        Path(args.output_jsonl) if args.output_jsonl else None,
    )


def main() -> None:
    agent_config, env_config, seeds, include_recurrent_delay_line, output_jsonl = parse_args()
    print(
        f"task observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"benchmark_id={env_config.task_metadata()['benchmark_id']} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"plasticity_rule={agent_config.plasticity_rule} "
        f"neuron_model={agent_config.neuron_model} "
        f"include_recurrent_delay_line={include_recurrent_delay_line}"
    )
    if output_jsonl is not None:
        prepare_jsonl(output_jsonl)
        print(f"output_jsonl={output_jsonl}")
    results = run_matrix(
        agent_config,
        env_config,
        seeds,
        include_recurrent_delay_line,
        output_jsonl=output_jsonl,
    )
    print("summary")
    for name in sorted(results):
        metrics = results[name]
        print(
            f"  {name} mean_eval_reward={metrics['mean_eval_reward']:.3f} "
            f"mean_eval_success={metrics['mean_eval_success']:.3f} "
            f"mean_elapsed_sec={metrics['mean_elapsed_sec']:.3f}"
        )
    print(
        f"  best_by_reward={best_condition(results, 'mean_eval_reward')['condition']} "
        f"best_by_success={best_condition(results, 'mean_eval_success')['condition']}"
    )


if __name__ == "__main__":
    main()
