"""Compare mainline component stages on the hard partial-goal-cue benchmark."""

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
        "description": "three-factor recurrent baseline",
        "added_component": "baseline",
        "agent_overrides": {
            "plasticity_rule": "three_factor",
            "modulation_mode": "scalar",
            "recurrent_delay_line": False,
            "metaplasticity": False,
        },
    },
    "h2_tess_recurrent": {
        "description": "swap in tess-like recurrent plasticity",
        "added_component": "tess_like_rule",
        "agent_overrides": {
            "plasticity_rule": "tess_like",
            "modulation_mode": "scalar",
            "recurrent_delay_line": False,
            "metaplasticity": False,
        },
    },
    "h3_tess_recurrent_delay": {
        "description": "add recurrent delay line on top of tess-like rule",
        "added_component": "recurrent_delay_line",
        "agent_overrides": {
            "plasticity_rule": "tess_like",
            "modulation_mode": "scalar",
            "recurrent_delay_line": True,
            "metaplasticity": False,
        },
    },
    "h4_eprop_like_v0": {
        "description": "switch to per-neuron modulation with recurrent delay line",
        "added_component": "per_neuron_modulation",
        "agent_overrides": {
            "plasticity_rule": "tess_like",
            "modulation_mode": "per_neuron",
            "recurrent_delay_line": True,
            "metaplasticity": False,
        },
    },
    "h5_metaplasticity_v0": {
        "description": "add metaplasticity on top of per-neuron modulation and delay",
        "added_component": "metaplasticity",
        "agent_overrides": {
            "plasticity_rule": "tess_like",
            "modulation_mode": "per_neuron",
            "recurrent_delay_line": True,
            "metaplasticity": True,
        },
    },
}

DEFAULT_STAGE_ORDER = [
    "h1_three_factor_recurrent",
    "h2_tess_recurrent",
    "h3_tess_recurrent_delay",
]


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def run_component_benchmark(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    stage_names: list[str],
    seeds: list[int],
    output_jsonl: Path | None = None,
) -> dict[str, dict[str, object]]:
    per_stage_runs: dict[str, list[dict[str, object]]] = {}
    for stage_name in stage_names:
        stage = STAGES[stage_name]
        per_stage_runs[stage_name] = []
        print(
            f"stage={stage_name} "
            f"change={stage['added_component']} "
            f"plasticity_rule={stage['agent_overrides']['plasticity_rule']} "
            f"modulation_mode={stage['agent_overrides']['modulation_mode']} "
            f"recurrent_delay_line={stage['agent_overrides']['recurrent_delay_line']} "
            f"metaplasticity={stage['agent_overrides']['metaplasticity']}"
        )
        for seed in seeds:
            agent_config = replace(
                base_config,
                seed=seed,
                **stage["agent_overrides"],
            )
            run_env_config = replace(env_config, seed=seed + 7)
            summary = train_agent(agent_config, run_env_config, verbose=False)
            per_stage_runs[stage_name].append(summary)
            print(
                f"  seed={seed} "
                f"eval_reward={summary['final_eval_reward']:.3f} "
                f"eval_success={summary['final_eval_success']:.3f} "
                f"elapsed_sec={summary['elapsed_sec']:.3f}"
            )
            if output_jsonl is not None:
                append_jsonl(
                    output_jsonl,
                    run_row(stage_name, stage, agent_config, run_env_config, summary),
                )
        print()

    aggregated = build_stage_summary(stage_names, per_stage_runs)
    if output_jsonl is not None:
        append_jsonl(
            output_jsonl,
            {
                "type": "summary",
                "config": config_to_dict(base_config, env_config, stage_names, seeds),
                "baseline_stage": stage_names[0],
                "stages": aggregated,
                "best_by_reward": best_stage(aggregated, "mean_eval_reward"),
                "best_by_success": best_stage(aggregated, "mean_eval_success"),
            },
        )
    return aggregated


def build_stage_summary(
    stage_names: list[str],
    per_stage_runs: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    aggregated: dict[str, dict[str, object]] = {}
    baseline_stage = stage_names[0]
    previous_stage_name: str | None = None
    for stage_name in stage_names:
        rows = per_stage_runs[stage_name]
        stage = STAGES[stage_name]
        metrics: dict[str, object] = {
            "description": stage["description"],
            "added_component": stage["added_component"],
            "plasticity_rule": stage["agent_overrides"]["plasticity_rule"],
            "modulation_mode": stage["agent_overrides"]["modulation_mode"],
            "recurrent_delay_line": stage["agent_overrides"]["recurrent_delay_line"],
            "metaplasticity": stage["agent_overrides"]["metaplasticity"],
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }
        baseline_metrics = aggregated.get(baseline_stage, metrics)
        metrics["reward_gain_vs_baseline"] = (
            metrics["mean_eval_reward"] - baseline_metrics["mean_eval_reward"]
        )
        metrics["success_gain_vs_baseline"] = (
            metrics["mean_eval_success"] - baseline_metrics["mean_eval_success"]
        )
        if previous_stage_name is None:
            metrics["previous_stage"] = None
            metrics["reward_gain_vs_previous"] = None
            metrics["success_gain_vs_previous"] = None
        else:
            previous_metrics = aggregated[previous_stage_name]
            metrics["previous_stage"] = previous_stage_name
            metrics["reward_gain_vs_previous"] = (
                metrics["mean_eval_reward"] - previous_metrics["mean_eval_reward"]
            )
            metrics["success_gain_vs_previous"] = (
                metrics["mean_eval_success"] - previous_metrics["mean_eval_success"]
            )
        aggregated[stage_name] = metrics
        previous_stage_name = stage_name
    return aggregated


def run_row(
    stage_name: str,
    stage: dict[str, object],
    agent_config: AgentConfig,
    env_config: PointRobotConfig,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "run",
        "stage": stage_name,
        "description": stage["description"],
        "added_component": stage["added_component"],
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
        "metaplasticity": agent_config.metaplasticity,
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


def best_stage(aggregated: dict[str, dict[str, object]], metric: str) -> dict[str, object]:
    if not aggregated:
        return {}
    stage_name = max(aggregated, key=lambda name: aggregated[name][metric])
    return {"stage": stage_name, metric: aggregated[stage_name][metric]}


def config_to_dict(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    stage_names: list[str],
    seeds: list[int],
) -> dict[str, object]:
    config = {
        "episodes": base_config.episodes,
        "eval_every": base_config.eval_every,
        "eval_episodes": base_config.eval_episodes,
        "n_neurons": base_config.n_neurons,
        "recurrent_degree": base_config.recurrent_degree,
        "neuron_model": base_config.neuron_model,
        "randomize_intrinsics": base_config.randomize_intrinsics,
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "stage_order": stage_names,
        "seeds": seeds,
    }
    config.update(env_config.task_metadata())
    return config


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[str], list[int], Path | None]:
    parser = argparse.ArgumentParser(
        description="Compare mainline component stages on point-robot control."
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
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=DEFAULT_STAGE_ORDER,
        help="default: stable ablation chain only; h4/h5 require explicit opt-in",
    )
    parser.add_argument("--seeds", type=int, default=5)
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
        list(args.stages),
        [args.seed_start + offset for offset in range(args.seeds)],
        Path(args.output_jsonl) if args.output_jsonl else None,
    )


def format_delta(value: object) -> str:
    if value is None:
        return "na"
    return f"{value:.3f}"


def main() -> None:
    agent_config, env_config, stage_names, seeds, output_jsonl = parse_args()
    task_metadata = env_config.task_metadata()
    print(
        f"benchmark_id={task_metadata['benchmark_id']} "
        f"observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"neuron_model={agent_config.neuron_model} "
        f"seeds={','.join(str(seed) for seed in seeds)}"
    )
    print(f"stage_order={','.join(stage_names)}")
    if output_jsonl is not None:
        prepare_jsonl(output_jsonl)
        print(f"output_jsonl={output_jsonl}")
    results = run_component_benchmark(
        agent_config,
        env_config,
        stage_names,
        seeds,
        output_jsonl=output_jsonl,
    )
    print("summary")
    for stage_name in stage_names:
        metrics = results[stage_name]
        print(
            f"  {stage_name} "
            f"mean_eval_reward={metrics['mean_eval_reward']:.3f} "
            f"mean_eval_success={metrics['mean_eval_success']:.3f} "
            f"mean_elapsed_sec={metrics['mean_elapsed_sec']:.3f} "
            f"reward_gain_vs_previous={format_delta(metrics['reward_gain_vs_previous'])} "
            f"success_gain_vs_previous={format_delta(metrics['success_gain_vs_previous'])} "
            f"reward_gain_vs_baseline={metrics['reward_gain_vs_baseline']:.3f} "
            f"success_gain_vs_baseline={metrics['success_gain_vs_baseline']:.3f}"
        )
    print(
        f"  best_by_reward={best_stage(results, 'mean_eval_reward')['stage']} "
        f"best_by_success={best_stage(results, 'mean_eval_success')['stage']}"
    )


if __name__ == "__main__":
    main()
