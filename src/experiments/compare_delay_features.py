"""CLI entry for plain RSNN vs delay-feature RSNN on point-robot control."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from envs.point_robot import PointRobotConfig
from models.common import mean, safe_ratio
from models.point_robot_closed_loop import AgentConfig, train_agent


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def run_comparison(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
    output_jsonl: Path | None = None,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, list[dict[str, float | str]]] = {"plain": [], "delay": []}
    for name, enabled in (("plain", False), ("delay", True)):
        print(f"condition={name}")
        for seed in seeds:
            agent_config = replace(base_config, delay_features=enabled, seed=seed)
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
                append_jsonl(
                    output_jsonl,
                    {
                        "type": "run",
                        "condition": name,
                        "seed": seed,
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
                        "observation_mode": run_env_config.observation_mode,
                        "goal_cue_steps": run_env_config.goal_cue_steps,
                        "max_steps": run_env_config.max_steps,
                        "delay_features": agent_config.delay_features,
                    },
                )
        print()

    aggregated = {
        name: {
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }
        for name, rows in summaries.items()
    }
    aggregated["delta"] = {
        "reward_gain_delay_minus_plain": (
            aggregated["delay"]["mean_eval_reward"] - aggregated["plain"]["mean_eval_reward"]
        ),
        "success_gain_delay_minus_plain": (
            aggregated["delay"]["mean_eval_success"] - aggregated["plain"]["mean_eval_success"]
        ),
        "speed_ratio_delay_vs_plain": safe_ratio(
            aggregated["delay"]["mean_elapsed_sec"],
            aggregated["plain"]["mean_elapsed_sec"],
        ),
    }
    if output_jsonl is not None:
        append_jsonl(
            output_jsonl,
            {
                "type": "summary",
                "plain": aggregated["plain"],
                "delay": aggregated["delay"],
                "delta": aggregated["delta"],
                "config": config_to_dict(base_config, env_config, seeds),
            },
        )
    return aggregated


def config_to_dict(base_config: AgentConfig, env_config: PointRobotConfig, seeds: list[int]) -> dict[str, object]:
    return {
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
        "seeds": seeds,
    }


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int], Path | None]:
    parser = argparse.ArgumentParser(
        description="Compare plain and delay-feature RSNN on point robot control."
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
        Path(args.output_jsonl) if args.output_jsonl else None,
    )


def main() -> None:
    agent_config, env_config, seeds, output_jsonl = parse_args()
    print(
        f"task observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"plasticity_rule={agent_config.plasticity_rule} "
        f"neuron_model={agent_config.neuron_model}"
    )
    if output_jsonl is not None:
        prepare_jsonl(output_jsonl)
        print(f"output_jsonl={output_jsonl}")
    results = run_comparison(agent_config, env_config, seeds, output_jsonl=output_jsonl)
    print("summary")
    print(
        f"  plain mean_eval_reward={results['plain']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['plain']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['plain']['mean_elapsed_sec']:.3f}"
    )
    print(
        f"  delay mean_eval_reward={results['delay']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['delay']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['delay']['mean_elapsed_sec']:.3f}"
    )
    print(
        "  delta "
        f"reward_gain_delay_minus_plain={results['delta']['reward_gain_delay_minus_plain']:.3f} "
        f"success_gain_delay_minus_plain={results['delta']['success_gain_delay_minus_plain']:.3f} "
        f"speed_ratio_delay_vs_plain={results['delta']['speed_ratio_delay_vs_plain']:.3f}"
    )


if __name__ == "__main__":
    main()
