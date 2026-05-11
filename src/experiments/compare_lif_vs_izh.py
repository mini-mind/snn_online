"""CLI entry for LIF vs Izh comparison on point-robot control."""

from __future__ import annotations

import argparse
from dataclasses import replace

from envs.point_robot import PointRobotConfig
from models.common import mean, safe_ratio
from models.point_robot_closed_loop import AgentConfig, train_agent


def run_comparison(base_config: AgentConfig, env_config: PointRobotConfig, seeds: list[int]) -> dict[str, dict[str, float]]:
    summaries: dict[str, list[dict[str, float | str]]] = {"lif": [], "izh": []}
    for neuron_model in ("lif", "izh"):
        print(f"model={neuron_model}")
        for seed in seeds:
            agent_config = replace(base_config, neuron_model=neuron_model, seed=seed)
            run_env_config = replace(env_config, seed=seed + 7)
            summary = train_agent(agent_config, run_env_config, verbose=False)
            summaries[neuron_model].append(summary)
            print(
                f"  seed={seed} "
                f"eval_reward={summary['final_eval_reward']:.3f} "
                f"eval_success={summary['final_eval_success']:.3f} "
                f"elapsed_sec={summary['elapsed_sec']:.3f}"
            )
        print()
    aggregated = {
        model: {
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }
        for model, rows in summaries.items()
    }
    aggregated["delta"] = {
        "reward_gain_izh_minus_lif": aggregated["izh"]["mean_eval_reward"] - aggregated["lif"]["mean_eval_reward"],
        "success_gain_izh_minus_lif": aggregated["izh"]["mean_eval_success"] - aggregated["lif"]["mean_eval_success"],
        "speed_ratio_izh_vs_lif": safe_ratio(
            aggregated["izh"]["mean_elapsed_sec"],
            aggregated["lif"]["mean_elapsed_sec"],
        ),
    }
    return aggregated


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int]]:
    parser = argparse.ArgumentParser(description="Compare LIF and Izhikevich on point robot control.")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=96)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--plasticity-rule", choices=["three_factor", "tess_like"], default=AgentConfig.plasticity_rule)
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--observation-mode", choices=["full", "partial_goal_cue"], default=PointRobotConfig.observation_mode)
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=31)
    args = parser.parse_args()
    return (
        AgentConfig(
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            n_neurons=args.n_neurons,
            recurrent_degree=args.recurrent_degree,
            plasticity_rule=args.plasticity_rule,
            neuron_model="lif",
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
    )


def main() -> None:
    agent_config, env_config, seeds = parse_args()
    print(
        f"task observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"plasticity_rule={agent_config.plasticity_rule}"
    )
    results = run_comparison(agent_config, env_config, seeds)
    print("summary")
    print(
        f"  lif mean_eval_reward={results['lif']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['lif']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['lif']['mean_elapsed_sec']:.3f}"
    )
    print(
        f"  izh mean_eval_reward={results['izh']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['izh']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['izh']['mean_elapsed_sec']:.3f}"
    )
    print(
        f"  delta reward_gain_izh_minus_lif={results['delta']['reward_gain_izh_minus_lif']:.3f} "
        f"success_gain_izh_minus_lif={results['delta']['success_gain_izh_minus_lif']:.3f} "
        f"speed_ratio_izh_vs_lif={results['delta']['speed_ratio_izh_vs_lif']:.3f}"
    )


if __name__ == "__main__":
    main()
