"""CLI entry for three_factor vs tess_like comparison on point-robot control."""

from __future__ import annotations

import argparse
from dataclasses import replace

from envs.point_robot import PointRobotConfig
from models.common import mean, safe_ratio
from models.point_robot_closed_loop import AgentConfig, train_agent


def run_comparison(base_config: AgentConfig, env_config: PointRobotConfig, seeds: list[int]) -> dict[str, dict[str, float]]:
    summaries: dict[str, list[dict[str, float | str]]] = {"three_factor": [], "tess_like": []}
    for plasticity_rule in ("three_factor", "tess_like"):
        print(f"rule={plasticity_rule}")
        for seed in seeds:
            agent_config = replace(base_config, plasticity_rule=plasticity_rule, seed=seed)
            run_env_config = replace(env_config, seed=seed + 7)
            summary = train_agent(agent_config, run_env_config, verbose=False)
            summaries[plasticity_rule].append(summary)
            print(
                f"  seed={seed} "
                f"eval_reward={summary['final_eval_reward']:.3f} "
                f"eval_success={summary['final_eval_success']:.3f} "
                f"elapsed_sec={summary['elapsed_sec']:.3f}"
            )
        print()

    aggregated = {
        rule: {
            "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
            "mean_eval_success": mean(row["final_eval_success"] for row in rows),
            "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
        }
        for rule, rows in summaries.items()
    }
    aggregated["delta"] = {
        "reward_gain_tess_like_minus_three_factor": (
            aggregated["tess_like"]["mean_eval_reward"] - aggregated["three_factor"]["mean_eval_reward"]
        ),
        "success_gain_tess_like_minus_three_factor": (
            aggregated["tess_like"]["mean_eval_success"] - aggregated["three_factor"]["mean_eval_success"]
        ),
        "speed_ratio_tess_like_vs_three_factor": safe_ratio(
            aggregated["tess_like"]["mean_elapsed_sec"],
            aggregated["three_factor"]["mean_elapsed_sec"],
        ),
    }
    return aggregated


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int]]:
    parser = argparse.ArgumentParser(
        description="Compare three_factor and tess_like on point robot control."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=96)
    parser.add_argument("--recurrent-degree", type=int, default=4)
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
    )


def main() -> None:
    agent_config, env_config, seeds = parse_args()
    print(
        f"task observation_mode={env_config.observation_mode} "
        f"goal_cue_steps={env_config.goal_cue_steps} "
        f"max_steps={env_config.max_steps} "
        f"n_neurons={agent_config.n_neurons} "
        f"recurrent_degree={agent_config.recurrent_degree} "
        f"neuron_model={agent_config.neuron_model}"
    )
    results = run_comparison(agent_config, env_config, seeds)
    print("summary")
    print(
        f"  three_factor mean_eval_reward={results['three_factor']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['three_factor']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['three_factor']['mean_elapsed_sec']:.3f}"
    )
    print(
        f"  tess_like mean_eval_reward={results['tess_like']['mean_eval_reward']:.3f} "
        f"mean_eval_success={results['tess_like']['mean_eval_success']:.3f} "
        f"mean_elapsed_sec={results['tess_like']['mean_elapsed_sec']:.3f}"
    )
    print(
        "  delta "
        f"reward_gain_tess_like_minus_three_factor="
        f"{results['delta']['reward_gain_tess_like_minus_three_factor']:.3f} "
        f"success_gain_tess_like_minus_three_factor="
        f"{results['delta']['success_gain_tess_like_minus_three_factor']:.3f} "
        f"speed_ratio_tess_like_vs_three_factor="
        f"{results['delta']['speed_ratio_tess_like_vs_three_factor']:.3f}"
    )


if __name__ == "__main__":
    main()
