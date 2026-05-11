"""CLI entry for three_factor vs tess_like comparison on point-robot control."""

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


def config_to_dict(base_config: AgentConfig, env_config: PointRobotConfig, seeds: list[int]) -> dict[str, object]:
    return {
        "episodes": base_config.episodes,
        "eval_every": base_config.eval_every,
        "eval_episodes": base_config.eval_episodes,
        "n_neurons": base_config.n_neurons,
        "recurrent_degree": base_config.recurrent_degree,
        "neuron_model": base_config.neuron_model,
        "observation_mode": env_config.observation_mode,
        "goal_cue_steps": env_config.goal_cue_steps,
        "max_steps": env_config.max_steps,
        "tess_fast_decay": base_config.tess_fast_decay,
        "tess_slow_decay": base_config.tess_slow_decay,
        "tess_post_decay": base_config.tess_post_decay,
        "tess_eligibility_decay": base_config.tess_eligibility_decay,
        "seeds": seeds,
    }


def run_comparison(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
    output_jsonl: Path | None = None,
) -> dict[str, dict[str, float]]:
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
            if output_jsonl is not None:
                append_jsonl(
                    output_jsonl,
                    {
                        "type": "run",
                        "plasticity_rule": plasticity_rule,
                        "seed": seed,
                        "eval_reward": summary["final_eval_reward"],
                        "eval_success": summary["final_eval_success"],
                        "elapsed_sec": summary["elapsed_sec"],
                        "episodes": agent_config.episodes,
                        "eval_every": agent_config.eval_every,
                        "eval_episodes": agent_config.eval_episodes,
                        "n_neurons": agent_config.n_neurons,
                        "recurrent_degree": agent_config.recurrent_degree,
                        "neuron_model": agent_config.neuron_model,
                        "observation_mode": run_env_config.observation_mode,
                        "goal_cue_steps": run_env_config.goal_cue_steps,
                        "max_steps": run_env_config.max_steps,
                        "tess_fast_decay": agent_config.tess_fast_decay,
                        "tess_slow_decay": agent_config.tess_slow_decay,
                        "tess_post_decay": agent_config.tess_post_decay,
                        "tess_eligibility_decay": agent_config.tess_eligibility_decay,
                    },
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
    if output_jsonl is not None:
        append_jsonl(
            output_jsonl,
            {
                "type": "summary",
                "three_factor": aggregated["three_factor"],
                "tess_like": aggregated["tess_like"],
                "delta": aggregated["delta"],
                "config": config_to_dict(base_config, env_config, seeds),
            },
        )
    return aggregated


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int], Path | None]:
    parser = argparse.ArgumentParser(
        description="Compare three_factor and tess_like on point robot control."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-neurons", type=int, default=96)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--tess-fast-decay", type=float, default=AgentConfig.tess_fast_decay)
    parser.add_argument("--tess-slow-decay", type=float, default=AgentConfig.tess_slow_decay)
    parser.add_argument("--tess-post-decay", type=float, default=AgentConfig.tess_post_decay)
    parser.add_argument("--tess-eligibility-decay", type=float, default=AgentConfig.tess_eligibility_decay)
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
            tess_fast_decay=args.tess_fast_decay,
            tess_slow_decay=args.tess_slow_decay,
            tess_post_decay=args.tess_post_decay,
            tess_eligibility_decay=args.tess_eligibility_decay,
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
        f"neuron_model={agent_config.neuron_model} "
        f"tess_fast_decay={agent_config.tess_fast_decay:.3f} "
        f"tess_slow_decay={agent_config.tess_slow_decay:.3f} "
        f"tess_post_decay={agent_config.tess_post_decay:.3f} "
        f"tess_eligibility_decay={agent_config.tess_eligibility_decay:.3f}"
    )
    if output_jsonl is not None:
        prepare_jsonl(output_jsonl)
        print(f"output_jsonl={output_jsonl}")
    results = run_comparison(agent_config, env_config, seeds, output_jsonl=output_jsonl)
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
