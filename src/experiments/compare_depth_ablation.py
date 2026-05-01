"""CLI entry for depth ablation on point-robot control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from envs.point_robot import PointRobotConfig
from models.common import mean, safe_ratio
from models.point_robot_closed_loop import AgentConfig, train_agent


@dataclass(frozen=True)
class ArchitectureSpec:
    """Single architecture choice."""

    label: str
    n_neurons: int
    n_layers: int


def run_depth_ablation(
    base_config: AgentConfig,
    env_config: PointRobotConfig,
    seeds: list[int],
    architectures: list[ArchitectureSpec],
) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, list[dict[str, float | str]]]] = {"lif": {}, "izh": {}}
    for neuron_model in ("lif", "izh"):
        print(f"model={neuron_model}")
        for architecture in architectures:
            print(f"  arch={architecture.label}")
            rows: list[dict[str, float | str]] = []
            for seed in seeds:
                agent_config = replace(
                    base_config,
                    neuron_model=neuron_model,
                    n_neurons=architecture.n_neurons,
                    n_layers=architecture.n_layers,
                    seed=seed,
                )
                run_env_config = replace(env_config, seed=seed + 7)
                summary = train_agent(agent_config, run_env_config, verbose=False)
                rows.append(summary)
                print(
                    f"    seed={seed} "
                    f"eval_reward={summary['final_eval_reward']:.3f} "
                    f"eval_success={summary['final_eval_success']:.3f} "
                    f"elapsed_sec={summary['elapsed_sec']:.3f}"
                )
            results[neuron_model][architecture.label] = rows
        print()
    return {
        model: {
            label: {
                "mean_eval_reward": mean(row["final_eval_reward"] for row in rows),
                "mean_eval_success": mean(row["final_eval_success"] for row in rows),
                "mean_elapsed_sec": mean(row["elapsed_sec"] for row in rows),
            }
            for label, rows in grouped_rows.items()
        }
        for model, grouped_rows in results.items()
    }


def print_summary(results: dict[str, dict[str, dict[str, float]]]) -> None:
    for neuron_model in ("lif", "izh"):
        print(f"summary model={neuron_model}")
        labels = list(results[neuron_model].keys())
        for label in labels:
            metrics = results[neuron_model][label]
            print(
                f"  arch={label} "
                f"mean_eval_reward={metrics['mean_eval_reward']:.3f} "
                f"mean_eval_success={metrics['mean_eval_success']:.3f} "
                f"mean_elapsed_sec={metrics['mean_elapsed_sec']:.3f}"
            )
        shallow_label, deep_label, wide_label = labels
        print(
            f"  delta_same_width deep_minus_shallow_reward="
            f"{results[neuron_model][deep_label]['mean_eval_reward'] - results[neuron_model][shallow_label]['mean_eval_reward']:.3f} "
            f"deep_minus_shallow_success="
            f"{results[neuron_model][deep_label]['mean_eval_success'] - results[neuron_model][shallow_label]['mean_eval_success']:.3f} "
            f"speed_ratio="
            f"{safe_ratio(results[neuron_model][deep_label]['mean_elapsed_sec'], results[neuron_model][shallow_label]['mean_elapsed_sec']):.3f}"
        )
        print(
            f"  delta_same_total_features deep_minus_wide_reward="
            f"{results[neuron_model][deep_label]['mean_eval_reward'] - results[neuron_model][wide_label]['mean_eval_reward']:.3f} "
            f"deep_minus_wide_success="
            f"{results[neuron_model][deep_label]['mean_eval_success'] - results[neuron_model][wide_label]['mean_eval_success']:.3f} "
            f"speed_ratio="
            f"{safe_ratio(results[neuron_model][deep_label]['mean_elapsed_sec'], results[neuron_model][wide_label]['mean_elapsed_sec']):.3f}"
        )
        print()


def parse_args() -> tuple[AgentConfig, PointRobotConfig, list[int], list[ArchitectureSpec]]:
    parser = argparse.ArgumentParser(description="Run strict depth ablation for point robot control.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--observation-mode", choices=["full", "partial_goal_cue"], default=PointRobotConfig.observation_mode)
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=31)
    parser.add_argument("--base-width", type=int, default=96)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--grid-width", type=int, default=0)
    parser.add_argument("--grid-height", type=int, default=0)
    parser.add_argument("--recurrent-radius", type=int, default=1)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--interlayer-radius", type=int, default=1)
    parser.add_argument("--interlayer-degree", type=int, default=4)
    args = parser.parse_args()
    base_config = AgentConfig(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        n_neurons=args.base_width,
        n_layers=1,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        recurrent_radius=args.recurrent_radius,
        recurrent_degree=args.recurrent_degree,
        interlayer_radius=args.interlayer_radius,
        interlayer_degree=args.interlayer_degree,
        neuron_model="lif",
        randomize_intrinsics=True,
        seed=args.seed_start,
    )
    env_config = PointRobotConfig(
        max_steps=args.max_steps,
        observation_mode=args.observation_mode,
        goal_cue_steps=args.goal_cue_steps,
        seed=args.seed_start + 7,
    )
    return (
        base_config,
        env_config,
        [args.seed_start + offset for offset in range(args.seeds)],
        [
            ArchitectureSpec(label=f"1x{args.base_width}", n_neurons=args.base_width, n_layers=1),
            ArchitectureSpec(label=f"{args.deep_layers}x{args.base_width}", n_neurons=args.base_width, n_layers=args.deep_layers),
            ArchitectureSpec(label=f"1x{args.base_width * args.deep_layers}", n_neurons=args.base_width * args.deep_layers, n_layers=1),
        ],
    )


def main() -> None:
    print_summary(run_depth_ablation(*parse_args()))


if __name__ == "__main__":
    main()
