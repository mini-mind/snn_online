"""CLI entry for the point-robot closed-loop experiment."""

from __future__ import annotations

import argparse

from envs.point_robot import PointRobotConfig
from models.point_robot_closed_loop import AgentConfig, train_agent


def parse_args() -> tuple[AgentConfig, PointRobotConfig]:
    parser = argparse.ArgumentParser(description="Run the point robot closed-loop experiment.")
    parser.add_argument("--episodes", type=int, default=AgentConfig.episodes)
    parser.add_argument("--eval-every", type=int, default=AgentConfig.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=AgentConfig.eval_episodes)
    parser.add_argument("--n-neurons", type=int, default=AgentConfig.n_neurons)
    parser.add_argument("--n-layers", type=int, default=AgentConfig.n_layers)
    parser.add_argument("--grid-width", type=int, default=AgentConfig.grid_width)
    parser.add_argument("--grid-height", type=int, default=AgentConfig.grid_height)
    parser.add_argument("--recurrent-radius", type=int, default=AgentConfig.recurrent_radius)
    parser.add_argument("--recurrent-degree", type=int, default=AgentConfig.recurrent_degree)
    parser.add_argument("--interlayer-radius", type=int, default=AgentConfig.interlayer_radius)
    parser.add_argument("--interlayer-degree", type=int, default=AgentConfig.interlayer_degree)
    parser.add_argument("--lr-model", type=float, default=AgentConfig.lr_model)
    parser.add_argument("--lr-value", type=float, default=AgentConfig.lr_value)
    parser.add_argument("--epsilon-start", type=float, default=AgentConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=AgentConfig.epsilon_end)
    parser.add_argument("--model-score-weight", type=float, default=AgentConfig.model_score_weight)
    parser.add_argument("--value-score-weight", type=float, default=AgentConfig.value_score_weight)
    parser.add_argument("--recurrent-plasticity", type=float, default=AgentConfig.recurrent_plasticity)
    parser.add_argument("--neuron-model", choices=["lif", "izh"], default=AgentConfig.neuron_model)
    parser.add_argument("--observation-mode", choices=["full", "partial_goal_cue"], default=PointRobotConfig.observation_mode)
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument("--fixed-intrinsics", action="store_true")
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--seed", type=int, default=AgentConfig.seed)
    args = parser.parse_args()
    return (
        AgentConfig(
            episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            n_neurons=args.n_neurons,
            n_layers=args.n_layers,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            recurrent_radius=args.recurrent_radius,
            recurrent_degree=args.recurrent_degree,
            interlayer_radius=args.interlayer_radius,
            interlayer_degree=args.interlayer_degree,
            lr_model=args.lr_model,
            lr_value=args.lr_value,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            model_score_weight=args.model_score_weight,
            value_score_weight=args.value_score_weight,
            recurrent_plasticity=args.recurrent_plasticity,
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
    )


def main() -> None:
    summary = train_agent(*parse_args(), verbose=True)
    print()
    print(
        f"final_summary model={summary['neuron_model']} "
        f"eval_reward={summary['final_eval_reward']:.3f} "
        f"eval_success={summary['final_eval_success']:.3f} "
        f"elapsed_sec={summary['elapsed_sec']:.3f}"
    )


if __name__ == "__main__":
    main()
