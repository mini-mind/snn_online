"""Observe quiet internal RSNN dynamics after online point-robot training."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from envs.point_robot import ACTIONS, PointRobotConfig, PointRobotEnv
from models.common import dot, mean
from models.point_robot_closed_loop import (
    AgentConfig,
    ClosedLoopPointRobotAgent,
    evaluate_agent,
    run_episode,
)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def train_agent_for_observation(
    config: AgentConfig,
    env_config: PointRobotConfig,
) -> tuple[ClosedLoopPointRobotAgent, dict[str, float]]:
    rng = random.Random(config.seed)
    env = PointRobotEnv(env_config, rng)
    agent = ClosedLoopPointRobotAgent(
        obs_dim=len(env.observation()),
        n_actions=len(ACTIONS),
        config=config,
        rng=rng,
        observation_mode=env_config.observation_mode,
    )
    final_train_reward = 0.0
    final_train_success = 0.0
    final_model_mse = 0.0
    episodes_since_eval = 0
    reward_window = 0.0
    success_window = 0
    model_error_window = 0.0
    for episode in range(1, config.episodes + 1):
        reward, reached, model_error, _ = run_episode(
            env,
            agent,
            config,
            episode=episode,
            learn=True,
        )
        reward_window += reward
        success_window += int(reached)
        model_error_window += model_error
        episodes_since_eval += 1
        if episode % config.eval_every == 0:
            final_train_reward = reward_window / episodes_since_eval
            final_train_success = success_window / episodes_since_eval
            final_model_mse = model_error_window / episodes_since_eval
            reward_window = 0.0
            success_window = 0
            model_error_window = 0.0
            episodes_since_eval = 0
    if episodes_since_eval > 0:
        final_train_reward = reward_window / episodes_since_eval
        final_train_success = success_window / episodes_since_eval
        final_model_mse = model_error_window / episodes_since_eval
    return agent, {
        "final_train_reward": final_train_reward,
        "final_train_success": final_train_success,
        "final_model_mse": final_model_mse,
    }


def collect_reference_features(
    agent: ClosedLoopPointRobotAgent,
    config: AgentConfig,
    env_config: PointRobotConfig,
    reference_episodes: int,
    seed: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    references = []
    for episode in range(reference_episodes):
        env = PointRobotEnv(env_config, rng)
        observation = env.reset()
        agent.reset_state()
        features = agent.observe(observation)
        done = False
        steps = 0
        while not done:
            action = agent.choose_action(observation, features, epsilon=0.0, learn=False)
            next_observation, _, done = env.step(action)
            next_features = agent.observe(next_observation)
            references.append(list(next_features))
            observation = next_observation
            features = next_features
            steps += 1
            if steps >= env_config.max_steps:
                break
    return references


def observe_quiet_phase(
    agent: ClosedLoopPointRobotAgent,
    input_dim: int,
    reference_features: list[list[float]],
    quiet_steps: int,
    reactivation_threshold: float,
) -> dict[str, float]:
    agent.reset_state()
    quiet_features = []
    quiet_input = [0.0 for _ in range(input_dim)]
    for _ in range(quiet_steps):
        quiet_features.append(agent.observe(quiet_input))

    activities = [feature_activity(features) for features in quiet_features]
    consecutive = [
        cosine_similarity(left, right)
        for left, right in zip(quiet_features, quiet_features[1:], strict=False)
    ]
    reference_similarities = [
        max((cosine_similarity(features, reference) for reference in reference_features), default=0.0)
        for features in quiet_features
    ]
    reactivations = [
        1.0 if similarity >= reactivation_threshold else 0.0
        for similarity in reference_similarities
    ]
    return {
        "quiet_mean_activity": mean(activities),
        "quiet_consecutive_similarity": mean(consecutive),
        "quiet_max_reference_similarity": mean(reference_similarities),
        "quiet_reactivation_fraction": mean(reactivations),
    }


def feature_activity(features: list[float]) -> float:
    return sum(abs(value) for value in features) / max(1, len(features))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(dot(left, left))
    right_norm = math.sqrt(dot(right, right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)


def run_experiment(
    config: AgentConfig,
    env_config: PointRobotConfig,
    reference_episodes: int,
    quiet_steps: int,
    reactivation_threshold: float,
) -> dict[str, object]:
    start_time = time.perf_counter()
    agent, train_summary = train_agent_for_observation(config, env_config)
    final_eval_reward, final_eval_success, final_eval_length = evaluate_agent(
        agent,
        config,
        env_config,
        seed=config.seed + 10000 + config.episodes,
    )
    reference_features = collect_reference_features(
        agent,
        config,
        env_config,
        reference_episodes=reference_episodes,
        seed=config.seed + 20000,
    )
    quiet_summary = observe_quiet_phase(
        agent,
        input_dim=len(PointRobotEnv(env_config, random.Random(config.seed + 30000)).observation()),
        reference_features=reference_features,
        quiet_steps=quiet_steps,
        reactivation_threshold=reactivation_threshold,
    )
    baseline_agent = build_untrained_agent(config, env_config, seed=config.seed + 40000)
    baseline_quiet_summary = observe_quiet_phase(
        baseline_agent,
        input_dim=len(PointRobotEnv(env_config, random.Random(config.seed + 50000)).observation()),
        reference_features=reference_features,
        quiet_steps=quiet_steps,
        reactivation_threshold=reactivation_threshold,
    )
    quiet_uplift = {
        "quiet_reference_similarity_uplift": (
            quiet_summary["quiet_max_reference_similarity"]
            - baseline_quiet_summary["quiet_max_reference_similarity"]
        ),
        "quiet_reactivation_fraction_uplift": (
            quiet_summary["quiet_reactivation_fraction"]
            - baseline_quiet_summary["quiet_reactivation_fraction"]
        ),
    }
    summary: dict[str, object] = {
        "type": "summary",
        "candidate": "h4_eprop_like_v0",
        "seed": config.seed,
        "episodes": config.episodes,
        "eval_every": config.eval_every,
        "eval_episodes": config.eval_episodes,
        "reference_episodes": reference_episodes,
        "quiet_steps": quiet_steps,
        "reactivation_threshold": reactivation_threshold,
        "n_neurons": config.n_neurons,
        "recurrent_degree": config.recurrent_degree,
        "plasticity_rule": config.plasticity_rule,
        "modulation_mode": config.modulation_mode,
        "recurrent_delay_line": config.recurrent_delay_line,
        "final_eval_reward": final_eval_reward,
        "final_eval_success": final_eval_success,
        "final_eval_length": final_eval_length,
        "reference_vectors": len(reference_features),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    summary.update(train_summary)
    summary.update(quiet_summary)
    summary.update(prefix_keys("untrained_", baseline_quiet_summary))
    summary.update(quiet_uplift)
    summary.update(env_config.task_metadata())
    return summary


def build_untrained_agent(
    config: AgentConfig,
    env_config: PointRobotConfig,
    seed: int,
) -> ClosedLoopPointRobotAgent:
    rng = random.Random(seed)
    env = PointRobotEnv(env_config, rng)
    return ClosedLoopPointRobotAgent(
        obs_dim=len(env.observation()),
        n_actions=len(ACTIONS),
        config=config,
        rng=rng,
        observation_mode=env_config.observation_mode,
    )


def prefix_keys(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def hard_point_robot_config(seed: int) -> PointRobotConfig:
    return PointRobotConfig(
        observation_mode="partial_goal_cue",
        goal_cue_steps=3,
        max_steps=80,
        seed=seed,
    )


def parse_args() -> tuple[AgentConfig, PointRobotConfig, int, int, float, Path | None]:
    parser = argparse.ArgumentParser(
        description=(
            "Train the hard-task mainline agent, then observe quiet internal dynamics "
            "without replay training."
        )
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--reference-episodes", type=int, default=4)
    parser.add_argument("--quiet-steps", type=int, default=40)
    parser.add_argument("--reactivation-threshold", type=float, default=0.75)
    parser.add_argument("--n-neurons", type=int, default=64)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-jsonl", default="")
    args = parser.parse_args()
    config = AgentConfig(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        n_neurons=args.n_neurons,
        recurrent_degree=args.recurrent_degree,
        plasticity_rule="tess_like",
        modulation_mode="per_neuron",
        recurrent_delay_line=True,
        randomize_intrinsics=True,
        seed=args.seed,
    )
    return (
        config,
        hard_point_robot_config(seed=args.seed + 7),
        args.reference_episodes,
        args.quiet_steps,
        args.reactivation_threshold,
        Path(args.output_jsonl) if args.output_jsonl else None,
    )


def main() -> None:
    config, env_config, reference_episodes, quiet_steps, threshold, output_jsonl = parse_args()
    print(
        f"quiet_observation candidate=h4_eprop_like_v0 "
        f"benchmark_id={env_config.task_metadata()['benchmark_id']} "
        f"episodes={config.episodes} reference_episodes={reference_episodes} "
        f"quiet_steps={quiet_steps} threshold={threshold}"
    )
    summary = run_experiment(
        config,
        env_config,
        reference_episodes=reference_episodes,
        quiet_steps=quiet_steps,
        reactivation_threshold=threshold,
    )
    print(
        "summary "
        f"final_eval_reward={summary['final_eval_reward']:.3f} "
        f"final_eval_success={summary['final_eval_success']:.3f} "
        f"quiet_mean_activity={summary['quiet_mean_activity']:.4f} "
        f"quiet_consecutive_similarity={summary['quiet_consecutive_similarity']:.3f} "
        f"quiet_max_reference_similarity={summary['quiet_max_reference_similarity']:.3f} "
        f"quiet_reactivation_fraction={summary['quiet_reactivation_fraction']:.3f} "
        f"untrained_quiet_max_reference_similarity="
        f"{summary['untrained_quiet_max_reference_similarity']:.3f} "
        f"untrained_quiet_reactivation_fraction="
        f"{summary['untrained_quiet_reactivation_fraction']:.3f} "
        f"quiet_reference_similarity_uplift="
        f"{summary['quiet_reference_similarity_uplift']:.3f} "
        f"quiet_reactivation_fraction_uplift="
        f"{summary['quiet_reactivation_fraction_uplift']:.3f} "
        f"reference_vectors={summary['reference_vectors']} "
        f"elapsed_sec={summary['elapsed_sec']:.3f}"
    )
    if output_jsonl is not None:
        append_jsonl(output_jsonl, summary)
        print(f"output_jsonl={output_jsonl}")


if __name__ == "__main__":
    main()
