"""R-SNN point robot control toy.

Run:
    python src/rsnn_point_robot_toy.py

The agent combines:

- recurrent spiking hidden state
- local linear world model readout trained by prediction error
- local action-value readout trained by TD error
- one-step model-based action scoring for control

This is a small control scaffold, not a benchmark implementation.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from point_robot_env import ACTIONS, PointRobotConfig, PointRobotEnv
from rsnn import RSNNConfig, RecurrentSpikingNetwork, clamp, dot


@dataclass
class AgentConfig:
    episodes: int = 320
    eval_every: int = 40
    eval_episodes: int = 60
    n_neurons: int = 64
    lr_model: float = 0.010
    lr_value: float = 0.018
    gamma: float = 0.94
    epsilon_start: float = 0.45
    epsilon_end: float = 0.06
    model_score_weight: float = 2.0
    value_score_weight: float = 0.55
    recurrent_plasticity: float = 0.0002
    seed: int = 31


class RSNNPointRobotAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: AgentConfig, rng: random.Random) -> None:
        self.config = config
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.rng = rng
        self.rsnn = RecurrentSpikingNetwork(
            RSNNConfig(
                input_dim=obs_dim,
                n_neurons=config.n_neurons,
                plastic_lr=config.recurrent_plasticity,
                seed=config.seed + 1,
            ),
            rng,
        )
        hidden_dim = config.n_neurons
        value_scale = 0.01 / math.sqrt(hidden_dim)
        model_scale = 0.01 / math.sqrt(hidden_dim)
        self.value_weights = [
            [rng.gauss(0.0, value_scale) for _ in range(hidden_dim)]
            for _ in range(n_actions)
        ]
        self.model_weights = [
            [
                [rng.gauss(0.0, model_scale) for _ in range(hidden_dim)]
                for _ in range(obs_dim)
            ]
            for _ in range(n_actions)
        ]

    def reset_state(self) -> None:
        self.rsnn.reset_state()

    def observe(self, observation: list[float], modulation: float = 0.0) -> list[float]:
        return normalize_features(self.rsnn.step(observation, modulation=modulation))

    def q_values(self, features: list[float]) -> list[float]:
        return [dot(weights, features) for weights in self.value_weights]

    def predict_next(self, observation: list[float], features: list[float], action: int) -> list[float]:
        predicted_delta = [dot(row, features) for row in self.model_weights[action]]
        predicted = [value + delta for value, delta in zip(observation, predicted_delta)]
        return clamp_observation(predicted)

    def choose_action(self, observation: list[float], features: list[float], epsilon: float, learn: bool) -> int:
        if learn and self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        q_values = self.q_values(features)
        scores = []
        for action in range(self.n_actions):
            predicted_next = self.predict_next(observation, features, action)
            predicted_distance = max(0.0, predicted_next[8])
            score = -self.config.model_score_weight * predicted_distance
            score += self.config.value_score_weight * q_values[action]
            if ACTIONS[action] == "stay":
                score -= 0.04
            scores.append(score)
        return argmax(scores)

    def learn_world_model(self, observation: list[float], features: list[float], action: int, next_observation: list[float]) -> float:
        predicted = self.predict_next(observation, features, action)
        errors = [target - output for target, output in zip(next_observation, predicted)]
        for output_index, error in enumerate(errors):
            row = self.model_weights[action][output_index]
            for hidden_index, feature in enumerate(features):
                row[hidden_index] += self.config.lr_model * clamp(error, -1.0, 1.0) * feature
                row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)
        return mean_squared(errors)

    def learn_value(self, features: list[float], action: int, td_error: float) -> None:
        row = self.value_weights[action]
        clipped_error = clamp(td_error, -2.0, 2.0)
        for hidden_index, feature in enumerate(features):
            row[hidden_index] += self.config.lr_value * clipped_error * feature
            row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)


def run_episode(
    env: PointRobotEnv,
    agent: RSNNPointRobotAgent,
    config: AgentConfig,
    episode: int,
    learn: bool,
) -> tuple[float, bool, float, int]:
    observation = env.reset()
    agent.reset_state()
    total_reward = 0.0
    prediction_error_sum = 0.0
    steps = 0
    epsilon = epsilon_for_episode(config, episode) if learn else 0.0

    features = agent.observe(observation)
    done = False
    reached = False
    while not done:
        action = agent.choose_action(observation, features, epsilon=epsilon, learn=learn)
        next_observation, reward, done = env.step(action)
        next_features = agent.observe(next_observation)
        prediction_mse = agent.learn_world_model(observation, features, action, next_observation) if learn else 0.0

        q_current = agent.q_values(features)[action]
        q_next = max(agent.q_values(next_features))
        td_error = reward + (0.0 if done else config.gamma * q_next) - q_current
        if learn:
            agent.learn_value(features, action, td_error)
            modulation = clamp(0.35 * td_error - 0.15 * prediction_mse, -1.0, 1.0)
            agent.rsnn.apply_recurrent_modulation(modulation)

        total_reward += reward
        prediction_error_sum += prediction_mse
        steps += 1
        observation = next_observation
        features = next_features
        reached = done and env.distance_to_goal() <= env.config.goal_radius

    return total_reward, reached, prediction_error_sum / max(1, steps), steps


def evaluate_agent(agent: RSNNPointRobotAgent, config: AgentConfig, env_config: PointRobotConfig, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    rewards = []
    successes = 0
    lengths = []
    for episode in range(config.eval_episodes):
        env = PointRobotEnv(env_config, rng)
        reward, reached, _, steps = run_episode(env, agent, config, episode=episode, learn=False)
        rewards.append(reward)
        successes += int(reached)
        lengths.append(steps)
    return sum(rewards) / len(rewards), successes / len(rewards), sum(lengths) / len(lengths)


def random_baseline(config: AgentConfig, env_config: PointRobotConfig, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    rewards = []
    successes = 0
    lengths = []
    for _ in range(config.eval_episodes):
        env = PointRobotEnv(env_config, rng)
        env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            _, reward, done = env.step(rng.randrange(len(ACTIONS)))
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        successes += int(env.distance_to_goal() <= env.config.goal_radius)
        lengths.append(steps)
    return sum(rewards) / len(rewards), successes / len(rewards), sum(lengths) / len(lengths)


def run(config: AgentConfig, env_config: PointRobotConfig) -> None:
    rng = random.Random(config.seed)
    env = PointRobotEnv(env_config, rng)
    agent = RSNNPointRobotAgent(obs_dim=len(env.observation()), n_actions=len(ACTIONS), config=config, rng=rng)

    print("R-SNN point robot control toy")
    print(
        f"seed={config.seed} episodes={config.episodes} n_neurons={config.n_neurons} "
        f"max_steps={env_config.max_steps}"
    )
    print("learn: world_model <- prediction_error, action_value <- TD_error")
    random_reward, random_success, random_length = random_baseline(config, env_config, seed=config.seed + 9000)
    print(
        f"random_baseline reward={random_reward:.3f} "
        f"success={random_success:.3f} length={random_length:.1f}"
    )
    print()

    reward_window = 0.0
    success_window = 0
    model_error_window = 0.0
    length_window = 0
    for episode in range(1, config.episodes + 1):
        reward, reached, model_error, steps = run_episode(env, agent, config, episode=episode, learn=True)
        reward_window += reward
        success_window += int(reached)
        model_error_window += model_error
        length_window += steps

        if episode % config.eval_every == 0:
            train_reward = reward_window / config.eval_every
            train_success = success_window / config.eval_every
            train_model_error = model_error_window / config.eval_every
            train_length = length_window / config.eval_every
            eval_reward, eval_success, eval_length = evaluate_agent(agent, config, env_config, seed=config.seed + 10000 + episode)
            print(
                f"episode={episode} "
                f"train_reward={train_reward:.3f} train_success={train_success:.3f} "
                f"model_mse={train_model_error:.4f} train_len={train_length:.1f} "
                f"eval_reward={eval_reward:.3f} eval_success={eval_success:.3f} "
                f"eval_len={eval_length:.1f}"
            )
            reward_window = 0.0
            success_window = 0
            model_error_window = 0.0
            length_window = 0




def normalize_features(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def clamp_observation(values: list[float]) -> list[float]:
    if len(values) < 10:
        return values
    clipped = list(values)
    for index in range(8):
        clipped[index] = clamp(clipped[index], -1.5, 1.5)
    clipped[8] = clamp(clipped[8], 0.0, 1.5)
    clipped[9] = 1.0
    return clipped

def epsilon_for_episode(config: AgentConfig, episode: int) -> float:
    progress = min(1.0, episode / max(1, config.episodes * 0.75))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def mean_squared(values: list[float]) -> float:
    return sum(value * value for value in values) / max(1, len(values))


def parse_args() -> tuple[AgentConfig, PointRobotConfig]:
    parser = argparse.ArgumentParser(description="Run the R-SNN point robot toy.")
    parser.add_argument("--episodes", type=int, default=AgentConfig.episodes)
    parser.add_argument("--eval-every", type=int, default=AgentConfig.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=AgentConfig.eval_episodes)
    parser.add_argument("--n-neurons", type=int, default=AgentConfig.n_neurons)
    parser.add_argument("--lr-model", type=float, default=AgentConfig.lr_model)
    parser.add_argument("--lr-value", type=float, default=AgentConfig.lr_value)
    parser.add_argument("--epsilon-start", type=float, default=AgentConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=AgentConfig.epsilon_end)
    parser.add_argument("--model-score-weight", type=float, default=AgentConfig.model_score_weight)
    parser.add_argument("--value-score-weight", type=float, default=AgentConfig.value_score_weight)
    parser.add_argument("--recurrent-plasticity", type=float, default=AgentConfig.recurrent_plasticity)
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--seed", type=int, default=AgentConfig.seed)
    args = parser.parse_args()
    agent_config = AgentConfig(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        n_neurons=args.n_neurons,
        lr_model=args.lr_model,
        lr_value=args.lr_value,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        model_score_weight=args.model_score_weight,
        value_score_weight=args.value_score_weight,
        recurrent_plasticity=args.recurrent_plasticity,
        seed=args.seed,
    )
    env_config = PointRobotConfig(max_steps=args.max_steps, seed=args.seed + 7)
    return agent_config, env_config


if __name__ == "__main__":
    run(*parse_args())
