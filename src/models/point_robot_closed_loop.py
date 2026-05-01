"""Point-robot closed-loop training built on top of `dynn`."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from envs.point_robot import ACTIONS, PointRobotConfig, PointRobotEnv
from models.common import argmax, clamp, dot, l2_normalize, mean_squared
from models.recurrent_spiking import RSNNConfig, build_spiking_network, resolve_grid_shape


@dataclass
class AgentConfig:
    """Point-robot agent configuration."""

    episodes: int = 320
    eval_every: int = 40
    eval_episodes: int = 60
    n_neurons: int = 64
    n_layers: int = 1
    grid_width: int = 0
    grid_height: int = 0
    recurrent_radius: int = 1
    recurrent_degree: int = 4
    interlayer_radius: int = 1
    interlayer_degree: int = 4
    lr_model: float = 0.010
    lr_value: float = 0.018
    gamma: float = 0.94
    epsilon_start: float = 0.45
    epsilon_end: float = 0.06
    model_score_weight: float = 2.0
    value_score_weight: float = 0.55
    recurrent_plasticity: float = 0.0002
    neuron_model: str = "lif"
    randomize_intrinsics: bool = True
    seed: int = 31


class LinearValueHead:
    """Linear action-value readout."""

    def __init__(self, n_actions: int, hidden_dim: int, lr: float, rng: random.Random) -> None:
        scale = 0.01 / math.sqrt(hidden_dim)
        self.lr = lr
        self.weights = [
            [rng.gauss(0.0, scale) for _ in range(hidden_dim)]
            for _ in range(n_actions)
        ]

    def q_values(self, features: list[float]) -> list[float]:
        return [dot(weights, features) for weights in self.weights]

    def update(self, features: list[float], action: int, td_error: float) -> None:
        row = self.weights[action]
        clipped_error = clamp(td_error, -2.0, 2.0)
        for hidden_index, feature in enumerate(features):
            row[hidden_index] += self.lr * clipped_error * feature
            row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)


class DeltaWorldModelHead:
    """One-step observation-delta predictor."""

    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        hidden_dim: int,
        lr: float,
        observation_mode: str,
        rng: random.Random,
    ) -> None:
        scale = 0.01 / math.sqrt(hidden_dim)
        self.lr = lr
        self.observation_mode = observation_mode
        self.weights = [
            [
                [rng.gauss(0.0, scale) for _ in range(hidden_dim)]
                for _ in range(obs_dim)
            ]
            for _ in range(n_actions)
        ]

    def predict(self, observation: list[float], features: list[float], action: int) -> list[float]:
        predicted_delta = [dot(row, features) for row in self.weights[action]]
        predicted = [
            value + delta
            for value, delta in zip(observation, predicted_delta, strict=True)
        ]
        return clamp_observation(predicted, self.observation_mode)

    def update(
        self,
        observation: list[float],
        features: list[float],
        action: int,
        next_observation: list[float],
    ) -> float:
        predicted = self.predict(observation, features, action)
        errors = [
            target - output
            for target, output in zip(next_observation, predicted, strict=True)
        ]
        for output_index, error in enumerate(errors):
            row = self.weights[action][output_index]
            clipped_error = clamp(error, -1.0, 1.0)
            for hidden_index, feature in enumerate(features):
                row[hidden_index] += self.lr * clipped_error * feature
                row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)
        return mean_squared(errors)


class ClosedLoopPointRobotAgent:
    """Closed-loop controller with recurrent spiking state."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: AgentConfig,
        rng: random.Random,
        observation_mode: str,
    ) -> None:
        self.config = config
        self.n_actions = n_actions
        self.rsnn = build_spiking_network(
            RSNNConfig(
                input_dim=obs_dim,
                n_neurons=config.n_neurons,
                n_layers=config.n_layers,
                grid_width=config.grid_width,
                grid_height=config.grid_height,
                recurrent_radius=config.recurrent_radius,
                recurrent_degree=config.recurrent_degree,
                interlayer_radius=config.interlayer_radius,
                interlayer_degree=config.interlayer_degree,
                neuron_model=config.neuron_model,
                plastic_lr=config.recurrent_plasticity,
                randomize_intrinsics=config.randomize_intrinsics,
                seed=config.seed + 1,
            ),
            rng,
        )
        hidden_dim = self.rsnn.feature_dim()
        self.value_head = LinearValueHead(n_actions, hidden_dim, config.lr_value, rng)
        self.world_model = DeltaWorldModelHead(
            n_actions=n_actions,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            lr=config.lr_model,
            observation_mode=observation_mode,
            rng=rng,
        )

    def reset_state(self) -> None:
        self.rsnn.reset_state()

    def observe(self, observation: list[float]) -> list[float]:
        return l2_normalize(self.rsnn.step(observation))

    def q_values(self, features: list[float]) -> list[float]:
        return self.value_head.q_values(features)

    def predict_next(self, observation: list[float], features: list[float], action: int) -> list[float]:
        return self.world_model.predict(observation, features, action)

    def choose_action(self, observation: list[float], features: list[float], epsilon: float, learn: bool) -> int:
        if learn and self.rsnn.rng.random() < epsilon:
            return self.rsnn.rng.randrange(self.n_actions)
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
        return self.world_model.update(observation, features, action, next_observation)

    def learn_value(self, features: list[float], action: int, td_error: float) -> None:
        self.value_head.update(features, action, td_error)


def run_episode(
    env: PointRobotEnv,
    agent: ClosedLoopPointRobotAgent,
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
        prediction_mse = (
            agent.learn_world_model(observation, features, action, next_observation)
            if learn
            else 0.0
        )

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


def evaluate_agent(agent: ClosedLoopPointRobotAgent, config: AgentConfig, env_config: PointRobotConfig, seed: int) -> tuple[float, float, float]:
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


def train_agent(config: AgentConfig, env_config: PointRobotConfig, verbose: bool = True) -> dict[str, float | str]:
    validate_agent_config(config)
    start_time = time.perf_counter()
    rng = random.Random(config.seed)
    env = PointRobotEnv(env_config, rng)
    agent = ClosedLoopPointRobotAgent(
        obs_dim=len(env.observation()),
        n_actions=len(ACTIONS),
        config=config,
        rng=rng,
        observation_mode=env_config.observation_mode,
    )

    if verbose:
        grid_width, grid_height = resolve_grid_shape(
            config.n_neurons,
            config.grid_width,
            config.grid_height,
        )
        print("R-SNN point robot closed loop")
        print(
            f"seed={config.seed} model={config.neuron_model} "
            f"episodes={config.episodes} n_neurons={config.n_neurons} "
            f"n_layers={config.n_layers} grid={grid_width}x{grid_height} "
            f"recurrent_radius={config.recurrent_radius} "
            f"recurrent_degree={config.recurrent_degree} "
            f"interlayer_radius={config.interlayer_radius} "
            f"interlayer_degree={config.interlayer_degree} "
            f"randomize_intrinsics={config.randomize_intrinsics} "
            f"max_steps={env_config.max_steps} "
            f"observation_mode={env_config.observation_mode} "
            f"goal_cue_steps={env_config.goal_cue_steps}"
        )
        print("learn: world_model <- prediction_error, action_value <- TD_error")
    random_reward, random_success, random_length = random_baseline(config, env_config, seed=config.seed + 9000)
    if verbose:
        print(
            f"random_baseline reward={random_reward:.3f} "
            f"success={random_success:.3f} length={random_length:.1f}"
        )
        print()

    reward_window = 0.0
    success_window = 0
    model_error_window = 0.0
    length_window = 0
    final_train_reward = 0.0
    final_train_success = 0.0
    final_model_error = 0.0
    final_train_length = 0.0
    final_eval_reward = random_reward
    final_eval_success = random_success
    final_eval_length = random_length
    episodes_since_eval = 0
    for episode in range(1, config.episodes + 1):
        reward, reached, model_error, steps = run_episode(env, agent, config, episode=episode, learn=True)
        reward_window += reward
        success_window += int(reached)
        model_error_window += model_error
        length_window += steps
        episodes_since_eval += 1

        if episode % config.eval_every == 0:
            final_train_reward = reward_window / episodes_since_eval
            final_train_success = success_window / episodes_since_eval
            final_model_error = model_error_window / episodes_since_eval
            final_train_length = length_window / episodes_since_eval
            final_eval_reward, final_eval_success, final_eval_length = evaluate_agent(
                agent,
                config,
                env_config,
                seed=config.seed + 10000 + episode,
            )
            if verbose:
                print(
                    f"episode={episode} "
                    f"train_reward={final_train_reward:.3f} train_success={final_train_success:.3f} "
                    f"model_mse={final_model_error:.4f} train_len={final_train_length:.1f} "
                    f"eval_reward={final_eval_reward:.3f} eval_success={final_eval_success:.3f} "
                    f"eval_len={final_eval_length:.1f}"
                )
            reward_window = 0.0
            success_window = 0
            model_error_window = 0.0
            length_window = 0
            episodes_since_eval = 0

    if episodes_since_eval > 0:
        final_train_reward = reward_window / episodes_since_eval
        final_train_success = success_window / episodes_since_eval
        final_model_error = model_error_window / episodes_since_eval
        final_train_length = length_window / episodes_since_eval
        final_eval_reward, final_eval_success, final_eval_length = evaluate_agent(
            agent,
            config,
            env_config,
            seed=config.seed + 10000 + config.episodes,
        )
        if verbose:
            print(
                f"episode={config.episodes} "
                f"train_reward={final_train_reward:.3f} train_success={final_train_success:.3f} "
                f"model_mse={final_model_error:.4f} train_len={final_train_length:.1f} "
                f"eval_reward={final_eval_reward:.3f} eval_success={final_eval_success:.3f} "
                f"eval_len={final_eval_length:.1f}"
            )

    elapsed_sec = time.perf_counter() - start_time
    return {
        "neuron_model": config.neuron_model,
        "seed": float(config.seed),
        "random_reward": random_reward,
        "random_success": random_success,
        "random_length": random_length,
        "final_train_reward": final_train_reward,
        "final_train_success": final_train_success,
        "final_model_mse": final_model_error,
        "final_train_length": final_train_length,
        "final_eval_reward": final_eval_reward,
        "final_eval_success": final_eval_success,
        "final_eval_length": final_eval_length,
        "elapsed_sec": elapsed_sec,
    }


def clamp_observation(values: list[float], observation_mode: str) -> list[float]:
    if len(values) < 10:
        return values
    clipped = list(values)
    for index in range(4):
        clipped[index] = clamp(clipped[index], -1.5, 1.5)
    if observation_mode == "full":
        for index in range(4, 8):
            clipped[index] = clamp(clipped[index], -1.5, 1.5)
    else:
        clipped[4] = clamp(clipped[4], -1.5, 1.5)
        clipped[5] = clamp(clipped[5], -1.5, 1.5)
        clipped[6] = clamp(clipped[6], 0.0, 1.0)
        clipped[7] = clamp(clipped[7], 0.0, 1.0)
    clipped[8] = clamp(clipped[8], 0.0, 1.5)
    clipped[9] = 1.0
    return clipped


def epsilon_for_episode(config: AgentConfig, episode: int) -> float:
    progress = min(1.0, episode / max(1, config.episodes * 0.75))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def validate_agent_config(config: AgentConfig) -> None:
    if config.episodes <= 0:
        raise ValueError(f"episodes must be positive, got {config.episodes}")
    if config.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {config.eval_every}")
    if config.eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {config.eval_episodes}")
    if config.n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {config.n_layers}")
    if config.n_neurons <= 0:
        raise ValueError(f"n_neurons must be positive, got {config.n_neurons}")
