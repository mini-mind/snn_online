"""点机器人闭环控制实验。

运行方式:
    python src/closed_loop/point_robot_closed_loop.py

该脚本把一个最小完整闭环拆成四层：

1. 循环脉冲隐状态；
2. 用预测误差训练的一步 world model；
3. 用 TD 误差训练的动作价值读出头；
4. 同时参考模型预测和价值估计的动作打分。

与之前的单一 LIF 版本不同，这里允许在 `lif` 和 `izh` 两种神经元模型之间
切换，从而比较更复杂脉冲动力学是否真的带来控制收益。
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

from point_robot_env import ACTIONS, PointRobotConfig, PointRobotEnv
from recurrent_spiking import RSNNConfig, build_spiking_network, clamp, dot, resolve_grid_shape


@dataclass
class AgentConfig:
    """点机器人智能体训练配置。"""

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
    """动作价值线性读出头。"""

    def __init__(self, n_actions: int, hidden_dim: int, lr: float, rng: random.Random) -> None:
        scale = 0.01 / math.sqrt(hidden_dim)
        self.lr = lr
        self.weights = [
            [rng.gauss(0.0, scale) for _ in range(hidden_dim)]
            for _ in range(n_actions)
        ]

    def q_values(self, features: list[float]) -> list[float]:
        """根据隐藏特征计算每个动作的价值。"""
        return [dot(weights, features) for weights in self.weights]

    def update(self, features: list[float], action: int, td_error: float) -> None:
        """按 TD 误差更新指定动作的线性读出。"""
        row = self.weights[action]
        clipped_error = clamp(td_error, -2.0, 2.0)
        for hidden_index, feature in enumerate(features):
            row[hidden_index] += self.lr * clipped_error * feature
            row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)


class DeltaWorldModelHead:
    """一步观测增量预测头。"""

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
        """预测执行动作后的下一步观测。"""
        predicted_delta = [dot(row, features) for row in self.weights[action]]
        predicted = [value + delta for value, delta in zip(observation, predicted_delta, strict=True)]
        return clamp_observation(predicted, self.observation_mode)

    def update(
        self,
        observation: list[float],
        features: list[float],
        action: int,
        next_observation: list[float],
    ) -> float:
        """用当前转移样本更新一步模型，并返回预测 MSE。"""
        predicted = self.predict(observation, features, action)
        errors = [target - output for target, output in zip(next_observation, predicted, strict=True)]
        for output_index, error in enumerate(errors):
            row = self.weights[action][output_index]
            clipped_error = clamp(error, -1.0, 1.0)
            for hidden_index, feature in enumerate(features):
                row[hidden_index] += self.lr * clipped_error * feature
                row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)
        return mean_squared(errors)


class ClosedLoopPointRobotAgent:
    """基于循环脉冲隐状态的点机器人控制智能体。

    结构上分三层：

    1. R-SNN 负责把时序观测压缩成动态特征；
    2. value head 估计每个动作的动作价值；
    3. world model head 预测执行动作后的下一步观测。

    动作选择时同时参考价值估计与一步模型预测，形成简化的 model-based control。
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: AgentConfig,
        rng: random.Random,
        observation_mode: str,
    ) -> None:
        """初始化 R-SNN、本地 world model 读出头和价值读出头。"""
        self.config = config
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.observation_mode = observation_mode
        self.rng = rng
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
        """重置 episode 内的循环隐状态。"""
        self.rsnn.reset_state()

    def observe(self, observation: list[float]) -> list[float]:
        """将环境观测送入 R-SNN，并返回归一化特征。"""
        return normalize_features(self.rsnn.step(observation))

    def q_values(self, features: list[float]) -> list[float]:
        """根据当前隐特征计算每个动作的价值。"""
        return self.value_head.q_values(features)

    def predict_next(self, observation: list[float], features: list[float], action: int) -> list[float]:
        """预测某个动作执行后的下一步观测。"""
        return self.world_model.predict(observation, features, action)

    def choose_action(self, observation: list[float], features: list[float], epsilon: float, learn: bool) -> int:
        """按 epsilon-greedy 方式选择动作。

        贪心分支里，动作分数同时考虑：
        - 预测下一状态是否更接近目标；
        - 当前价值头给出的动作价值。
        """
        if learn and self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        q_values = self.q_values(features)
        scores = []
        for action in range(self.n_actions):
            predicted_next = self.predict_next(observation, features, action)
            predicted_distance = max(0.0, predicted_next[8])

            # model score 偏向“朝目标更近”的动作，value score 则保留长期回报估计。
            score = -self.config.model_score_weight * predicted_distance
            score += self.config.value_score_weight * q_values[action]
            if ACTIONS[action] == "stay":
                score -= 0.04
            scores.append(score)
        return argmax(scores)

    def learn_world_model(self, observation: list[float], features: list[float], action: int, next_observation: list[float]) -> float:
        """用当前转移样本更新一步 world model，并返回预测 MSE。"""
        return self.world_model.update(observation, features, action, next_observation)

    def learn_value(self, features: list[float], action: int, td_error: float) -> None:
        """按 TD 误差更新指定动作的价值读出头。"""
        self.value_head.update(features, action, td_error)


def run_episode(
    env: PointRobotEnv,
    agent: ClosedLoopPointRobotAgent,
    config: AgentConfig,
    episode: int,
    learn: bool,
) -> tuple[float, bool, float, int]:
    """运行单个 episode，并在需要时执行在线学习。"""
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

        # 价值学习仍然使用标准一步 TD 目标，但隐状态来自脉冲网络。
        q_current = agent.q_values(features)[action]
        q_next = max(agent.q_values(next_features))
        td_error = reward + (0.0 if done else config.gamma * q_next) - q_current
        if learn:
            agent.learn_value(features, action, td_error)

            # 递归连接的调制信号同时参考 TD 误差与模型预测误差：
            # 奖励偏高时鼓励当前递归动力学，模型误差过大时则抑制。
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
    """在关闭学习的条件下评估智能体平均表现。"""
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
    """计算随机策略基线，便于观察训练是否真正带来控制增益。"""
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
    """运行训练主循环，并返回最终汇总指标。

    该函数既服务命令行入口，也供神经元模型对比脚本复用。
    """
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

    # 窗口统计用于观察最近阶段的训练趋势，避免仅看全局平均掩盖退化。
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


def run(config: AgentConfig, env_config: PointRobotConfig) -> None:
    """命令行入口。"""
    summary = train_agent(config, env_config, verbose=True)
    print()
    print(
        f"final_summary model={summary['neuron_model']} "
        f"eval_reward={summary['final_eval_reward']:.3f} "
        f"eval_success={summary['final_eval_success']:.3f} "
        f"elapsed_sec={summary['elapsed_sec']:.3f}"
    )


def normalize_features(values: list[float]) -> list[float]:
    """将特征向量归一化，控制不同 episode 之间的尺度波动。"""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def clamp_observation(values: list[float], observation_mode: str) -> list[float]:
    """对预测观测做简单裁剪，避免 world model 输出失控。"""
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
    """按线性退火策略计算当前 episode 的探索率。"""
    progress = min(1.0, episode / max(1, config.episodes * 0.75))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def validate_agent_config(config: AgentConfig) -> None:
    """校验训练配置，避免静默地产生无意义汇总。"""
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


def argmax(values: list[float]) -> int:
    """返回最大值对应索引。"""
    return max(range(len(values)), key=lambda index: values[index])


def mean_squared(values: list[float]) -> float:
    """计算均方值。"""
    return sum(value * value for value in values) / max(1, len(values))


def parse_args() -> tuple[AgentConfig, PointRobotConfig]:
    """解析命令行参数并构造智能体与环境配置。"""
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
    parser.add_argument(
        "--observation-mode",
        choices=["full", "partial_goal_cue"],
        default=PointRobotConfig.observation_mode,
    )
    parser.add_argument("--goal-cue-steps", type=int, default=PointRobotConfig.goal_cue_steps)
    parser.add_argument(
        "--fixed-intrinsics",
        action="store_true",
        help="disable per-neuron randomized intrinsic parameters",
    )
    parser.add_argument("--max-steps", type=int, default=PointRobotConfig.max_steps)
    parser.add_argument("--seed", type=int, default=AgentConfig.seed)
    args = parser.parse_args()
    agent_config = AgentConfig(
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
    )
    env_config = PointRobotConfig(
        max_steps=args.max_steps,
        observation_mode=args.observation_mode,
        goal_cue_steps=args.goal_cue_steps,
        seed=args.seed + 7,
    )
    return agent_config, env_config


if __name__ == "__main__":
    run(*parse_args())
