"""R-SNN 点机器人控制玩具实验。

运行方式:
    python src/rsnn_point_robot_toy.py

该智能体组合了四个最小部件：

1. 循环脉冲隐状态；
2. 用预测误差训练的局部线性 world model 读出头；
3. 用 TD 误差训练的动作价值读出头；
4. 基于一步前瞻的 model-based 动作打分。

这不是基准实现，而是一个用于验证“在线局部学习 + 简化规划”能否形成控制
闭环的实验脚手架。
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
    """点机器人智能体训练配置。"""

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
    """基于循环脉冲隐状态的点机器人控制智能体。

    结构上分三层：

    1. R-SNN 负责把时序观测压缩成动态特征；
    2. value head 估计每个动作的动作价值；
    3. world model head 预测执行动作后的下一步观测。

    动作选择时同时参考价值估计与一步模型预测，形成简化的 model-based control。
    """

    def __init__(self, obs_dim: int, n_actions: int, config: AgentConfig, rng: random.Random) -> None:
        """初始化 R-SNN、本地 world model 读出头和价值读出头。"""
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
        """重置 episode 内的循环隐状态。"""
        self.rsnn.reset_state()

    def observe(self, observation: list[float], modulation: float = 0.0) -> list[float]:
        """将环境观测送入 R-SNN，并返回归一化特征。"""
        return normalize_features(self.rsnn.step(observation, modulation=modulation))

    def q_values(self, features: list[float]) -> list[float]:
        """根据当前隐特征计算每个动作的价值。"""
        return [dot(weights, features) for weights in self.value_weights]

    def predict_next(self, observation: list[float], features: list[float], action: int) -> list[float]:
        """预测某个动作执行后的下一步观测。"""
        predicted_delta = [dot(row, features) for row in self.model_weights[action]]
        predicted = [value + delta for value, delta in zip(observation, predicted_delta)]
        return clamp_observation(predicted)

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
        predicted = self.predict_next(observation, features, action)
        errors = [target - output for target, output in zip(next_observation, predicted)]

        # 这里使用最直接的局部线性误差校正：每个输出维都按误差和隐藏特征更新。
        for output_index, error in enumerate(errors):
            row = self.model_weights[action][output_index]
            for hidden_index, feature in enumerate(features):
                row[hidden_index] += self.config.lr_model * clamp(error, -1.0, 1.0) * feature
                row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)
        return mean_squared(errors)

    def learn_value(self, features: list[float], action: int, td_error: float) -> None:
        """按 TD 误差更新指定动作的价值读出头。"""
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


def evaluate_agent(agent: RSNNPointRobotAgent, config: AgentConfig, env_config: PointRobotConfig, seed: int) -> tuple[float, float, float]:
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


def run(config: AgentConfig, env_config: PointRobotConfig) -> None:
    """运行训练主循环并定期输出评估指标。"""
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

    # 窗口统计用于观察最近阶段的训练趋势，避免仅看全局平均掩盖退化。
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
    """将特征向量归一化，控制不同 episode 之间的尺度波动。"""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def clamp_observation(values: list[float]) -> list[float]:
    """对预测观测做简单裁剪，避免 world model 输出失控。"""
    if len(values) < 10:
        return values
    clipped = list(values)
    for index in range(8):
        clipped[index] = clamp(clipped[index], -1.5, 1.5)
    clipped[8] = clamp(clipped[8], 0.0, 1.5)
    clipped[9] = 1.0
    return clipped

def epsilon_for_episode(config: AgentConfig, episode: int) -> float:
    """按线性退火策略计算当前 episode 的探索率。"""
    progress = min(1.0, episode / max(1, config.episodes * 0.75))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def argmax(values: list[float]) -> int:
    """返回最大值对应索引。"""
    return max(range(len(values)), key=lambda index: values[index])


def mean_squared(values: list[float]) -> float:
    """计算均方值。"""
    return sum(value * value for value in values) / max(1, len(values))


def parse_args() -> tuple[AgentConfig, PointRobotConfig]:
    """解析命令行参数并构造智能体与环境配置。"""
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
