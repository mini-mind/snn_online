"""无第三方依赖的二维点机器人控制环境。

环境只保留最小控制闭环所需的元素：

1. 位置、速度和目标点；
2. 有阻尼的离散动作加速度；
3. 兼顾“靠近目标”和“动作代价”的奖励函数。

它不追求物理逼真度，而是为在线学习算法提供一个足够紧凑、可快速迭代的
控制任务。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


ACTIONS = ["up", "down", "left", "right", "stay"]
ACTION_ACCEL = {
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "stay": (0.0, 0.0),
}


@dataclass
class PointRobotConfig:
    """点机器人环境超参数。"""

    world_size: float = 1.0
    max_steps: int = 60
    acceleration: float = 0.06
    velocity_decay: float = 0.82
    max_speed: float = 0.18
    goal_radius: float = 0.12
    action_cost: float = 0.006
    seed: int = 23


class PointRobotEnv:
    """二维平面上的点机器人任务。

    状态由机器人位置、速度以及目标位置组成。每一步接收一个离散动作，
    将其转成加速度，再结合速度衰减更新动力学。环境边界使用简单的夹紧与
    反弹处理，以避免状态无限发散。
    """

    def __init__(self, config: PointRobotConfig, rng: random.Random | None = None) -> None:
        """创建环境并立即采样一个初始状态。"""
        self.config = config
        self.rng = rng or random.Random(config.seed)
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.steps = 0
        self.reset()

    def reset(self) -> list[float]:
        """重置机器人与目标位置，并返回初始观测。"""
        self.x = self.rng.uniform(-0.8, 0.8)
        self.y = self.rng.uniform(-0.8, 0.8)
        self.vx = 0.0
        self.vy = 0.0
        self.goal_x = self.rng.uniform(-0.85, 0.85)
        self.goal_y = self.rng.uniform(-0.85, 0.85)
        while self.distance_to_goal() < 0.45:
            self.goal_x = self.rng.uniform(-0.85, 0.85)
            self.goal_y = self.rng.uniform(-0.85, 0.85)
        self.steps = 0
        return self.observation()

    def step(self, action_index: int) -> tuple[list[float], float, bool]:
        """执行一步环境转移。

        返回:
            一个三元组 `(observation, reward, done)`。
        """
        action = ACTIONS[action_index]
        old_distance = self.distance_to_goal()
        ax, ay = ACTION_ACCEL[action]

        # 动力学模型非常简单：速度先衰减，再叠加动作加速度，最后做最大速度裁剪。
        self.vx = self.config.velocity_decay * self.vx + self.config.acceleration * ax
        self.vy = self.config.velocity_decay * self.vy + self.config.acceleration * ay
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        # 位置更新后做边界约束；触边时给速度一个反向衰减，模拟简化碰撞。
        self.x = clamp(self.x + self.vx, -self.config.world_size, self.config.world_size)
        self.y = clamp(self.y + self.vy, -self.config.world_size, self.config.world_size)
        if abs(self.x) >= self.config.world_size:
            self.vx *= -0.2
        if abs(self.y) >= self.config.world_size:
            self.vy *= -0.2

        self.steps += 1
        new_distance = self.distance_to_goal()
        reached = new_distance <= self.config.goal_radius
        timeout = self.steps >= self.config.max_steps
        # 奖励以“向目标前进的距离改善”为主，同时引入动作成本，避免无意义抖动。
        reward = (old_distance - new_distance) * 4.0 - self.config.action_cost
        if action == "stay":
            reward -= self.config.action_cost
        if reached:
            reward += 1.5
        if timeout and not reached:
            reward -= 0.4
        return self.observation(), reward, reached or timeout

    def observation(self) -> list[float]:
        """构造供智能体使用的连续观测向量。"""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)
        return [
            self.x,
            self.y,
            self.vx / self.config.max_speed,
            self.vy / self.config.max_speed,
            self.goal_x,
            self.goal_y,
            dx,
            dy,
            distance / (2.0 * self.config.world_size),
            1.0,
        ]

    def distance_to_goal(self) -> float:
        """计算当前位置到目标点的欧氏距离。"""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        return math.sqrt(dx * dx + dy * dy)


def clamp(value: float, low: float, high: float) -> float:
    """将标量限制在 `[low, high]` 区间内。"""
    return max(low, min(high, value))
