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
    """点机器人环境超参数。

    属性:
        world_size: 正方形世界边界的一半边长。
        max_steps: 单个 episode 最多执行的动作步数。
        acceleration: 离散动作映射到速度增量时的加速度尺度。
        velocity_decay: 每一步的速度衰减系数。
        max_speed: 允许的最大速度模长。
        goal_radius: 视为到达目标的距离阈值。
        action_cost: 每一步动作的固定代价。
        observation_mode: 观测模式，支持 `full` 和 `partial_goal_cue`。
        goal_cue_steps: 在部分可观测模式下，前多少步保留目标方向提示。
        seed: 默认随机种子。
    """

    world_size: float = 1.0
    max_steps: int = 60
    acceleration: float = 0.06
    velocity_decay: float = 0.82
    max_speed: float = 0.18
    goal_radius: float = 0.12
    action_cost: float = 0.006
    observation_mode: str = "full"
    goal_cue_steps: int = 6
    seed: int = 23


class PointRobotEnv:
    """二维平面上的点机器人任务。

    状态由机器人位置、速度以及目标位置组成。每一步接收一个离散动作，
    将其转成加速度，再结合速度衰减更新动力学。环境边界使用简单的夹紧与
    反弹处理，以避免状态无限发散。
    """

    def __init__(self, config: PointRobotConfig, rng: random.Random | None = None) -> None:
        """创建环境并立即采样一个初始状态。"""
        if config.observation_mode not in {"full", "partial_goal_cue"}:
            raise ValueError(f"unsupported observation_mode: {config.observation_mode}")
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
        """构造供智能体使用的连续观测向量。

        `full` 模式提供完整目标坐标与相对位移，属于近似全可观测任务。

        `partial_goal_cue` 模式只在 episode 前 `goal_cue_steps` 步提供目标相对方向
        提示；之后只保留自位置、自速度、步进进度和目标距离。这样智能体必须在
        循环状态里保留“目标大致在哪里”的记忆，才能持续导航。
        """
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)
        normalized_distance = distance / (2.0 * self.config.world_size)
        velocity_x = self.vx / self.config.max_speed
        velocity_y = self.vy / self.config.max_speed
        if self.config.observation_mode == "full":
            return [
                self.x,
                self.y,
                velocity_x,
                velocity_y,
                self.goal_x,
                self.goal_y,
                dx,
                dy,
                normalized_distance,
                1.0,
            ]

        goal_visible = self.goal_direction_visible()
        cue_dx = dx if goal_visible else 0.0
        cue_dy = dy if goal_visible else 0.0
        step_progress = self.steps / max(1, self.config.max_steps)
        return [
            self.x,
            self.y,
            velocity_x,
            velocity_y,
            cue_dx,
            cue_dy,
            1.0 if goal_visible else 0.0,
            step_progress,
            normalized_distance,
            1.0,
        ]

    def distance_to_goal(self) -> float:
        """计算当前位置到目标点的欧氏距离。"""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def goal_direction_visible(self) -> bool:
        """返回当前时间步是否仍向智能体暴露目标方向提示。"""
        return self.steps < self.config.goal_cue_steps


def clamp(value: float, low: float, high: float) -> float:
    """将标量限制在 `[low, high]` 区间内。"""
    return max(low, min(high, value))
