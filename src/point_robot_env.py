"""A dependency-free 2D point robot control environment."""

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
    world_size: float = 1.0
    max_steps: int = 60
    acceleration: float = 0.06
    velocity_decay: float = 0.82
    max_speed: float = 0.18
    goal_radius: float = 0.12
    action_cost: float = 0.006
    seed: int = 23


class PointRobotEnv:
    def __init__(self, config: PointRobotConfig, rng: random.Random | None = None) -> None:
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
        action = ACTIONS[action_index]
        old_distance = self.distance_to_goal()
        ax, ay = ACTION_ACCEL[action]
        self.vx = self.config.velocity_decay * self.vx + self.config.acceleration * ax
        self.vy = self.config.velocity_decay * self.vy + self.config.acceleration * ay
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            self.vx *= scale
            self.vy *= scale

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
        reward = (old_distance - new_distance) * 4.0 - self.config.action_cost
        if action == "stay":
            reward -= self.config.action_cost
        if reached:
            reward += 1.5
        if timeout and not reached:
            reward -= 0.4
        return self.observation(), reward, reached or timeout

    def observation(self) -> list[float]:
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
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        return math.sqrt(dx * dx + dy * dy)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
