"""Minimal point-robot control environment."""

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
TASK_FAMILY = "point_robot"


@dataclass
class PointRobotConfig:
    """Point-robot task configuration."""

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

    def task_metadata(self) -> dict[str, str]:
        """Stable task labels for summaries and JSONL artifacts."""
        return {
            "task_family": TASK_FAMILY,
            "benchmark_id": point_robot_benchmark_id(
                self.observation_mode,
                self.goal_cue_steps,
                self.max_steps,
            ),
            "observability_level": point_robot_observability_level(
                self.observation_mode,
                self.goal_cue_steps,
            ),
            "horizon_level": point_robot_horizon_level(self.max_steps),
            "reward_level": "dense_reward",
            "dynamics_level": "deterministic_dynamics",
            "goal_structure_level": "single_random_goal",
            "action_level": "discrete_actions",
            "distribution_level": "train_eval_same",
        }


def point_robot_observability_level(observation_mode: str, goal_cue_steps: int) -> str:
    if observation_mode == "full":
        return "full"
    if observation_mode == "partial_goal_cue":
        return f"partial_goal_cue_{goal_cue_steps}"
    raise ValueError(f"unsupported observation_mode: {observation_mode}")


def point_robot_horizon_level(max_steps: int) -> str:
    if max_steps <= 20:
        return f"short_horizon_{max_steps}"
    if max_steps <= 60:
        return f"medium_horizon_{max_steps}"
    return f"long_horizon_{max_steps}"


def point_robot_benchmark_id(
    observation_mode: str,
    goal_cue_steps: int,
    max_steps: int,
) -> str:
    observability_level = point_robot_observability_level(observation_mode, goal_cue_steps)
    return f"{TASK_FAMILY}_{observability_level}_h{max_steps}"


class PointRobotEnv:
    """Continuous 2D point robot with discrete actions."""

    def __init__(self, config: PointRobotConfig, rng: random.Random | None = None) -> None:
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

    def task_metadata(self) -> dict[str, str]:
        return self.config.task_metadata()

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
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def goal_direction_visible(self) -> bool:
        return self.steps < self.config.goal_cue_steps


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
