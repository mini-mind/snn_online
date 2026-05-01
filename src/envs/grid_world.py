"""Grid-world environment for the cognitive-map toy."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from models.common import dot, l2_normalize


ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class GridWorldConfig:
    """Grid-world task configuration."""

    grid_size: int = 5
    feature_dim: int = 40
    noise: float = 0.0
    seed: int = 11


class GridWorld:
    """Grid world with a vertical wall and one gap."""

    def __init__(self, config: GridWorldConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.obstacles = self._build_obstacles(config.grid_size)
        self.states = [
            (x, y)
            for y in range(config.grid_size)
            for x in range(config.grid_size)
            if (x, y) not in self.obstacles
        ]
        self.state_set = set(self.states)
        self.state_codes = self._build_place_codes()
        self.state = self.rng.choice(self.states)

    def reset(self) -> tuple[int, int]:
        self.state = self.rng.choice(self.states)
        return self.state

    def step(self, action: str) -> tuple[int, int]:
        self.state = self.transition(self.state, action)
        return self.state

    def transition(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        dx, dy = ACTION_DELTAS[action]
        candidate = (state[0] + dx, state[1] + dy)
        if candidate in self.state_set:
            return candidate
        return state

    def encode(self, state: tuple[int, int]) -> list[float]:
        code = self.state_codes[state]
        if self.config.noise <= 0.0:
            return list(code)
        return [max(0.0, value + self.rng.gauss(0.0, self.config.noise)) for value in code]

    def decode(self, code: list[float]) -> tuple[int, int]:
        return max(self.states, key=lambda state: cosine(code, self.state_codes[state]))

    def true_shortest_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[str] | None:
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            state, path = queue.popleft()
            if state == goal:
                return path
            for action in ACTIONS:
                next_state = self.transition(state, action)
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, path + [action]))
        return None

    def _build_obstacles(self, size: int) -> set[tuple[int, int]]:
        if size < 5:
            return set()
        wall_x = size // 2
        gap_y = size // 2
        return {(wall_x, y) for y in range(size) if y != gap_y}

    def _build_place_codes(self) -> dict[tuple[int, int], list[float]]:
        centers = [
            (
                self.rng.uniform(0, self.config.grid_size - 1),
                self.rng.uniform(0, self.config.grid_size - 1),
            )
            for _ in range(self.config.feature_dim)
        ]
        codes: dict[tuple[int, int], list[float]] = {}
        sigma = max(1.0, self.config.grid_size / 3.0)
        for state in self.states:
            x, y = state
            code = []
            for center_x, center_y in centers:
                squared_distance = (x - center_x) ** 2 + (y - center_y) ** 2
                code.append(math.exp(-squared_distance / (2.0 * sigma * sigma)))
            code.extend(
                [
                    x / max(1, self.config.grid_size - 1),
                    y / max(1, self.config.grid_size - 1),
                    1.0,
                ]
            )
            codes[state] = l2_normalize(code)
        self.config.feature_dim = len(next(iter(codes.values())))
        return codes


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot(left, right) / (left_norm * right_norm)
