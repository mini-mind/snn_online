"""Cognitive Map Learner + ETLP-like local prediction toy.

Run:
    python src/cognitive_map_etlp_toy.py

This toy combines two ideas used throughout the project:

1. Cognitive Map Learner:
   learn an action-conditioned transition structure, then plan with it.

2. ETLP / three-factor local update:
   update = pre_trace * prediction_error

The environment is a small gridworld with a wall and a gap. Observations are
continuous high-dimensional place codes, not one-hot table IDs. The learner
builds one local linear transition model per action:

    z_hat_next = W[action] @ pre_trace
    delta W[action][out][in] = lr * prediction_error[out] * pre_trace[in]

After online exploration, the model is decoded into a transition graph and used
for shortest-path planning to arbitrary goals.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass


ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class CMLConfig:
    grid_size: int = 5
    feature_dim: int = 40
    train_steps: int = 2500
    eval_every: int = 500
    eval_pairs: int = 0
    planning_horizon: int = 12
    lr: float = 0.12
    trace_decay: float = 0.15
    weight_decay: float = 0.0002
    noise: float = 0.0
    seed: int = 11


class GridWorld:
    def __init__(self, config: CMLConfig, rng: random.Random) -> None:
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
        centers = []
        for _ in range(self.config.feature_dim):
            centers.append((self.rng.uniform(0, self.config.grid_size - 1), self.rng.uniform(0, self.config.grid_size - 1)))

        codes: dict[tuple[int, int], list[float]] = {}
        sigma = max(1.0, self.config.grid_size / 3.0)
        for state in self.states:
            x, y = state
            code = []
            for center_x, center_y in centers:
                squared_distance = (x - center_x) ** 2 + (y - center_y) ** 2
                code.append(math.exp(-squared_distance / (2.0 * sigma * sigma)))
            code.extend([x / max(1, self.config.grid_size - 1), y / max(1, self.config.grid_size - 1), 1.0])
            codes[state] = normalize(code)
        self.config.feature_dim = len(next(iter(codes.values())))
        return codes


class LocalTransitionLearner:
    def __init__(self, config: CMLConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        scale = 0.03 / math.sqrt(config.feature_dim)
        self.weights = {
            action: [
                [rng.gauss(0.0, scale) for _ in range(config.feature_dim)]
                for _ in range(config.feature_dim)
            ]
            for action in ACTIONS
        }
        self.traces = {action: [0.0 for _ in range(config.feature_dim)] for action in ACTIONS}

    def predict(self, state_code: list[float], action: str) -> list[float]:
        matrix = self.weights[action]
        return [dot(row, state_code) for row in matrix]

    def learn(self, state_code: list[float], action: str, next_code: list[float]) -> float:
        trace = self.traces[action]
        for index, value in enumerate(state_code):
            trace[index] = self.config.trace_decay * trace[index] + value

        predicted = self.predict(trace, action)
        error = [target - output for target, output in zip(next_code, predicted)]
        matrix = self.weights[action]
        for output_index, error_value in enumerate(error):
            row = matrix[output_index]
            for input_index, trace_value in enumerate(trace):
                row[input_index] += self.config.lr * error_value * trace_value
                row[input_index] *= 1.0 - self.config.weight_decay
                row[input_index] = clamp(row[input_index], -2.0, 2.0)
        return mean_squared(error)

    def decoded_transition(self, world: GridWorld, state: tuple[int, int], action: str) -> tuple[int, int]:
        predicted_code = self.predict(world.state_codes[state], action)
        return world.decode(predicted_code)

    def learned_graph(self, world: GridWorld) -> dict[tuple[int, int], dict[str, tuple[int, int]]]:
        return {
            state: {action: self.decoded_transition(world, state, action) for action in ACTIONS}
            for state in world.states
        }


class Planner:
    def __init__(self, graph: dict[tuple[int, int], dict[str, tuple[int, int]]]) -> None:
        self.graph = graph

    def plan(self, start: tuple[int, int], goal: tuple[int, int], max_depth: int) -> list[str] | None:
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            state, path = queue.popleft()
            if state == goal:
                return path
            if len(path) >= max_depth:
                continue
            for action in ACTIONS:
                next_state = self.graph[state][action]
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, path + [action]))
        return None


def train_step(world: GridWorld, learner: LocalTransitionLearner, rng: random.Random, step: int) -> float:
    if step % 37 == 0:
        world.reset()
    state = world.state
    action = rng.choice(ACTIONS)
    state_code = world.encode(state)
    next_state = world.step(action)
    next_code = world.encode(next_state)
    return learner.learn(state_code, action, next_code)


def evaluate(world: GridWorld, learner: LocalTransitionLearner, rng: random.Random, config: CMLConfig) -> tuple[float, float, float]:
    transition_correct = 0
    transition_total = 0
    graph = learner.learned_graph(world)
    planner = Planner(graph)
    planning_success = 0
    path_ratio_sum = 0.0
    path_ratio_count = 0

    for state in world.states:
        for action in ACTIONS:
            predicted = graph[state][action]
            expected = world.transition(state, action)
            transition_correct += int(predicted == expected)
            transition_total += 1

    if config.eval_pairs <= 0:
        pairs = [(start, goal) for start in world.states for goal in world.states]
    else:
        pairs = [(rng.choice(world.states), rng.choice(world.states)) for _ in range(config.eval_pairs)]

    for start, goal in pairs:
        if start == goal:
            planning_success += 1
            path_ratio_sum += 1.0
            path_ratio_count += 1
            continue

        learned_path = planner.plan(start, goal, max_depth=config.planning_horizon)
        true_path = world.true_shortest_path(start, goal)
        if learned_path is None or true_path is None:
            continue

        state = start
        for action in learned_path:
            state = world.transition(state, action)
        if state == goal:
            planning_success += 1
            path_ratio_sum += len(true_path) / max(len(learned_path), 1)
            path_ratio_count += 1

    transition_accuracy = transition_correct / transition_total
    planning_success_rate = planning_success / len(pairs)
    path_efficiency = path_ratio_sum / max(1, path_ratio_count)
    return transition_accuracy, planning_success_rate, path_efficiency


def run(config: CMLConfig) -> None:
    rng = random.Random(config.seed)
    world = GridWorld(config, rng)
    learner = LocalTransitionLearner(config, rng)

    print("Cognitive Map + ETLP-like local prediction toy")
    print(
        f"seed={config.seed} grid={config.grid_size} states={len(world.states)} "
        f"feature_dim={config.feature_dim} train_steps={config.train_steps}"
    )
    print("rule: delta_w[action][out][in] = lr * prediction_error[out] * pre_trace[in]")
    print()

    transition_accuracy, planning_success, path_efficiency = evaluate(world, learner, rng, config)
    print(
        f"step=0 transition_acc={transition_accuracy:.3f} "
        f"planning_success={planning_success:.3f} path_efficiency={path_efficiency:.3f}"
    )

    error_window = 0.0
    for step in range(1, config.train_steps + 1):
        error_window += train_step(world, learner, rng, step)
        if step % config.eval_every == 0:
            transition_accuracy, planning_success, path_efficiency = evaluate(world, learner, rng, config)
            prediction_mse = error_window / config.eval_every
            print(
                f"step={step} prediction_mse={prediction_mse:.4f} "
                f"transition_acc={transition_accuracy:.3f} "
                f"planning_success={planning_success:.3f} "
                f"path_efficiency={path_efficiency:.3f}"
            )
            error_window = 0.0


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot(left, right) / (left_norm * right_norm)


def mean_squared(values: list[float]) -> float:
    return sum(value * value for value in values) / max(1, len(values))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_args() -> CMLConfig:
    parser = argparse.ArgumentParser(description="Run the Cognitive Map + ETLP-like toy.")
    parser.add_argument("--grid-size", type=int, default=CMLConfig.grid_size)
    parser.add_argument("--feature-dim", type=int, default=CMLConfig.feature_dim)
    parser.add_argument("--train-steps", type=int, default=CMLConfig.train_steps)
    parser.add_argument("--eval-every", type=int, default=CMLConfig.eval_every)
    parser.add_argument("--eval-pairs", type=int, default=CMLConfig.eval_pairs, help="planning pairs per eval; <=0 means all state pairs")
    parser.add_argument("--planning-horizon", type=int, default=CMLConfig.planning_horizon)
    parser.add_argument("--lr", type=float, default=CMLConfig.lr)
    parser.add_argument("--trace-decay", type=float, default=CMLConfig.trace_decay)
    parser.add_argument("--noise", type=float, default=CMLConfig.noise)
    parser.add_argument("--seed", type=int, default=CMLConfig.seed)
    args = parser.parse_args()
    return CMLConfig(
        grid_size=args.grid_size,
        feature_dim=args.feature_dim,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        eval_pairs=args.eval_pairs,
        planning_horizon=args.planning_horizon,
        lr=args.lr,
        trace_decay=args.trace_decay,
        noise=args.noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
