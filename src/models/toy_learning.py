"""Pure-Python toy learners for local online learning experiments."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from envs.grid_world import ACTIONS as GRID_ACTIONS
from envs.grid_world import GridWorld, GridWorldConfig
from models.common import argmax, clamp, mean_squared


@dataclass
class ContinuousToyConfig:
    """Configuration for the continuous-input toy."""

    input_dim: int = 6
    n_classes: int = 2
    seq_len: int = 24
    train_steps: int = 2000
    eval_every: int = 200
    eval_samples: int = 400
    lr: float = 0.035
    weight_decay: float = 0.0004
    membrane_decay: float = 0.82
    trace_decay: float = 0.88
    threshold: float = 1.0
    pseudo_width: float = 0.75
    noise: float = 0.18
    drift: float = 0.35
    seed: int = 7


class ContinuousTemporalTask:
    """Binary sequential classification task with slow drift."""

    def __init__(self, config: ContinuousToyConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.prototypes = self._build_prototypes()

    def sample(self, step: int) -> tuple[list[list[float]], int]:
        label = self.rng.randrange(self.config.n_classes)
        phase = 2.0 * math.pi * step / max(1, self.config.train_steps)
        drift_vec = [
            math.sin(phase),
            math.cos(phase),
            math.sin(phase + 0.8),
            math.cos(phase + 0.8),
            math.sin(phase + 1.6),
            math.cos(phase + 1.6),
        ]
        sequence: list[list[float]] = []
        for t, prototype_step in enumerate(self.prototypes[label]):
            drift_scale = self.config.drift * t / max(1, self.config.seq_len - 1)
            analog_input = []
            for feature, drift in zip(prototype_step, drift_vec, strict=True):
                value = feature + drift_scale * drift + self.rng.gauss(0.0, self.config.noise)
                analog_input.append(max(0.0, value))
            sequence.append(analog_input)
        return sequence, label

    def _build_prototypes(self) -> list[list[list[float]]]:
        class_zero: list[list[float]] = []
        class_one: list[list[float]] = []
        for index in range(self.config.seq_len):
            time = index / max(1, self.config.seq_len - 1)
            early_bump = math.exp(-((time - 0.30) ** 2) / 0.018)
            late_bump = math.exp(-((time - 0.70) ** 2) / 0.018)
            shared_wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * time)
            shared_context = 0.5 + 0.5 * math.cos(2.0 * math.pi * time)
            class_zero.append([1.2 * early_bump + 0.2, 0.25 * late_bump, time, 1.0 - time, shared_wave, shared_context])
            class_one.append([0.25 * early_bump, 1.2 * late_bump + 0.2, time, 1.0 - time, shared_wave, shared_context])
        return [class_zero, class_one]


class ContinuousLocalClassifier:
    """ETLP-like readout with local pre traces and class teaching signals."""

    def __init__(self, config: ContinuousToyConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.weights = [
            [rng.gauss(0.0, 0.18) for _ in range(config.input_dim)]
            for _ in range(config.n_classes)
        ]
        self.bias = [0.0 for _ in range(config.n_classes)]

    def predict(self, sequence: list[list[float]]) -> int:
        logits = self._run(sequence, label=None, learn=False)
        return argmax(logits)

    def train_one(self, sequence: list[list[float]], label: int) -> int:
        logits = self._run(sequence, label=label, learn=True)
        return argmax(logits)

    def weight_norm(self) -> float:
        return math.sqrt(sum(weight * weight for row in self.weights for weight in row))

    def _run(self, sequence: list[list[float]], label: int | None, learn: bool) -> list[float]:
        pre_trace = [0.0 for _ in range(self.config.input_dim)]
        voltage = [0.0 for _ in range(self.config.n_classes)]
        readout = [0.0 for _ in range(self.config.n_classes)]
        target = [0.0 for _ in range(self.config.n_classes)] if label is not None else None
        if target is not None and label is not None:
            target[label] = 1.0
        for analog_input in sequence:
            pre_trace = [
                self.config.trace_decay * previous + value
                for previous, value in zip(pre_trace, analog_input, strict=True)
            ]
            voltage = [
                self.config.membrane_decay * previous
                + sum(weight * value for weight, value in zip(row, analog_input, strict=True))
                + self.bias[class_index]
                for class_index, (previous, row) in enumerate(zip(voltage, self.weights, strict=True))
            ]
            readout = [0.88 * prev + current for prev, current in zip(readout, voltage, strict=True)]
            if learn and target is not None:
                probabilities = softmax(readout)
                teaching = [
                    goal - probability
                    for goal, probability in zip(target, probabilities, strict=True)
                ]
                self._apply_teaching(pre_trace, voltage, teaching)
        return readout

    def _apply_teaching(
        self,
        pre_trace: list[float],
        voltage: list[float],
        teaching: list[float],
    ) -> None:
        for class_index in range(self.config.n_classes):
            post_factor = triangular_pseudo_derivative(
                voltage[class_index],
                self.config.threshold,
                self.config.pseudo_width,
            )
            for input_index, trace in enumerate(pre_trace):
                delta = teaching[class_index] * post_factor * trace
                weight = self.weights[class_index][input_index]
                next_weight = (weight + self.config.lr * delta) * (1.0 - self.config.weight_decay)
                self.weights[class_index][input_index] = clamp(next_weight, -3.0, 3.0)


@dataclass
class CognitiveMapConfig:
    """Configuration for the cognitive-map toy."""

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


class LocalTransitionLearner:
    """Action-conditioned local predictor with one weight matrix per action."""

    def __init__(self, config: CognitiveMapConfig, rng: random.Random, feature_dim: int) -> None:
        self.config = config
        self.rng = rng
        self.feature_dim = feature_dim
        scale = 0.03 / math.sqrt(feature_dim)
        self.weights = {
            action: [
                [rng.gauss(0.0, scale) for _ in range(feature_dim)]
                for _ in range(feature_dim)
            ]
            for action in GRID_ACTIONS
        }
        self.pre_trace = [0.0 for _ in range(feature_dim)]

    def predict(self, state_code: list[float], action: str) -> list[float]:
        return [
            sum(weight * value for weight, value in zip(row, state_code, strict=True))
            for row in self.weights[action]
        ]

    def learn(self, state_code: list[float], action: str, next_code: list[float]) -> float:
        self.pre_trace = [
            self.config.trace_decay * previous + value
            for previous, value in zip(self.pre_trace, state_code, strict=True)
        ]
        predicted = self.predict(state_code, action)
        errors = [
            target - output
            for target, output in zip(next_code, predicted, strict=True)
        ]
        action_weights = self.weights[action]
        for output_index, error in enumerate(errors):
            row = action_weights[output_index]
            clipped_error = clamp(error, -1.0, 1.0)
            for input_index, trace in enumerate(self.pre_trace):
                next_weight = (row[input_index] + self.config.lr * clipped_error * trace) * (
                    1.0 - self.config.weight_decay
                )
                row[input_index] = clamp(next_weight, -2.0, 2.0)
        return mean_squared(errors)

    def decoded_transition(self, world: GridWorld, state: tuple[int, int], action: str) -> tuple[int, int]:
        predicted_code = self.predict(world.state_codes[state], action)
        return world.decode(predicted_code)

    def learned_graph(self, world: GridWorld) -> dict[tuple[int, int], dict[str, tuple[int, int]]]:
        return {
            state: {action: self.decoded_transition(world, state, action) for action in GRID_ACTIONS}
            for state in world.states
        }


class Planner:
    """Breadth-first planner over the learned graph."""

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
            for action in GRID_ACTIONS:
                next_state = self.graph[state][action]
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, path + [action]))
        return None


def evaluate_continuous_toy(model: ContinuousLocalClassifier, task: ContinuousTemporalTask, step: int, samples: int) -> float:
    correct = 0
    for offset in range(samples):
        sequence, label = task.sample(step + offset)
        correct += int(model.predict(sequence) == label)
    return correct / samples


def train_continuous_toy(config: ContinuousToyConfig) -> None:
    rng = random.Random(config.seed)
    task = ContinuousTemporalTask(config, rng)
    model = ContinuousLocalClassifier(config, rng)

    print("ETLP-like continuous-input toy")
    print(f"seed={config.seed} train_steps={config.train_steps} seq_len={config.seq_len}")
    print("rule: delta_w = lr * pre_trace * post_membrane_factor * teaching_signal")
    print()

    initial_accuracy = evaluate_continuous_toy(model, task, step=0, samples=config.eval_samples)
    print(f"step=0 eval_accuracy={initial_accuracy:.3f}")

    online_correct = 0
    window_correct = 0
    for step in range(1, config.train_steps + 1):
        sequence, label = task.sample(step)
        prediction = model.train_one(sequence, label)
        online_correct += int(prediction == label)
        window_correct += int(prediction == label)
        if step % config.eval_every == 0:
            eval_accuracy = evaluate_continuous_toy(model, task, step=step, samples=config.eval_samples)
            online_accuracy = online_correct / step
            window_accuracy = window_correct / config.eval_every
            print(
                f"step={step} "
                f"online_acc={online_accuracy:.3f} "
                f"window_acc={window_accuracy:.3f} "
                f"eval_accuracy={eval_accuracy:.3f} "
                f"weight_norm={model.weight_norm():.3f}"
            )
            window_correct = 0


def train_step_cognitive_map(world: GridWorld, learner: LocalTransitionLearner, rng: random.Random, step: int) -> float:
    if step % 37 == 0:
        world.reset()
    state = world.state
    action = rng.choice(GRID_ACTIONS)
    state_code = world.encode(state)
    next_state = world.step(action)
    next_code = world.encode(next_state)
    return learner.learn(state_code, action, next_code)


def evaluate_cognitive_map(world: GridWorld, learner: LocalTransitionLearner, rng: random.Random, config: CognitiveMapConfig) -> tuple[float, float, float]:
    transition_correct = 0
    transition_total = 0
    graph = learner.learned_graph(world)
    planner = Planner(graph)
    planning_success = 0
    path_ratio_sum = 0.0
    path_ratio_count = 0

    for state in world.states:
        for action in GRID_ACTIONS:
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

    return (
        transition_correct / transition_total,
        planning_success / len(pairs),
        path_ratio_sum / max(1, path_ratio_count),
    )


def train_cognitive_map(config: CognitiveMapConfig) -> None:
    validate_cognitive_map_config(config)
    rng = random.Random(config.seed)
    world = GridWorld(
        GridWorldConfig(
            grid_size=config.grid_size,
            feature_dim=config.feature_dim,
            noise=config.noise,
            seed=config.seed,
        ),
        rng,
    )
    learner = LocalTransitionLearner(config, rng, world.config.feature_dim)

    print("Cognitive Map + ETLP-like local prediction toy")
    print(
        f"seed={config.seed} grid={config.grid_size} states={len(world.states)} "
        f"feature_dim={world.config.feature_dim} train_steps={config.train_steps}"
    )
    print("rule: delta_w[action][out][in] = lr * prediction_error[out] * pre_trace[in]")
    print()

    transition_accuracy, planning_success, path_efficiency = evaluate_cognitive_map(world, learner, rng, config)
    print(
        f"step=0 transition_acc={transition_accuracy:.3f} "
        f"planning_success={planning_success:.3f} path_efficiency={path_efficiency:.3f}"
    )

    error_window = 0.0
    for step in range(1, config.train_steps + 1):
        error_window += train_step_cognitive_map(world, learner, rng, step)
        if step % config.eval_every == 0:
            transition_accuracy, planning_success, path_efficiency = evaluate_cognitive_map(world, learner, rng, config)
            prediction_mse = error_window / config.eval_every
            print(
                f"step={step} prediction_mse={prediction_mse:.4f} "
                f"transition_acc={transition_accuracy:.3f} "
                f"planning_success={planning_success:.3f} "
                f"path_efficiency={path_efficiency:.3f}"
            )
            error_window = 0.0


def validate_cognitive_map_config(config: CognitiveMapConfig) -> None:
    if config.grid_size <= 1:
        raise ValueError(f"grid_size must be greater than 1, got {config.grid_size}")
    if config.feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {config.feature_dim}")
    if config.train_steps <= 0:
        raise ValueError(f"train_steps must be positive, got {config.train_steps}")
    if config.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {config.eval_every}")
    if config.planning_horizon <= 0:
        raise ValueError(f"planning_horizon must be positive, got {config.planning_horizon}")


def triangular_pseudo_derivative(value: float, threshold: float, width: float) -> float:
    distance = abs(value - threshold)
    return max(0.20, 1.0 - distance / width)


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]
