"""Toy learners rewritten to use `dynn.Net`."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from envs.grid_world import ACTIONS as GRID_ACTIONS
from envs.grid_world import GridWorld, GridWorldConfig
from models.common import argmax, clamp, mean_squared
from models.dynn_support import dynn


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


class _ContinuousRule:
    """Local ETLP-style readout update for the continuous toy."""

    def __init__(self, config: ContinuousToyConfig) -> None:
        self.config = config

    def initialize_traces(self, *, edge_block) -> dict[str, tuple[float, ...]]:
        return {"pre": tuple(0.0 for _ in range(edge_block.source_count))}

    def step(
        self,
        *,
        edge_block,
        traces,
        pre_activity,
        post_activity,
        learning_rate,
        modulation,
        node_states,
        step_index=0,
        return_weights=False,
    ):
        del learning_rate, modulation, step_index
        trace_state = traces or {"pre": tuple(0.0 for _ in range(edge_block.source_count))}
        prev_pre = tuple(float(value) for value in trace_state.get("pre", ()))
        next_pre = tuple(
            self.config.trace_decay * prev + float(activity)
            for prev, activity in zip(prev_pre, pre_activity, strict=True)
        )
        state = node_states.get(edge_block.target_node_set, {})
        voltage = tuple(float(value) for value in state.get("voltage", ()))
        post_factor = tuple(
            triangular_pseudo_derivative(value, self.config.threshold, self.config.pseudo_width)
            for value in voltage
        )
        teaching_signal = tuple(float(value) for value in post_activity)
        deltas = tuple(
            teaching_signal[target] * post_factor[target] * next_pre[source]
            for source, target in zip(edge_block.source_indices, edge_block.target_indices, strict=True)
        )
        next_weights = tuple(
            clamp(
                (float(weight) + self.config.lr * delta) * (1.0 - self.config.weight_decay),
                -3.0,
                3.0,
            )
            for weight, delta in zip(edge_block.weights, deltas, strict=True)
        )
        abs_deltas = tuple(
            abs(after - before)
            for before, after in zip(edge_block.weights, next_weights, strict=True)
        )
        result = {
            "traces": {"pre": next_pre},
            "weight_update_count": len(deltas),
            "mean_abs_weight_delta": (sum(abs_deltas) / len(abs_deltas)) if abs_deltas else 0.0,
            "max_abs_weight_delta": max(abs_deltas, default=0.0),
        }
        if return_weights:
            result["weights"] = next_weights
        return result


class DynnContinuousToy:
    """Continuous classification toy using `dynn` as the execution core."""

    def __init__(self, config: ContinuousToyConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.graph = dynn.build(
            {"id": "continuous-toy"},
            {
                "node_sets": [
                    {"id": "obs", "size": config.input_dim, "node_type": "linear"},
                    {
                        "id": "cls",
                        "size": config.n_classes,
                        "node_type": "linear",
                        "parameters": {"bias": [0.0] * config.n_classes},
                    },
                ],
                "edge_sets": [
                    {
                        "id": "obs_to_cls",
                        "source": {"node_set": "obs"},
                        "target": {"node_set": "cls"},
                        "representation": {
                            "kind": "explicit",
                            "edges": [
                                {
                                    "source": source,
                                    "target": target,
                                    "weight": rng.gauss(0.0, 0.18),
                                }
                                for target in range(config.n_classes)
                                for source in range(config.input_dim)
                            ],
                        },
                    }
                ],
                "ports": [
                    {"id": "obs", "node_set": "obs", "kind": "input"},
                    {"id": "cls", "node_set": "cls", "kind": "output"},
                ],
            },
        )
        self.net = dynn.Net(self.graph, plasticity=_ContinuousRule(config), learning_rate=config.lr)

    def predict(self, sequence: list[list[float]]) -> int:
        logits = self._run(sequence, label=None, learn=False)
        return argmax(logits)

    def train_one(self, sequence: list[list[float]], label: int) -> int:
        logits = self._run(sequence, label=label, learn=True)
        return argmax(logits)

    def weight_norm(self) -> float:
        weights = getattr(self.net, "_edge_weights", {}).get("obs_to_cls", ())
        return math.sqrt(sum(weight * weight for weight in weights))

    def _run(self, sequence: list[list[float]], label: int | None, learn: bool) -> list[float]:
        _reset_net_state(self.net, keep_traces=False)
        readout = [0.0 for _ in range(self.config.n_classes)]
        target = [0.0 for _ in range(self.config.n_classes)] if label is not None else None
        if target is not None and label is not None:
            target[label] = 1.0
        for analog_input in sequence:
            target_signal = [0.0 for _ in range(self.config.n_classes)]
            if learn and target is not None:
                probabilities = softmax(readout)
                target_signal = [goal - prob for goal, prob in zip(target, probabilities, strict=True)]
            output = self.net.step(
                {"obs": analog_input},
                outputs={"cls": target_signal},
                modulation=1.0 if learn and target is not None else 0.0,
            )
            logits = list(output.node("cls"))
            readout = [0.88 * prev + current for prev, current in zip(readout, logits, strict=True)]
        return readout


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


class _TransitionRule:
    """Action-conditioned local prediction update."""

    def __init__(self, config: CognitiveMapConfig, feature_dim: int) -> None:
        self.config = config
        self.feature_dim = feature_dim

    def initialize_traces(self, *, edge_block) -> dict[str, tuple[float, ...]]:
        return {"pre": tuple(0.0 for _ in range(edge_block.source_count))}

    def step(
        self,
        *,
        edge_block,
        traces,
        pre_activity,
        post_activity,
        learning_rate,
        modulation,
        node_states,
        step_index=0,
        return_weights=False,
    ):
        del learning_rate, modulation, step_index
        target_signal = tuple(float(value) for value in post_activity)
        if not target_signal:
            result = {
                "traces": traces or {"pre": tuple(0.0 for _ in range(edge_block.source_count))},
                "weight_update_count": 0,
                "mean_abs_weight_delta": 0.0,
                "max_abs_weight_delta": 0.0,
            }
            if return_weights:
                result["weights"] = tuple(edge_block.weights)
            return result
        trace_state = traces or {"pre": tuple(0.0 for _ in range(edge_block.source_count))}
        prev_pre = tuple(float(value) for value in trace_state.get("pre", ()))
        next_pre = tuple(
            self.config.trace_decay * prev + float(activity)
            for prev, activity in zip(prev_pre, pre_activity, strict=True)
        )
        state = node_states.get(edge_block.target_node_set, {})
        predicted = tuple(float(value) for value in state.get("voltage", ()))
        if len(predicted) != len(target_signal):
            predicted = tuple(0.0 for _ in range(len(target_signal)))
        error = tuple(
            target - output
            for target, output in zip(target_signal, predicted, strict=True)
        )
        deltas = tuple(
            error[target] * next_pre[source]
            for source, target in zip(edge_block.source_indices, edge_block.target_indices, strict=True)
        )
        next_weights = tuple(
            clamp(
                (float(weight) + self.config.lr * delta) * (1.0 - self.config.weight_decay),
                -2.0,
                2.0,
            )
            for weight, delta in zip(edge_block.weights, deltas, strict=True)
        )
        abs_deltas = tuple(
            abs(after - before)
            for before, after in zip(edge_block.weights, next_weights, strict=True)
        )
        result = {
            "traces": {"pre": next_pre},
            "weight_update_count": len(deltas),
            "mean_abs_weight_delta": (sum(abs_deltas) / len(abs_deltas)) if abs_deltas else 0.0,
            "max_abs_weight_delta": max(abs_deltas, default=0.0),
        }
        if return_weights:
            result["weights"] = next_weights
        return result


class DynnLocalTransitionLearner:
    """Local transition learner using one `dynn` graph for all actions."""

    def __init__(self, config: CognitiveMapConfig, rng: random.Random, feature_dim: int) -> None:
        self.config = config
        self.rng = rng
        self.feature_dim = feature_dim
        node_sets = [{"id": "state", "size": feature_dim, "node_type": "linear"}]
        edge_sets = []
        ports = [{"id": "state", "node_set": "state", "kind": "input"}]
        scale = 0.03 / math.sqrt(feature_dim)
        for action in GRID_ACTIONS:
            pred_id = f"pred_{action}"
            node_sets.append({"id": pred_id, "size": feature_dim, "node_type": "linear"})
            ports.append({"id": pred_id, "node_set": pred_id, "kind": "output"})
            edge_sets.append(
                {
                    "id": f"state_to_{pred_id}",
                    "source": {"node_set": "state"},
                    "target": {"node_set": pred_id},
                    "representation": {
                        "kind": "explicit",
                        "edges": [
                            {
                                "source": source,
                                "target": target,
                                "weight": rng.gauss(0.0, scale),
                            }
                            for target in range(feature_dim)
                            for source in range(feature_dim)
                        ],
                    },
                }
            )
        self.port_to_node = {action: f"pred_{action}" for action in GRID_ACTIONS}
        self.graph = dynn.build(
            {"id": "cognitive-map-toy"},
            {"node_sets": node_sets, "edge_sets": edge_sets, "ports": ports},
        )
        self.net = dynn.Net(
            self.graph,
            plasticity=_TransitionRule(config, feature_dim),
            learning_rate=config.lr,
        )

    def predict(self, state_code: list[float], action: str) -> list[float]:
        _reset_net_state(self.net, keep_traces=True)
        output = self.net.step(
            {"state": state_code},
            outputs={self.port_to_node[action]: [0.0] * self.feature_dim},
            modulation=0.0,
        )
        return list(output.node(self.port_to_node[action]))

    def learn(self, state_code: list[float], action: str, next_code: list[float]) -> float:
        _reset_net_state(self.net, keep_traces=True)
        output = self.net.step(
            {"state": state_code},
            outputs={self.port_to_node[action]: next_code},
            modulation=1.0,
        )
        predicted = list(output.node(self.port_to_node[action]))
        error = [target - output for target, output in zip(next_code, predicted, strict=True)]
        return mean_squared(error)

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


def evaluate_continuous_toy(model: DynnContinuousToy, task: ContinuousTemporalTask, step: int, samples: int) -> float:
    correct = 0
    for offset in range(samples):
        sequence, label = task.sample(step + offset)
        correct += int(model.predict(sequence) == label)
    return correct / samples


def train_continuous_toy(config: ContinuousToyConfig) -> None:
    rng = random.Random(config.seed)
    task = ContinuousTemporalTask(config, rng)
    model = DynnContinuousToy(config, rng)

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


def train_step_cognitive_map(world: GridWorld, learner: DynnLocalTransitionLearner, rng: random.Random, step: int) -> float:
    if step % 37 == 0:
        world.reset()
    state = world.state
    action = rng.choice(GRID_ACTIONS)
    state_code = world.encode(state)
    next_state = world.step(action)
    next_code = world.encode(next_state)
    return learner.learn(state_code, action, next_code)


def evaluate_cognitive_map(world: GridWorld, learner: DynnLocalTransitionLearner, rng: random.Random, config: CognitiveMapConfig) -> tuple[float, float, float]:
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
    learner = DynnLocalTransitionLearner(config, rng, world.config.feature_dim)

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


def triangular_pseudo_derivative(value: float, threshold: float, width: float) -> float:
    distance = abs(value - threshold)
    return max(0.20, 1.0 - distance / width)


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _reset_net_state(net, *, keep_traces: bool) -> None:
    weights = dict(getattr(net, "_edge_weights", {}))
    traces = dict(getattr(net, "_plasticity_traces", {})) if keep_traces else None
    net.reset()
    net._edge_weights = weights
    if traces is not None:
        net._plasticity_traces = traces
