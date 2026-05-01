"""Thin `dynn` wrapper for recurrent spiking networks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from models.common import clamp
from models.dynn_support import dynn


@dataclass
class RSNNConfig:
    """Recurrent spiking network configuration."""

    input_dim: int
    n_neurons: int = 48
    n_layers: int = 1
    grid_width: int = 0
    grid_height: int = 0
    input_grid_width: int = 0
    input_grid_height: int = 0
    recurrent_radius: int = 1
    recurrent_degree: int = 4
    interlayer_radius: int = 1
    interlayer_degree: int = 4
    neuron_model: str = "lif"
    membrane_decay: float = 0.86
    trace_decay: float = 0.90
    threshold: float = 1.0
    recurrent_scale: float = 0.22
    input_scale: float = 0.65
    bias_scale: float = 0.18
    plastic_lr: float = 0.0008
    weight_decay: float = 0.00005
    randomize_intrinsics: bool = True
    membrane_decay_jitter: float = 0.035
    threshold_jitter: float = 0.12
    izh_a: float = 0.02
    izh_b: float = 0.20
    izh_c: float = -65.0
    izh_d: float = 8.0
    izh_a_jitter: float = 0.008
    izh_b_jitter: float = 0.040
    izh_c_jitter: float = 4.0
    izh_d_jitter: float = 2.5
    izh_dt: float = 0.5
    izh_substeps: int = 2
    izh_spike_threshold: float = 30.0
    izh_spike_threshold_jitter: float = 4.0
    izh_input_gain: float = 8.0
    izh_input_gain_jitter: float = 1.2
    seed: int = 13


class _ThreeFactorRule:
    """Local recurrent plasticity used by the point-robot experiment."""

    def __init__(
        self,
        config: RSNNConfig,
        layer_hidden_ids: list[str],
        layer_thresholds: dict[str, tuple[float, ...]],
    ) -> None:
        self.config = config
        self.layer_hidden_ids = tuple(layer_hidden_ids)
        self.trace_decay = float(config.trace_decay)
        self.weight_decay = float(config.weight_decay)
        self._threshold_map = dict(layer_thresholds)
        self._width_map = {
            node_set_id: _plastic_width(config)
            for node_set_id in self.layer_hidden_ids
        }

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
        del post_activity, learning_rate, step_index
        trace_state = traces or {"pre": tuple(0.0 for _ in range(edge_block.source_count))}
        prev_pre = tuple(float(value) for value in trace_state.get("pre", ()))
        if len(prev_pre) != edge_block.source_count:
            prev_pre = tuple(0.0 for _ in range(edge_block.source_count))
        next_pre = tuple(
            self.trace_decay * prev + float(activity)
            for prev, activity in zip(prev_pre, pre_activity, strict=True)
        )
        if edge_block.source_node_set != edge_block.target_node_set:
            result: dict[str, object] = {
                "traces": {"pre": next_pre},
                "weight_update_count": 0,
                "mean_abs_weight_delta": 0.0,
                "max_abs_weight_delta": 0.0,
            }
            if return_weights:
                result["weights"] = tuple(edge_block.weights)
            return result

        state = node_states.get(edge_block.target_node_set, {})
        post_voltage = tuple(float(value) for value in state.get("voltage", ()))
        thresholds = self._threshold_map.get(edge_block.target_node_set, ())
        width = self._width_map.get(edge_block.target_node_set, _plastic_width(self.config))
        post_factor = tuple(
            triangular_pseudo_derivative(voltage, threshold=threshold, width=width)
            for voltage, threshold in zip(post_voltage, thresholds, strict=True)
        )
        clipped_modulation = clamp(float(modulation), -1.0, 1.0)
        deltas = tuple(
            clipped_modulation * post_factor[target] * next_pre[source]
            for source, target in zip(edge_block.source_indices, edge_block.target_indices, strict=True)
        )
        next_weights = tuple(
            clamp(
                (float(weight) + self.config.plastic_lr * delta) * (1.0 - self.weight_decay),
                -1.5,
                1.5,
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


class DynnRecurrentSpikingNetwork:
    """Multi-layer recurrent spiking feature extractor backed by `dynn`."""

    def __init__(self, config: RSNNConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random(config.seed)
        self.grid_width, self.grid_height = resolve_grid_shape(
            config.n_neurons,
            config.grid_width,
            config.grid_height,
        )
        self.layer_ids = [f"hidden_{index}" for index in range(config.n_layers)]
        topology = _build_topology(config, self.rng, self.grid_width, self.grid_height)
        layer_parameters = _layer_parameters_from_topology(topology, self.layer_ids)
        self._layer_thresholds = _layer_thresholds_from_parameters(config, layer_parameters)
        self._layer_reset_values = _layer_reset_values_from_parameters(config, layer_parameters)
        self.graph = dynn.build(
            {"id": f"{config.neuron_model}-closed-loop-rsnn"},
            topology,
            seed=config.seed,
        )
        self.rule = _ThreeFactorRule(config, self.layer_ids, self._layer_thresholds)
        self.net = dynn.Net(
            self.graph,
            plasticity=self.rule,
            learning_rate=config.plastic_lr,
        )
        self._spike_traces = {
            layer_id: [0.0 for _ in range(config.n_neurons)]
            for layer_id in self.layer_ids
        }
        self._features = [0.0 for _ in range(self.feature_dim())]

    def reset_state(self) -> None:
        self.net.reset()
        for layer_id in self.layer_ids:
            self._spike_traces[layer_id] = [0.0 for _ in range(self.config.n_neurons)]
        self._features = [0.0 for _ in range(self.feature_dim())]

    def step(self, inputs: list[float]) -> list[float]:
        output = self.net.step({"obs": list(inputs)}, modulation=0.0)
        self._features = self._read_features(output)
        return list(self._features)

    def apply_recurrent_modulation(self, modulation: float) -> None:
        self.net.apply_plasticity(modulation)

    def features(self) -> list[float]:
        return list(self._features)

    def feature_dim(self) -> int:
        return self.config.n_neurons * self.config.n_layers

    def _read_features(self, output) -> list[float]:
        dynamics_state = getattr(self.net, "_dynamics_state", {})
        features: list[float] = []
        for layer_id in self.layer_ids:
            state = dynamics_state.get(layer_id, {}) if isinstance(dynamics_state, dict) else {}
            spikes = _float_series(
                output.node(layer_id) or state.get("activity", ()),
                self.config.n_neurons,
                0.0,
            )
            voltage = _float_series(state.get("voltage", ()), self.config.n_neurons, 0.0)
            prev_trace = self._spike_traces[layer_id]
            next_trace = [
                self.config.trace_decay * previous + spike
                for previous, spike in zip(prev_trace, spikes, strict=True)
            ]
            self._spike_traces[layer_id] = next_trace
            if self.config.neuron_model == "lif":
                thresholds = self._layer_thresholds[layer_id]
                features.extend(
                    trace + 0.15 * max(0.0, membrane / max(threshold, 1e-6))
                    for trace, membrane, threshold in zip(next_trace, voltage, thresholds, strict=True)
                )
            else:
                thresholds = self._layer_thresholds[layer_id]
                reset_values = self._layer_reset_values[layer_id]
                features.extend(
                    trace + 0.15 * max(0.0, (membrane - reset) / max(1.0, threshold - reset))
                    for trace, membrane, threshold, reset in zip(
                        next_trace,
                        voltage,
                        thresholds,
                        reset_values,
                        strict=True,
                    )
                )
        return features


def build_spiking_network(
    config: RSNNConfig,
    rng: random.Random | None = None,
) -> DynnRecurrentSpikingNetwork:
    return DynnRecurrentSpikingNetwork(config, rng)


def _build_topology(
    config: RSNNConfig,
    rng: random.Random,
    grid_width: int,
    grid_height: int,
) -> dict[str, Any]:
    node_sets: list[dict[str, Any]] = [
        {
            "id": "obs",
            "size": config.input_dim,
            "node_type": "linear",
            "parameters": {"bias": 0.0},
        }
    ]
    edge_sets: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = [{"id": "obs", "node_set": "obs", "kind": "input"}]

    input_dim = config.input_dim
    input_grid_width = 0
    input_grid_height = 0
    for layer_index in range(config.n_layers):
        hidden_id = f"hidden_{layer_index}"
        node_sets.append(
            {
                "id": hidden_id,
                "size": config.n_neurons,
                "node_type": _node_type(config),
                "parameters": _node_parameters(config, rng),
            }
        )
        ports.append({"id": hidden_id, "node_set": hidden_id, "kind": "output"})

        source_id = "obs" if layer_index == 0 else f"hidden_{layer_index - 1}"
        input_indices = build_input_indices(
            config,
            grid_width,
            grid_height,
            input_dim=input_dim,
            input_grid_width=input_grid_width,
            input_grid_height=input_grid_height,
            rng=rng,
        )
        edge_sets.append(
            _explicit_edge_set(
                edge_id=f"{source_id}_to_{hidden_id}",
                source_id=source_id,
                target_id=hidden_id,
                rows=input_indices,
                scale=config.input_scale / math.sqrt(mean_fanin(input_indices)),
                rng=rng,
            )
        )

        recurrent_indices = build_recurrent_indices(config, grid_width, grid_height, rng)
        edge_sets.append(
            _explicit_edge_set(
                edge_id=f"{hidden_id}_recurrent",
                source_id=hidden_id,
                target_id=hidden_id,
                rows=recurrent_indices,
                scale=config.recurrent_scale / math.sqrt(mean_fanin(recurrent_indices)),
                rng=rng,
            )
        )
        input_dim = config.n_neurons
        input_grid_width = grid_width
        input_grid_height = grid_height

    return {
        "node_sets": node_sets,
        "edge_sets": edge_sets,
        "ports": ports,
    }


def _explicit_edge_set(
    *,
    edge_id: str,
    source_id: str,
    target_id: str,
    rows: list[list[int]],
    scale: float,
    rng: random.Random,
) -> dict[str, Any]:
    edges: list[dict[str, float | int]] = []
    for target_index, row in enumerate(rows):
        for source_index in row:
            edges.append(
                {
                    "source": int(source_index),
                    "target": int(target_index),
                    "weight": rng.gauss(0.0, scale),
                }
            )
    return {
        "id": edge_id,
        "source": {"node_set": source_id},
        "target": {"node_set": target_id},
        "representation": {"kind": "explicit", "edges": edges},
    }


def _node_type(config: RSNNConfig) -> str:
    if config.neuron_model == "lif":
        return "lif"
    if config.neuron_model == "izh":
        return "izh"
    raise ValueError(f"unsupported neuron model: {config.neuron_model}")


def _node_parameters(config: RSNNConfig, rng: random.Random) -> dict[str, Any]:
    if config.neuron_model == "lif":
        tau_m_mean = 1.0 / max(1e-6, 1.0 - config.membrane_decay)
        tau_m = []
        threshold = []
        bias = []
        for _ in range(config.n_neurons):
            if config.randomize_intrinsics:
                sampled_decay = clamp(
                    rng.gauss(config.membrane_decay, config.membrane_decay_jitter),
                    0.60,
                    0.99,
                )
                tau_m.append(1.0 / max(1e-6, 1.0 - sampled_decay))
                threshold.append(
                    clamp(rng.gauss(config.threshold, config.threshold_jitter), 0.55, 1.60)
                )
            else:
                tau_m.append(tau_m_mean)
                threshold.append(config.threshold)
            bias.append(rng.gauss(0.0, config.bias_scale))
        return {
            "tau_m": tau_m,
            "v_rest": 0.0,
            "v_reset": 0.0,
            "v_threshold": threshold,
            "bias": bias,
        }

    a_values = []
    b_values = []
    c_values = []
    d_values = []
    v_peak_values = []
    input_gain_values = []
    bias = []
    for _ in range(config.n_neurons):
        if config.randomize_intrinsics:
            a_values.append(clamp(rng.gauss(config.izh_a, config.izh_a_jitter), 0.005, 0.08))
            b_values.append(clamp(rng.gauss(config.izh_b, config.izh_b_jitter), 0.08, 0.35))
            c_values.append(clamp(rng.gauss(config.izh_c, config.izh_c_jitter), -78.0, -50.0))
            d_values.append(clamp(rng.gauss(config.izh_d, config.izh_d_jitter), 2.0, 16.0))
            v_peak_values.append(
                clamp(
                    rng.gauss(config.izh_spike_threshold, config.izh_spike_threshold_jitter),
                    18.0,
                    45.0,
                )
            )
            input_gain_values.append(
                clamp(
                    rng.gauss(config.izh_input_gain, config.izh_input_gain_jitter),
                    4.0,
                    14.0,
                )
            )
        else:
            a_values.append(config.izh_a)
            b_values.append(config.izh_b)
            c_values.append(config.izh_c)
            d_values.append(config.izh_d)
            v_peak_values.append(config.izh_spike_threshold)
            input_gain_values.append(config.izh_input_gain)
        bias.append(rng.gauss(0.0, config.bias_scale))
    return {
        "a": a_values,
        "b": b_values,
        "c": c_values,
        "d": d_values,
        "v_peak": v_peak_values,
        "input_gain": input_gain_values,
        "dt": config.izh_dt,
        "substeps": config.izh_substeps,
        "bias": bias,
    }


def _layer_parameters_from_topology(
    topology: dict[str, Any],
    layer_ids: list[str],
) -> dict[str, dict[str, Any]]:
    wanted = set(layer_ids)
    parameters: dict[str, dict[str, Any]] = {}
    for node_set in topology.get("node_sets", []):
        node_set_id = str(node_set.get("id", ""))
        if node_set_id in wanted:
            raw_parameters = node_set.get("parameters", {})
            parameters[node_set_id] = dict(raw_parameters) if isinstance(raw_parameters, dict) else {}
    return parameters


def _layer_thresholds_from_parameters(
    config: RSNNConfig,
    parameters: dict[str, dict[str, Any]],
) -> dict[str, tuple[float, ...]]:
    thresholds: dict[str, tuple[float, ...]] = {}
    key = "v_threshold" if config.neuron_model == "lif" else "v_peak"
    default = config.threshold if config.neuron_model == "lif" else config.izh_spike_threshold
    for layer_index in range(config.n_layers):
        layer_id = f"hidden_{layer_index}"
        thresholds[layer_id] = _float_series(
            parameters.get(layer_id, {}).get(key, ()),
            config.n_neurons,
            default,
        )
    return thresholds


def _layer_reset_values_from_parameters(
    config: RSNNConfig,
    parameters: dict[str, dict[str, Any]],
) -> dict[str, tuple[float, ...]]:
    reset_values: dict[str, tuple[float, ...]] = {}
    key = "v_reset" if config.neuron_model == "lif" else "c"
    default = 0.0 if config.neuron_model == "lif" else config.izh_c
    for layer_index in range(config.n_layers):
        layer_id = f"hidden_{layer_index}"
        reset_values[layer_id] = _float_series(
            parameters.get(layer_id, {}).get(key, ()),
            config.n_neurons,
            default,
        )
    return reset_values


def _float_series(value: Any, count: int, default: float) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(count))
    if isinstance(value, (list, tuple)):
        values = tuple(float(item) for item in value)
        if len(values) == count:
            return values
        if values:
            return tuple(values[index % len(values)] for index in range(count))
    return tuple(float(default) for _ in range(count))


def _plastic_width(config: RSNNConfig) -> float:
    return 0.8 if config.neuron_model == "lif" else 20.0


def resolve_grid_shape(n_neurons: int, grid_width: int, grid_height: int) -> tuple[int, int]:
    if grid_width > 0 and grid_height > 0:
        if grid_width * grid_height != n_neurons:
            raise ValueError(
                f"grid shape {grid_width}x{grid_height} does not match n_neurons={n_neurons}"
            )
        return grid_width, grid_height
    for candidate_height in range(int(math.sqrt(n_neurons)), 0, -1):
        if n_neurons % candidate_height == 0:
            return n_neurons // candidate_height, candidate_height
    return n_neurons, 1


def build_input_indices(
    config: RSNNConfig,
    grid_width: int,
    grid_height: int,
    *,
    input_dim: int,
    input_grid_width: int,
    input_grid_height: int,
    rng: random.Random,
) -> list[list[int]]:
    if (
        input_grid_width <= 0
        or input_grid_height <= 0
        or input_grid_width * input_grid_height != input_dim
    ):
        return [list(range(input_dim)) for _ in range(config.n_neurons)]

    rows: list[list[int]] = []
    for neuron in range(config.n_neurons):
        x, y = neuron_to_xy(neuron, grid_width)
        source_x = map_coordinate(x, grid_width, input_grid_width)
        source_y = map_coordinate(y, grid_height, input_grid_height)
        candidates = neighborhood_indices(
            center_x=source_x,
            center_y=source_y,
            width=input_grid_width,
            height=input_grid_height,
            radius=config.interlayer_radius,
            include_center=True,
        )
        rows.append(sample_candidates(candidates, config.interlayer_degree, rng))
    return rows


def build_recurrent_indices(
    config: RSNNConfig,
    grid_width: int,
    grid_height: int,
    rng: random.Random,
) -> list[list[int]]:
    rows: list[list[int]] = []
    for neuron in range(config.n_neurons):
        x, y = neuron_to_xy(neuron, grid_width)
        candidates = neighborhood_indices(
            center_x=x,
            center_y=y,
            width=grid_width,
            height=grid_height,
            radius=config.recurrent_radius,
            include_center=False,
        )
        if not candidates:
            candidates = [neuron]
        rows.append(sample_candidates(candidates, config.recurrent_degree, rng))
    return rows


def neuron_to_xy(index: int, width: int) -> tuple[int, int]:
    return index % width, index // width


def xy_to_neuron(x: int, y: int, width: int) -> int:
    return y * width + x


def map_coordinate(index: int, source_size: int, target_size: int) -> int:
    if target_size <= 1 or source_size <= 1:
        return 0
    scaled = round(index * (target_size - 1) / (source_size - 1))
    return int(clamp(scaled, 0, target_size - 1))


def neighborhood_indices(
    *,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    radius: int,
    include_center: bool,
) -> list[int]:
    indices: list[int] = []
    for neighbor_y in range(max(0, center_y - radius), min(height - 1, center_y + radius) + 1):
        for neighbor_x in range(max(0, center_x - radius), min(width - 1, center_x + radius) + 1):
            if not include_center and neighbor_x == center_x and neighbor_y == center_y:
                continue
            indices.append(xy_to_neuron(neighbor_x, neighbor_y, width))
    return indices


def sample_candidates(candidates: list[int], degree: int, rng: random.Random) -> list[int]:
    if not candidates:
        return []
    if degree <= 0 or degree >= len(candidates):
        return list(candidates)
    return rng.sample(candidates, degree)


def mean_fanin(rows: list[list[int]]) -> float:
    total = sum(len(row) for row in rows)
    return max(1.0, total / max(1, len(rows)))


def triangular_pseudo_derivative(value: float, threshold: float, width: float = 0.8) -> float:
    distance = abs(value - threshold)
    width = max(width, 1e-6)
    return max(0.05, 1.0 - distance / width)
