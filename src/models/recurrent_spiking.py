"""Thin `dynn` wrapper for a single-layer recurrent spiking network."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from models.common import clamp
from models.dynn_support import dynn


@dataclass
class RSNNConfig:
    """Single-layer recurrent spiking network configuration."""

    input_dim: int
    n_neurons: int = 48
    recurrent_degree: int = 4
    neuron_model: str = "lif"
    membrane_decay: float = 0.86
    trace_decay: float = 0.90
    threshold: float = 1.0
    recurrent_scale: float = 0.22
    input_scale: float = 0.65
    bias_scale: float = 0.18
    plastic_lr: float = 0.0008
    weight_decay: float = 0.00005
    plasticity_rule: str = "three_factor"
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
    tess_fast_decay: float = 0.55
    tess_slow_decay: float = 0.92
    tess_post_decay: float = 0.80
    tess_eligibility_decay: float = 0.88
    seed: int = 13


class _ThreeFactorRule:
    """Baseline local plasticity: pre-trace x post sensitivity x modulation."""

    def __init__(self, config: RSNNConfig, thresholds: tuple[float, ...]) -> None:
        self.config = config
        self.trace_decay = float(config.trace_decay)
        self.weight_decay = float(config.weight_decay)
        self.thresholds = thresholds
        self.width = _plastic_width(config)

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
        prev_pre = _safe_trace(trace_state.get("pre", ()), edge_block.source_count)
        next_pre = tuple(
            self.trace_decay * prev + float(activity)
            for prev, activity in zip(prev_pre, pre_activity, strict=True)
        )
        if edge_block.source_node_set != "hidden" or edge_block.target_node_set != "hidden":
            return _plasticity_result(edge_block, {"pre": next_pre}, return_weights)

        post_factor = _post_factors(
            node_states=node_states,
            node_set_id=edge_block.target_node_set,
            thresholds=self.thresholds,
            width=self.width,
        )
        clipped_modulation = clamp(float(modulation), -1.0, 1.0)
        deltas = tuple(
            clipped_modulation * post_factor[target] * next_pre[source]
            for source, target in zip(edge_block.source_indices, edge_block.target_indices, strict=True)
        )
        return _apply_weight_deltas(
            config=self.config,
            edge_block=edge_block,
            deltas=deltas,
            traces={"pre": next_pre},
            return_weights=return_weights,
        )


class _TessLikeRule:
    """TESS-like local rule with fast/slow pre traces and a local eligibility state."""

    def __init__(self, config: RSNNConfig, thresholds: tuple[float, ...]) -> None:
        self.config = config
        self.weight_decay = float(config.weight_decay)
        self.thresholds = thresholds
        self.width = _plastic_width(config)

    def initialize_traces(self, *, edge_block) -> dict[str, tuple[float, ...]]:
        return {
            "fast_pre": tuple(0.0 for _ in range(edge_block.source_count)),
            "slow_pre": tuple(0.0 for _ in range(edge_block.source_count)),
            "post": tuple(0.0 for _ in range(edge_block.target_count)),
            "eligibility": tuple(0.0 for _ in range(len(edge_block.weights))),
        }

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
        del learning_rate, step_index
        trace_state = traces or {}
        fast_pre = _safe_trace(trace_state.get("fast_pre", ()), edge_block.source_count)
        slow_pre = _safe_trace(trace_state.get("slow_pre", ()), edge_block.source_count)
        post_trace = _safe_trace(trace_state.get("post", ()), edge_block.target_count)
        prev_eligibility = _safe_trace(trace_state.get("eligibility", ()), len(edge_block.weights))
        next_fast_pre = tuple(
            self.config.tess_fast_decay * prev + float(activity)
            for prev, activity in zip(fast_pre, pre_activity, strict=True)
        )
        next_slow_pre = tuple(
            self.config.tess_slow_decay * prev + float(activity)
            for prev, activity in zip(slow_pre, pre_activity, strict=True)
        )
        next_post_trace = tuple(
            self.config.tess_post_decay * prev + float(activity)
            for prev, activity in zip(post_trace, post_activity, strict=True)
        )
        next_traces = {
            "fast_pre": next_fast_pre,
            "slow_pre": next_slow_pre,
            "post": next_post_trace,
            "eligibility": prev_eligibility,
        }
        if edge_block.source_node_set != "hidden" or edge_block.target_node_set != "hidden":
            return _plasticity_result(edge_block, next_traces, return_weights)

        post_factor = _post_factors(
            node_states=node_states,
            node_set_id=edge_block.target_node_set,
            thresholds=self.thresholds,
            width=self.width,
        )
        clipped_modulation = clamp(float(modulation), -1.0, 1.0)
        next_eligibility = []
        deltas = []
        for edge_index, (source, target) in enumerate(
            zip(edge_block.source_indices, edge_block.target_indices, strict=True)
        ):
            synchrony = 0.5 * (
                next_fast_pre[source] * next_post_trace[target]
                + next_slow_pre[source] * post_factor[target]
            )
            eligibility = (
                self.config.tess_eligibility_decay * prev_eligibility[edge_index] + synchrony
            )
            next_eligibility.append(eligibility)
            deltas.append(clipped_modulation * post_factor[target] * eligibility)
        next_traces["eligibility"] = tuple(next_eligibility)
        return _apply_weight_deltas(
            config=self.config,
            edge_block=edge_block,
            deltas=tuple(deltas),
            traces=next_traces,
            return_weights=return_weights,
        )


class DynnRecurrentSpikingNetwork:
    """Single hidden-population recurrent spiking feature extractor."""

    def __init__(self, config: RSNNConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random(config.seed)
        topology = _build_topology(config, self.rng)
        parameters = _hidden_parameters_from_topology(topology)
        self._thresholds = _thresholds_from_parameters(config, parameters)
        self._reset_values = _reset_values_from_parameters(config, parameters)
        self.graph = dynn.build(
            {"id": f"{config.neuron_model}-closed-loop-rsnn"},
            topology,
            seed=config.seed,
        )
        self.rule = _build_plasticity_rule(config, self._thresholds)
        self.net = dynn.Net(self.graph, plasticity=self.rule, learning_rate=config.plastic_lr)
        self._spike_trace = [0.0 for _ in range(config.n_neurons)]
        self._features = [0.0 for _ in range(config.n_neurons)]

    def reset_state(self) -> None:
        self.net.reset()
        self._spike_trace = [0.0 for _ in range(self.config.n_neurons)]
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
        return self.config.n_neurons

    def _read_features(self, output) -> list[float]:
        dynamics_state = getattr(self.net, "_dynamics_state", {})
        state = dynamics_state.get("hidden", {}) if isinstance(dynamics_state, dict) else {}
        spikes = _float_series(output.node("hidden") or state.get("activity", ()), self.config.n_neurons, 0.0)
        voltage = _float_series(state.get("voltage", ()), self.config.n_neurons, 0.0)
        next_trace = [
            self.config.trace_decay * previous + spike
            for previous, spike in zip(self._spike_trace, spikes, strict=True)
        ]
        self._spike_trace = next_trace
        if self.config.neuron_model == "lif":
            return [
                trace + 0.15 * max(0.0, membrane / max(threshold, 1e-6))
                for trace, membrane, threshold in zip(
                    next_trace, voltage, self._thresholds, strict=True
                )
            ]
        return [
            trace + 0.15 * max(0.0, (membrane - reset) / max(1.0, threshold - reset))
            for trace, membrane, threshold, reset in zip(
                next_trace,
                voltage,
                self._thresholds,
                self._reset_values,
                strict=True,
            )
        ]


def build_spiking_network(
    config: RSNNConfig,
    rng: random.Random | None = None,
) -> DynnRecurrentSpikingNetwork:
    return DynnRecurrentSpikingNetwork(config, rng)


def _build_topology(config: RSNNConfig, rng: random.Random) -> dict[str, Any]:
    node_sets: list[dict[str, Any]] = [
        {
            "id": "obs",
            "size": config.input_dim,
            "node_type": "linear",
            "parameters": {"bias": 0.0},
        },
        {
            "id": "hidden",
            "size": config.n_neurons,
            "node_type": _node_type(config),
            "parameters": _node_parameters(config, rng),
        },
    ]
    edge_sets = [
        _explicit_edge_set(
            edge_id="obs_to_hidden",
            source_id="obs",
            target_id="hidden",
            rows=[list(range(config.input_dim)) for _ in range(config.n_neurons)],
            scale=config.input_scale / math.sqrt(max(1, config.input_dim)),
            rng=rng,
        ),
        _explicit_edge_set(
            edge_id="hidden_recurrent",
            source_id="hidden",
            target_id="hidden",
            rows=_build_recurrent_indices(config, rng),
            scale=config.recurrent_scale / math.sqrt(max(1, config.recurrent_degree)),
            rng=rng,
        ),
    ]
    ports = [
        {"id": "obs", "node_set": "obs", "kind": "input"},
        {"id": "hidden", "node_set": "hidden", "kind": "output"},
    ]
    return {"node_sets": node_sets, "edge_sets": edge_sets, "ports": ports}


def _build_recurrent_indices(config: RSNNConfig, rng: random.Random) -> list[list[int]]:
    rows: list[list[int]] = []
    for target in range(config.n_neurons):
        candidates = [index for index in range(config.n_neurons) if index != target]
        if not candidates:
            candidates = [target]
        rows.append(sample_candidates(candidates, config.recurrent_degree, rng))
    return rows


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


def _build_plasticity_rule(
    config: RSNNConfig,
    thresholds: tuple[float, ...],
) -> _ThreeFactorRule | _TessLikeRule:
    if config.plasticity_rule == "three_factor":
        return _ThreeFactorRule(config, thresholds)
    if config.plasticity_rule == "tess_like":
        return _TessLikeRule(config, thresholds)
    raise ValueError(f"unsupported plasticity_rule: {config.plasticity_rule}")


def _hidden_parameters_from_topology(topology: dict[str, Any]) -> dict[str, Any]:
    for node_set in topology.get("node_sets", []):
        if str(node_set.get("id", "")) == "hidden":
            raw_parameters = node_set.get("parameters", {})
            return dict(raw_parameters) if isinstance(raw_parameters, dict) else {}
    return {}


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


def _thresholds_from_parameters(
    config: RSNNConfig,
    parameters: dict[str, Any],
) -> tuple[float, ...]:
    key = "v_threshold" if config.neuron_model == "lif" else "v_peak"
    default = config.threshold if config.neuron_model == "lif" else config.izh_spike_threshold
    return _float_series(parameters.get(key, ()), config.n_neurons, default)


def _reset_values_from_parameters(
    config: RSNNConfig,
    parameters: dict[str, Any],
) -> tuple[float, ...]:
    key = "v_reset" if config.neuron_model == "lif" else "c"
    default = 0.0 if config.neuron_model == "lif" else config.izh_c
    return _float_series(parameters.get(key, ()), config.n_neurons, default)


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


def _safe_trace(values: tuple[float, ...] | list[float] | Any, count: int) -> tuple[float, ...]:
    if isinstance(values, (list, tuple)) and len(values) == count:
        return tuple(float(value) for value in values)
    return tuple(0.0 for _ in range(count))


def _post_factors(
    *,
    node_states,
    node_set_id: str,
    thresholds: tuple[float, ...],
    width: float,
) -> tuple[float, ...]:
    state = node_states.get(node_set_id, {})
    post_voltage = tuple(float(value) for value in state.get("voltage", ()))
    if len(post_voltage) != len(thresholds):
        post_voltage = tuple(0.0 for _ in range(len(thresholds)))
    return tuple(
        triangular_pseudo_derivative(voltage, threshold=threshold, width=width)
        for voltage, threshold in zip(post_voltage, thresholds, strict=True)
    )


def _plasticity_result(edge_block, traces, return_weights: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "traces": traces,
        "weight_update_count": 0,
        "mean_abs_weight_delta": 0.0,
        "max_abs_weight_delta": 0.0,
    }
    if return_weights:
        result["weights"] = tuple(edge_block.weights)
    return result


def _apply_weight_deltas(
    *,
    config: RSNNConfig,
    edge_block,
    deltas: tuple[float, ...],
    traces: dict[str, tuple[float, ...]],
    return_weights: bool,
) -> dict[str, object]:
    next_weights = tuple(
        clamp(
            (float(weight) + config.plastic_lr * delta) * (1.0 - config.weight_decay),
            -1.5,
            1.5,
        )
        for weight, delta in zip(edge_block.weights, deltas, strict=True)
    )
    abs_deltas = tuple(
        abs(after - before)
        for before, after in zip(edge_block.weights, next_weights, strict=True)
    )
    result: dict[str, object] = {
        "traces": traces,
        "weight_update_count": len(deltas),
        "mean_abs_weight_delta": (sum(abs_deltas) / len(abs_deltas)) if abs_deltas else 0.0,
        "max_abs_weight_delta": max(abs_deltas, default=0.0),
    }
    if return_weights:
        result["weights"] = next_weights
    return result


def sample_candidates(candidates: list[int], degree: int, rng: random.Random) -> list[int]:
    if not candidates:
        return []
    if degree <= 0 or degree >= len(candidates):
        return list(candidates)
    return rng.sample(candidates, degree)


def triangular_pseudo_derivative(value: float, threshold: float, width: float = 0.8) -> float:
    distance = abs(value - threshold)
    width = max(width, 1e-6)
    return max(0.05, 1.0 - distance / width)
