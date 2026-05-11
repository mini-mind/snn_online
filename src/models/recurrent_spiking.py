"""Pure-Python single-layer recurrent spiking network."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from models.common import clamp


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
    delay_features: bool = False
    delay_mix_lr: float = 0.0004
    seed: int = 13


class LocalRecurrentSpikingNetwork:
    """Small RSNN used directly by experiments without external runtime adapters."""

    def __init__(self, config: RSNNConfig, rng: random.Random | None = None) -> None:
        if config.plasticity_rule not in {"three_factor", "tess_like"}:
            raise ValueError(f"unsupported plasticity_rule: {config.plasticity_rule}")
        if config.neuron_model not in {"lif", "izh"}:
            raise ValueError(f"unsupported neuron_model: {config.neuron_model}")
        self.config = config
        self.rng = rng or random.Random(config.seed)
        self.input_weights = _dense_weights(
            rows=config.n_neurons,
            cols=config.input_dim,
            scale=config.input_scale / math.sqrt(max(1, config.input_dim)),
            rng=self.rng,
        )
        self.recurrent_sources = _build_recurrent_sources(config, self.rng)
        self.recurrent_weights = [
            [
                self.rng.gauss(
                    0.0,
                    config.recurrent_scale / math.sqrt(max(1, len(row))),
                )
                for _ in row
            ]
            for row in self.recurrent_sources
        ]
        self.bias = [self.rng.gauss(0.0, config.bias_scale) for _ in range(config.n_neurons)]
        self.thresholds = _thresholds(config, self.rng)
        self.reset_values = _reset_values(config, self.rng)
        self.izh_a, self.izh_b, self.izh_d, self.izh_gain = _izh_params(config, self.rng)
        self.delay_decays = [0.72, 0.88, 0.97]
        self.delay_mix_weights = _delay_mix_weights(config, self.delay_decays, self.rng)
        self.reset_state()

    def reset_state(self) -> None:
        self.voltage = [0.0 for _ in range(self.config.n_neurons)]
        self.recovery = [
            self.izh_b[index] * self.reset_values[index]
            for index in range(self.config.n_neurons)
        ]
        self.spikes = [0.0 for _ in range(self.config.n_neurons)]
        self.spike_trace = [0.0 for _ in range(self.config.n_neurons)]
        self.pre_trace = [0.0 for _ in range(self.config.n_neurons)]
        self.fast_pre_trace = [0.0 for _ in range(self.config.n_neurons)]
        self.slow_pre_trace = [0.0 for _ in range(self.config.n_neurons)]
        self.post_trace = [0.0 for _ in range(self.config.n_neurons)]
        self.delay_traces = _empty_delay_traces(self.config, self.delay_decays)
        self.eligibility = [
            [0.0 for _ in row]
            for row in self.recurrent_sources
        ]
        self._features = [0.0 for _ in range(self.feature_dim())]

    def step(self, inputs: list[float]) -> list[float]:
        current = self._current(inputs)
        if self.config.neuron_model == "lif":
            new_spikes = self._lif_step(current)
        else:
            new_spikes = self._izh_step(current)
        self.spikes = new_spikes
        self.spike_trace = [
            self.config.trace_decay * trace + spike
            for trace, spike in zip(self.spike_trace, self.spikes, strict=True)
        ]
        self._update_delay_traces()
        self._features = self._read_features()
        return list(self._features)

    def apply_recurrent_modulation(self, modulation: float) -> None:
        if self.config.plastic_lr == 0.0:
            return
        clipped = clamp(float(modulation), -1.0, 1.0)
        post_factor = [
            triangular_pseudo_derivative(
                voltage,
                threshold=threshold,
                width=_plastic_width(self.config),
            )
            for voltage, threshold in zip(self.voltage, self.thresholds, strict=True)
        ]
        if self.config.plasticity_rule == "three_factor":
            self._apply_three_factor(clipped, post_factor)
        else:
            self._apply_tess_like(clipped, post_factor)
        self._apply_delay_mix_modulation(clipped, post_factor)

    def features(self) -> list[float]:
        return list(self._features)

    def feature_dim(self) -> int:
        return self.config.n_neurons * (2 if self.config.delay_features else 1)

    def _current(self, inputs: list[float]) -> list[float]:
        values = []
        for target in range(self.config.n_neurons):
            input_drive = sum(
                weight * value
                for weight, value in zip(self.input_weights[target], inputs, strict=True)
            )
            recurrent_drive = sum(
                weight * self.spikes[source]
                for source, weight in zip(
                    self.recurrent_sources[target],
                    self.recurrent_weights[target],
                    strict=True,
                )
            )
            values.append(input_drive + recurrent_drive + self.bias[target])
        return values

    def _lif_step(self, current: list[float]) -> list[float]:
        spikes = []
        for index, drive in enumerate(current):
            voltage = self.config.membrane_decay * self.voltage[index] + drive
            if voltage >= self.thresholds[index]:
                spikes.append(1.0)
                self.voltage[index] = 0.0
            else:
                spikes.append(0.0)
                self.voltage[index] = voltage
        return spikes

    def _izh_step(self, current: list[float]) -> list[float]:
        spikes = [0.0 for _ in range(self.config.n_neurons)]
        for index, drive in enumerate(current):
            voltage = self.voltage[index] or self.reset_values[index]
            recovery = self.recovery[index]
            for _ in range(max(1, self.config.izh_substeps)):
                scaled_drive = self.izh_gain[index] * drive
                voltage += self.config.izh_dt * (
                    0.04 * voltage * voltage + 5.0 * voltage + 140.0 - recovery + scaled_drive
                )
                recovery += self.config.izh_dt * self.izh_a[index] * (
                    self.izh_b[index] * voltage - recovery
                )
                if voltage >= self.thresholds[index]:
                    spikes[index] = 1.0
                    voltage = self.reset_values[index]
                    recovery += self.izh_d[index]
                    break
            self.voltage[index] = voltage
            self.recovery[index] = recovery
        return spikes

    def _read_features(self) -> list[float]:
        if self.config.neuron_model == "lif":
            base_features = [
                trace + 0.15 * max(0.0, voltage / max(threshold, 1e-6))
                for trace, voltage, threshold in zip(
                    self.spike_trace, self.voltage, self.thresholds, strict=True
                )
            ]
        else:
            base_features = [
                trace + 0.15 * max(0.0, (voltage - reset) / max(1.0, threshold - reset))
                for trace, voltage, threshold, reset in zip(
                    self.spike_trace,
                    self.voltage,
                    self.thresholds,
                    self.reset_values,
                    strict=True,
                )
            ]
        if not self.config.delay_features:
            return base_features
        return base_features + self._delay_features()

    def _update_delay_traces(self) -> None:
        if not self.config.delay_features:
            return
        for neuron, spike in enumerate(self.spikes):
            for slot, decay in enumerate(self.delay_decays):
                self.delay_traces[neuron][slot] = decay * self.delay_traces[neuron][slot] + spike

    def _delay_features(self) -> list[float]:
        return [
            sum(weight * trace for weight, trace in zip(weights, traces, strict=True))
            for weights, traces in zip(self.delay_mix_weights, self.delay_traces, strict=True)
        ]

    def _apply_three_factor(self, modulation: float, post_factor: list[float]) -> None:
        self.pre_trace = [
            self.config.trace_decay * trace + spike
            for trace, spike in zip(self.pre_trace, self.spikes, strict=True)
        ]
        for target, sources in enumerate(self.recurrent_sources):
            for edge_index, source in enumerate(sources):
                delta = modulation * post_factor[target] * self.pre_trace[source]
                self._update_recurrent_weight(target, edge_index, delta)

    def _apply_tess_like(self, modulation: float, post_factor: list[float]) -> None:
        self.fast_pre_trace = [
            self.config.tess_fast_decay * trace + spike
            for trace, spike in zip(self.fast_pre_trace, self.spikes, strict=True)
        ]
        self.slow_pre_trace = [
            self.config.tess_slow_decay * trace + spike
            for trace, spike in zip(self.slow_pre_trace, self.spikes, strict=True)
        ]
        self.post_trace = [
            self.config.tess_post_decay * trace + spike
            for trace, spike in zip(self.post_trace, self.spikes, strict=True)
        ]
        for target, sources in enumerate(self.recurrent_sources):
            for edge_index, source in enumerate(sources):
                synchrony = 0.5 * (
                    self.fast_pre_trace[source] * self.post_trace[target]
                    + self.slow_pre_trace[source] * post_factor[target]
                )
                eligibility = (
                    self.config.tess_eligibility_decay * self.eligibility[target][edge_index]
                    + synchrony
                )
                self.eligibility[target][edge_index] = eligibility
                delta = modulation * post_factor[target] * eligibility
                self._update_recurrent_weight(target, edge_index, delta)

    def _apply_delay_mix_modulation(self, modulation: float, post_factor: list[float]) -> None:
        if not self.config.delay_features or self.config.delay_mix_lr == 0.0:
            return
        for neuron, traces in enumerate(self.delay_traces):
            for slot, trace in enumerate(traces):
                delta = modulation * post_factor[neuron] * trace
                weight = self.delay_mix_weights[neuron][slot]
                next_weight = (weight + self.config.delay_mix_lr * delta) * (
                    1.0 - self.config.weight_decay
                )
                self.delay_mix_weights[neuron][slot] = clamp(next_weight, -0.8, 0.8)

    def _update_recurrent_weight(self, target: int, edge_index: int, delta: float) -> None:
        weight = self.recurrent_weights[target][edge_index]
        next_weight = (weight + self.config.plastic_lr * delta) * (1.0 - self.config.weight_decay)
        self.recurrent_weights[target][edge_index] = clamp(next_weight, -1.5, 1.5)


def build_spiking_network(
    config: RSNNConfig,
    rng: random.Random | None = None,
) -> LocalRecurrentSpikingNetwork:
    return LocalRecurrentSpikingNetwork(config, rng)


def _dense_weights(rows: int, cols: int, scale: float, rng: random.Random) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def _delay_mix_weights(
    config: RSNNConfig,
    delay_decays: list[float],
    rng: random.Random,
) -> list[list[float]]:
    if not config.delay_features:
        return []
    return [
        [rng.uniform(0.20, 0.45) / len(delay_decays) for _ in delay_decays]
        for _ in range(config.n_neurons)
    ]


def _empty_delay_traces(config: RSNNConfig, delay_decays: list[float]) -> list[list[float]]:
    if not config.delay_features:
        return []
    return [
        [0.0 for _ in delay_decays]
        for _ in range(config.n_neurons)
    ]


def _build_recurrent_sources(config: RSNNConfig, rng: random.Random) -> list[list[int]]:
    rows: list[list[int]] = []
    for target in range(config.n_neurons):
        candidates = [index for index in range(config.n_neurons) if index != target]
        if not candidates:
            candidates = [target]
        rows.append(sample_candidates(candidates, config.recurrent_degree, rng))
    return rows


def _thresholds(config: RSNNConfig, rng: random.Random) -> list[float]:
    if config.neuron_model == "lif":
        return [
            clamp(rng.gauss(config.threshold, config.threshold_jitter), 0.55, 1.60)
            if config.randomize_intrinsics
            else config.threshold
            for _ in range(config.n_neurons)
        ]
    return [
        clamp(
            rng.gauss(config.izh_spike_threshold, config.izh_spike_threshold_jitter),
            18.0,
            45.0,
        )
        if config.randomize_intrinsics
        else config.izh_spike_threshold
        for _ in range(config.n_neurons)
    ]


def _reset_values(config: RSNNConfig, rng: random.Random) -> list[float]:
    if config.neuron_model == "lif":
        return [0.0 for _ in range(config.n_neurons)]
    return [
        clamp(rng.gauss(config.izh_c, config.izh_c_jitter), -78.0, -50.0)
        if config.randomize_intrinsics
        else config.izh_c
        for _ in range(config.n_neurons)
    ]


def _izh_params(
    config: RSNNConfig,
    rng: random.Random,
) -> tuple[list[float], list[float], list[float], list[float]]:
    a_values = []
    b_values = []
    d_values = []
    gain_values = []
    for _ in range(config.n_neurons):
        if config.randomize_intrinsics:
            a_values.append(clamp(rng.gauss(config.izh_a, config.izh_a_jitter), 0.005, 0.08))
            b_values.append(clamp(rng.gauss(config.izh_b, config.izh_b_jitter), 0.08, 0.35))
            d_values.append(clamp(rng.gauss(config.izh_d, config.izh_d_jitter), 2.0, 16.0))
            gain_values.append(clamp(rng.gauss(config.izh_input_gain, config.izh_input_gain_jitter), 4.0, 14.0))
        else:
            a_values.append(config.izh_a)
            b_values.append(config.izh_b)
            d_values.append(config.izh_d)
            gain_values.append(config.izh_input_gain)
    return a_values, b_values, d_values, gain_values


def sample_candidates(candidates: list[int], degree: int, rng: random.Random) -> list[int]:
    if not candidates:
        return []
    if degree <= 0 or degree >= len(candidates):
        return list(candidates)
    return rng.sample(candidates, degree)


def _plastic_width(config: RSNNConfig) -> float:
    return 0.8 if config.neuron_model == "lif" else 20.0


def triangular_pseudo_derivative(value: float, threshold: float, width: float = 0.8) -> float:
    distance = abs(value - threshold)
    width = max(width, 1e-6)
    return max(0.05, 1.0 - distance / width)
