"""Small recurrent spiking network utilities for toy control experiments.

The module is intentionally dependency-free. It provides a recurrent LIF-like
state update and a local three-factor recurrent plasticity hook. Readout heads
can use `features()` as eligibility-like hidden traces.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class RSNNConfig:
    input_dim: int
    n_neurons: int = 48
    membrane_decay: float = 0.86
    trace_decay: float = 0.90
    threshold: float = 1.0
    recurrent_scale: float = 0.22
    input_scale: float = 0.65
    bias_scale: float = 0.18
    plastic_lr: float = 0.0008
    weight_decay: float = 0.00005
    seed: int = 13


class RecurrentSpikingNetwork:
    """A compact LIF-like R-SNN with local recurrent eligibility traces."""

    def __init__(self, config: RSNNConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random(config.seed)
        input_scale = config.input_scale / math.sqrt(config.input_dim)
        recurrent_scale = config.recurrent_scale / math.sqrt(config.n_neurons)
        self.input_weights = [
            [self.rng.gauss(0.0, input_scale) for _ in range(config.input_dim)]
            for _ in range(config.n_neurons)
        ]
        self.recurrent_weights = [
            [self.rng.gauss(0.0, recurrent_scale) for _ in range(config.n_neurons)]
            for _ in range(config.n_neurons)
        ]
        self.bias = [self.rng.gauss(0.0, config.bias_scale) for _ in range(config.n_neurons)]
        self.membrane = [0.0 for _ in range(config.n_neurons)]
        self.spikes = [0.0 for _ in range(config.n_neurons)]
        self.spike_trace = [0.0 for _ in range(config.n_neurons)]
        self.pre_trace = [0.0 for _ in range(config.n_neurons)]

    def reset_state(self) -> None:
        for index in range(self.config.n_neurons):
            self.membrane[index] = 0.0
            self.spikes[index] = 0.0
            self.spike_trace[index] = 0.0
            self.pre_trace[index] = 0.0

    def step(self, inputs: list[float], modulation: float = 0.0) -> list[float]:
        previous_spikes = list(self.spikes)
        next_spikes = [0.0 for _ in range(self.config.n_neurons)]

        for neuron in range(self.config.n_neurons):
            current = dot(self.input_weights[neuron], inputs)
            current += dot(self.recurrent_weights[neuron], previous_spikes)
            current += self.bias[neuron]
            self.membrane[neuron] = self.config.membrane_decay * self.membrane[neuron] + current
            if self.membrane[neuron] >= self.config.threshold:
                next_spikes[neuron] = 1.0
                self.membrane[neuron] -= self.config.threshold

        self.spikes = next_spikes
        for neuron, spike in enumerate(next_spikes):
            self.spike_trace[neuron] = self.config.trace_decay * self.spike_trace[neuron] + spike
            self.pre_trace[neuron] = self.config.trace_decay * self.pre_trace[neuron] + previous_spikes[neuron]

        if modulation != 0.0 and self.config.plastic_lr > 0.0:
            self.apply_recurrent_modulation(modulation)

        return self.features()

    def features(self) -> list[float]:
        return [
            self.spike_trace[index] + 0.15 * max(0.0, self.membrane[index] / self.config.threshold)
            for index in range(self.config.n_neurons)
        ]

    def apply_recurrent_modulation(self, modulation: float) -> None:
        clipped_modulation = clamp(modulation, -1.0, 1.0)
        for post in range(self.config.n_neurons):
            post_factor = triangular_pseudo_derivative(self.membrane[post], self.config.threshold)
            for pre in range(self.config.n_neurons):
                update = clipped_modulation * post_factor * self.pre_trace[pre]
                weight = self.recurrent_weights[post][pre]
                weight += self.config.plastic_lr * update
                weight *= 1.0 - self.config.weight_decay
                self.recurrent_weights[post][pre] = clamp(weight, -1.5, 1.5)


def triangular_pseudo_derivative(value: float, threshold: float, width: float = 0.8) -> float:
    distance = abs(value - threshold)
    return max(0.05, 1.0 - distance / width)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
