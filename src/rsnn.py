"""小型循环脉冲网络工具，供玩具控制实验复用。

该模块刻意保持无第三方依赖，只实现两件核心能力：

1. 一个近似 LIF 的循环脉冲状态更新过程；
2. 一个局部三因子递归可塑性接口，用于在运行时调节循环连接。

外部读出头可以直接使用 :meth:`features` 返回的隐藏特征，这些特征同时
包含了脉冲轨迹与膜电位信息，可视作一种近似 eligibility trace 的隐状态摘要。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class RSNNConfig:
    """R-SNN 的超参数配置。

    属性:
        input_dim: 输入向量维度。
        n_neurons: 隐层脉冲神经元数量。
        membrane_decay: 膜电位泄露衰减系数。
        trace_decay: 脉冲痕迹与前突触痕迹的低通衰减系数。
        threshold: 触发脉冲的阈值。
        recurrent_scale: 循环权重初始化尺度。
        input_scale: 输入权重初始化尺度。
        bias_scale: 偏置初始化尺度。
        plastic_lr: 局部循环可塑性的学习率。
        weight_decay: 循环权重更新后的衰减项。
        seed: 默认随机种子。
    """

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
    """紧凑的类 LIF 循环脉冲网络。

    该实现追求的是“足够可解释、足够轻量”的实验骨架，而不是严格的生物
    真实感。网络内部维护膜电位、当前脉冲、脉冲轨迹以及前突触轨迹，便于：

    1. 用循环连接形成时序隐状态；
    2. 用局部痕迹近似保存近期因果信息；
    3. 在外部调制信号到来时执行三因子式的循环权重更新。
    """

    def __init__(self, config: RSNNConfig, rng: random.Random | None = None) -> None:
        """初始化网络参数和内部状态。"""
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
        """清空单个 episode 内的动态状态，不重置权重。"""
        for index in range(self.config.n_neurons):
            self.membrane[index] = 0.0
            self.spikes[index] = 0.0
            self.spike_trace[index] = 0.0
            self.pre_trace[index] = 0.0

    def step(self, inputs: list[float], modulation: float = 0.0) -> list[float]:
        """推进一个时间步并返回当前隐藏特征。

        参数:
            inputs: 当前时刻输入。
            modulation: 可选的全局调制信号；非零时会触发循环连接更新。

        返回:
            供读出层使用的隐藏特征向量。
        """
        previous_spikes = list(self.spikes)
        next_spikes = [0.0 for _ in range(self.config.n_neurons)]

        # 先基于输入、电路递归输入与偏置更新膜电位，再决定是否发放脉冲。
        for neuron in range(self.config.n_neurons):
            current = dot(self.input_weights[neuron], inputs)
            current += dot(self.recurrent_weights[neuron], previous_spikes)
            current += self.bias[neuron]
            self.membrane[neuron] = self.config.membrane_decay * self.membrane[neuron] + current
            if self.membrane[neuron] >= self.config.threshold:
                next_spikes[neuron] = 1.0
                self.membrane[neuron] -= self.config.threshold

        self.spikes = next_spikes

        # 轨迹变量保留了最近一段时间内的活动强度，后续局部学习规则直接使用它们。
        for neuron, spike in enumerate(next_spikes):
            self.spike_trace[neuron] = self.config.trace_decay * self.spike_trace[neuron] + spike
            self.pre_trace[neuron] = self.config.trace_decay * self.pre_trace[neuron] + previous_spikes[neuron]

        if modulation != 0.0 and self.config.plastic_lr > 0.0:
            self.apply_recurrent_modulation(modulation)

        return self.features()

    def features(self) -> list[float]:
        """组合脉冲轨迹与膜电位，生成稳定的读出特征。"""
        return [
            self.spike_trace[index] + 0.15 * max(0.0, self.membrane[index] / self.config.threshold)
            for index in range(self.config.n_neurons)
        ]

    def apply_recurrent_modulation(self, modulation: float) -> None:
        """对循环权重执行局部三因子更新。

        更新形式可概括为：

            delta_w ~ modulation * post_factor * pre_trace

        其中：
        - `modulation` 表示来自任务层面的全局调制；
        - `post_factor` 用膜电位附近的伪导数近似后突触可塑性敏感度；
        - `pre_trace` 保存前突触近期活动。
        """
        clipped_modulation = clamp(modulation, -1.0, 1.0)

        # 这里逐个 post-pre 突触更新，不依赖反向传播图，适合作为在线局部学习钩子。
        for post in range(self.config.n_neurons):
            post_factor = triangular_pseudo_derivative(self.membrane[post], self.config.threshold)
            for pre in range(self.config.n_neurons):
                update = clipped_modulation * post_factor * self.pre_trace[pre]
                weight = self.recurrent_weights[post][pre]
                weight += self.config.plastic_lr * update
                weight *= 1.0 - self.config.weight_decay
                self.recurrent_weights[post][pre] = clamp(weight, -1.5, 1.5)


def triangular_pseudo_derivative(value: float, threshold: float, width: float = 0.8) -> float:
    """计算三角形伪导数。

    当膜电位靠近阈值时返回较大值，使局部学习更关注“差一点发放”的神经元。
    """
    distance = abs(value - threshold)
    return max(0.05, 1.0 - distance / width)


def dot(left: list[float], right: list[float]) -> float:
    """计算两个等长向量的点积。"""
    return sum(a * b for a, b in zip(left, right))


def clamp(value: float, low: float, high: float) -> float:
    """将标量裁剪到指定闭区间。"""
    return max(low, min(high, value))
