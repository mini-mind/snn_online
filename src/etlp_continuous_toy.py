"""连续输入流上的 ETLP 风格局部可塑性玩具实验。

运行方式:
    python src/etlp_continuous_toy.py

这个脚本刻意做得很小，并且完全无第三方依赖。它不是 ETLP 论文的严格复现，
而是保留其核心约束的研究玩具：

    update = pre_trace * post_membrane_factor * teaching_signal

这里的输入不是二值脉冲，而是连续传感信号。相应地，`pre_trace` 不再表示
离散事件计数，而是一个低通滤波后的模拟痕迹，更接近真实在线传感流或上游
潜变量特征。
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


@dataclass
class ToyConfig:
    """连续序列分类实验的配置。"""

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
    """带缓慢分布漂移的二分类连续序列任务。

    每个类别都对应一条原型轨迹，训练时会在原型上叠加噪声与逐步漂移，从而
    逼迫模型在在线学习过程中持续适应，而不是记住静态模板。
    """

    def __init__(self, config: ToyConfig, rng: random.Random) -> None:
        """预构建类别原型序列。"""
        self.config = config
        self.rng = rng
        self.prototypes = self._build_prototypes()

    def sample(self, step: int) -> tuple[list[list[float]], int]:
        """采样一条序列及其标签。

        `step` 会控制漂移相位，使任务分布随训练进程缓慢变化。
        """
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

        # 漂移随时间步递增，意味着同一条序列内部也存在上下文相关扰动。
        sequence: list[list[float]] = []
        for t, prototype_step in enumerate(self.prototypes[label]):
            drift_scale = self.config.drift * t / max(1, self.config.seq_len - 1)
            analog_input = []
            for feature, drift in zip(prototype_step, drift_vec):
                value = feature + drift_scale * drift + self.rng.gauss(0.0, self.config.noise)
                analog_input.append(max(0.0, value))
            sequence.append(analog_input)
        return sequence, label

    def _build_prototypes(self) -> list[list[list[float]]]:
        """构建两类原型序列。

        两个类别共享时间上下文特征，但在“早峰值”和“晚峰值”通道上有不同
        强度，从而形成一个需要时序整合才能区分的任务。
        """
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


class ETLPContinuousClassifier:
    """单层类 LIF 分类器，使用 ETLP 风格局部规则训练。"""

    def __init__(self, config: ToyConfig, rng: random.Random) -> None:
        """初始化读出层权重和偏置。"""
        self.config = config
        self.weights = [
            [rng.gauss(0.0, 0.18) for _ in range(config.input_dim)]
            for _ in range(config.n_classes)
        ]
        self.bias = [0.0 for _ in range(config.n_classes)]

    def predict(self, sequence: list[list[float]]) -> int:
        """执行前向推理并返回预测类别。"""
        logits = self._run(sequence, label=None, learn=False)
        return argmax(logits)

    def train_one(self, sequence: list[list[float]], label: int) -> int:
        """对单个序列执行一次在线训练，并返回训练时的预测类别。"""
        logits = self._run(sequence, label=label, learn=True)
        return argmax(logits)

    def weight_norm(self) -> float:
        """计算权重矩阵的 L2 范数，用于监控训练是否发散。"""
        total = 0.0
        for row in self.weights:
            for weight in row:
                total += weight * weight
        return math.sqrt(total)

    def _run(self, sequence: list[list[float]], label: int | None, learn: bool) -> list[float]:
        """在整条序列上运行神经元动力学，并按需执行局部学习。"""
        membrane = [0.0 for _ in range(self.config.n_classes)]
        pre_trace = [0.0 for _ in range(self.config.input_dim)]
        readout = [0.0 for _ in range(self.config.n_classes)]
        target = None
        if label is not None:
            target = [0.0 for _ in range(self.config.n_classes)]
            target[label] = 1.0

        for analog_input in sequence:
            # 输入痕迹是局部学习规则中的前突触项，累计了近期连续输入强度。
            for input_index, value in enumerate(analog_input):
                pre_trace[input_index] = self.config.trace_decay * pre_trace[input_index] + value

            # 每个类别神经元各自维护膜电位与读出积累值。
            for class_index in range(self.config.n_classes):
                current = dot(self.weights[class_index], analog_input) + self.bias[class_index]
                membrane[class_index] = self.config.membrane_decay * membrane[class_index] + current
                spiked = membrane[class_index] >= self.config.threshold
                readout[class_index] = 0.88 * readout[class_index] + membrane[class_index] + (0.5 if spiked else 0.0)
                if spiked:
                    membrane[class_index] -= 0.5 * self.config.threshold

            if learn and target is not None:
                # 教学信号来自目标分布与当前 softmax 概率之差，保持监督信号简单直观。
                probabilities = softmax(readout)
                for class_index in range(self.config.n_classes):
                    teaching_signal = target[class_index] - probabilities[class_index]
                    post_factor = triangular_pseudo_derivative(
                        membrane[class_index],
                        threshold=self.config.threshold,
                        width=self.config.pseudo_width,
                    )
                    for input_index in range(self.config.input_dim):
                        # 这里直接体现三因子形式：
                        # teaching_signal * post_factor * pre_trace
                        local_update = teaching_signal * post_factor * pre_trace[input_index]
                        weight = self.weights[class_index][input_index]
                        weight += self.config.lr * local_update
                        weight *= 1.0 - self.config.weight_decay
                        self.weights[class_index][input_index] = clamp(weight, -3.0, 3.0)
                    self.bias[class_index] += self.config.lr * teaching_signal * post_factor

        return readout


def dot(left: list[float], right: list[float]) -> float:
    """计算点积。"""
    return sum(a * b for a, b in zip(left, right))


def argmax(values: list[float]) -> int:
    """返回最大元素的索引。"""
    return max(range(len(values)), key=lambda index: values[index])


def clamp(value: float, low: float, high: float) -> float:
    """裁剪标量。"""
    return max(low, min(high, value))


def triangular_pseudo_derivative(value: float, threshold: float, width: float) -> float:
    """在阈值附近给出较大响应的三角形伪导数近似。"""
    distance = abs(value - threshold)
    return max(0.20, 1.0 - distance / width)


def softmax(values: list[float]) -> list[float]:
    """计算数值稳定版 softmax。"""
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def evaluate(model: ETLPContinuousClassifier, task: ContinuousTemporalTask, step: int, samples: int) -> float:
    """在给定步数对应的数据分布上评估分类准确率。"""
    correct = 0
    for offset in range(samples):
        sequence, label = task.sample(step + offset)
        correct += int(model.predict(sequence) == label)
    return correct / samples


def run(config: ToyConfig) -> None:
    """运行完整训练与周期性评估流程。"""
    rng = random.Random(config.seed)
    task = ContinuousTemporalTask(config, rng)
    model = ETLPContinuousClassifier(config, rng)

    print("ETLP-like continuous-input toy")
    print(f"seed={config.seed} train_steps={config.train_steps} seq_len={config.seq_len}")
    print("rule: delta_w = lr * pre_trace * post_membrane_factor * teaching_signal")
    print()

    initial_accuracy = evaluate(model, task, step=0, samples=config.eval_samples)
    print(f"step=0 eval_accuracy={initial_accuracy:.3f}")

    # `online_correct` 衡量从训练开始累积的在线预测表现；
    # `window_correct` 则帮助观察最近一段训练窗口的局部适应效果。
    online_correct = 0
    window_correct = 0
    for step in range(1, config.train_steps + 1):
        sequence, label = task.sample(step)
        prediction = model.train_one(sequence, label)
        online_correct += int(prediction == label)
        window_correct += int(prediction == label)

        if step % config.eval_every == 0:
            eval_accuracy = evaluate(model, task, step=step, samples=config.eval_samples)
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


def parse_args() -> ToyConfig:
    """解析命令行参数并生成实验配置。"""
    parser = argparse.ArgumentParser(description="Run the ETLP-like continuous-input toy.")
    parser.add_argument("--train-steps", type=int, default=ToyConfig.train_steps)
    parser.add_argument("--eval-every", type=int, default=ToyConfig.eval_every)
    parser.add_argument("--eval-samples", type=int, default=ToyConfig.eval_samples)
    parser.add_argument("--lr", type=float, default=ToyConfig.lr)
    parser.add_argument("--noise", type=float, default=ToyConfig.noise)
    parser.add_argument("--drift", type=float, default=ToyConfig.drift)
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    args = parser.parse_args()
    return ToyConfig(
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        lr=args.lr,
        noise=args.noise,
        drift=args.drift,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
