"""Small math helpers shared by experiments."""

from __future__ import annotations

import math
from typing import Iterable


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def mean(values: Iterable[float]) -> float:
    value_list = list(values)
    return sum(value_list) / max(1, len(value_list))


def mean_squared(values: list[float]) -> float:
    return sum(value * value for value in values) / max(1, len(values))


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
