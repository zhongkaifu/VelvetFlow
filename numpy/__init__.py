"""Lightweight numpy stub for offline test execution."""

from __future__ import annotations

import math
from typing import Iterable, List


float32 = float


def array(values: Iterable[float], dtype=None) -> List[float]:
    return [float(v) for v in values]


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class _Linalg:
    @staticmethod
    def norm(v: Iterable[float]) -> float:
        return math.sqrt(sum(x * x for x in v))


linalg = _Linalg()


__all__ = ["array", "dot", "linalg", "float32"]
